"""
Unit tests for retrieval/hybrid.py — Group B bug fixes (B-1 through B-4).

All external dependencies (Qdrant, BM25 pickle, embed model) are fully mocked.
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import TextNode

import retrieval.hybrid as hybrid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scored_point(text: str, metadata: dict) -> MagicMock:
    """Return a mock ScoredPoint whose payload mirrors a real Qdrant payload."""
    sp = MagicMock()
    sp.payload = {"text": text, **metadata}
    return sp


def _make_bm25_node(text: str, metadata: dict) -> TextNode:
    return TextNode(text=text, metadata=metadata)


def _make_bm25_retriever(nodes: list[TextNode], corpus_size: int = 100) -> MagicMock:
    retriever = MagicMock()
    retriever.bm25 = MagicMock()
    retriever.bm25.corpus_size = corpus_size
    retriever.retrieve.return_value = [MagicMock(node=n) for n in nodes]
    retriever.similarity_top_k = 20
    return retriever


def _qdrant_client_mock(points: list[MagicMock]) -> MagicMock:
    """Return a mock QdrantClient whose query_points returns given points."""
    client = MagicMock()
    result = MagicMock()
    result.points = points
    client.query_points.return_value = result
    return client


# ---------------------------------------------------------------------------
# B-1: BM25 nodes filtered by company
# ---------------------------------------------------------------------------

def test_bm25_nodes_filtered_by_company(monkeypatch, tmp_path):
    """Nodes with a non-matching company must be excluded from BM25 hits."""
    fake_pkl = tmp_path / "bm25_index.pkl"
    fake_pkl.touch()
    monkeypatch.setattr(hybrid, "BM25_INDEX_PATH", fake_pkl)

    bm25_nodes = [
        _make_bm25_node("correct node", {"company": "3M", "year": "2022"}),
        _make_bm25_node("wrong company", {"company": "Apple", "year": "2022"}),
    ]
    retriever = _make_bm25_retriever(bm25_nodes)

    embed_mock = MagicMock()
    embed_mock.get_text_embedding.return_value = [0.1] * 768
    qdrant_mock = _qdrant_client_mock(
        [_make_scored_point("dense node", {"company": "3M", "year": "2022"})]
    )

    with (
        patch("ingestion.index.load_bm25_index", return_value=retriever),
        patch("ingestion.embed.get_embed_model", return_value=embed_mock),
        patch("retrieval.hybrid.get_qdrant_client", return_value=qdrant_mock),
    ):
        result = hybrid.hybrid_retrieve("net sales", top_k=10, company="3M", year="2022")

    texts = [n.text for n in result]
    assert "correct node" in texts
    assert "wrong company" not in texts


# ---------------------------------------------------------------------------
# B-1: BM25 nodes filtered by year
# ---------------------------------------------------------------------------

def test_bm25_nodes_filtered_by_year(monkeypatch, tmp_path):
    """Nodes with a non-matching year must be excluded from BM25 hits."""
    fake_pkl = tmp_path / "bm25_index.pkl"
    fake_pkl.touch()
    monkeypatch.setattr(hybrid, "BM25_INDEX_PATH", fake_pkl)

    bm25_nodes = [
        _make_bm25_node("correct year", {"company": "3M", "year": "2022"}),
        _make_bm25_node("wrong year",   {"company": "3M", "year": "2021"}),
    ]
    retriever = _make_bm25_retriever(bm25_nodes)

    embed_mock = MagicMock()
    embed_mock.get_text_embedding.return_value = [0.1] * 768
    qdrant_mock = _qdrant_client_mock(
        [_make_scored_point("dense node", {"company": "3M", "year": "2022"})]
    )

    with (
        patch("ingestion.index.load_bm25_index", return_value=retriever),
        patch("ingestion.embed.get_embed_model", return_value=embed_mock),
        patch("retrieval.hybrid.get_qdrant_client", return_value=qdrant_mock),
    ):
        result = hybrid.hybrid_retrieve("net sales", top_k=10, company="3M", year="2022")

    texts = [n.text for n in result]
    assert "correct year" in texts
    assert "wrong year" not in texts


# ---------------------------------------------------------------------------
# B-2: Corrupt BM25 falls back to dense-only
# ---------------------------------------------------------------------------

def test_corrupt_bm25_falls_back_to_dense_only(monkeypatch, tmp_path):
    """If load_bm25_index raises, hybrid_retrieve must still return dense results."""
    fake_pkl = tmp_path / "bm25_index.pkl"
    fake_pkl.touch()
    monkeypatch.setattr(hybrid, "BM25_INDEX_PATH", fake_pkl)

    embed_mock = MagicMock()
    embed_mock.get_text_embedding.return_value = [0.1] * 768
    qdrant_mock = _qdrant_client_mock(
        [_make_scored_point("dense only result", {"company": "3M", "year": "2022"})]
    )

    with (
        patch("ingestion.index.load_bm25_index", side_effect=ValueError("corrupt pickle")),
        patch("ingestion.embed.get_embed_model", return_value=embed_mock),
        patch("retrieval.hybrid.get_qdrant_client", return_value=qdrant_mock),
        warnings.catch_warnings(record=True) as w,
    ):
        warnings.simplefilter("always")
        result = hybrid.hybrid_retrieve("net sales", top_k=10)

    assert any("BM25 index load failed" in str(warning.message) for warning in w)
    assert len(result) >= 1
    assert result[0].text == "dense only result"


# ---------------------------------------------------------------------------
# B-3: corpus_size caps similarity_top_k
# ---------------------------------------------------------------------------

def test_corpus_size_caps_similarity_top_k(monkeypatch, tmp_path):
    """similarity_top_k must be set to min(top_k, corpus_size) when corpus is small."""
    fake_pkl = tmp_path / "bm25_index.pkl"
    fake_pkl.touch()
    monkeypatch.setattr(hybrid, "BM25_INDEX_PATH", fake_pkl)

    small_corpus_size = 3
    retriever = _make_bm25_retriever([], corpus_size=small_corpus_size)

    embed_mock = MagicMock()
    embed_mock.get_text_embedding.return_value = [0.1] * 768
    qdrant_mock = _qdrant_client_mock([])

    with (
        patch("ingestion.index.load_bm25_index", return_value=retriever),
        patch("ingestion.embed.get_embed_model", return_value=embed_mock),
        patch("retrieval.hybrid.get_qdrant_client", return_value=qdrant_mock),
    ):
        hybrid.hybrid_retrieve("net sales", top_k=20)

    assert retriever.similarity_top_k == min(20, small_corpus_size)


# ---------------------------------------------------------------------------
# B-4: Qdrant payload not mutated
# ---------------------------------------------------------------------------

def test_qdrant_payload_not_mutated(monkeypatch, tmp_path):
    """After hybrid_retrieve, original ScoredPoint.payload must still contain 'text'."""
    # No BM25 index — keep it simple
    monkeypatch.setattr(hybrid, "BM25_INDEX_PATH", tmp_path / "nonexistent.pkl")

    sp = _make_scored_point("dense payload text", {"company": "3M", "year": "2022"})
    original_text = sp.payload["text"]

    embed_mock = MagicMock()
    embed_mock.get_text_embedding.return_value = [0.1] * 768
    qdrant_mock = _qdrant_client_mock([sp])

    with (
        patch("ingestion.embed.get_embed_model", return_value=embed_mock),
        patch("retrieval.hybrid.get_qdrant_client", return_value=qdrant_mock),
    ):
        hybrid.hybrid_retrieve("net sales", top_k=10)

    # payload must not have been mutated — "text" key must still be present
    assert sp.payload.get("text") == original_text
