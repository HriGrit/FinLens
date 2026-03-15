"""
Unit tests for backend/ingestion/index.py

Covers:
  Q1 — non-completed upsert status raises RuntimeError
  Q2 — embedding batch shorter than nodes raises ValueError
  Q3 — None metadata field in point-ID construction raises ValueError
  I5 — BM25 save/load round-trip preserves ranking; unrecognized version raises
"""
from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import TextNode

from ingestion.index import (
    _BM25_FORMAT_VERSION,
    build_bm25_index,
    build_qdrant_index,
    load_bm25_index,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_node(
    text: str = "Revenue grew.",
    *,
    filename: str = "3M_2022_10K.pdf",
    company: str = "3M",
    year: str = "2022",
    doc_type: str = "10-K",
    page_number: int | None = 1,
    element_type: str = "paragraph",
    chunk_index: int = 0,
) -> TextNode:
    return TextNode(
        text=text,
        metadata={
            "filename": filename,
            "company": company,
            "year": year,
            "doc_type": doc_type,
            "page_number": page_number,
            "element_type": element_type,
            "chunk_index": chunk_index,
        },
    )


def _embedding(dim: int = 4) -> list[float]:
    return [0.1] * dim


# ---------------------------------------------------------------------------
# Q1 — Non-completed upsert status raises RuntimeError
# ---------------------------------------------------------------------------

def test_upsert_non_completed_status_raises(tmp_path):
    """Q1: build_qdrant_index raises RuntimeError when Qdrant reports a non-completed upsert."""
    node = _make_node()

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    # Simulate a non-completed upsert result
    mock_upsert_result = MagicMock()
    mock_upsert_result.status.value = "failed"
    mock_client.upsert.return_value = mock_upsert_result

    mock_embed_model = MagicMock()
    mock_embed_model.get_text_embedding.return_value = _embedding()
    mock_embed_model.get_text_embedding_batch.return_value = [_embedding()]

    with (
        patch("ingestion.index.get_qdrant_client", return_value=mock_client),
        patch("ingestion.embed.get_embed_model", return_value=mock_embed_model),
    ):
        with pytest.raises(RuntimeError, match="Qdrant upsert did not complete"):
            build_qdrant_index([node])


def test_upsert_completed_status_does_not_raise():
    """Q1: build_qdrant_index does NOT raise when Qdrant reports completed status."""
    node = _make_node()

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    mock_upsert_result = MagicMock()
    mock_upsert_result.status.value = "completed"
    mock_client.upsert.return_value = mock_upsert_result

    mock_embed_model = MagicMock()
    mock_embed_model.get_text_embedding.return_value = _embedding()
    mock_embed_model.get_text_embedding_batch.return_value = [_embedding()]

    with (
        patch("ingestion.index.get_qdrant_client", return_value=mock_client),
        patch("ingestion.embed.get_embed_model", return_value=mock_embed_model),
    ):
        build_qdrant_index([node])  # must not raise


def test_upsert_no_status_attribute_does_not_raise():
    """Q1: build_qdrant_index is permissive when result has no status attribute."""
    node = _make_node()

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    # Return value with no .status attribute
    mock_upsert_result = MagicMock(spec=[])  # spec=[] means no attributes
    mock_client.upsert.return_value = mock_upsert_result

    mock_embed_model = MagicMock()
    mock_embed_model.get_text_embedding.return_value = _embedding()
    mock_embed_model.get_text_embedding_batch.return_value = [_embedding()]

    with (
        patch("ingestion.index.get_qdrant_client", return_value=mock_client),
        patch("ingestion.embed.get_embed_model", return_value=mock_embed_model),
    ):
        build_qdrant_index([node])  # must not raise


# ---------------------------------------------------------------------------
# Q2 — Embedding batch length mismatch raises ValueError
# ---------------------------------------------------------------------------

def test_embedding_batch_shorter_than_nodes_raises():
    """Q2: build_qdrant_index raises ValueError when embeddings count != nodes count."""
    nodes = [_make_node("Text one."), _make_node("Text two.")]

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    mock_embed_model = MagicMock()
    mock_embed_model.get_text_embedding.return_value = _embedding()
    # Only one embedding for two nodes
    mock_embed_model.get_text_embedding_batch.return_value = [_embedding()]

    with (
        patch("ingestion.index.get_qdrant_client", return_value=mock_client),
        patch("ingestion.embed.get_embed_model", return_value=mock_embed_model),
    ):
        with pytest.raises(ValueError, match="Embedding batch length mismatch"):
            build_qdrant_index(nodes)


def test_embedding_batch_correct_length_does_not_raise():
    """Q2: build_qdrant_index proceeds normally when embeddings count matches nodes count."""
    nodes = [_make_node("Text one."), _make_node("Text two.")]

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    mock_upsert_result = MagicMock()
    mock_upsert_result.status.value = "completed"
    mock_client.upsert.return_value = mock_upsert_result

    mock_embed_model = MagicMock()
    mock_embed_model.get_text_embedding.return_value = _embedding()
    mock_embed_model.get_text_embedding_batch.return_value = [_embedding(), _embedding()]

    with (
        patch("ingestion.index.get_qdrant_client", return_value=mock_client),
        patch("ingestion.embed.get_embed_model", return_value=mock_embed_model),
    ):
        build_qdrant_index(nodes)  # must not raise


# ---------------------------------------------------------------------------
# Q3 — None metadata field in point-ID construction raises ValueError
# ---------------------------------------------------------------------------

def test_none_page_number_raises_before_upsert():
    """Q3: build_qdrant_index raises ValueError when page_number is None."""
    node = _make_node(page_number=None)

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    mock_embed_model = MagicMock()
    mock_embed_model.get_text_embedding.return_value = _embedding()
    mock_embed_model.get_text_embedding_batch.return_value = [_embedding()]

    with (
        patch("ingestion.index.get_qdrant_client", return_value=mock_client),
        patch("ingestion.embed.get_embed_model", return_value=mock_embed_model),
    ):
        with pytest.raises(ValueError, match="point-ID fields"):
            build_qdrant_index([node])


def test_none_filename_raises_before_upsert():
    """Q3: build_qdrant_index raises ValueError when filename is None."""
    node = _make_node(filename=None)  # type: ignore[arg-type]

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    mock_embed_model = MagicMock()
    mock_embed_model.get_text_embedding.return_value = _embedding()
    mock_embed_model.get_text_embedding_batch.return_value = [_embedding()]

    with (
        patch("ingestion.index.get_qdrant_client", return_value=mock_client),
        patch("ingestion.embed.get_embed_model", return_value=mock_embed_model),
    ):
        with pytest.raises(ValueError, match="point-ID fields"):
            build_qdrant_index([node])


# ---------------------------------------------------------------------------
# I5 — BM25 save/load round-trip preserves ranking; unrecognized version raises
# ---------------------------------------------------------------------------

def test_bm25_round_trip_preserves_retriever(tmp_path: Path):
    """I5: save + load produces a functional BM25Retriever that can retrieve nodes."""
    nodes = [
        _make_node("Revenue grew significantly in Q4.", chunk_index=0),
        _make_node("Net sales increased by 12 percent.", chunk_index=1),
        _make_node("Operating expenses decreased.", chunk_index=2),
    ]
    index_path = tmp_path / "bm25_index.pkl"
    build_bm25_index(nodes, output_path=index_path)

    retriever = load_bm25_index(index_path)
    results = retriever.retrieve("revenue sales")
    assert len(results) > 0, "Loaded BM25 retriever should return results for a relevant query"


def test_bm25_saved_artifact_contains_version_key(tmp_path: Path):
    """I5: The saved artifact dict must include a 'version' key equal to _BM25_FORMAT_VERSION."""
    nodes = [_make_node()]
    index_path = tmp_path / "bm25_index.pkl"
    build_bm25_index(nodes, output_path=index_path)

    with open(index_path, "rb") as f:
        payload = pickle.load(f)

    assert isinstance(payload, dict), "Artifact must be a dict"
    assert payload.get("version") == _BM25_FORMAT_VERSION, (
        f"Expected version key {_BM25_FORMAT_VERSION!r}, got {payload.get('version')!r}"
    )


def test_bm25_load_raises_for_unrecognized_version(tmp_path: Path):
    """I5: load_bm25_index raises ValueError for an artifact with an unrecognized version."""
    index_path = tmp_path / "bm25_index.pkl"
    bad_payload = {"version": "v99", "nodes": []}
    with open(index_path, "wb") as f:
        pickle.dump(bad_payload, f)

    with pytest.raises(ValueError, match="Unrecognized BM25 index version"):
        load_bm25_index(index_path)


def test_bm25_load_raises_for_legacy_dict_without_version(tmp_path: Path):
    """I5: load_bm25_index raises ValueError for an old-format dict missing 'version'."""
    index_path = tmp_path / "bm25_index.pkl"
    legacy_payload = {"nodes": [], "similarity_top_k": 10}  # no 'version' key
    with open(index_path, "wb") as f:
        pickle.dump(legacy_payload, f)

    with pytest.raises(ValueError, match="Unrecognized BM25 index version"):
        load_bm25_index(index_path)


def test_bm25_load_raises_for_non_dict_payload(tmp_path: Path):
    """I5: load_bm25_index raises ValueError when the pickle contains a non-dict object."""
    index_path = tmp_path / "bm25_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump("not a dict", f)

    with pytest.raises(ValueError, match="expected a dict"):
        load_bm25_index(index_path)
