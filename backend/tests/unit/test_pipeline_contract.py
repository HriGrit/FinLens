from __future__ import annotations

from llama_index.core.schema import TextNode

from retrieval import pipeline
from retrieval.pipeline import RetrievalResult


def _node(text: str) -> TextNode:
    return TextNode(text=text, metadata={})


def test_retrieve_and_rerank_returns_named_contract(monkeypatch) -> None:
    candidates = [_node("a"), _node("b")]
    reranked = [_node("b")]

    monkeypatch.setattr(pipeline, "hybrid_retrieve", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr(pipeline, "rerank", lambda *_args, **_kwargs: reranked)

    result = pipeline.retrieve_and_rerank(query="q", retrieval_top_k=2, rerank_top_k=1)

    assert isinstance(result, RetrievalResult)
    assert result.nodes == reranked
    assert result.candidate_count == 2


def test_retrieve_and_rerank_empty_candidates_contract(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "hybrid_retrieve", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(pipeline, "rerank", lambda *_args, **_kwargs: [_node("should-not-run")])

    result = pipeline.retrieve_and_rerank(query="q")

    assert isinstance(result, RetrievalResult)
    assert result.nodes == []
    assert result.candidate_count == 0
