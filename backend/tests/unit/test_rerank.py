from __future__ import annotations

from llama_index.core.schema import TextNode

import pytest

from retrieval.rerank import rerank


class _FakeScores(list):
    def tolist(self):
        return list(self)


class _FakeCrossEncoder:
    def __init__(self, scores: list[float]):
        self._scores = scores

    def predict(self, _pairs):
        return _FakeScores(self._scores)


def test_rerank_orders_nodes_by_cross_encoder_score(monkeypatch):
    nodes = [
        TextNode(text="first"),
        TextNode(text="second"),
        TextNode(text="third"),
    ]

    monkeypatch.setattr(
        "retrieval.rerank._get_cross_encoder",
        lambda: _FakeCrossEncoder([0.1, 0.9, 0.3]),
    )

    reranked = rerank("query", nodes, top_k=2)

    assert [node.text for node in reranked] == ["second", "third"]


def test_rerank_returns_empty_for_no_nodes(monkeypatch):
    # _get_cross_encoder should not be called when there are no nodes.
    monkeypatch.setattr("retrieval.rerank._get_cross_encoder", lambda: None)
    assert rerank("query", []) == []
