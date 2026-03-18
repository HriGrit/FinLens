"""
rerank.py — Cross-encoder reranker with lazy model loading.

Milestone coverage: M0.6 (cross-encoder reranker).
"""
from __future__ import annotations

import os

from llama_index.core.schema import TextNode

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = None
_RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() in {"1", "true", "yes", "on"}


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def rerank(query: str, nodes: list[TextNode], top_k: int = 5) -> list[TextNode]:
    """Score query-node pairs with cross-encoder and return top_k by descending score."""
    if not _RERANK_ENABLED:
        return nodes[:top_k]

    if not nodes:
        return []

    model = _get_cross_encoder()
    pairs = [(query, node.text) for node in nodes]
    scores: list[float] = model.predict(pairs).tolist()

    scored = sorted(zip(scores, nodes), key=lambda x: x[0], reverse=True)
    return [node for _, node in scored[:top_k]]
