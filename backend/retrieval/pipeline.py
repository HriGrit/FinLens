"""
pipeline.py — Seam between ingestion and generation.

retrieve_and_rerank() is the single public entry point called by the API.
"""
from __future__ import annotations

from llama_index.core.schema import TextNode

from observability.tracing import span
from .hybrid import hybrid_retrieve
from .rerank import rerank

DEFAULT_RETRIEVAL_TOP_K = 20
DEFAULT_RERANK_TOP_K = 5


def retrieve_and_rerank(
    query: str,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    company: str | None = None,
    year: str | None = None,
    trace=None,
) -> tuple[list[TextNode], int]:
    """Hybrid-retrieve then cross-encoder rerank. Returns rerank_top_k nodes."""
    if trace is not None:
        with span(trace, "hybrid_retrieve", input={"query": query, "top_k": retrieval_top_k}):
            candidates = hybrid_retrieve(query, top_k=retrieval_top_k, company=company, year=year)
    else:
        candidates = hybrid_retrieve(query, top_k=retrieval_top_k, company=company, year=year)

    if not candidates:
        return [], 0

    if trace is not None:
        with span(trace, "cross_encoder_rerank", input={"n_candidates": len(candidates)}):
            return rerank(query, candidates, top_k=rerank_top_k), len(candidates)
    return rerank(query, candidates, top_k=rerank_top_k), len(candidates)
