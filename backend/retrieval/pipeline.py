"""
pipeline.py — Seam between ingestion and generation.

retrieve_and_rerank() is the single public entry point called by the API.
"""
from __future__ import annotations

from dataclasses import dataclass

from llama_index.core.schema import TextNode

from .hybrid import hybrid_retrieve
from .rerank import rerank

DEFAULT_RETRIEVAL_TOP_K = 20
DEFAULT_RERANK_TOP_K = 5


@dataclass(frozen=True)
class RetrievalResult:
    """Stable retrieval contract for downstream callers."""

    nodes: list[TextNode]
    candidate_count: int


def retrieve_and_rerank(
    query: str,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    company: str | None = None,
    year: str | None = None,
    trace=None,
) -> RetrievalResult:
    """Hybrid-retrieve then cross-encoder rerank."""
    if trace is not None:
        s = trace.span(name="hybrid_retrieve", input={"query": query, "top_k": retrieval_top_k})
        candidates = hybrid_retrieve(query, top_k=retrieval_top_k, company=company, year=year)
        s.end(output={"n_candidates": len(candidates)})
    else:
        candidates = hybrid_retrieve(query, top_k=retrieval_top_k, company=company, year=year)

    if not candidates:
        return RetrievalResult(nodes=[], candidate_count=0)

    if trace is not None:
        s = trace.span(name="cross_encoder_rerank", input={"n_candidates": len(candidates)})
        results = rerank(query, candidates, top_k=rerank_top_k)
        s.end(output={"n_results": len(results)})
        return RetrievalResult(nodes=results, candidate_count=len(candidates))
    return RetrievalResult(
        nodes=rerank(query, candidates, top_k=rerank_top_k),
        candidate_count=len(candidates),
    )
