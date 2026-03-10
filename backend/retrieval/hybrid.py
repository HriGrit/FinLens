"""
hybrid.py — Hybrid BM25 + Qdrant retriever with Reciprocal Rank Fusion.

Combines sparse (BM25) and dense (Qdrant cosine) retrieval, fuses with RRF,
and returns a deduplicated ranked list of TextNodes.
"""
from __future__ import annotations

from pathlib import Path

from llama_index.core.schema import TextNode
from shared.qdrant import QDRANT_COLLECTION, get_qdrant_client
BM25_INDEX_PATH = Path(__file__).resolve().parents[2] / "data" / "bm25_index.pkl"
RRF_K = 60  # standard RRF constant


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def _fuse_results(
    bm25_hits: list[TextNode],
    dense_hits: list[TextNode],
) -> list[TextNode]:
    """Reciprocal Rank Fusion over two ranked lists, deduplicated by text."""
    scores: dict[str, float] = {}
    nodes_by_key: dict[str, TextNode] = {}

    for rank, node in enumerate(bm25_hits, 1):
        key = node.text
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        nodes_by_key[key] = node

    for rank, node in enumerate(dense_hits, 1):
        key = node.text
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        nodes_by_key[key] = node

    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [nodes_by_key[k] for k in ranked_keys]


def hybrid_retrieve(
    query: str,
    top_k: int = 20,
    company: str | None = None,
    year: str | None = None,
) -> list[TextNode]:
    """Run BM25 + Qdrant retrieval, fuse with RRF, return top_k nodes."""
    from ingestion.index import load_bm25_index
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    from ingestion.embed import get_embed_model

    # --- BM25 ---
    bm25_nodes: list[TextNode] = []
    if BM25_INDEX_PATH.exists():
        try:
            bm25_retriever = load_bm25_index(BM25_INDEX_PATH)
            if bm25_retriever.bm25:                                      # B-3
                bm25_corpus_size = bm25_retriever.bm25.corpus_size
                if bm25_corpus_size > 0:
                    bm25_retriever.similarity_top_k = min(top_k, bm25_corpus_size)
            bm25_results = bm25_retriever.retrieve(query)
            bm25_nodes = [r.node for r in bm25_results]
            if company:                                                   # B-1
                bm25_nodes = [n for n in bm25_nodes if n.metadata.get("company") == company]
            if year:
                bm25_nodes = [n for n in bm25_nodes if n.metadata.get("year") == year]
        except Exception as exc:                                          # B-2
            import warnings
            warnings.warn(
                f"BM25 index load failed ({exc}); falling back to dense-only retrieval.",
                stacklevel=2,
            )

    # --- Qdrant dense ---
    dense_nodes: list[TextNode] = []
    embed_model = get_embed_model()
    query_embedding = embed_model.get_text_embedding(query)

    client = get_qdrant_client()

    filter_conditions = []
    if company:
        filter_conditions.append(FieldCondition(key="company", match=MatchValue(value=company)))
    if year:
        filter_conditions.append(FieldCondition(key="year", match=MatchValue(value=year)))

    qdrant_filter = Filter(must=filter_conditions) if filter_conditions else None

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        using="dense",
        query_filter=qdrant_filter,
        limit=top_k,
    )

    for r in results.points:                                             # B-4
        text = r.payload.get("text", "")
        dense_nodes.append(
            TextNode(text=text, metadata={k: v for k, v in r.payload.items() if k != "text"})
        )

    # --- Fuse ---
    return _fuse_results(bm25_nodes, dense_nodes)[:top_k]
