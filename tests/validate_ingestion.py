"""
Post-ingestion smoke test — validates Qdrant collection + BM25 index after run_ingestion.py.

Run with: cd backend && uv run python ../tests/validate_ingestion.py
Requires: run_ingestion.py completed successfully and Qdrant reachable at configured QDRANT_URL.
"""
from __future__ import annotations

from pathlib import Path

from ingestion.embed import get_embed_model
from ingestion.index import load_bm25_index
from qdrant_client.models import FieldCondition, Filter, MatchValue
from shared.qdrant import get_qdrant_client, get_qdrant_collection

REPO_ROOT = Path(__file__).resolve().parents[1]
BM25_PATH = REPO_ROOT / "data" / "bm25_index.pkl"
REQUIRED_METADATA = {"element_type", "page_number", "filename", "company", "year", "doc_type"}


def validate_ingestion() -> None:
    client = get_qdrant_client()
    collection = get_qdrant_collection()

    # --- 1. Collection exists ---
    assert client.collection_exists(collection), f"Collection '{collection}' not found in Qdrant"
    info = client.get_collection(collection)
    point_count = info.points_count
    assert point_count > 0, "Collection exists but has 0 points"
    print(f"[PASS] Collection '{collection}' exists with {point_count} points.")

    # --- 2. Dense query with payload filter ---
    embedding_dim = len(get_embed_model().get_text_embedding("test"))
    results = client.query_points(
        collection_name=collection,
        query=[0.0] * embedding_dim,  # zero vector — validates filter + retrieval plumbing
        using="dense",
        query_filter=Filter(
            must=[
                FieldCondition(key="company", match=MatchValue(value="3M")),
                FieldCondition(key="year", match=MatchValue(value="2022")),
            ]
        ),
        limit=5,
    )
    assert len(results.points) > 0, "Query with company=3M year=2022 filter returned 0 results"
    print(f"[PASS] Filtered query (company=3M, year=2022) returned {len(results.points)} result(s).")

    # --- 3. Metadata completeness ---
    for point in results.points:
        payload = point.payload or {}
        missing = REQUIRED_METADATA - set(payload.keys())
        assert not missing, f"Point {point.id} missing metadata fields: {missing}"
    print("[PASS] All 6 required metadata fields present on sampled points.")

    # --- 4. BM25 index ---
    assert BM25_PATH.exists(), f"BM25 index not found at {BM25_PATH}"
    bm25_retriever = load_bm25_index(BM25_PATH)
    bm25_results = bm25_retriever.retrieve("net sales revenue")
    assert len(bm25_results) > 0, "BM25 retrieve returned 0 results"
    print(f"[PASS] BM25 index loaded from {BM25_PATH}, returned {len(bm25_results)} result(s) for test query.")

    # --- Summary ---
    print("\n--- Ingestion Validation Summary ---")
    print(f"  Collection  : {collection}")
    print(f"  Point count : {point_count}")
    print("  BM25 nodes  : n/a (retriever object)")
    print("\nAll checks passed.")


def main() -> None:
    validate_ingestion()


if __name__ == "__main__":
    main()
