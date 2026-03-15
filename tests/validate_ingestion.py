"""
Post-ingestion smoke test — validates Qdrant collection + BM25 index after run_ingestion.py.

Run with: cd backend && uv run python ../tests/validate_ingestion.py
Requires: run_ingestion.py completed successfully, Qdrant running on localhost:6333.
"""
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from ingestion.embed import get_embed_model
from ingestion.index import load_bm25_index  # X1: use the official load function

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "finlens_chunks_dev"
BM25_PATH = REPO_ROOT / "data" / "bm25_index.pkl"
REQUIRED_METADATA = {"element_type", "page_number", "filename", "company", "year", "doc_type"}

client = QdrantClient(url="http://localhost:6333")

# --- 1. Collection exists ---
assert client.collection_exists(COLLECTION), f"Collection '{COLLECTION}' not found in Qdrant"
info = client.get_collection(COLLECTION)
point_count = info.points_count
assert point_count > 0, "Collection exists but has 0 points"
print(f"[PASS] Collection '{COLLECTION}' exists with {point_count} points.")

# --- 2. Dense query with payload filter ---
embedding_dim = len(get_embed_model().get_text_embedding("test"))
results = client.query_points(
    collection_name=COLLECTION,
    query=[0.0] * embedding_dim,  # zero vector — just checks filter + retrieval plumbing
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
print(f"[PASS] All 6 required metadata fields present on sampled points.")

# --- 4. BM25 index ---
# X1: Use load_bm25_index() instead of pickle.load() directly so the versioned
#     serialization contract is respected and the retriever is properly reconstructed.
assert BM25_PATH.exists(), f"BM25 index not found at {BM25_PATH}"
bm25 = load_bm25_index(BM25_PATH)
bm25_results = bm25.retrieve("net sales revenue")
assert len(bm25_results) > 0, "BM25 retrieve returned 0 results"
print(f"[PASS] BM25 index loaded from {BM25_PATH}, returned {len(bm25_results)} result(s) for test query.")

# --- Summary ---
print(f"\n--- Ingestion Validation Summary ---")
print(f"  Collection  : {COLLECTION}")
print(f"  Point count : {point_count}")
print(f"  BM25 nodes  : n/a (retriever object)")
print(f"\nAll checks passed.")
