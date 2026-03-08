"""
setup_collection.py — Idempotent Qdrant collection setup for FinLens.

Run once before ingestion. Safe to re-run — exits cleanly if collection already exists.

Usage (from backend/):
    uv run python ingestion/setup_collection.py
"""
import os

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "finlens_chunks_dev")


def setup_collection() -> None:
    try:
        from ingestion.embed import get_embed_model
    except ModuleNotFoundError:
        # Supports execution via `uv run python backend/ingestion/setup_collection.py`.
        from embed import get_embed_model
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        HnswConfigDiff,
        PayloadSchemaType,
        VectorParams,
    )

    dense_dim = len(get_embed_model().get_text_embedding("test"))
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=dense_dim,
                    distance=Distance.COSINE,
                    on_disk=False,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
                    datatype="float32",
                )
            },
        )
        print(f"Created collection '{COLLECTION_NAME}' (dim={dense_dim}, cosine, m=16, ef_construct=200).")

    # Payload indexes — idempotent (Qdrant is a no-op if index already exists)
    FILTER_FIELDS = {
        "company": PayloadSchemaType.KEYWORD,
        "year": PayloadSchemaType.KEYWORD,
        "doc_type": PayloadSchemaType.KEYWORD,
        "element_type": PayloadSchemaType.KEYWORD,
    }
    for field, schema in FILTER_FIELDS.items():
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema,
        )
        print(f"  Payload index ensured: {field} ({schema})")

    print("Setup complete.")


if __name__ == "__main__":
    setup_collection()
