"""
setup_collection.py — Idempotent Qdrant collection setup for FinLens.

Run once before ingestion. Safe to re-run — exits cleanly if collection already exists.

Usage (from backend/):
    uv run python ingestion/setup_collection.py
"""
from dotenv import load_dotenv

load_dotenv()

from shared.qdrant import get_qdrant_client, get_qdrant_collection


def setup_collection() -> None:
    try:
        from ingestion.embed import get_embed_model
    except ModuleNotFoundError:
        # Supports execution via `uv run python backend/ingestion/setup_collection.py`.
        from embed import get_embed_model
    from qdrant_client.models import (
        Distance,
        HnswConfigDiff,
        PayloadSchemaType,
        VectorParams,
    )

    dense_dim = len(get_embed_model().get_text_embedding("test"))
    client = get_qdrant_client()
    collection_name = get_qdrant_collection()

    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists — skipping creation.")
    else:
        client.create_collection(
            collection_name=collection_name,
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
        print(f"Created collection '{collection_name}' (dim={dense_dim}, cosine, m=16, ef_construct=200).")

    # Payload indexes — idempotent (Qdrant is a no-op if index already exists)
    FILTER_FIELDS = {
        "company": PayloadSchemaType.KEYWORD,
        "year": PayloadSchemaType.KEYWORD,
        "doc_type": PayloadSchemaType.KEYWORD,
        "element_type": PayloadSchemaType.KEYWORD,
    }
    for field, schema in FILTER_FIELDS.items():
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=schema,
        )
        print(f"  Payload index ensured: {field} ({schema})")

    print("Setup complete.")


if __name__ == "__main__":
    setup_collection()
