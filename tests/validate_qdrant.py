"""
Milestone M0.4 — Qdrant round-trip validation (named-vector schema).

Run with: cd backend && uv run python ../tests/validate_qdrant.py
Requires: Qdrant reachable at configured QDRANT_URL.
"""
from __future__ import annotations

import os
import uuid

from ingestion.embed import get_embed_model
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from shared.qdrant import get_qdrant_client, get_qdrant_collection


def validate_qdrant_round_trip() -> None:
    base_collection = get_qdrant_collection()
    collection = os.getenv("VALIDATE_QDRANT_COLLECTION", f"{base_collection}_validate")
    dim = len(get_embed_model().get_text_embedding("test"))
    client = get_qdrant_client()

    # --- Setup ---
    if client.collection_exists(collection):
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": VectorParams(
                size=dim,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
                datatype="float32",
            )
        },
    )
    print(f"Collection '{collection}' created (named vector 'dense', dim={dim}).")

    for field in ("company", "year", "doc_type", "element_type"):
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    print("Payload indexes created: company, year, doc_type, element_type.")

    # --- Insert ---
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=point_id,
                vector={"dense": [0.1] * dim},
                payload={
                    "text": "3M reported net sales of $35.4 billion in 2022.",
                    "element_type": "paragraph",
                    "page_number": 42,
                    "filename": "3M_2022_10K.pdf",
                    "company": "3M",
                    "year": "2022",
                    "doc_type": "10-K",
                },
            )
        ],
    )
    print(f"Upserted point id={point_id}.")

    # --- Query ---
    results = client.query_points(
        collection_name=collection,
        query=[0.1] * dim,
        using="dense",
        query_filter=Filter(
            must=[
                FieldCondition(key="company", match=MatchValue(value="3M")),
                FieldCondition(key="year", match=MatchValue(value="2022")),
            ]
        ),
        limit=5,
    )
    print(f"Query returned {len(results.points)} result(s).")
    for result in results.points:
        print(f"  id={result.id}  score={result.score:.4f}  payload={result.payload}")

    assert len(results.points) == 1, "Expected exactly 1 result from payload-filter query"
    assert results.points[0].payload["company"] == "3M"
    assert results.points[0].payload["doc_type"] == "10-K"
    assert results.points[0].score > 0.99, (
        f"Expected near-perfect score for identical vector, got {results.points[0].score}"
    )

    # --- Cleanup ---
    client.delete_collection(collection)
    print(f"Collection '{collection}' deleted (cleanup).")
    print("\nM0.4 PASS: Qdrant named-vector round-trip works.")


def main() -> None:
    validate_qdrant_round_trip()


if __name__ == "__main__":
    main()
