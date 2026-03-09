from __future__ import annotations

import pytest
from llama_index.core.schema import TextNode

from ingestion import embed
from ingestion.parse import REQUIRED_METADATA_FIELDS
from ingestion.index import build_qdrant_index


@pytest.mark.integration
def test_none_page_number_stored_as_null_in_qdrant(
    qdrant_client,
    isolated_qdrant_collection,
):
    """A-2: page_number=None must be stored as JSON null, not omitted from the payload."""
    node = TextNode(
        text="Revenue details with unknown page.",
        metadata={
            "element_type": "paragraph",
            "page_number": None,
            "filename": "3M_2022_10K.pdf",
            "company": "3M",
            "year": "2022",
            "doc_type": "10-K",
            "chunk_index": 0,
        },
    )
    build_qdrant_index([node], collection_name=isolated_qdrant_collection)

    results = qdrant_client.scroll(
        collection_name=isolated_qdrant_collection,
        limit=1,
        with_payload=True,
    )
    point = results[0][0]
    payload = point.payload or {}
    assert "page_number" in payload, "page_number key must be present in Qdrant payload"
    assert payload["page_number"] is None, "page_number must be null (not absent)"


@pytest.mark.integration
def test_deterministic_ids_make_upsert_idempotent(
    qdrant_client,
    isolated_qdrant_collection,
    sample_nodes: list[TextNode],
):
    """A-3: calling build_qdrant_index twice with the same nodes must not increase point count."""
    build_qdrant_index(sample_nodes, collection_name=isolated_qdrant_collection)
    count_after_first = qdrant_client.get_collection(isolated_qdrant_collection).points_count

    build_qdrant_index(sample_nodes, collection_name=isolated_qdrant_collection)
    count_after_second = qdrant_client.get_collection(isolated_qdrant_collection).points_count

    assert count_after_second == count_after_first, (
        f"Re-ingestion inserted duplicates: {count_after_first} → {count_after_second}"
    )


@pytest.mark.integration
def test_build_qdrant_index_upserts_documents(
    qdrant_client,
    isolated_qdrant_collection,
    sample_nodes: list[TextNode],
):
    build_qdrant_index(sample_nodes, collection_name=isolated_qdrant_collection)

    collection = qdrant_client.get_collection(isolated_qdrant_collection)
    assert collection.points_count > 0

    model_dim = len(embed.get_embed_model().get_text_embedding("test"))
    results = qdrant_client.query_points(
        collection_name=isolated_qdrant_collection,
        query=[0.0] * model_dim,
        using="dense",
        query_filter=None,
        limit=5,
    )
    assert len(results.points) == len(sample_nodes)
    for point in results.points:
        missing = REQUIRED_METADATA_FIELDS - set((point.payload or {}).keys())
        assert not missing
