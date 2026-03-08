from __future__ import annotations

import pytest
from llama_index.core.schema import TextNode

from ingestion import embed
from ingestion.parse import REQUIRED_METADATA_FIELDS
from ingestion.index import build_qdrant_index


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
