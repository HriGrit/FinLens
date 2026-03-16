from __future__ import annotations

import importlib

import pytest


def _extract_dense_size(collection_info) -> int:
    vectors = collection_info.config.params.vectors
    if isinstance(vectors, dict):
        return vectors["dense"].size
    return vectors.size


@pytest.mark.integration
def test_setup_collection_is_idempotent(
    monkeypatch,
    qdrant_client,
    isolated_qdrant_collection,
    qdrant_url,
):
    monkeypatch.setenv("QDRANT_URL", qdrant_url)
    monkeypatch.setenv("QDRANT_COLLECTION", isolated_qdrant_collection)

    module = importlib.import_module("ingestion.setup_collection")
    importlib.reload(module)

    module.setup_collection()
    assert qdrant_client.collection_exists(isolated_qdrant_collection)

    # Re-run should not fail when collection already exists.
    module.setup_collection()
    assert qdrant_client.collection_exists(isolated_qdrant_collection)


@pytest.mark.integration
def test_setup_collection_uses_runtime_embedding_dimension(
    monkeypatch,
    qdrant_client,
    isolated_qdrant_collection,
    qdrant_url,
    test_embedding_dim: int,
):
    monkeypatch.setenv("QDRANT_URL", qdrant_url)
    monkeypatch.setenv("QDRANT_COLLECTION", isolated_qdrant_collection)

    module = importlib.import_module("ingestion.setup_collection")
    importlib.reload(module)
    module.setup_collection()

    collection_info = qdrant_client.get_collection(isolated_qdrant_collection)
    assert _extract_dense_size(collection_info) == test_embedding_dim
