from __future__ import annotations

import importlib

import pytest


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
