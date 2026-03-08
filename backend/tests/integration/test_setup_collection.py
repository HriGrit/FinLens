from __future__ import annotations

import importlib

import pytest


@pytest.mark.integration
def test_setup_collection_is_idempotent(
    qdrant_client,
    isolated_qdrant_collection,
    qdrant_url,
):
    module = importlib.import_module("ingestion.setup_collection")
    importlib.reload(module)
    module.COLLECTION_NAME = isolated_qdrant_collection
    module.QDRANT_URL = qdrant_url

    module.setup_collection()
    assert qdrant_client.collection_exists(isolated_qdrant_collection)

    # Re-run should not fail when collection already exists.
    module.setup_collection()
    assert qdrant_client.collection_exists(isolated_qdrant_collection)
