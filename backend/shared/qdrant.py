"""Shared Qdrant client helpers used by ingestion and API codepaths."""
from __future__ import annotations

import os
from typing import Any


def get_qdrant_url(default: str = "http://localhost:6333") -> str:
    """Return the active Qdrant URL."""
    return (os.getenv("QDRANT_URL") or "").strip() or default


def get_qdrant_api_key() -> str | None:
    """Return an API key only when configured."""
    key = (os.getenv("QDRANT_API_KEY") or "").strip()
    return key or None


def get_qdrant_collection(default: str = "finlens_chunks_dev") -> str:
    """Return the active Qdrant collection name."""
    return os.getenv("QDRANT_COLLECTION", default)


def get_qdrant_client(**kwargs: Any):
    """Create a Qdrant client with local or cloud auth."""
    from qdrant_client import QdrantClient

    timeout = kwargs.pop("timeout", 30)
    url = kwargs.pop("url", get_qdrant_url())
    api_key = kwargs.pop("api_key", get_qdrant_api_key())

    client_kwargs: dict[str, Any] = {"url": url, "timeout": timeout}
    client_kwargs.update(kwargs)
    if api_key:
        client_kwargs["api_key"] = api_key

    return QdrantClient(**client_kwargs)

