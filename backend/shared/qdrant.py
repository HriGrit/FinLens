import os

from qdrant_client import QdrantClient

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "finlens_chunks_dev"

_DEFAULT_QDRANT_TIMEOUT = 60  # seconds; raise via QDRANT_TIMEOUT env var if upserts still timeout


def get_qdrant_url() -> str:
    """Return the configured Qdrant URL."""
    return os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)


def get_qdrant_collection() -> str:
    """Return the configured Qdrant collection name."""
    return os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION)


QDRANT_URL = get_qdrant_url()
QDRANT_COLLECTION = get_qdrant_collection()


def get_qdrant_client() -> QdrantClient:
    """Return a QdrantClient configured from environment variables."""
    kwargs: dict = {"url": get_qdrant_url()}
    api_key = os.getenv("QDRANT_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    kwargs["timeout"] = int(os.getenv("QDRANT_TIMEOUT", str(_DEFAULT_QDRANT_TIMEOUT)))
    return QdrantClient(**kwargs)
