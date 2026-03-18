import os

from qdrant_client import QdrantClient

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "finlens_chunks_dev"
LOCAL_MODE = "local"
HOSTED_MODE = "hosted"

_DEFAULT_QDRANT_TIMEOUT = 60  # seconds; raise via QDRANT_TIMEOUT env var if upserts still timeout


def _normalize_qdrant_mode(mode: str | None) -> str:
    if not mode:
        return LOCAL_MODE
    normalized = mode.strip().lower()
    if normalized not in {LOCAL_MODE, HOSTED_MODE}:
        return LOCAL_MODE
    return normalized


def get_qdrant_mode() -> str:
    """Return the Qdrant deployment mode."""
    return _normalize_qdrant_mode(os.getenv("QDRANT_MODE"))


def get_qdrant_url() -> str:
    """Return the configured Qdrant URL."""
    return os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)


def get_qdrant_collection() -> str:
    """Return the configured Qdrant collection name."""
    return os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION)


QDRANT_URL = get_qdrant_url()
QDRANT_COLLECTION = get_qdrant_collection()


def has_qdrant_credentials() -> bool:
    """Return whether current mode has the required credentials."""
    if get_qdrant_mode() == LOCAL_MODE:
        return True
    return bool(os.getenv("QDRANT_URL", "").strip()) and bool(os.getenv("QDRANT_API_KEY", "").strip())


def get_qdrant_config_error() -> str | None:
    """Return a human-readable config error when hosted mode is misconfigured."""
    if get_qdrant_mode() == LOCAL_MODE:
        return None
    if has_qdrant_credentials():
        return None
    return "QDRANT_MODE=hosted requires QDRANT_URL and QDRANT_API_KEY"


def get_qdrant_client() -> QdrantClient:
    """Return a QdrantClient configured from environment variables."""
    if get_qdrant_config_error():
        raise RuntimeError(get_qdrant_config_error() or "Invalid Qdrant configuration")
    kwargs: dict = {"url": get_qdrant_url()}
    api_key = os.getenv("QDRANT_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    timeout = os.getenv("QDRANT_TIMEOUT", str(_DEFAULT_QDRANT_TIMEOUT))
    try:
        kwargs["timeout"] = int(timeout)
    except ValueError as exc:
        raise RuntimeError(f"Invalid QDRANT_TIMEOUT={timeout!r}: expected integer value.") from exc
    try:
        return QdrantClient(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Qdrant client: {exc}") from exc
