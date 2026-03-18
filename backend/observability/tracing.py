"""
tracing.py — Langfuse span wrappers for pipeline observability (Langfuse v2).

Public API:
  create_trace(name, metadata)  -> StatefulTraceClient
  flush()                       -> force-flush background queue
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_langfuse = None
LOCAL_MODE = "local"
HOSTED_MODE = "hosted"


class _NoOpSpan:
    def end(self, output: Any = None, **kwargs: Any) -> None:
        """No-op span that safely ignores end output/metadata."""
        return None


class _NoOpTrace:
    id: str = "noop-trace-id"

    def update(self, **kwargs: Any) -> None:
        """No-op trace update."""
        return None

    def span(self, name: str = "", input: Any = None, **kwargs: Any) -> _NoOpSpan:
        """No-op child span factory."""
        return _NoOpSpan()


class _NoOpLangfuse:
    def trace(self, name: str = "", user_id=None, metadata=None) -> _NoOpTrace:
        """No-op Langfuse client."""
        return _NoOpTrace()

    def flush(self) -> None:
        """No-op flush."""
        return None


def _normalize_langfuse_mode(mode: str | None) -> str:
    """Normalize env var value into a known trace mode."""
    return HOSTED_MODE if (mode or "").strip().lower() == HOSTED_MODE else LOCAL_MODE


def get_langfuse_mode() -> str:
    """Return active Langfuse mode (`local` default, `hosted` when explicitly selected)."""
    return _normalize_langfuse_mode(os.getenv("LANGFUSE_MODE"))


def get_langfuse_host() -> str:
    """Return configured Langfuse host URL."""
    return os.getenv("LANGFUSE_HOST", "http://localhost:3000")


def has_langfuse_credentials() -> bool:
    """Whether Langfuse API credentials are both present and non-empty."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    return bool(public_key) and bool(secret_key)


def _get_langfuse():
    global _langfuse
    if _langfuse is None:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        if not public_key or not secret_key:
            logger.warning("LANGFUSE keys not set; tracing disabled (no-op mode).")
            _langfuse = _NoOpLangfuse()
        else:
            try:
                from langfuse import Langfuse

                _langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=get_langfuse_host(),
                )
            except Exception as exc:
                logger.warning("Langfuse init failed (%s); no-op mode.", exc)
                _langfuse = _NoOpLangfuse()
    return _langfuse


def create_trace(name: str, user_id: str | None = None, metadata: dict | None = None):
    """Create a top-level Langfuse trace for a user query."""
    lf = _get_langfuse()
    return lf.trace(name=name, user_id=user_id, metadata=metadata or {})


def flush() -> None:
    """Force-flush the Langfuse background queue (useful in tests)."""
    _get_langfuse().flush()


def _reset_for_testing() -> None:
    """Reset the module cache used in tests."""
    global _langfuse
    _langfuse = None
