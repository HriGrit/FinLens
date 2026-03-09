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
                    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
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
