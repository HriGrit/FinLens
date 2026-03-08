"""
tracing.py — Langfuse span wrappers for pipeline observability (Langfuse v2).

Public API:
  create_trace(name, metadata)  -> StatefulTraceClient
  span(trace, name, input)      -> context manager yielding StatefulSpanClient
  flush()                       -> force-flush background queue
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

_langfuse = None


def _get_langfuse():
    global _langfuse
    if _langfuse is None:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        )
    return _langfuse


def create_trace(name: str, user_id: str | None = None, metadata: dict | None = None):
    """Create a top-level Langfuse trace for a user query."""
    lf = _get_langfuse()
    return lf.trace(name=name, user_id=user_id, metadata=metadata or {})


@contextmanager
def span(trace, name: str, input: Any = None) -> Generator:
    """Context manager wrapping a Langfuse span."""
    s = trace.span(name=name, input=input)
    try:
        yield s
    finally:
        s.end()


def flush() -> None:
    """Force-flush the Langfuse background queue (useful in tests)."""
    _get_langfuse().flush()
