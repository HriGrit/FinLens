from __future__ import annotations

import builtins
import importlib
import logging
import sys
import types

import litellm
import pytest
from llama_index.core.schema import TextNode

import retrieval.pipeline
from observability import tracing


class _SpySpan:
    def __init__(self) -> None:
        self.outputs: list[tuple[object | None, dict]] = []

    def end(self, output: object | None = None, **kwargs: object) -> None:
        self.outputs.append((output, kwargs))


class _SpyTrace:
    def __init__(self) -> None:
        self.spans: list[tuple[str, object, _SpySpan]] = []

    def span(self, name: str, input: object = None, **kwargs: object) -> _SpySpan:
        s = _SpySpan()
        self.spans.append((name, input, s))
        return s


def _node(text: str) -> TextNode:
    return TextNode(text=text, metadata={})


def test_missing_keys_returns_noop_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracing._reset_for_testing()

    lf = tracing._get_langfuse()
    assert isinstance(lf, tracing._NoOpLangfuse)


def test_empty_string_keys_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    tracing._reset_for_testing()

    lf = tracing._get_langfuse()
    assert isinstance(lf, tracing._NoOpLangfuse)


def test_missing_keys_logs_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracing._reset_for_testing()

    with caplog.at_level(logging.WARNING):
        tracing._get_langfuse()

    assert any("tracing disabled" in record.message for record in caplog.records)


def test_langfuse_constructor_failure_returns_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")

    def raising_langfuse(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("initialization failed")

    fake_langfuse_module = types.SimpleNamespace(Langfuse=raising_langfuse)
    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse_module)
    tracing._reset_for_testing()

    lf = tracing._get_langfuse()
    assert isinstance(lf, tracing._NoOpLangfuse)


def test_langfuse_import_error_returns_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")

    original_import = builtins.__import__

    def import_language(*args: object, **kwargs: object) -> object:
        if args and args[0] == "langfuse":
            raise ModuleNotFoundError("langfuse not installed")
        return original_import(*args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_language)
    tracing._reset_for_testing()

    lf = tracing._get_langfuse()
    assert isinstance(lf, tracing._NoOpLangfuse)


def test_create_trace_does_not_raise_without_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracing._reset_for_testing()

    trace = tracing.create_trace("rag_query", metadata={"company": "3M"})
    assert trace.id == "noop-trace-id"


def test_noop_trace_has_id_attribute() -> None:
    trace = tracing._NoOpTrace()
    assert isinstance(trace.id, str)
    assert trace.id


def test_noop_trace_update_does_not_raise() -> None:
    trace = tracing._NoOpTrace()
    trace.update(foo=1, bar="baz")


def test_noop_trace_span_returns_noop_span() -> None:
    trace = tracing._NoOpTrace()
    span = trace.span(name="child", input={"q": "v"})
    assert isinstance(span, tracing._NoOpSpan)


def test_noop_span_end_does_not_raise() -> None:
    span = tracing._NoOpSpan()
    span.end()
    span.end(output={"ok": True})


def test_noop_langfuse_flush_does_not_raise() -> None:
    lf = tracing._NoOpLangfuse()
    lf.flush()


def test_get_langfuse_is_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracing._reset_for_testing()

    first = tracing._get_langfuse()
    second = tracing._get_langfuse()
    assert first is second


def test_pipeline_records_hybrid_retrieve_span_output(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = _SpyTrace()
    candidates = [_node("a"), _node("b"), _node("c")]
    reranked = [_node("a")]

    monkeypatch.setattr(retrieval.pipeline, "hybrid_retrieve", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr(retrieval.pipeline, "rerank", lambda *_args, **_kwargs: reranked)

    result = retrieval.pipeline.retrieve_and_rerank(
        query="revenue",
        retrieval_top_k=3,
        rerank_top_k=1,
        company="3M",
        year="2022",
        trace=trace,
    )

    assert result.nodes == reranked
    assert result.candidate_count == len(candidates)
    assert len(trace.spans) == 2
    assert trace.spans[0][0] == "hybrid_retrieve"
    assert trace.spans[0][1] == {"query": "revenue", "top_k": 3}
    assert trace.spans[0][2].outputs == [({"n_candidates": len(candidates)}, {})]
    assert trace.spans[1][0] == "cross_encoder_rerank"
    assert trace.spans[1][1] == {"n_candidates": len(candidates)}
    assert trace.spans[1][2].outputs == [({"n_results": len(reranked)}, {})]


def test_pipeline_no_double_end_on_empty_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = _SpyTrace()
    rerank_called = False

    def fake_rerank(*_args: object, **_kwargs: object) -> list[TextNode]:
        nonlocal rerank_called
        rerank_called = True
        return []

    monkeypatch.setattr(retrieval.pipeline, "hybrid_retrieve", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(retrieval.pipeline, "rerank", fake_rerank)

    result = retrieval.pipeline.retrieve_and_rerank(
        query="empty",
        retrieval_top_k=5,
        rerank_top_k=2,
        company=None,
        year=None,
        trace=trace,
    )

    assert result.nodes == []
    assert result.candidate_count == 0
    assert not rerank_called
    assert len(trace.spans) == 1
    assert trace.spans[0][0] == "hybrid_retrieve"
    assert trace.spans[0][2].outputs == [({"n_candidates": 0}, {})]


def test_pipeline_with_none_trace_does_not_use_span(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = [_node("x"), _node("y")]
    calls: list[str] = []

    monkeypatch.setattr(retrieval.pipeline, "hybrid_retrieve", lambda *_args, **_kwargs: _record_call(calls, "hybrid") or candidates)
    monkeypatch.setattr(retrieval.pipeline, "rerank", lambda *_args, **_kwargs: _record_call(calls, "rerank") or candidates[:1])

    result = retrieval.pipeline.retrieve_and_rerank(
        query="q", retrieval_top_k=2, rerank_top_k=1, trace=None
    )

    assert result.nodes == candidates[:1]
    assert result.candidate_count == len(candidates)
    assert calls == ["hybrid", "rerank"]


def _record_call(log: list[str], name: str) -> None:
    log.append(name)


def test_litellm_callbacks_not_set_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import generation.generate

    success_marker = ["success-callback-marker"]
    failure_marker = ["failure-callback-marker"]
    monkeypatch.setattr(litellm, "success_callback", success_marker)
    monkeypatch.setattr(litellm, "failure_callback", failure_marker)

    importlib.reload(generation.generate)

    assert litellm.success_callback is success_marker
    assert litellm.failure_callback is failure_marker


def test_register_langfuse_callbacks_installs_callbacks() -> None:
    litellm.success_callback = []
    litellm.failure_callback = []

    from generation.generate import register_langfuse_callbacks

    register_langfuse_callbacks()
    assert litellm.success_callback == ["langfuse"]
    assert litellm.failure_callback == ["langfuse"]
