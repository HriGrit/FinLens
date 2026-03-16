from __future__ import annotations

from types import SimpleNamespace

import pytest
from llama_index.core.schema import TextNode

import generation.generate as gen


def _valid_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Valid answer"))],
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
        ),
    )


def test_generate_raises_on_empty_choices(monkeypatch) -> None:
    def fake_completion(*args, **kwargs):
        return SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    monkeypatch.setattr("generation.generate.litellm.completion", fake_completion)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(gen.MalformedGenerationResponseError, match="missing choices"):
        gen.generate(
            query="What is this?",
            context_nodes=[TextNode(text="chunk", metadata={})],
            model="openrouter/stepfun/step-3.5-flash:free",
        )


def test_generate_raises_on_missing_message_content(monkeypatch) -> None:
    def fake_completion(*args, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr("generation.generate.litellm.completion", fake_completion)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(gen.MalformedGenerationResponseError, match="missing message content"):
        gen.generate(
            query="What is this?",
            context_nodes=[TextNode(text="chunk", metadata={})],
            model="openrouter/stepfun/step-3.5-flash:free",
        )


def test_generate_raises_on_missing_usage(monkeypatch) -> None:
    def fake_completion(*args, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))],
            usage=None,
        )

    monkeypatch.setattr("generation.generate.litellm.completion", fake_completion)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(gen.MalformedGenerationResponseError, match="missing usage"):
        gen.generate(
            query="What is this?",
            context_nodes=[TextNode(text="chunk", metadata={})],
            model="openrouter/stepfun/step-3.5-flash:free",
        )


def test_generate_raises_on_partial_usage(monkeypatch) -> None:
    def fake_completion(*args, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    monkeypatch.setattr("generation.generate.litellm.completion", fake_completion)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(gen.MalformedGenerationResponseError, match="usage.total_tokens"):
        gen.generate(
            query="What is this?",
            context_nodes=[TextNode(text="chunk", metadata={})],
            model="openrouter/stepfun/step-3.5-flash:free",
        )


def test_generate_sets_cost_to_none_when_cost_calculation_fails(monkeypatch) -> None:
    monkeypatch.setattr("generation.generate.litellm.completion", lambda *args, **kwargs: _valid_response())
    monkeypatch.setattr(
        "generation.generate.litellm.completion_cost",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("cost unavailable")),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    result = gen.generate(
        query="What is this?",
        context_nodes=[TextNode(text="chunk", metadata={})],
        model="openrouter/stepfun/step-3.5-flash:free",
    )

    assert result["usage"]["cost_usd"] is None


def test_generate_normalizes_providerless_openrouter_model_ids(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_completion(*args, **kwargs):
        captured["model"] = kwargs["model"]
        return _valid_response()

    monkeypatch.setattr("generation.generate.litellm.completion", fake_completion)
    monkeypatch.setattr("generation.generate.litellm.completion_cost", lambda **kwargs: 0.0)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    result = gen.generate(
        query="What is this?",
        context_nodes=[TextNode(text="chunk", metadata={})],
        model="qwen/qwen3-4b:free",
    )

    assert captured["model"] == "openrouter/qwen/qwen3-4b:free"
    assert result["model"] == "openrouter/qwen/qwen3-4b:free"
    assert "model_reasoning" in result


# ── _extract_think tests ──────────────────────────────────────────────────────


def test_extract_think_strips_think_block() -> None:
    clean, reasoning = gen._extract_think("<think>reasoning</think>answer")
    assert clean == "answer"
    assert reasoning == "reasoning"


def test_extract_think_no_think_block() -> None:
    text = "plain answer with no think tags"
    clean, reasoning = gen._extract_think(text)
    assert clean == text
    assert reasoning is None


def test_extract_think_multiline_block() -> None:
    content = "<think>\nline one\nline two\n</think>final answer"
    clean, reasoning = gen._extract_think(content)
    assert clean == "final answer"
    assert "line one" in reasoning
    assert "line two" in reasoning


def test_generate_includes_model_reasoning_when_present(monkeypatch) -> None:
    def fake_completion(*args, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="<think>my reasoning</think>clean answer"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    monkeypatch.setattr("generation.generate.litellm.completion", fake_completion)
    monkeypatch.setattr("generation.generate.litellm.completion_cost", lambda **kwargs: 0.0)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    result = gen.generate(
        query="What is this?",
        context_nodes=[TextNode(text="chunk", metadata={})],
    )

    assert result["answer"] == "clean answer"
    assert result["model_reasoning"] == "my reasoning"


def test_generate_model_reasoning_none_when_absent(monkeypatch) -> None:
    monkeypatch.setattr("generation.generate.litellm.completion", lambda *args, **kwargs: _valid_response())
    monkeypatch.setattr("generation.generate.litellm.completion_cost", lambda **kwargs: 0.0)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    result = gen.generate(
        query="What is this?",
        context_nodes=[TextNode(text="chunk", metadata={})],
    )

    assert result["model_reasoning"] is None
