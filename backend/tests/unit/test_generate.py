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
