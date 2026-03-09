from __future__ import annotations

from types import SimpleNamespace

import pytest
from llama_index.core.schema import TextNode

from generation.generate import generate


def test_generate_raises_value_error_on_none_answer(monkeypatch) -> None:
    def fake_completion(*args, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )

    monkeypatch.setattr("generation.generate.litellm.completion", fake_completion)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(ValueError, match="empty response"):
        generate(
            query="What is this?",
            context_nodes=[TextNode(text="chunk", metadata={})],
            model="openrouter/stepfun/step-3.5-flash:free",
        )
