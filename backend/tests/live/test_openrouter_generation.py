from __future__ import annotations

import os

import pytest
from llama_index.core.schema import TextNode

from generation.generate import generate


@pytest.mark.live
def test_openrouter_generation_returns_answer(requires_openrouter):
    model = os.getenv("OPENROUTER_TEST_MODEL", "qwen/qwen3-4b:free")
    nodes = [
        TextNode(
            text="3M reported net sales of $35.4 billion in 2022.",
            metadata={"company": "3M", "year": "2022", "doc_type": "10-K", "page_number": 47, "filename": "3M_2022_10K.pdf"},
        )
    ]

    output = generate(query="What is 2+2?", context_nodes=nodes, model=model)

    assert output["answer"]
    assert output["usage"]["total_tokens"] > 0
