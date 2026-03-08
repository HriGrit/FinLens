from __future__ import annotations

from llama_index.core.schema import TextNode

from generation.prompt import build_prompt


def test_build_prompt_formats_context_with_citations_and_query():
    context = [
        TextNode(
            text="3M reported net sales of $35.4 billion.",
            metadata={
                "company": "3M",
                "year": "2022",
                "doc_type": "10-K",
                "page_number": 47,
                "filename": "3M_2022_10K.pdf",
            },
        ),
        TextNode(
            text="Net sales increased 11% from last year.",
            metadata={
                "company": "3M",
                "year": "2022",
                "doc_type": "10-K",
                "page_number": 48,
                "filename": "3M_2022_10K.pdf",
            },
        ),
    ]

    messages = build_prompt("What were 3M's net sales in 2022?", context)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are FinLens")
    assert messages[1]["role"] == "user"

    rendered = messages[1]["content"]
    assert "[1] (3M 2022 10-K p.47)" in rendered
    assert "[2] (3M 2022 10-K p.48)" in rendered
    assert "What were 3M's net sales in 2022?" in rendered


def test_build_prompt_handles_missing_metadata():
    context = [
        TextNode(
            text="Revenue increased.",
            metadata={"company": "3M", "year": "2022", "filename": "3M_2022_10K.pdf"},
        )
    ]

    rendered = build_prompt("Any highlights?", context)[1]["content"]

    assert "10-K" in rendered
    assert "Any highlights?" in rendered
    assert "p.?" in rendered
