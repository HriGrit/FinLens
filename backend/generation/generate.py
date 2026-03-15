"""
generate.py — LiteLLM call with citation assembly.

Milestone coverage: M0.5 (LiteLLM -> OpenRouter -> Mistral).
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import litellm
from llama_index.core.schema import TextNode

from .prompt import build_prompt

DEFAULT_MODEL = "openrouter/stepfun/step-3.5-flash:free"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


class MalformedGenerationResponseError(ValueError):
    """Raised when upstream LiteLLM response shape is malformed."""


def register_langfuse_callbacks() -> None:
    """Install LiteLLM -> Langfuse callbacks. Call at app startup, never at import."""
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


def _validated_answer_and_usage(response: Any) -> tuple[str, dict[str, int]]:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, Sequence) or len(choices) == 0:
        raise MalformedGenerationResponseError("LLM response is missing choices.")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise MalformedGenerationResponseError("LLM response is missing message content.")

    usage = getattr(response, "usage", None)
    if usage is None:
        raise MalformedGenerationResponseError("LLM response is missing usage.")

    usage_dict: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, field, None)
        if not isinstance(value, int):
            raise MalformedGenerationResponseError(
                f"LLM response usage.{field} is missing or invalid."
            )
        usage_dict[field] = value

    return content, usage_dict


def generate(
    query: str,
    context_nodes: list[TextNode],
    model: str = DEFAULT_MODEL,
    trace=None,
) -> dict:
    """Call LLM with context nodes; return answer, citations, and token usage."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set.")

    messages = build_prompt(query, context_nodes)

    lf_metadata = {"existing_trace_id": trace.id, "generation_name": "llm_call"} if trace else {}

    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=api_key,
        api_base=OPENROUTER_API_BASE,
        metadata=lf_metadata,
    )

    answer, usage_dict = _validated_answer_and_usage(response)

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None

    citations = [
        {
            "index": i + 1,
            "company": node.metadata.get("company"),
            "year": node.metadata.get("year"),
            "doc_type": node.metadata.get("doc_type"),
            "page_number": node.metadata.get("page_number"),
            "filename": node.metadata.get("filename"),
        }
        for i, node in enumerate(context_nodes)
    ]

    return {
        "answer": answer,
        "citations": citations,
        "usage": {**usage_dict, "cost_usd": cost},
        "model": model,
    }
