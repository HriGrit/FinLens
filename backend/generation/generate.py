"""
generate.py — LiteLLM call with citation assembly.

Milestone coverage: M0.5 (LiteLLM -> OpenRouter -> Mistral).
"""
from __future__ import annotations

import os

import litellm
from llama_index.core.schema import TextNode

from .prompt import build_prompt

DEFAULT_MODEL = "openrouter/stepfun/step-3.5-flash:free"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]


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

    answer = response.choices[0].message.content
    usage = response.usage
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = 0.0

    usage_dict = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }

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
