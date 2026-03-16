"""
generate.py — LiteLLM call with citation assembly.

Milestone coverage: M0.5 (LiteLLM -> OpenRouter -> Mistral).
"""
from __future__ import annotations

import os
import random
from collections.abc import Sequence
from typing import Any

import litellm
from llama_index.core.schema import TextNode
from pydantic import BaseModel

from .prompt import build_prompt
from .openrouter_models import get_free_models

DEFAULT_MODEL = "qwen/qwen3-4b:free"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_PROVIDER_PREFIX = "openrouter/"
KNOWN_PROVIDER_PREFIXES = (
    "openrouter/",
    "openai/",
    "azure/",
    "anthropic/",
    "bedrock/",
    "vertex_ai/",
    "gemini/",
    "google/",
    "huggingface/",
    "ollama/",
    "replicate/",
    "groq/",
    "cohere/",
    "mistral/",
    "together_ai/",
    "fireworks_ai/",
    "xai/",
    "deepseek/",
    "perplexity/",
    "cerebras/",
    "databricks/",
    "predibase/",
    "watsonx/",
)


class MalformedGenerationResponseError(ValueError):
    """Raised when upstream LiteLLM response shape is malformed."""


class AllModelsTooHotError(RuntimeError):
    """Raised when no candidate can complete due to rate-limiting."""

    def __init__(
        self,
        requested_model: str,
        active_model: str,
        attempted_models: list[str],
        fallback_attempts: int,
        events: list["FallbackEvent"],
    ) -> None:
        self.requested_model = requested_model
        self.active_model = active_model
        self.attempted_models = attempted_models
        self.fallback_attempts = fallback_attempts
        self.events = events
        super().__init__("All candidate models are currently rate limited.")


class FallbackEvent(BaseModel):
    from_model: str
    to_model: str
    reason: str


def register_langfuse_callbacks() -> None:
    """Install LiteLLM -> Langfuse callbacks. Call at app startup, never at import."""
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


def _normalize_model_name(model: str) -> str:
    """Normalize frontend/OpenRouter model ids into LiteLLM-compatible names."""

    if model == "openrouter/free":
        return DEFAULT_MODEL

    if model.startswith(KNOWN_PROVIDER_PREFIXES):
        return model

    if "/" in model:
        return f"{OPENROUTER_PROVIDER_PREFIX}{model}"

    return model


def _strip_openrouter_prefix(model: str) -> str:
    return model[len(OPENROUTER_PROVIDER_PREFIX) :] if model.startswith(OPENROUTER_PROVIDER_PREFIX) else model


def _is_rate_limited_error(exc: Exception) -> bool:
    status_candidates = [
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(getattr(exc, "response", None), "status_code", None),
        getattr(getattr(exc, "httpx_response", None), "status_code", None),
    ]
    for status in status_candidates:
        if status == 429:
            return True
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg


def _candidate_fallback_models(requested_model: str, max_fallbacks: int = 2) -> list[str]:
    try:
        free_models = get_free_models()
    except Exception:
        return []

    canonical_requested = _strip_openrouter_prefix(requested_model)
    normalized_requested = _normalize_model_name(requested_model)
    deduped = []
    seen = set()

    for option in free_models:
        candidate = _normalize_model_name(option.id)
        canonical_candidate = _strip_openrouter_prefix(candidate)
        if canonical_candidate == canonical_requested:
            continue
        if canonical_candidate in seen or canonical_requested == candidate or canonical_candidate == normalized_requested:
            continue
        deduped.append(candidate)
        seen.add(canonical_candidate)

    if len(deduped) <= max_fallbacks:
        random.shuffle(deduped)
        return deduped

    return random.sample(deduped, k=max_fallbacks)


def _build_fallback_metadata(
    requested_model: str,
    active_model: str,
    fallback_attempts: int,
    attempted_models: list[str],
    events: list[FallbackEvent],
) -> dict[str, Any]:
    return {
        "requested_model": requested_model,
        "active_model": active_model,
        "fallback_used": fallback_attempts > 0,
        "fallback_attempts": fallback_attempts,
        "attempted_models": attempted_models,
        "events": [event.model_dump() for event in events],
    }


def _call_litellm_once(
    model: str,
    query: str,
    context_nodes: list[TextNode],
    api_key: str,
    lf_metadata: dict[str, str | None],
) -> tuple[dict[str, Any], dict[str, int]]:
    response = litellm.completion(
        model=model,
        messages=build_prompt(query, context_nodes),
        api_key=api_key,
        api_base=OPENROUTER_API_BASE,
        metadata=lf_metadata,
    )
    answer, usage_dict = _validated_answer_and_usage(response)
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None

    return {
        "answer": answer,
        "model": model,
        "usage": {**usage_dict, "cost_usd": cost},
    }, usage_dict


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

    lf_metadata = {"existing_trace_id": trace.id, "generation_name": "llm_call"} if trace else {}
    normalized_request = _normalize_model_name(model)
    events: list[FallbackEvent] = []
    attempted_models = [normalized_request]
    fallback_models: list[str] | None = None

    final_result: dict[str, Any] | None = None

    for idx in range(3):
        if idx == 0:
            model_id = normalized_request
        else:
            if fallback_models is None:
                fallback_models = _candidate_fallback_models(normalized_request, max_fallbacks=2)
            if idx - 1 >= len(fallback_models):
                break
            model_id = fallback_models[idx - 1]
        try:
            response_payload, _ = _call_litellm_once(
                model=model_id,
                query=query,
                context_nodes=context_nodes,
                api_key=api_key,
                lf_metadata=lf_metadata,
            )
            final_result = response_payload
            break
        except Exception as exc:
            if not _is_rate_limited_error(exc):
                raise
            if idx >= 2:
                break
            if fallback_models is None:
                fallback_models = _candidate_fallback_models(normalized_request, max_fallbacks=2)
            if idx >= len(fallback_models):
                break
            next_model = fallback_models[idx]
            events.append(FallbackEvent(from_model=model_id, to_model=next_model, reason="rate_limited"))
            attempted_models.append(next_model)

    if final_result is None:
        raise AllModelsTooHotError(
            requested_model=normalized_request,
            active_model=attempted_models[-1],
            attempted_models=attempted_models,
            fallback_attempts=min(len(attempted_models) - 1, 2),
            events=events,
        )

    final_answer = final_result["answer"]
    final_usage = final_result["usage"]
    final_model = final_result["model"]

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
        "answer": final_answer,
        "citations": citations,
        "usage": final_usage,
        "model": final_model,
        "fallback": _build_fallback_metadata(
            requested_model=normalized_request,
            active_model=final_model,
            fallback_attempts=len(events),
            attempted_models=attempted_models,
            events=events,
        ),
    }
