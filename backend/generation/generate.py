"""
generate.py — LiteLLM call with citation assembly.

Milestone coverage: M0.5 (LiteLLM -> OpenRouter -> Mistral).
"""
from __future__ import annotations

import os
import re
import random
from collections.abc import Sequence
from typing import Any

import litellm
from llama_index.core.schema import TextNode
from pydantic import BaseModel

from .prompt import build_prompt
from .openrouter_models import get_free_models
from .groq_models import get_groq_models

DEFAULT_MODEL = "qwen/qwen3-4b:free"
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
GROQ_API_BASE = "https://api.groq.com/openai/v1"
OPENROUTER_PROVIDER_PREFIX = "openrouter/"
GROQ_PROVIDER_PREFIX = "groq/"
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


def _normalize_model_provider(model: str) -> str:
    if model.startswith(GROQ_PROVIDER_PREFIX):
        return "groq"
    if model.startswith(OPENROUTER_PROVIDER_PREFIX):
        return "openrouter"
    return "openrouter"


def _provider_credentials(model: str) -> tuple[str, str]:
    provider = _normalize_model_provider(model)
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")
        return api_key, GROQ_API_BASE

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set.")
    return api_key, OPENROUTER_API_BASE


def _strip_provider_prefix(model: str, provider: str) -> str:
    if provider == "groq" and model.startswith(GROQ_PROVIDER_PREFIX):
        return model[len(GROQ_PROVIDER_PREFIX) :]
    if provider == "openrouter" and model.startswith(OPENROUTER_PROVIDER_PREFIX):
        return model[len(OPENROUTER_PROVIDER_PREFIX) :]
    return model


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
    provider = _normalize_model_provider(requested_model)
    try:
        if provider == "groq":
            candidate_models = [option.id for option in get_groq_models()]
        else:
            candidate_models = [option.id for option in get_free_models()]
    except Exception:
        return []

    canonical_requested = _strip_provider_prefix(requested_model, provider)
    normalized_requested = _normalize_model_name(requested_model)
    deduped = []
    seen = set()

    for model_id in candidate_models:
        raw_model = (
            f"{GROQ_PROVIDER_PREFIX}{model_id}"
            if provider == "groq" and not model_id.startswith(GROQ_PROVIDER_PREFIX)
            else model_id
        )
        candidate = _normalize_model_name(raw_model)
        canonical_candidate = _strip_provider_prefix(candidate, provider)
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
    lf_metadata: dict[str, str | None],
) -> tuple[dict[str, Any], dict[str, int]]:
    api_key, api_base = _provider_credentials(model)
    response = litellm.completion(
        model=model,
        messages=build_prompt(query, context_nodes),
        api_key=api_key,
        api_base=api_base,
        metadata=lf_metadata,
    )
    answer, model_reasoning, usage_dict = _validated_answer_and_usage(response)
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None

    return {
        "answer": answer,
        "model": model,
        "model_reasoning": model_reasoning,
        "usage": {**usage_dict, "cost_usd": cost},
    }, usage_dict


def _extract_think(content: str) -> tuple[str, str | None]:
    match = _THINK_RE.search(content)
    if match is None:
        return content.strip(), None
    reasoning_text = match.group(1).strip()
    clean = _THINK_RE.sub("", content).strip()
    return clean, reasoning_text or None


def _validated_answer_and_usage(response: Any) -> tuple[str, str | None, dict[str, int]]:
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

    clean_content, model_reasoning = _extract_think(content)
    return clean_content, model_reasoning, usage_dict


def generate(
    query: str,
    context_nodes: list[TextNode],
    model: str = DEFAULT_MODEL,
    trace=None,
) -> dict:
    """Call LLM with context nodes; return answer, citations, and token usage."""
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
    final_model_reasoning = final_result.get("model_reasoning")

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
        "model_reasoning": final_model_reasoning,
        "fallback": _build_fallback_metadata(
            requested_model=normalized_request,
            active_model=final_model,
            fallback_attempts=len(events),
            attempted_models=attempted_models,
            events=events,
        ),
    }
