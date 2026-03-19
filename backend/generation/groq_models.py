from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
CACHE_TTL_SECONDS = 60 * 60


@dataclass(frozen=True)
class GroqModelOption:
    id: str
    name: str
    context_length: int | None = None
    description: str | None = None


FALLBACK_GROQ_MODELS: tuple[GroqModelOption, ...] = (
    GroqModelOption(
        id="llama-3.3-70b-versatile",
        name="Groq: Llama 3.3 70B Versatile",
        context_length=32768,
        description="General chat model from Meta",
    ),
    GroqModelOption(
        id="llama-3.1-8b-instant",
        name="Groq: Llama 3.1 8B Instant",
        context_length=131072,
    ),
    GroqModelOption(
        id="llama-3.2-11b-vision-preview",
        name="Groq: Llama 3.2 11B Vision Preview",
        context_length=8192,
        description="Vision-capable variant",
    ),
    GroqModelOption(
        id="qwen2.5-72b-instruct",
        name="Groq: Qwen2.5 72B Instruct",
        context_length=32768,
    ),
)

_groq_models_cache: list[GroqModelOption] | None = None
_groq_models_cache_expires_at = 0.0


def _is_groq_model(model: dict[str, Any]) -> bool:
    model_id = str(model.get("id", ""))
    if not model_id:
        return False
    # Keep chat-capable text models only.
    lowered = model_id.lower()
    return not (
        lowered.startswith("whisper")
        or lowered.startswith("playai")
        or lowered.startswith("distil-whisper")
        or lowered.startswith("llava")
        or lowered.startswith("pixtral")
        or "embed" in lowered
    )


def _to_groq_model_option(model: dict[str, Any]) -> GroqModelOption:
    context_length = model.get("context_length")
    if context_length is None:
        context_length = model.get("context_window")
    return GroqModelOption(
        id=str(model["id"]),
        name=str(model.get("name") or model["id"]),
        context_length=context_length,
        description=model.get("description"),
    )


def _sort_key(model: GroqModelOption) -> tuple[str, str]:
    return (model.name.lower(), model.id.lower())


def _dedupe_models(models: list[GroqModelOption]) -> list[GroqModelOption]:
    deduped: dict[str, GroqModelOption] = {}
    for model in models:
        deduped[model.id] = model
    return sorted(deduped.values(), key=_sort_key)


def _fetch_groq_models() -> list[GroqModelOption]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")

    response = httpx.get(
        GROQ_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    models = [_to_groq_model_option(model) for model in data if _is_groq_model(model)]
    if not models:
        raise ValueError("Groq models API returned no chat models.")
    return _dedupe_models(models)


def get_groq_models(force_refresh: bool = False) -> list[GroqModelOption]:
    global _groq_models_cache, _groq_models_cache_expires_at

    now = time.time()
    if not force_refresh and _groq_models_cache is not None and now < _groq_models_cache_expires_at:
        return list(_groq_models_cache)

    try:
        models = _fetch_groq_models()
    except Exception:
        models = _dedupe_models(list(FALLBACK_GROQ_MODELS))

    _groq_models_cache = models
    _groq_models_cache_expires_at = now + CACHE_TTL_SECONDS
    return list(models)
