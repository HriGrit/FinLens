from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_TTL_SECONDS = 60 * 60


@dataclass(frozen=True)
class FreeModelOption:
    id: str
    name: str
    context_length: int | None = None
    description: str | None = None


FALLBACK_FREE_MODELS: tuple[FreeModelOption, ...] = (
    FreeModelOption(
        id="arcee-ai/trinity-large-preview:free",
        name="Arcee AI: Trinity Large Preview (free)",
        context_length=131000,
    ),
    FreeModelOption(
        id="arcee-ai/trinity-mini:free",
        name="Arcee AI: Trinity Mini (free)",
        context_length=131072,
    ),
    FreeModelOption(
        id="nvidia/nemotron-3-nano-30b-a3b:free",
        name="NVIDIA: Nemotron 3 Nano 30B A3B (free)",
        context_length=256000,
    ),
    FreeModelOption(
        id="liquid/lfm-2.5-1.2b-thinking:free",
        name="LiquidAI: LFM2.5-1.2B-Thinking (free)",
        context_length=32768,
    ),
    FreeModelOption(
        id="liquid/lfm-2.5-1.2b-instruct:free",
        name="LiquidAI: LFM2.5-1.2B-Instruct (free)",
        context_length=32768,
    ),
    FreeModelOption(
        id="qwen/qwen3-4b:free",
        name="Qwen: Qwen3 4B (free)",
        context_length=40960,
    ),
)

_free_models_cache: list[FreeModelOption] | None = None
_free_models_cache_expires_at = 0.0


def _is_free_model(model: dict[str, Any]) -> bool:
    model_id = str(model.get("id", ""))
    pricing = model.get("pricing") or {}
    prompt_price = str(pricing.get("prompt", ""))
    completion_price = str(pricing.get("completion", ""))
    return model_id == "openrouter/free" or model_id.endswith(":free") or (
        prompt_price == "0" and completion_price == "0"
    )


def _is_supported_free_model(model_id: str) -> bool:
    """Filter out known unsupported aliases that regress in this deployment."""

    return model_id != "openrouter/free"


def _to_free_model_option(model: dict[str, Any]) -> FreeModelOption:
    return FreeModelOption(
        id=str(model["id"]),
        name=str(model.get("name") or model["id"]),
        context_length=model.get("context_length"),
        description=model.get("description"),
    )


def _sort_key(model: FreeModelOption) -> tuple[int, str]:
    return (0 if model.id == "openrouter/free" else 1, model.name.lower())


def _dedupe_models(models: list[FreeModelOption]) -> list[FreeModelOption]:
    deduped: dict[str, FreeModelOption] = {}
    for model in models:
        deduped[model.id] = model
    return sorted(deduped.values(), key=_sort_key)


def _fetch_free_models() -> list[FreeModelOption]:
    response = httpx.get(OPENROUTER_MODELS_URL, timeout=10)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    free_models = [
        _to_free_model_option(model)
        for model in data
        if _is_free_model(model) and _is_supported_free_model(str(model.get("id", "")))
    ]
    if not free_models:
        raise ValueError("OpenRouter models API returned no free models.")
    return _dedupe_models(free_models)


def get_free_models(force_refresh: bool = False) -> list[FreeModelOption]:
    global _free_models_cache, _free_models_cache_expires_at

    now = time.time()
    if not force_refresh and _free_models_cache is not None and now < _free_models_cache_expires_at:
        return list(_free_models_cache)

    try:
        models = _fetch_free_models()
    except Exception:
        models = _dedupe_models(list(FALLBACK_FREE_MODELS))

    _free_models_cache = models
    _free_models_cache_expires_at = now + CACHE_TTL_SECONDS
    return list(models)
