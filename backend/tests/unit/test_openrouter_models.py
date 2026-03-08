from __future__ import annotations

from generation.openrouter_models import (
    FALLBACK_FREE_MODELS,
    FreeModelOption,
    _dedupe_models,
    _is_free_model,
    get_free_models,
)


def test_is_free_model_accepts_free_suffix_and_zero_pricing():
    assert _is_free_model({"id": "qwen/qwen3-4b:free", "pricing": {"prompt": "0", "completion": "0"}})
    assert _is_free_model({"id": "openrouter/free", "pricing": {"prompt": "1", "completion": "1"}})
    assert _is_free_model({"id": "vendor/model", "pricing": {"prompt": "0", "completion": "0"}})
    assert not _is_free_model({"id": "vendor/model", "pricing": {"prompt": "1", "completion": "0"}})


def test_dedupe_models_keeps_last_copy_and_sorts_router_first():
    models = [
        FreeModelOption(id="vendor/b", name="B"),
        FreeModelOption(id="openrouter/free", name="Router"),
        FreeModelOption(id="vendor/a", name="A"),
        FreeModelOption(id="vendor/b", name="B newer"),
    ]

    deduped = _dedupe_models(models)

    assert [model.id for model in deduped] == ["openrouter/free", "vendor/a", "vendor/b"]
    assert deduped[-1].name == "B newer"


def test_get_free_models_falls_back_when_fetch_fails(monkeypatch):
    monkeypatch.setattr("generation.openrouter_models._free_models_cache", None)
    monkeypatch.setattr("generation.openrouter_models._free_models_cache_expires_at", 0.0)
    monkeypatch.setattr("generation.openrouter_models._fetch_free_models", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    models = get_free_models(force_refresh=True)

    assert models
    assert models[0].id == "openrouter/free"
    assert {model.id for model in models}.issuperset({model.id for model in FALLBACK_FREE_MODELS})
