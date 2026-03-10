"""
test_embed.py — Unit tests for the embedding singleton cache logic.

All tests mock HuggingFaceEmbedding so no real model is loaded.
"""
from unittest.mock import MagicMock, patch

import pytest


def _reset_singleton() -> None:
    """Reset module-level singleton state between tests."""
    import ingestion.embed as embed_mod
    embed_mod._embed_model = None
    embed_mod._embed_device = None


@pytest.fixture(autouse=True)
def _patch_embedding_model():
    """Override conftest autouse fixture — we need the real get_embed_model here."""
    yield


@pytest.fixture(autouse=True)
def reset_embed_singleton():
    _reset_singleton()
    yield
    _reset_singleton()


@patch("ingestion.embed.HuggingFaceEmbedding")
def test_singleton_reused_same_model_and_device(mock_hf: MagicMock) -> None:
    """Calling get_embed_model twice with same args should only construct once."""
    mock_instance = MagicMock()
    mock_instance.model_name = "Alibaba-NLP/gte-modernbert-base"
    mock_hf.return_value = mock_instance

    from ingestion.embed import get_embed_model

    m1 = get_embed_model()
    m2 = get_embed_model()

    assert m1 is m2
    mock_hf.assert_called_once()


@patch("ingestion.embed.HuggingFaceEmbedding")
def test_singleton_reinitialised_on_model_change(mock_hf: MagicMock) -> None:
    """Changing model_name should trigger a second construction."""
    model_a = MagicMock()
    model_a.model_name = "model-a"
    model_b = MagicMock()
    model_b.model_name = "model-b"
    mock_hf.side_effect = [model_a, model_b]

    from ingestion.embed import get_embed_model

    r1 = get_embed_model("model-a")
    r2 = get_embed_model("model-b")

    assert r1 is model_a
    assert r2 is model_b
    assert mock_hf.call_count == 2


@patch("ingestion.embed.HuggingFaceEmbedding")
def test_singleton_reinitialised_on_device_change(mock_hf: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing the resolved device should trigger a second construction."""
    import ingestion.embed as embed_mod

    model_cpu = MagicMock()
    model_cpu.model_name = embed_mod.DEFAULT_MODEL
    model_mps = MagicMock()
    model_mps.model_name = embed_mod.DEFAULT_MODEL
    mock_hf.side_effect = [model_cpu, model_mps]

    # Force device to "cpu" for first call
    monkeypatch.setattr(embed_mod, "DEFAULT_DEVICE", None)
    with patch("torch.backends.mps.is_available", return_value=False):
        r1 = embed_mod.get_embed_model()

    assert r1 is model_cpu
    assert embed_mod._embed_device == "cpu"

    # Now switch to "mps" by resetting DEFAULT_DEVICE and making mps available
    monkeypatch.setattr(embed_mod, "DEFAULT_DEVICE", None)
    with patch("torch.backends.mps.is_available", return_value=True):
        r2 = embed_mod.get_embed_model()

    assert r2 is model_mps
    assert embed_mod._embed_device == "mps"
    assert mock_hf.call_count == 2
