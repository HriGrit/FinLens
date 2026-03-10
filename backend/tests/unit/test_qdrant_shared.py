from __future__ import annotations

from unittest.mock import patch

from shared import qdrant


def test_qdrant_client_picks_up_api_key_from_env(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "https://cloud.qdrant.test")
    monkeypatch.setenv("QDRANT_API_KEY", "sk-test")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_collection")

    with patch("shared.qdrant.QdrantClient") as mocked_client:
        qdrant.get_qdrant_client()
        mocked_client.assert_called_once_with(
            url="https://cloud.qdrant.test",
            timeout=30,
            api_key="sk-test",
        )


def test_qdrant_url_falls_back_to_default_when_empty(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "")
    assert qdrant.get_qdrant_url("http://localhost:6333") == "http://localhost:6333"
