from __future__ import annotations

import importlib
from unittest.mock import patch


def test_get_qdrant_client_no_api_key(monkeypatch):
    """Without QDRANT_API_KEY, client is constructed without api_key kwarg."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    with patch("qdrant_client.QdrantClient") as mock_client:
        import shared.qdrant as m

        importlib.reload(m)
        m.get_qdrant_client()

    call_kwargs = mock_client.call_args.kwargs
    assert call_kwargs.get("url") == "http://localhost:6333"
    assert "api_key" not in call_kwargs


def test_get_qdrant_client_with_api_key(monkeypatch):
    """With QDRANT_API_KEY set, client is constructed with api_key kwarg."""
    monkeypatch.setenv("QDRANT_URL", "https://cloud.qdrant.io")
    monkeypatch.setenv("QDRANT_API_KEY", "secret-key")

    with patch("qdrant_client.QdrantClient") as mock_client:
        import shared.qdrant as m

        importlib.reload(m)
        m.get_qdrant_client()

    call_kwargs = mock_client.call_args.kwargs
    assert call_kwargs.get("url") == "https://cloud.qdrant.io"
    assert call_kwargs.get("api_key") == "secret-key"


def test_get_qdrant_collection_reads_current_env(monkeypatch):
    import shared.qdrant as m

    monkeypatch.setenv("QDRANT_COLLECTION", "finlens_chunks_test")

    assert m.get_qdrant_collection() == "finlens_chunks_test"


def test_point_id_stable_for_same_chunk():
    from ingestion.index import _make_point_id

    id1 = _make_point_id("doc.pdf", 5, "some text")
    id2 = _make_point_id("doc.pdf", 5, "some text")

    assert id1 == id2


def test_point_id_different_chunks_same_page():
    from ingestion.index import _make_point_id

    id1 = _make_point_id("doc.pdf", 5, "first chunk")
    id2 = _make_point_id("doc.pdf", 5, "second chunk")

    assert id1 != id2


def test_point_id_same_text_different_page():
    from ingestion.index import _make_point_id

    id1 = _make_point_id("doc.pdf", 5, "same text")
    id2 = _make_point_id("doc.pdf", 6, "same text")

    assert id1 != id2
