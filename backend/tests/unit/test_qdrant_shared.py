from __future__ import annotations

import importlib

import pytest
from unittest.mock import patch


def test_qdrant_mode_defaults_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_MODE", raising=False)
    import shared.qdrant as m

    assert m.get_qdrant_mode() == m.LOCAL_MODE


def test_qdrant_mode_parses_hosted_and_local(monkeypatch: pytest.MonkeyPatch) -> None:
    import shared.qdrant as m

    monkeypatch.setenv("QDRANT_MODE", "hosted")
    assert m.get_qdrant_mode() == m.HOSTED_MODE
    monkeypatch.setenv("QDRANT_MODE", "LOCAL")
    assert m.get_qdrant_mode() == m.LOCAL_MODE


def test_get_qdrant_client_default_timeout(monkeypatch):
    """QdrantClient receives timeout=60 by default."""
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.delenv("QDRANT_TIMEOUT", raising=False)
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    with patch("qdrant_client.QdrantClient") as mock_client:
        import shared.qdrant as m

        importlib.reload(m)
        m.get_qdrant_client()

    assert mock_client.call_args.kwargs["timeout"] == 60


def test_get_qdrant_client_custom_timeout(monkeypatch):
    """QdrantClient receives timeout from QDRANT_TIMEOUT env var."""
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_TIMEOUT", "120")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")

    with patch("qdrant_client.QdrantClient") as mock_client:
        import shared.qdrant as m

        importlib.reload(m)
        m.get_qdrant_client()

    assert mock_client.call_args.kwargs["timeout"] == 120


def test_get_qdrant_client_no_api_key(monkeypatch):
    """Without QDRANT_API_KEY, client is constructed without api_key kwarg."""
    monkeypatch.setenv("QDRANT_MODE", "local")
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
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_URL", "https://cloud.qdrant.io")
    monkeypatch.setenv("QDRANT_API_KEY", "secret-key")

    with patch("qdrant_client.QdrantClient") as mock_client:
        import shared.qdrant as m

        importlib.reload(m)
        m.get_qdrant_client()

    call_kwargs = mock_client.call_args.kwargs
    assert call_kwargs.get("url") == "https://cloud.qdrant.io"
    assert call_kwargs.get("api_key") == "secret-key"


def test_get_qdrant_client_hosted_requires_credentials(monkeypatch):
    monkeypatch.setenv("QDRANT_MODE", "hosted")
    monkeypatch.setenv("QDRANT_URL", "https://cloud.qdrant.io")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    with patch("qdrant_client.QdrantClient") as mock_client:
        import shared.qdrant as m

        importlib.reload(m)
        with pytest.raises(RuntimeError, match="QDRANT_MODE=hosted requires QDRANT_URL and QDRANT_API_KEY"):
            m.get_qdrant_client()

    assert mock_client.call_count == 0


def test_get_qdrant_config_error_for_hosted_mode(monkeypatch):
    import shared.qdrant as m

    monkeypatch.setenv("QDRANT_MODE", "hosted")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    assert m.get_qdrant_config_error() == (
        "QDRANT_MODE=hosted requires QDRANT_URL and QDRANT_API_KEY"
    )


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


def test_point_id_same_page_same_text_different_chunk_index():
    """Same filename/page/text but different chunk_index must not collide."""
    from ingestion.index import _make_point_id

    id1 = _make_point_id("doc.pdf", 5, "Revenue: $10M", element_type="paragraph", chunk_index=0)
    id2 = _make_point_id("doc.pdf", 5, "Revenue: $10M", element_type="paragraph", chunk_index=1)

    assert id1 != id2


def test_point_id_stable_same_inputs():
    """Same inputs always produce the same ID (stable hashing)."""
    from ingestion.index import _make_point_id

    id1 = _make_point_id("doc.pdf", 5, "Revenue: $10M", element_type="paragraph", chunk_index=0)
    id2 = _make_point_id("doc.pdf", 5, "Revenue: $10M", element_type="paragraph", chunk_index=0)

    assert id1 == id2
