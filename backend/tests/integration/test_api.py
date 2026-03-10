from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from llama_index.core.schema import TextNode
from generation.openrouter_models import FreeModelOption

from api.main import app


class _FakeAsyncResponse:
    def __init__(self, status_code: int = 200, text: str = "ok"):
        self.status_code = status_code
        self.text = text


class _FakeAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url: str, timeout: int = 2):
        return _FakeAsyncResponse()


@pytest.mark.integration
def test_health_endpoint_returns_ok():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.integration
def test_free_models_endpoint_returns_model_options(monkeypatch):
    monkeypatch.setattr(
        "api.main.get_free_models",
        lambda: [
            FreeModelOption(id="openrouter/free", name="Free Router", context_length=1000, description="router"),
            FreeModelOption(id="qwen/qwen3-4b:free", name="Qwen", context_length=40000, description=None),
        ],
    )

    with TestClient(app) as client:
        response = client.get("/models/free")

        assert response.status_code == 200
        assert response.json()[0]["id"] == "openrouter/free"
        assert response.json()[1]["id"] == "qwen/qwen3-4b:free"


@pytest.mark.integration
def test_chat_endpoint_routes_retrieve_and_generate(monkeypatch):
    def _nodes(*args, **kwargs):
        return (
            [
                TextNode(
                    text="3M reported net sales of $35.4 billion.",
                    metadata={
                        "company": "3M",
                        "year": "2022",
                        "doc_type": "10-K",
                        "page_number": 47,
                        "filename": "3M_2022_10K.pdf",
                    },
                ),
            ],
            1,
        )

    def _response(*args, **kwargs):
        assert kwargs["model"] == "qwen/qwen3-4b:free"
        return {
            "answer": "The company reported strong net sales.",
            "citations": [{"company": "3M", "year": "2022", "doc_type": "10-K"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22, "cost_usd": 0.0},
            "model": "mock",
        }

    monkeypatch.setattr("api.main.retrieve_and_rerank", _nodes)
    monkeypatch.setattr("api.main.generate", _response)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "query": "What were 3M sales in 2022?",
                "company": "3M",
                "year": "2022",
                "model": "qwen/qwen3-4b:free",
                "rerank_top_k": 1,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "The company reported strong net sales."
        assert body["usage"]["total_tokens"] == 22
        assert body["trace_id"]
        assert body["reasoning"]["summary"].startswith("FinLens answered")
        assert body["reasoning"]["retrieval"]["retrieval_top_k"] == 20
        assert body["reasoning"]["rerank"]["requested_rerank_top_k"] == 1
        assert body["reasoning"]["generation"]["model"] == "mock"
        assert body["reasoning"]["generation"]["cost_usd"] == 0.0


@pytest.mark.integration
def test_chat_endpoint_returns_not_found_when_no_nodes(monkeypatch):
    monkeypatch.setattr("api.main.retrieve_and_rerank", lambda **kwargs: ([], 0))

    with TestClient(app) as client:
        response = client.post("/chat", json={"query": "No match"})

        assert response.status_code == 404
        assert response.json()["detail"] == "No relevant context found for this query."


@pytest.mark.integration
def test_chat_returns_502_when_model_returns_empty(monkeypatch):
    def _nodes(*args, **kwargs):
        return (
            [
                TextNode(
                    text="3M reported net sales of $35.4 billion.",
                    metadata={
                        "company": "3M",
                        "year": "2022",
                        "doc_type": "10-K",
                        "page_number": 47,
                        "filename": "3M_2022_10K.pdf",
                    },
                )
            ],
            1,
        )

    monkeypatch.setattr("api.main.retrieve_and_rerank", _nodes)
    monkeypatch.setattr("api.main.generate", lambda **kwargs: (_ for _ in ()).throw(ValueError("empty")))

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "query": "What were 3M sales in 2022?",
                "company": "3M",
                "year": "2022",
                "model": "qwen/qwen3-4b:free",
            },
        )

    assert response.status_code == 502


@pytest.mark.integration
def test_chat_rejects_empty_query():
    with TestClient(app) as client:
        response = client.post("/chat", json={"query": ""})

    assert response.status_code == 422


@pytest.mark.integration
def test_chat_rejects_whitespace_only_query():
    with TestClient(app) as client:
        response = client.post("/chat", json={"query": "   "})

    assert response.status_code == 422


@pytest.mark.integration
def test_chat_rejects_oversized_query():
    with TestClient(app) as client:
        response = client.post("/chat", json={"query": "x" * 2001})

    assert response.status_code == 422


@pytest.mark.integration
def test_reasoning_retrieved_candidates_reflect_pre_rerank_count(monkeypatch):
    def _nodes(*args, **kwargs):
        return (
            [
                TextNode(
                    text="3M reported net sales of $35.4 billion.",
                    metadata={
                        "company": "3M",
                        "year": "2022",
                        "doc_type": "10-K",
                        "page_number": 47,
                        "filename": "3M_2022_10K.pdf",
                    },
                )
            ],
            20,
        )

    monkeypatch.setattr(
        "api.main.retrieve_and_rerank",
        _nodes,
    )
    monkeypatch.setattr(
        "api.main.generate",
        lambda *args, **kwargs: {
            "answer": "The company reported strong net sales.",
            "citations": [{"company": "3M", "year": "2022", "doc_type": "10-K"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22, "cost_usd": 0.0},
            "model": "mock",
        },
    )

    with TestClient(app) as client:
        response = client.post("/chat", json={"query": "test", "rerank_top_k": 1})
        assert response.status_code == 200

    body = response.json()
    assert body["reasoning"]["retrieval"]["retrieved_candidates"] == 20
    assert body["reasoning"]["rerank"]["selected_nodes"] == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_services_status_builds_qdrant_client_inside_to_thread(monkeypatch):
    calls = {"client_created": 0, "get_collections": 0}

    class _FakeQdrantClient:
        def get_collections(self):
            calls["get_collections"] += 1
            return []

    def _get_qdrant_client():
        calls["client_created"] += 1
        return _FakeQdrantClient()

    async def _to_thread(func, /, *args, **kwargs):
        assert calls["client_created"] == 0
        return func(*args, **kwargs)

    monkeypatch.setattr("api.main.get_qdrant_client", _get_qdrant_client)
    monkeypatch.setattr("api.main.asyncio.to_thread", _to_thread)
    monkeypatch.setattr("api.main.httpx.AsyncClient", _FakeAsyncClient)

    from api.main import services_status

    status = await services_status()

    assert calls["client_created"] == 1
    assert calls["get_collections"] == 1
    assert status["qdrant"]["status"] == "ok"
