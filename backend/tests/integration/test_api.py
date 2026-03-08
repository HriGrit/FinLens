from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from llama_index.core.schema import TextNode
from types import SimpleNamespace

from api.main import app


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
            SimpleNamespace(id="openrouter/free", name="Free Router", context_length=1000, description="router"),
            SimpleNamespace(id="qwen/qwen3-4b:free", name="Qwen", context_length=40000, description=None),
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
        return [
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
        ]

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


@pytest.mark.integration
def test_chat_endpoint_returns_not_found_when_no_nodes(monkeypatch):
    monkeypatch.setattr("api.main.retrieve_and_rerank", lambda **kwargs: [])

    with TestClient(app) as client:
        response = client.post("/chat", json={"query": "No match"})

        assert response.status_code == 404
        assert response.json()["detail"] == "No relevant context found for this query."
