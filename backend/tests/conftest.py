from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest
from llama_index.core.schema import TextNode


class _DummyEmbeddingModel:
    """Fast embedding stub used by tests that need a stable vector shape."""

    vector_dim = 1024

    def get_text_embedding(self, _text: str) -> list[float]:
        return [0.0] * self.vector_dim

    def get_text_embedding_batch(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        return [[0.0] * self.vector_dim for _ in texts]


def _default_sample_nodes() -> list[TextNode]:
    return [
        TextNode(
            text="3M reported net sales of $35.4 billion in 2022.",
            metadata={
                "element_type": "paragraph",
                "page_number": 47,
                "filename": "3M_2022_10K.pdf",
                "company": "3M",
                "year": "2022",
                "doc_type": "10-K",
            },
        ),
        TextNode(
            text="3M segment revenue details in the 2023 quarter.",
            metadata={
                "element_type": "paragraph",
                "page_number": 55,
                "filename": "3M_2023Q2_10Q.pdf",
                "company": "3M",
                "year": "2023",
                "doc_type": "10-Q",
            },
        ),
        TextNode(
            text="Adobe filed a 10-K for 2022 with comparable operating metrics.",
            metadata={
                "element_type": "paragraph",
                "page_number": 30,
                "filename": "ADOBE_2022_10K.pdf",
                "company": "Adobe",
                "year": "2022",
                "doc_type": "10-K",
            },
        ),
    ]


@pytest.fixture(scope="session")
def qdrant_url() -> str:
    return os.getenv("QDRANT_URL", "http://localhost:6333")


@pytest.fixture(scope="session")
def test_collection_name() -> str:
    return os.getenv("TEST_QDRANT_COLLECTION", "finlens_chunks_test")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture()
def sample_pdf_path(repo_root: Path) -> Path:
    return repo_root.parent / os.getenv("TEST_DOC_PATH", "3M_2022_10K_10.pdf")


@pytest.fixture()
def sample_nodes() -> list[TextNode]:
    return _default_sample_nodes()


@pytest.fixture()
def fake_embed_model():
    return _DummyEmbeddingModel()


@pytest.fixture(autouse=True)
def _patch_embedding_model(monkeypatch: pytest.MonkeyPatch, fake_embed_model: _DummyEmbeddingModel):
    from ingestion import embed

    monkeypatch.setattr(
        embed,
        "get_embed_model",
        lambda _model_name="Alibaba-NLP/gte-modernbert-base": fake_embed_model,
    )
    yield


@pytest.fixture()
def qdrant_client(qdrant_url: str, request: pytest.FixtureRequest):
    if "integration" not in request.keywords and "live" not in request.keywords:
        return None

    try:
        from qdrant_client import QdrantClient
    except Exception as exc:
        pytest.skip(f"qdrant-client not available: {exc}")

    client = QdrantClient(url=qdrant_url)
    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"Qdrant not reachable at {qdrant_url}: {exc}")
    return client


@pytest.fixture()
def isolated_qdrant_collection(qdrant_client, test_collection_name: str) -> Generator[str, None, None]:
    if qdrant_client is None:
        pytest.skip("Integration test requested without qdrant client.")

    if qdrant_client.collection_exists(test_collection_name):
        qdrant_client.delete_collection(test_collection_name)

    try:
        yield test_collection_name
    finally:
        if qdrant_client.collection_exists(test_collection_name):
            qdrant_client.delete_collection(test_collection_name)


@pytest.fixture()
def seeded_qdrant_collection(
    qdrant_client,
    isolated_qdrant_collection: str,
    sample_nodes: list[TextNode],
):
    if qdrant_client is None:
        pytest.skip("Integration test requested without qdrant client.")

    from ingestion.index import build_qdrant_index

    build_qdrant_index(sample_nodes, collection_name=isolated_qdrant_collection)
    return isolated_qdrant_collection


@pytest.fixture()
def requires_openrouter():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY is not set.")
    return key


@pytest.fixture()
def requires_langfuse():
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        pytest.skip("Langfuse keys are missing.")
    return {"public_key": public_key, "secret_key": secret_key}
