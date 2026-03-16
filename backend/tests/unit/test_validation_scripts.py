from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_ingestion_uses_configured_collection(monkeypatch, tmp_path):
    module = _load_module("validate_ingestion_script", "tests/validate_ingestion.py")

    class _FakeClient:
        def __init__(self) -> None:
            self.collections_seen: list[str] = []

        def collection_exists(self, collection_name: str) -> bool:
            self.collections_seen.append(collection_name)
            return True

        def get_collection(self, collection_name: str):
            self.collections_seen.append(collection_name)
            return SimpleNamespace(points_count=3)

        def query_points(self, collection_name: str, **kwargs):
            self.collections_seen.append(collection_name)
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="p1",
                        payload={
                            "element_type": "paragraph",
                            "page_number": 1,
                            "filename": "3M_2022_10K.pdf",
                            "company": "3M",
                            "year": "2022",
                            "doc_type": "10-K",
                        },
                    )
                ]
            )

    fake_client = _FakeClient()
    monkeypatch.setattr(module, "get_qdrant_client", lambda: fake_client)
    monkeypatch.setattr(module, "get_qdrant_collection", lambda: "finlens_chunks_ci")
    monkeypatch.setattr(module, "get_embed_model", lambda: SimpleNamespace(get_text_embedding=lambda _text: [0.0] * 4))
    bm25_path = tmp_path / "bm25_index.pkl"
    bm25_path.write_bytes(b"ok")
    monkeypatch.setattr(module, "BM25_PATH", bm25_path)
    monkeypatch.setattr(
        module,
        "load_bm25_index",
        lambda _path: SimpleNamespace(retrieve=lambda _query: [SimpleNamespace(node="ok")]),
    )

    module.validate_ingestion()

    assert fake_client.collections_seen
    assert set(fake_client.collections_seen) == {"finlens_chunks_ci"}


def test_validate_ingestion_uses_public_bm25_loader_without_private_attrs(monkeypatch, tmp_path):
    module = _load_module("validate_ingestion_script_bm25", "tests/validate_ingestion.py")
    monkeypatch.setattr(module, "get_qdrant_client", lambda: SimpleNamespace(
        collection_exists=lambda _collection: True,
        get_collection=lambda _collection: SimpleNamespace(points_count=2),
        query_points=lambda **kwargs: SimpleNamespace(points=[
            SimpleNamespace(
                id="1",
                payload={
                    "element_type": "paragraph",
                    "page_number": 1,
                    "filename": "f.pdf",
                    "company": "3M",
                    "year": "2022",
                    "doc_type": "10-K",
                },
            )
        ]),
    ))
    monkeypatch.setattr(module, "get_qdrant_collection", lambda: "finlens_chunks_ci")
    monkeypatch.setattr(module, "get_embed_model", lambda: SimpleNamespace(get_text_embedding=lambda _text: [0.0] * 4))

    class _FakeRetriever:
        def retrieve(self, _query: str):
            return [SimpleNamespace(node="ok")]

        def __getattr__(self, name: str):
            if name == "bm25":
                raise AssertionError("validation script must not read private bm25 internals")
            raise AttributeError(name)

    bm25_path = tmp_path / "bm25_index.pkl"
    bm25_path.write_bytes(b"ok")
    monkeypatch.setattr(module, "BM25_PATH", bm25_path)
    monkeypatch.setattr(module, "load_bm25_index", lambda _path: _FakeRetriever())

    module.validate_ingestion()
