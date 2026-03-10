from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
from llama_index.core.schema import TextNode

from ingestion.run_ingestion import (
    DocSpec,
    _build_batch,
    _load_chunk_artifacts_for_successful_docs,
    _load_manifest,
    _mark_failed,
    _mark_pending,
    _mark_success,
    _migrate_registry_to_manifest,
    _parse_pdf_filename,
    _parse_batch,
    _rebuild_bm25_for_indexed_docs,
    _save_chunk_artifact,
    _save_manifest,
)


def _path(name: str) -> Path:
    return Path(name)


def test_standard_three_part_filename():
    spec = _parse_pdf_filename(_path("3M_2022_10K.pdf"))
    assert spec is not None
    assert spec.company == "3M"
    assert spec.year == "2022"
    assert spec.doc_type == "10-K"


def test_company_with_underscore():
    spec = _parse_pdf_filename(_path("JOHNSON_JOHNSON_2022_10K.pdf"))
    assert spec is not None
    assert spec.company == "JOHNSON_JOHNSON"
    assert spec.year == "2022"
    assert spec.doc_type == "10-K"


def test_four_part_numeric_suffix_rejected():
    spec = _parse_pdf_filename(_path("3M_2022_10K_10.pdf"))
    assert spec is None


def test_too_few_parts_rejected():
    spec = _parse_pdf_filename(_path("BADNAME.pdf"))
    assert spec is None


def test_10q_doc_type_hyphenated():
    spec = _parse_pdf_filename(_path("APPLE_2023_10Q.pdf"))
    assert spec is not None
    assert spec.doc_type == "10-Q"


def test_10q_with_quarter_year():
    spec = _parse_pdf_filename(_path("MSFT_2023Q1_10Q.pdf"))
    assert spec is not None
    assert spec.year == "2023Q1"
    assert spec.doc_type == "10-Q"


def test_manifest_roundtrip_success(tmp_path, monkeypatch):
    from ingestion import run_ingestion as ri

    manifest_path = tmp_path / "ingestion_manifest.json"
    monkeypatch.setattr(ri, "MANIFEST_PATH", manifest_path)
    manifest = {
        "3M_2022_10K.pdf": {
            "status": "success",
            "attempts": 1,
            "last_error": None,
            "chunk_count": 106,
            "updated_at": "2026-03-11T10:30:00Z",
        }
    }
    _save_manifest(manifest)
    loaded = _load_manifest()
    assert loaded == manifest


def test_manifest_roundtrip_failed(tmp_path, monkeypatch):
    from ingestion import run_ingestion as ri

    manifest_path = tmp_path / "ingestion_manifest.json"
    monkeypatch.setattr(ri, "MANIFEST_PATH", manifest_path)
    manifest = {
        "3M_2023_10K.pdf": {
            "status": "failed",
            "attempts": 2,
            "last_error": "FileNotFoundError: missing",
            "chunk_count": None,
            "updated_at": "2026-03-11T10:31:45Z",
        }
    }
    _save_manifest(manifest)
    loaded = _load_manifest()
    assert loaded == manifest


def test_mark_pending_increments_attempts():
    manifest = {
        "3M_2022_10K.pdf": {
            "status": "failed",
            "attempts": 2,
            "last_error": "boom",
            "chunk_count": None,
            "updated_at": "2026-03-11T10:31:45Z",
        }
    }
    _mark_pending(manifest, "3M_2022_10K.pdf")
    entry = manifest["3M_2022_10K.pdf"]
    assert entry["status"] == "pending"
    assert entry["attempts"] == 3


def test_mark_success_sets_chunk_count():
    manifest = {
        "3M_2022_10K.pdf": {
            "status": "pending",
            "attempts": 1,
            "last_error": "old",
            "chunk_count": None,
            "updated_at": "2026-03-11T10:31:45Z",
        }
    }
    _mark_success(manifest, "3M_2022_10K.pdf", chunk_count=12)
    entry = manifest["3M_2022_10K.pdf"]
    assert entry["status"] == "success"
    assert entry["chunk_count"] == 12
    assert entry["last_error"] is None


def test_mark_failed_records_error():
    manifest = {
        "3M_2022_10K.pdf": {
            "status": "pending",
            "attempts": 1,
            "last_error": None,
            "chunk_count": 10,
            "updated_at": "2026-03-11T10:31:45Z",
        }
    }
    _mark_failed(manifest, "3M_2022_10K.pdf", "oops")
    entry = manifest["3M_2022_10K.pdf"]
    assert entry["status"] == "failed"
    assert entry["last_error"] == "oops"
    assert entry["chunk_count"] is None


def _specs() -> list[DocSpec]:
    return [
        DocSpec(path=Path("/tmp/A_2022_10K.pdf"), company="A", year="2022"),
        DocSpec(path=Path("/tmp/B_2022_10K.pdf"), company="B", year="2022"),
        DocSpec(path=Path("/tmp/C_2022_10K.pdf"), company="C", year="2022"),
    ]


def test_batch_selection_excludes_success():
    manifest = {
        "A_2022_10K.pdf": {"status": "success"},
        "B_2022_10K.pdf": {"status": "failed"},
    }
    batch, _, _ = _build_batch(_specs(), manifest, retry_failed=False, limit=None)
    names = {s.path.name for s in batch}
    assert "A_2022_10K.pdf" not in names


def test_batch_selection_retry_failed_includes_failed():
    manifest = {
        "A_2022_10K.pdf": {"status": "success"},
        "B_2022_10K.pdf": {"status": "failed"},
    }
    batch, _, _ = _build_batch(_specs(), manifest, retry_failed=True, limit=None)
    names = {s.path.name for s in batch}
    assert "B_2022_10K.pdf" in names
    assert "A_2022_10K.pdf" not in names


def test_batch_selection_pending_always_requeued():
    manifest = {
        "A_2022_10K.pdf": {"status": "success"},
        "B_2022_10K.pdf": {"status": "pending"},
    }
    batch, _, _ = _build_batch(_specs(), manifest, retry_failed=False, limit=None)
    names = {s.path.name for s in batch}
    assert "B_2022_10K.pdf" in names


def test_migrate_registry_to_manifest(tmp_path, monkeypatch):
    from ingestion import run_ingestion as ri

    reg_path = tmp_path / "ingestion_registry.json"
    reg_path.write_text(json.dumps({"ingested": ["A_2022_10K.pdf", "B_2023_10K.pdf"]}))
    monkeypatch.setattr(ri, "REGISTRY_PATH", reg_path)

    manifest = _migrate_registry_to_manifest()
    assert set(manifest.keys()) == {"A_2022_10K.pdf", "B_2023_10K.pdf"}
    assert all(entry["status"] == "success" for entry in manifest.values())
    assert all(entry["attempts"] == 1 for entry in manifest.values())
    assert all(entry["chunk_count"] is None for entry in manifest.values())


def test_chunk_artifact_save_and_load(tmp_path, monkeypatch):
    from ingestion import run_ingestion as ri

    chunks_dir = tmp_path / "chunks"
    bm25_path = tmp_path / "bm25_index.pkl"
    monkeypatch.setattr(ri, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(ri, "BM25_INDEX_PATH", bm25_path)
    nodes = [
        TextNode(
            text="Revenue grew.",
            metadata={"filename": "A_2022_10K.pdf", "company": "A", "year": "2022"},
        )
    ]
    _save_chunk_artifact("A_2022_10K.pdf", nodes)
    manifest = {"A_2022_10K.pdf": {"status": "success"}}
    loaded = _load_chunk_artifacts_for_successful_docs(manifest)
    assert len(loaded) == 1
    assert loaded[0].text == "Revenue grew."


def test_load_chunks_skips_failed_docs(tmp_path, monkeypatch):
    from ingestion import run_ingestion as ri

    chunks_dir = tmp_path / "chunks"
    bm25_path = tmp_path / "bm25_index.pkl"
    monkeypatch.setattr(ri, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(ri, "BM25_INDEX_PATH", bm25_path)
    nodes = [
        TextNode(
            text="Only successful docs should load.",
            metadata={"filename": "A_2022_10K.pdf", "company": "A", "year": "2022"},
        )
    ]
    _save_chunk_artifact("A_2022_10K.pdf", nodes)
    _save_chunk_artifact("B_2022_10K.pdf", nodes)

    manifest = {
        "A_2022_10K.pdf": {"status": "success"},
        "B_2022_10K.pdf": {"status": "failed"},
    }
    loaded = _load_chunk_artifacts_for_successful_docs(manifest)
    assert len(loaded) == 1
    assert loaded[0].metadata["filename"] == "A_2022_10K.pdf"


def test_load_chunks_falls_back_to_existing_bm25_for_migrated_success(tmp_path, monkeypatch):
    from ingestion import run_ingestion as ri

    chunks_dir = tmp_path / "chunks"
    bm25_path = tmp_path / "bm25_index.pkl"
    monkeypatch.setattr(ri, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(ri, "BM25_INDEX_PATH", bm25_path)

    fallback_node = TextNode(
        text="Legacy BM25 node.",
        metadata={"filename": "A_2022_10K.pdf", "company": "A", "year": "2022"},
    )
    bm25_path.write_bytes(pickle.dumps({"nodes": [fallback_node]}))

    manifest = {"A_2022_10K.pdf": {"status": "success"}}
    loaded = _load_chunk_artifacts_for_successful_docs(manifest)
    assert len(loaded) == 1
    assert loaded[0].text == "Legacy BM25 node."


def test_parse_batch_marks_failed_docs_and_continues(monkeypatch):
    from ingestion import run_ingestion as ri

    class _FakeFuture:
        def __init__(self, fn, spec):
            self._fn = fn
            self._spec = spec

        def result(self):
            return self._fn(self._spec)

        def cancel(self):
            return True

    class _FakeExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, spec):
            return _FakeFuture(fn, spec)

    def _fake_parse_document(spec: DocSpec):
        if spec.path.name == "A_2022_10K.pdf":
            raise RuntimeError("broken pdf")
        return [TextNode(text="ok", metadata={"filename": spec.path.name})]

    monkeypatch.setattr(ri, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(ri, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(ri, "parse_document", _fake_parse_document)
    monkeypatch.setattr(ri, "_save_manifest", lambda manifest: None)

    manifest: dict[str, dict] = {}
    batch = _specs()[:2]
    parsed_by_filename, parse_failed = _parse_batch(
        batch=batch,
        manifest=manifest,
        workers=2,
        continue_on_error=True,
    )

    assert parse_failed is True
    assert "B_2022_10K.pdf" in parsed_by_filename
    assert manifest["A_2022_10K.pdf"]["status"] == "failed"
    assert manifest["B_2022_10K.pdf"]["status"] == "pending"


def test_rebuild_bm25_does_not_mark_success_when_write_fails(tmp_path, monkeypatch):
    from ingestion import run_ingestion as ri

    chunks_dir = tmp_path / "chunks"
    bm25_path = tmp_path / "bm25_index.pkl"
    monkeypatch.setattr(ri, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(ri, "BM25_INDEX_PATH", bm25_path)
    monkeypatch.setattr(ri, "build_bm25_index", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk full")))

    nodes = [
        TextNode(
            text="Revenue grew.",
            metadata={"filename": "A_2022_10K.pdf", "company": "A", "year": "2022"},
        )
    ]
    _save_chunk_artifact("A_2022_10K.pdf", nodes)
    manifest = {
        "A_2022_10K.pdf": {
            "status": "pending",
            "attempts": 1,
            "last_error": None,
            "chunk_count": None,
            "updated_at": "2026-03-11T10:30:00Z",
        }
    }

    with pytest.raises(RuntimeError, match="disk full"):
        _rebuild_bm25_for_indexed_docs(manifest, {"A_2022_10K.pdf": 1})

    assert manifest["A_2022_10K.pdf"]["status"] == "pending"


def test_save_manifest_is_atomic(tmp_path, monkeypatch):
    from ingestion import run_ingestion as ri

    manifest_path = tmp_path / "ingestion_manifest.json"
    monkeypatch.setattr(ri, "MANIFEST_PATH", manifest_path)
    manifest = {
        "A_2022_10K.pdf": {
            "status": "success",
            "attempts": 1,
            "last_error": None,
            "chunk_count": 3,
            "updated_at": "2026-03-11T10:30:00Z",
        }
    }
    _save_manifest(manifest)
    payload = json.loads(manifest_path.read_text())
    assert payload == manifest
