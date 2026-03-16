from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from llama_index.core.schema import TextNode

from ingestion.chunk import split_paragraph_nodes
from ingestion.index import build_bm25_index
from ingestion.run_ingestion import (
    DocSpec,
    _build_batch,
    _load_chunk_artifacts_for_successful_docs,
    _mark_failed,
    _mark_pending,
    _mark_success,
    _save_chunk_artifact,
)


@pytest.mark.integration
def test_mixed_batch_records_success_and_failure(tmp_path, monkeypatch, sample_nodes: list[TextNode]):
    from ingestion import run_ingestion as ri

    chunks_dir = tmp_path / "chunks"
    monkeypatch.setattr(ri, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(ri, "build_qdrant_index", lambda *args, **kwargs: None)

    manifest: dict[str, dict] = {}
    good_name = "A_2022_10K.pdf"
    bad_name = "B_2022_10K.pdf"
    good_enriched = [sample_nodes[0]]
    bad_enriched: list[TextNode] = []

    _mark_pending(manifest, good_name)
    good_chunks = split_paragraph_nodes(good_enriched)
    _save_chunk_artifact(good_name, good_chunks)
    ri.build_qdrant_index(good_chunks, collection_name="test")
    _mark_success(manifest, good_name, chunk_count=len(good_chunks))

    _mark_pending(manifest, bad_name)
    if not bad_enriched:
        _mark_failed(manifest, bad_name, "parse_document returned 0 nodes")

    good_artifact = chunks_dir / "A_2022_10K.pkl"
    bad_artifact = chunks_dir / "B_2022_10K.pkl"
    assert manifest[good_name]["status"] == "success"
    assert manifest[bad_name]["status"] == "failed"
    assert good_artifact.exists()
    assert not bad_artifact.exists()


@pytest.mark.integration
def test_bm25_no_duplicates_across_reruns(tmp_path, monkeypatch, sample_nodes: list[TextNode]):
    from ingestion import run_ingestion as ri

    chunks_dir = tmp_path / "chunks"
    bm25_path = tmp_path / "bm25_index.pkl"
    monkeypatch.setattr(ri, "CHUNKS_DIR", chunks_dir)

    chunks = split_paragraph_nodes(sample_nodes)
    _save_chunk_artifact("A_2022_10K.pdf", chunks)
    manifest = {"A_2022_10K.pdf": {"status": "success"}}

    nodes_first = _load_chunk_artifacts_for_successful_docs(manifest)
    build_bm25_index(nodes_first, output_path=bm25_path)
    payload_first = pickle.loads(bm25_path.read_bytes())
    count_first = len(payload_first["nodes"])

    nodes_second = _load_chunk_artifacts_for_successful_docs(manifest)
    build_bm25_index(nodes_second, output_path=bm25_path)
    payload_second = pickle.loads(bm25_path.read_bytes())
    count_second = len(payload_second["nodes"])

    assert count_first == count_second


@pytest.mark.integration
def test_retry_failed_selects_correct_docs():
    all_specs = [
        DocSpec(path=Path("/tmp/A_2022_10K.pdf"), company="A", year="2022"),
        DocSpec(path=Path("/tmp/B_2022_10K.pdf"), company="B", year="2022"),
        DocSpec(path=Path("/tmp/C_2022_10K.pdf"), company="C", year="2022"),
        DocSpec(path=Path("/tmp/D_2022_10K.pdf"), company="D", year="2022"),
    ]
    manifest = {
        "A_2022_10K.pdf": {"status": "success"},
        "B_2022_10K.pdf": {"status": "failed"},
        "C_2022_10K.pdf": {"status": "pending"},
    }

    batch, _, _ = _build_batch(all_specs, manifest, retry_failed=True, limit=None)
    names = {spec.path.name for spec in batch}
    assert "A_2022_10K.pdf" not in names
    assert "B_2022_10K.pdf" in names
    assert "C_2022_10K.pdf" in names
    assert "D_2022_10K.pdf" in names
