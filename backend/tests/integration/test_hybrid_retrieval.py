from __future__ import annotations

import os
import pytest
from llama_index.core.schema import TextNode

from ingestion.index import build_bm25_index
from retrieval import hybrid


@pytest.mark.integration
def test_hybrid_retrieve_merges_bm25_and_qdrant_candidates(
    tmp_path,
    monkeypatch,
    seeded_qdrant_collection,
    sample_nodes: list[TextNode],
):
    bm25_path = tmp_path / "bm25_index.pkl"
    build_bm25_index(sample_nodes, output_path=bm25_path)
    monkeypatch.setattr(hybrid, "BM25_INDEX_PATH", bm25_path)
    monkeypatch.setattr(hybrid, "QDRANT_COLLECTION", seeded_qdrant_collection)
    if "QDRANT_URL" not in os.environ:
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setattr(hybrid, "QDRANT_URL", os.environ.get("QDRANT_URL", "http://localhost:6333"))

    nodes = hybrid.hybrid_retrieve("What were 3M's net sales in 2022?", top_k=5, company="3M", year="2022")

    assert nodes
    assert all(hasattr(node, "text") for node in nodes)
    assert len(nodes) <= 5
    assert all(node.metadata.get("filename") for node in nodes)
