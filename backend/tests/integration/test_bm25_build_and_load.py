from __future__ import annotations

from typing import cast

from llama_index.core.schema import TextNode
import pytest

from ingestion.index import build_bm25_index, load_bm25_index


@pytest.mark.integration
def test_build_and_load_bm25_index_roundtrips_nodes(tmp_path, sample_nodes: list[TextNode]):
    bm25_path = tmp_path / "bm25_index.pkl"
    build_bm25_index(sample_nodes, output_path=bm25_path)

    retriever = load_bm25_index(index_path=bm25_path)
    results = retriever.retrieve("net sales")

    assert results
    assert len(results) <= len(sample_nodes)

    assert any("sales" in cast(str, result.node.text).lower() for result in results)
