from __future__ import annotations

from llama_index.core.schema import TextNode

from retrieval import hybrid


def _node(text: str, source: str) -> TextNode:
    return TextNode(text=text, metadata={"source": source, "element_type": "paragraph"})


def test_fuse_results_deduplicates_by_text_and_prefers_dense_metadata_when_duplicate():
    bm25_nodes = [
        _node("A", "bm25"),
        _node("B", "bm25"),
        _node("C", "bm25"),
    ]
    dense_nodes = [
        _node("A", "dense"),
        _node("D", "dense"),
    ]

    fused = hybrid._fuse_results(bm25_nodes, dense_nodes)

    assert [n.text for n in fused] == ["A", "B", "D", "C"]
    assert fused[0].metadata["source"] == "dense"


def test_fuse_results_respects_rank_weighting():
    bm25_nodes = [
        _node("A", "bm25"),
        _node("B", "bm25"),
    ]
    dense_nodes = [
        _node("C", "dense"),
        _node("A", "dense"),
    ]

    fused = hybrid._fuse_results(bm25_nodes, dense_nodes)
    assert fused[0].text == "A"
    assert fused[1].text == "C"
