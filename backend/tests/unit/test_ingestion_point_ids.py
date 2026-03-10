from __future__ import annotations

from llama_index.core.schema import TextNode

from ingestion.index import _make_point_id


def _node(chunk_index: int) -> TextNode:
    return TextNode(
        text="same text",
        metadata={
            "filename": "3M_2022_10K.pdf",
            "page_number": 3,
            "element_type": "paragraph",
            "chunk_index": chunk_index,
            "tree_level": 1,
            "reading_order": 7,
        },
    )


def test_point_id_is_stable_for_the_same_node():
    first = _node(0)
    second = _node(0)
    assert _make_point_id(first) == _make_point_id(second)


def test_point_id_changes_when_chunk_index_changes():
    first = _node(0)
    second = _node(1)
    assert _make_point_id(first) != _make_point_id(second)
