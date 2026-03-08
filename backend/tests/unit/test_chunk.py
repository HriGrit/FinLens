from __future__ import annotations

import pytest
from llama_index.core.schema import TextNode

from ingestion.chunk import split_paragraph_nodes


class _FakeChunkNode(TextNode):
    pass


class _TwoChunksSplitter:
    def get_nodes_from_documents(self, _documents):
        return [
            _FakeChunkNode(text="Income rose in 2022."),
            _FakeChunkNode(text="Net sales increased."),
        ]


class _EmptySplitter:
    def get_nodes_from_documents(self, _documents):
        return []


def test_split_paragraph_nodes_chunks_only_paragraph_nodes():
    paragraph_node = TextNode(
        text="Income rose in 2022. Net sales increased.",
        metadata={
            "element_type": "paragraph",
            "page_number": 47,
            "filename": "3M_2022_10K.pdf",
            "company": "3M",
            "year": "2022",
            "doc_type": "10-K",
        },
    )
    heading_node = TextNode(
        text="Segment Results",
        metadata={
            "element_type": "heading",
            "page_number": 49,
            "filename": "3M_2022_10K.pdf",
            "company": "3M",
            "year": "2022",
            "doc_type": "10-K",
        },
    )

    chunks = split_paragraph_nodes([paragraph_node, heading_node], splitter=_TwoChunksSplitter())

    assert [chunk.metadata["element_type"] for chunk in chunks] == ["paragraph", "paragraph", "heading"]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 0]


def test_split_paragraph_nodes_keeps_original_text_when_splitter_is_empty():
    paragraph_node = TextNode(
        text="Income rose in 2022.",
        metadata={
            "element_type": "paragraph",
            "page_number": 47,
            "filename": "3M_2022_10K.pdf",
            "company": "3M",
            "year": "2022",
            "doc_type": "10-K",
        },
    )

    chunks = split_paragraph_nodes([paragraph_node], splitter=_EmptySplitter())

    assert len(chunks) == 1
    assert chunks[0].text == "Income rose in 2022."
    assert chunks[0].metadata["chunk_index"] == 0


def test_split_paragraph_nodes_preserves_required_metadata():
    bad_node = TextNode(
        text="Missing year metadata.",
        metadata={"element_type": "paragraph", "page_number": 1, "filename": "a.pdf", "company": "3M", "doc_type": "10-K"},
    )

    with pytest.raises(AssertionError, match="missing metadata fields"):
        split_paragraph_nodes([bad_node], splitter=_TwoChunksSplitter())
