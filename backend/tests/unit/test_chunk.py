from __future__ import annotations

import pytest
from llama_index.core.schema import TextNode

from ingestion.chunk import split_paragraph_nodes


class _FakeChunkNode(TextNode):
    pass


class _TwoChunksSplitter:
    def get_nodes_from_documents(self, documents):
        # Must propagate _source_idx from each source document to its output chunks
        # (the real SemanticSplitterNodeParser does this; I3 now asserts it)
        src_idx = documents[0].metadata.get("_source_idx", 0) if documents else 0
        return [
            _FakeChunkNode(text="Income rose in 2022.", metadata={"_source_idx": src_idx}),
            _FakeChunkNode(text="Net sales increased.", metadata={"_source_idx": src_idx}),
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


def test_page_number_none_does_not_raise():
    """A-1: node with page_number=None but all other required fields passes through."""
    node = TextNode(
        text="Revenue details.",
        metadata={
            "element_type": "paragraph",
            "page_number": None,
            "filename": "3M_2022_10K.pdf",
            "company": "3M",
            "year": "2022",
            "doc_type": "10-K",
        },
    )
    chunks = split_paragraph_nodes([node], splitter=_EmptySplitter())
    assert len(chunks) == 1
    assert chunks[0].metadata["page_number"] is None


def test_split_paragraph_nodes_preserves_required_metadata():
    bad_node = TextNode(
        text="Missing year metadata.",
        metadata={"element_type": "paragraph", "page_number": 1, "filename": "a.pdf", "company": "3M", "doc_type": "10-K"},
    )

    with pytest.raises(AssertionError, match="missing metadata fields"):
        split_paragraph_nodes([bad_node], splitter=_TwoChunksSplitter())


# ---------------------------------------------------------------------------
# I3 — _source_idx missing from chunk output raises AssertionError
# ---------------------------------------------------------------------------

class _NoSourceIdxSplitter:
    """Returns chunks that do NOT carry _source_idx — simulates a broken splitter."""

    def get_nodes_from_documents(self, documents):
        # Return one chunk per document but intentionally omit _source_idx
        return [TextNode(text="chunk text", metadata={"other_key": "value"}) for _ in documents]


def test_split_raises_when_chunk_missing_source_idx():
    """I3: split_paragraph_nodes raises AssertionError when _source_idx is absent from a chunk."""
    node = TextNode(
        text="Income rose in 2022.",
        metadata={
            "element_type": "paragraph",
            "page_number": 1,
            "filename": "3M_2022_10K.pdf",
            "company": "3M",
            "year": "2022",
            "doc_type": "10-K",
        },
    )
    with pytest.raises(AssertionError, match="_source_idx"):
        split_paragraph_nodes([node], splitter=_NoSourceIdxSplitter())


# ---------------------------------------------------------------------------
# I4 — Splitter cannot overwrite parse-time metadata
# ---------------------------------------------------------------------------

class _MetadataOverwriteSplitter:
    """Returns chunks that overwrite parse-time metadata fields."""

    def get_nodes_from_documents(self, documents):
        chunks = []
        for doc in documents:
            src_idx = doc.metadata.get("_source_idx", 0)
            chunk = TextNode(
                text="chunk text",
                metadata={
                    "_source_idx": src_idx,
                    # Try to overwrite these authoritative parse-time fields:
                    "element_type": "HACKED",
                    "page_number": 9999,
                    "filename": "evil.pdf",
                    "company": "EvilCorp",
                    "year": "9999",
                    "doc_type": "HACKED",
                },
            )
            chunks.append(chunk)
        return chunks


def test_splitter_cannot_overwrite_parse_time_metadata():
    """I4: parse-time metadata fields are restored from the original node after splitting."""
    original_node = TextNode(
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

    chunks = split_paragraph_nodes([original_node], splitter=_MetadataOverwriteSplitter())

    assert len(chunks) == 1
    meta = chunks[0].metadata
    assert meta["element_type"] == "paragraph", "element_type must be restored from original node"
    assert meta["page_number"] == 47, "page_number must be restored from original node"
    assert meta["filename"] == "3M_2022_10K.pdf", "filename must be restored from original node"
    assert meta["company"] == "3M", "company must be restored from original node"
    assert meta["year"] == "2022", "year must be restored from original node"
    assert meta["doc_type"] == "10-K", "doc_type must be restored from original node"
