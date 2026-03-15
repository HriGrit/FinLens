from __future__ import annotations

import pytest

from ingestion.parse import (
    assert_node_metadata,
    convert_pdf,
    doc_to_nodes,
    inject_heading_context,
)


class _FakeProv:
    def __init__(self, page_no: int):
        self.page_no = page_no


class _FakeNode:
    def __init__(self, label: str, text: str, page_no: int = 1, level: int = 0):
        self.label = label
        self.text = text
        self.prov = [_FakeProv(page_no)]
        self.tree_level = level


class _FakeTableNode(_FakeNode):
    def __init__(self, text: str, page_no: int = 1, level: int = 0):
        super().__init__("table", "", page_no=page_no, level=level)
        self._markdown = text

    def export_to_markdown(self, doc=None) -> str:
        return self._markdown


class _FakeDoc:
    def __init__(self, items):
        self._items = items

    def iterate_items(self):
        for item in self._items:
            yield item, 0


_DOC_KWARGS = dict(filename="sample.pdf", company="3M", year="2022", doc_type="10-K")


def test_doc_to_nodes_includes_tables_and_orders_reading_position():
    doc = _FakeDoc(
        [
            _FakeNode("section_header", "Item 1"),
            _FakeNode("text", "Revenue was up"),
            _FakeTableNode("| Quarter | Revenue |\n|---|---|\n| Q1 | 100 |"),
            _FakeNode("text", "CAPITAL DISCLOSURE"),
        ]
    )

    nodes = doc_to_nodes(doc, **_DOC_KWARGS)

    assert [n.metadata["element_type"] for n in nodes] == ["heading", "paragraph", "table", "paragraph"]
    assert nodes[2].text.startswith("| Quarter |")


def test_heading_context_injection_applies_to_next_nodes():
    doc = _FakeDoc(
        [
            _FakeNode("section_header", "Results"),
            _FakeNode("text", "Revenue was up."),
            _FakeTableNode("|Q1|"),
        ]
    )

    nodes = doc_to_nodes(doc, **_DOC_KWARGS)
    enriched = inject_heading_context(nodes)

    assert enriched[0].metadata["section_heading"] == "Results"
    assert enriched[0].text.startswith("[Section: Results]")
    assert enriched[1].metadata["section_heading"] == "Results"


def test_parse_populates_required_metadata_fields():
    doc = _FakeDoc([_FakeNode("text", "Some revenue text.")])
    nodes = doc_to_nodes(doc, **_DOC_KWARGS)
    for node in nodes:
        required = {"element_type", "page_number", "filename", "company", "year", "doc_type"}
        missing = required - set(node.metadata.keys())
        assert not missing


# ---------------------------------------------------------------------------
# I1 — Docling conversion non-success is rejected
# ---------------------------------------------------------------------------

class _FakeConversionResult:
    """Mimics a Docling ConversionResult with a configurable status."""

    class _Status:
        def __init__(self, name: str):
            self.name = name

        def __str__(self) -> str:
            return self.name

    def __init__(self, *, success: bool = True):
        self.status = self._Status("SUCCESS" if success else "FAILURE")
        self.document = _FakeDoc([_FakeNode("text", "parsed text.")])


class _FakeConverter:
    def __init__(self, *, success: bool = True):
        self._success = success

    def convert(self, pdf_path):
        return _FakeConversionResult(success=self._success)


def test_convert_pdf_succeeds_when_status_is_success():
    """I1: convert_pdf returns the document when Docling status is SUCCESS."""
    converter = _FakeConverter(success=True)
    doc = convert_pdf("dummy.pdf", converter)
    assert doc is not None


def test_convert_pdf_raises_when_conversion_fails():
    """I1: convert_pdf raises ValueError when Docling reports a non-SUCCESS status."""
    converter = _FakeConverter(success=False)
    with pytest.raises(ValueError, match="Docling conversion failed"):
        convert_pdf("dummy.pdf", converter)


def test_convert_pdf_accepts_result_without_status_attribute():
    """I1: convert_pdf is permissive when result.status is absent (legacy compat)."""

    class _NoStatusResult:
        document = _FakeDoc([_FakeNode("text", "parsed text.")])

    class _NoStatusConverter:
        def convert(self, _path):
            return _NoStatusResult()

    doc = convert_pdf("dummy.pdf", _NoStatusConverter())
    assert doc is not None


# ---------------------------------------------------------------------------
# I2 — None metadata values are rejected by assert_node_metadata
# ---------------------------------------------------------------------------

def test_assert_node_metadata_passes_when_all_fields_populated():
    """I2: assert_node_metadata passes for fully-populated nodes."""
    from llama_index.core.schema import TextNode

    node = TextNode(
        text="Revenue details.",
        metadata={
            "element_type": "paragraph",
            "page_number": 5,
            "filename": "sample.pdf",
            "company": "3M",
            "year": "2022",
            "doc_type": "10-K",
        },
    )
    assert_node_metadata([node])  # must not raise


def test_assert_node_metadata_raises_for_none_value():
    """I2: assert_node_metadata raises AssertionError when a required field is None."""
    from llama_index.core.schema import TextNode

    node = TextNode(
        text="Revenue details.",
        metadata={
            "element_type": "paragraph",
            "page_number": None,  # None value — should be rejected
            "filename": "sample.pdf",
            "company": "3M",
            "year": "2022",
            "doc_type": "10-K",
        },
    )
    with pytest.raises(AssertionError, match="None values"):
        assert_node_metadata([node])


def test_assert_node_metadata_raises_for_missing_field():
    """I2: assert_node_metadata still raises AssertionError when a field is absent."""
    from llama_index.core.schema import TextNode

    node = TextNode(
        text="Revenue details.",
        metadata={
            "element_type": "paragraph",
            "page_number": 5,
            # "filename" is missing
            "company": "3M",
            "year": "2022",
            "doc_type": "10-K",
        },
    )
    with pytest.raises(AssertionError, match="missing metadata fields"):
        assert_node_metadata([node])


# ---------------------------------------------------------------------------
# I7 — Unrecognized doc_type raises ValueError
# ---------------------------------------------------------------------------

def test_doc_to_nodes_accepts_valid_doc_types():
    """I7: doc_to_nodes accepts all entries from the allowlist."""
    doc = _FakeDoc([_FakeNode("text", "Some text.")])
    for valid_type in ("10-K", "10-Q", "8-K", "DEF14A"):
        nodes = doc_to_nodes(doc, filename="f.pdf", company="X", year="2022", doc_type=valid_type)
        assert len(nodes) == 1


def test_doc_to_nodes_accepts_non_hyphenated_variants():
    """I7: doc_to_nodes normalises non-hyphenated forms like 10K -> 10-K."""
    doc = _FakeDoc([_FakeNode("text", "Some text.")])
    nodes = doc_to_nodes(doc, filename="f.pdf", company="X", year="2022", doc_type="10K")
    assert nodes[0].metadata["doc_type"] == "10-K"


def test_doc_to_nodes_raises_for_unrecognized_doc_type():
    """I7: doc_to_nodes raises ValueError for unrecognized doc_type values."""
    doc = _FakeDoc([_FakeNode("text", "Some text.")])
    with pytest.raises(ValueError, match="Unrecognized doc_type"):
        doc_to_nodes(doc, filename="f.pdf", company="X", year="2022", doc_type="UNKNOWN")
