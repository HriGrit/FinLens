from __future__ import annotations

from ingestion.chunk import split_paragraph_nodes
from ingestion.parse import doc_to_nodes, inject_heading_context


class _FakeProv:
    def __init__(self, page_no: int):
        self.page_no = page_no


class _FakeNode:
    def __init__(self, label: str, text: str, page_no: int = 1):
        self.label = label
        self.text = text
        self.prov = [_FakeProv(page_no)]


class _FakeTableNode(_FakeNode):
    def __init__(self, text: str, page_no: int = 1):
        super().__init__("table", "", page_no=page_no)
        self._markdown = text

    def export_to_markdown(self, doc=None) -> str:
        return self._markdown


class _FakeDoc:
    def __init__(self, items):
        self._items = items

    def iterate_items(self):
        for idx, item in enumerate(self._items):
            yield item, idx


class _FakeChunkNode:
    def __init__(self, text: str, metadata: dict | None = None):
        self.text = text
        self.metadata = metadata or {}


class _FakeSplitter:
    def get_nodes_from_documents(self, documents):
        src_idx = documents[0].metadata.get("_source_idx", 0) if documents else 0
        return [
            _FakeChunkNode("income 2022", metadata={"_source_idx": src_idx}),
            _FakeChunkNode("income 2023", metadata={"_source_idx": src_idx}),
        ]


_DOC_KWARGS = dict(filename="sample.pdf", company="3M", year="2022")


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

    assert [n.metadata["element_type"] for n in nodes] == [
        "heading",
        "paragraph",
        "table",
        "paragraph",
    ]
    assert [n.metadata["reading_order"] for n in nodes] == [0, 1, 2, 3]
    assert nodes[2].text.startswith("| Quarter |")


def test_doc_to_nodes_uses_table_markdown_export():
    table_text = "| Ticker | Price |\n| --- | --- |\n| MMM | 250 |"
    doc = _FakeDoc([_FakeTableNode(table_text)])

    nodes = doc_to_nodes(doc, **_DOC_KWARGS)

    assert len(nodes) == 1
    assert nodes[0].text == table_text


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


def test_split_paragraph_nodes_keeps_nonparagraph_in_place_with_chunks():
    nodes = doc_to_nodes(
        _FakeDoc(
            [
                _FakeNode("section_header", "Overview"),
                _FakeNode("text", "Income statement."),
                _FakeTableNode("|Q1|"),
            ]
        ),
        **_DOC_KWARGS,
    )
    enriched = inject_heading_context(nodes)
    chunks = split_paragraph_nodes(enriched, _FakeSplitter())

    assert chunks[0].metadata["element_type"] == "paragraph"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["element_type"] == "paragraph"
    assert chunks[1].metadata["chunk_index"] == 1
    assert chunks[2].metadata["element_type"] == "table"
    assert chunks[2].metadata["chunk_index"] == 0


def test_doc_to_nodes_populates_all_required_metadata_fields():
    from ingestion.parse import REQUIRED_METADATA_FIELDS

    doc = _FakeDoc([_FakeNode("text", "Some revenue text.")])
    nodes = doc_to_nodes(doc, **_DOC_KWARGS)

    for node in nodes:
        missing = REQUIRED_METADATA_FIELDS - node.metadata.keys()
        assert not missing, f"Missing fields: {missing}"
