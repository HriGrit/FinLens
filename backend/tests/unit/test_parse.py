from __future__ import annotations

from ingestion.parse import doc_to_nodes, inject_heading_context


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
