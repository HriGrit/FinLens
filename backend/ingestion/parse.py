"""
parse.py — DoclingDocument -> LlamaIndex TextNode conversion.

Milestone coverage: M2.1 (parse.py extracted), M2.2 (assertion block).
"""
from typing import Any, Iterable, Optional

from llama_index.core.schema import TextNode


def _normalize_label(label_raw: object) -> str:
    label = getattr(label_raw, "value", str(label_raw))
    if not label:
        return "unknown"

    label_lower = label.lower()
    if "." in label_lower:
        label_lower = label_lower.split(".")[-1]

    if "section_header" in label_lower:
        return "heading"
    if label_lower == "text":
        return "paragraph"

    return label_lower


def _iter_doc_items(dl_doc: Any) -> Iterable[tuple[Any, Optional[int]]]:
    for entry in dl_doc.iterate_items():
        if isinstance(entry, tuple) and len(entry) >= 2:
            yield entry[0], entry[1]
        else:
            yield entry, None


def _extract_page_number(item: Any) -> Optional[int]:
    if hasattr(item, "prov") and item.prov:
        prov = item.prov[0]
        # Docling uses `.page` (not `.page_no`) on the Prov model
        return getattr(prov, "page", None) or getattr(prov, "page_no", None)
    return None


def _extract_item_text(item: Any, doc: Any, label: str) -> str:
    if label == "table" and hasattr(item, "export_to_markdown"):
        return item.export_to_markdown(doc=doc).strip()
    return str(getattr(item, "text", "")).strip()


def doc_to_nodes(
    dl_doc: Any,
    filename: str,
    company: str,
    year: str,
    doc_type: str = "10-K",
) -> list[TextNode]:
    """Walk DoclingDocument in reading order and emit typed TextNode objects.

    All 6 required metadata fields are populated on every node:
    element_type, page_number, filename, company, year, doc_type.
    """
    nodes: list[TextNode] = []
    skip_labels = {"page_header", "page_footer"}

    for reading_order, (item, tree_level) in enumerate(_iter_doc_items(dl_doc)):
        label = _normalize_label(getattr(item, "label", "unknown"))
        if label in skip_labels:
            continue

        text = _extract_item_text(item, dl_doc, label)
        if not text:
            continue

        metadata: dict[str, Any] = {
            "element_type": label,
            "page_number": _extract_page_number(item),
            "tree_level": tree_level,
            "reading_order": reading_order,
            "filename": filename,
            "company": company,
            "year": year,
            "doc_type": doc_type,
            "source_item_class": item.__class__.__name__,
        }

        nodes.append(TextNode(text=text, metadata=metadata))

    return nodes


def inject_heading_context(nodes: list[TextNode]) -> list[TextNode]:
    """Prepend section heading to each non-heading node; drop heading nodes."""
    current_heading = ""
    enriched: list[TextNode] = []

    for node in nodes:
        if node.metadata.get("element_type") == "heading":
            current_heading = node.text.strip()
            continue

        node_metadata = dict(node.metadata)
        node_text = node.text
        if current_heading:
            node_metadata["section_heading"] = current_heading
            node_text = f"[Section: {current_heading}]\n\n{node_text}"

        enriched.append(TextNode(text=node_text, metadata=node_metadata))

    return enriched


REQUIRED_METADATA_FIELDS = frozenset(
    {"element_type", "page_number", "filename", "company", "year", "doc_type"}
)


def assert_node_metadata(nodes: list[TextNode]) -> None:
    """Assert all 6 required metadata fields are present on every node."""
    for i, node in enumerate(nodes):
        missing = REQUIRED_METADATA_FIELDS - node.metadata.keys()
        assert not missing, f"Node {i} missing metadata fields: {missing}"
