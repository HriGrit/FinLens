"""
chunk.py — SemanticSplitter wrapper with metadata assertion.

Milestone coverage: M2.2 (assert page_number in node.metadata for every node).
"""
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import TextNode

from .embed import get_embed_model
from .parse import REQUIRED_METADATA_FIELDS


def build_splitter(buffer_size: int = 1, breakpoint_percentile_threshold: int = 95) -> SemanticSplitterNodeParser:
    return SemanticSplitterNodeParser(
        embed_model=get_embed_model(),
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
    )


def split_paragraph_nodes(
    nodes: list[TextNode],
    splitter: SemanticSplitterNodeParser | None = None,
) -> list[TextNode]:
    """Semantically split paragraph nodes; pass non-paragraph nodes through unchanged.

    Batches all paragraph nodes into a single SemanticSplitter call to reduce
    per-call embedding overhead.
    """
    from collections import defaultdict

    if splitter is None:
        splitter = build_splitter()

    para_nodes = [(i, n) for i, n in enumerate(nodes) if n.metadata.get("element_type") == "paragraph"]
    non_para = [(i, n) for i, n in enumerate(nodes) if n.metadata.get("element_type") != "paragraph"]

    # Build source docs tagged with their original position index
    source_docs = []
    for src_idx, (_, node) in enumerate(para_nodes):
        meta = dict(node.metadata)
        meta["_source_idx"] = src_idx
        source_docs.append(Document(text=node.text, metadata=meta))

    # One batched splitter call instead of N individual calls
    all_split = splitter.get_nodes_from_documents(source_docs) if source_docs else []

    # I3: assert every output chunk from the splitter carries _source_idx
    for chunk_idx, chunk_node in enumerate(all_split):
        assert "_source_idx" in chunk_node.metadata, (
            f"Chunk {chunk_idx} produced by splitter is missing '_source_idx' in metadata. "
            "The splitter may not be preserving source metadata."
        )

    # Group output chunks by source paragraph
    grouped: dict[int, list] = defaultdict(list)
    for chunk_node in all_split:
        src_idx = chunk_node.metadata.get("_source_idx", 0)
        grouped[src_idx].append(chunk_node)

    # Reconstruct output preserving original ordering
    out: list[TextNode | list] = [None] * len(nodes)  # type: ignore[assignment]

    for orig_pos, node in non_para:
        meta = dict(node.metadata)
        meta.setdefault("chunk_index", 0)
        out[orig_pos] = TextNode(text=node.text, metadata=meta)

    for src_idx, (orig_pos, node) in enumerate(para_nodes):
        chunks = grouped.get(src_idx, [])
        if not chunks:
            meta = dict(node.metadata)
            meta["chunk_index"] = 0
            out[orig_pos] = TextNode(text=node.text, metadata=meta)
            continue

        result_nodes = []
        for chunk_index, chunk_node in enumerate(chunks):
            chunk_text = getattr(chunk_node, "text", "").strip()
            if not chunk_text:
                continue
            meta = dict(node.metadata)
            meta.update({k: v for k, v in chunk_node.metadata.items() if not k.startswith("_")})
            # I4: restore authoritative parse-time metadata fields from the original node,
            # overriding whatever the splitter may have set.
            _PARSE_TIME_FIELDS = (
                "element_type", "page_number", "filename", "company", "year", "doc_type"
            )
            for field in _PARSE_TIME_FIELDS:
                if field in node.metadata:
                    meta[field] = node.metadata[field]
            meta["chunk_index"] = chunk_index
            result_nodes.append(TextNode(text=chunk_text, metadata=meta))
        out[orig_pos] = result_nodes  # will be flattened below

    # Flatten paragraph slots (which may expand to multiple chunks)
    flat_out: list[TextNode] = []
    for item in out:
        if isinstance(item, list):
            flat_out.extend(item)
        elif item is not None:
            flat_out.append(item)

    _assert_chunk_metadata(flat_out)
    return flat_out


def _assert_chunk_metadata(nodes: list[TextNode]) -> None:
    """M2.2 pass condition: all 6 required fields present on every chunk."""
    for i, node in enumerate(nodes):
        missing = REQUIRED_METADATA_FIELDS - node.metadata.keys()
        assert not missing, (
            f"Chunk {i} (element_type={node.metadata.get('element_type')!r}) "
            f"missing metadata fields: {missing}"
        )
