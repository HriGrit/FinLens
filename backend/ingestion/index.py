"""
index.py — Build and persist Qdrant vector index and BM25 index.

Milestone coverage: M2.3 (Qdrant index built), M2.4 (BM25 index persisted).
"""
import pickle
import uuid
from pathlib import Path
from typing import Any

from llama_index.core.schema import TextNode
from shared.qdrant import QDRANT_COLLECTION, get_qdrant_client


_POINT_ID_FIELDS = ("filename", "company", "year", "doc_type", "page_number", "element_type")


def _make_point_id(
    filename: str,
    page_number: str | int,
    text: str,
    element_type: str = "",
    chunk_index: str | int = "",
) -> str:
    key = f"{filename}:{page_number}:{element_type}:{chunk_index}:{text}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def _validate_point_id_fields(node: "TextNode", node_index: int) -> None:
    """Q3: Assert all fields used in point ID construction are non-None."""
    null_fields = [f for f in _POINT_ID_FIELDS if node.metadata.get(f) is None]
    if null_fields:
        raise ValueError(
            f"Node {node_index} has None value(s) for point-ID fields: {null_fields}. "
            "Deterministic point IDs cannot be built with None metadata."
        )


def build_qdrant_index(nodes: list[TextNode], collection_name: str = QDRANT_COLLECTION) -> None:
    """Embed nodes and upsert into Qdrant. Creates collection if it doesn't exist."""
    from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

    from .embed import get_embed_model

    embed_model = get_embed_model()
    client = get_qdrant_client()

    # Determine embedding dimension from a test embed
    sample_embedding = embed_model.get_text_embedding("test")
    dim = len(sample_embedding)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                    on_disk=False,
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
                    datatype="float32",
                )
            },
        )
        print(f"Created Qdrant collection '{collection_name}' (dim={dim}, cosine, m=16, ef_construct=200).")

    from qdrant_client.models import PointStruct

    points = []
    texts = [node.text for node in nodes]
    print(f"Embedding {len(nodes)} nodes ...")
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)

    # Q2: Validate embedding batch length before zip
    if len(embeddings) != len(nodes):
        raise ValueError(
            f"Embedding batch length mismatch: expected {len(nodes)} embeddings "
            f"for {len(nodes)} nodes, got {len(embeddings)}."
        )

    for node_index, (node, embedding) in enumerate(zip(nodes, embeddings)):
        # Q3: Validate that all point-ID fields are non-None before constructing the ID
        _validate_point_id_fields(node, node_index)
        payload = {
            "text": node.text,
            **node.metadata,
        }
        points.append(
            PointStruct(
                id=_make_point_id(
                    str(node.metadata.get("filename", "")),
                    str(node.metadata.get("page_number", "")),
                    node.text,
                    element_type=str(node.metadata.get("element_type", "")),
                    chunk_index=str(node.metadata.get("chunk_index", "")),
                ),
                vector={"dense": embedding},
                payload=payload,
            )
        )

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        # Q1: Check upsert status and raise if not completed
        result = client.upsert(collection_name=collection_name, points=batch)
        status = getattr(result, "status", None)
        if status is not None:
            status_name = getattr(status, "value", str(status)).lower()
            if status_name != "completed":
                raise RuntimeError(
                    f"Qdrant upsert did not complete successfully: status={status!r} "
                    f"(batch starting at index {i})"
                )

    print(f"Upserted {len(points)} points into '{collection_name}'.")


_BM25_FORMAT_VERSION = "v1"


def build_bm25_index(nodes: list[TextNode], output_path: str | Path = "data/bm25_index.pkl") -> None:
    """Build BM25 index from nodes and persist to disk as a versioned pickle file.

    I5: The artifact is saved as a dict with an explicit 'version' key so that
    load_bm25_index can verify the format before reconstruction.
    """
    from llama_index.retrievers.bm25 import BM25Retriever

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
    payload: dict[str, Any] = {
        "version": _BM25_FORMAT_VERSION,
        "nodes": nodes,
        "similarity_top_k": retriever.similarity_top_k,
        "skip_stemming": retriever.skip_stemming,
        "token_pattern": retriever.token_pattern,
        "language": "en",
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"BM25 index with {len(nodes)} nodes persisted to {output_path}.")


def load_bm25_index(index_path: str | Path = "data/bm25_index.pkl") -> "BM25Retriever":
    """Load a previously persisted BM25Retriever from disk.

    I5: Checks the 'version' key and raises ValueError for unrecognized versions.
    """
    from llama_index.retrievers.bm25 import BM25Retriever  # noqa: F401

    with open(index_path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid BM25 index payload: expected a dict, got {type(payload).__name__}. "
            "Rebuild the index."
        )

    version = payload.get("version")
    if version != _BM25_FORMAT_VERSION:
        raise ValueError(
            f"Unrecognized BM25 index version: {version!r}. "
            f"Expected {_BM25_FORMAT_VERSION!r}. Rebuild the index."
        )

    if "nodes" not in payload:
        raise ValueError("Invalid BM25 index payload: missing 'nodes' key. Rebuild the index.")

    return BM25Retriever.from_defaults(
        nodes=payload["nodes"],
        similarity_top_k=payload.get("similarity_top_k", 10),
        skip_stemming=payload.get("skip_stemming", False),
        token_pattern=payload.get("token_pattern", "(?u)\\b\\w\\w+\\b"),
    )
