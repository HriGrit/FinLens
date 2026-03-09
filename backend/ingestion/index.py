"""
index.py — Build and persist Qdrant vector index and BM25 index.

Milestone coverage: M2.3 (Qdrant index built), M2.4 (BM25 index persisted).
"""
import os
import pickle
from pathlib import Path
from typing import Any

from llama_index.core.schema import TextNode

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "finlens_chunks_dev")


def build_qdrant_index(nodes: list[TextNode], collection_name: str = QDRANT_COLLECTION) -> None:
    """Embed nodes and upsert into Qdrant. Creates collection if it doesn't exist."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

    from .embed import get_embed_model

    embed_model = get_embed_model()
    client = QdrantClient(url=QDRANT_URL)

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
    import uuid

    points = []
    texts = [node.text for node in nodes]
    print(f"Embedding {len(nodes)} nodes ...")
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)

    for node, embedding in zip(nodes, embeddings):
        payload = {
            "text": node.text,
            **node.metadata,
        }
        points.append(
            PointStruct(
                id=str(uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{node.metadata.get('filename', '')}:{node.metadata.get('page_number', '')}:{node.metadata.get('chunk_index', '')}",
                )),
                vector={"dense": embedding},
                payload=payload,
            )
        )

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    print(f"Upserted {len(points)} points into '{collection_name}'.")


def build_bm25_index(nodes: list[TextNode], output_path: str | Path = "data/bm25_index.pkl") -> None:
    """Build BM25 index from nodes and persist to disk as a pickle file."""
    from llama_index.retrievers.bm25 import BM25Retriever

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
    payload: dict[str, Any] = {
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
    """Load a previously persisted BM25Retriever from disk."""
    from llama_index.retrievers.bm25 import BM25Retriever  # noqa: F401

    with open(index_path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "nodes" not in payload:
        raise ValueError("Invalid BM25 index payload. Rebuild the index.")

    return BM25Retriever.from_defaults(
        nodes=payload["nodes"],
        similarity_top_k=payload.get("similarity_top_k", 10),
        skip_stemming=payload.get("skip_stemming", False),
        token_pattern=payload.get("token_pattern", "(?u)\\b\\w\\w+\\b"),
    )
