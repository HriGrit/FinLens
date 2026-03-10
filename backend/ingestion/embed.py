"""
embed.py — Singleton HuggingFace embedding model loader.

Loads once per process; subsequent calls return the cached instance.
"""
import os

import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "Alibaba-NLP/gte-modernbert-base")
DEFAULT_DEVICE = os.getenv("EMBEDDING_DEVICE")

_embed_model: HuggingFaceEmbedding | None = None
_embed_device: str | None = None


def get_embed_model(model_name: str | None = None) -> HuggingFaceEmbedding:
    """Return (and cache) the HuggingFaceEmbedding singleton."""
    global _embed_model, _embed_device
    model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL)
    device = DEFAULT_DEVICE or ("mps" if torch.backends.mps.is_available() else "cpu")
    if _embed_model is None or _embed_model.model_name != model_name or _embed_device != device:
        print(f"Loading embedding model: {model_name}")
        _embed_model = HuggingFaceEmbedding(model_name=model_name, device=device)
        _embed_device = device
    return _embed_model
