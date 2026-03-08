"""
main.py — FastAPI application.

Routes:
  POST /chat                — query the RAG pipeline
  GET  /health              — liveness check
  GET  /status/ingestion    — Qdrant collection stats
  GET  /status/services     — dependency health checks
"""
from __future__ import annotations

import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient

from generation.generate import DEFAULT_MODEL, generate
from generation.openrouter_models import get_free_models
from observability.tracing import create_trace
from retrieval.pipeline import retrieve_and_rerank

app = FastAPI(title="FinLens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "finlens_chunks_dev")
LANGFUSE_URL = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
TOTAL_DOCUMENTS = 5  # manifest size in run_ingestion.py


class ChatRequest(BaseModel):
    query: str
    company: str | None = None
    year: str | None = None
    model: str | None = None
    retrieval_top_k: int = 20
    rerank_top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    usage: dict
    model: str
    latency_ms: int


class FreeModelResponse(BaseModel):
    id: str
    name: str
    context_length: int | None = None
    description: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models/free", response_model=list[FreeModelResponse])
def free_models() -> list[FreeModelResponse]:
    return [FreeModelResponse(**model.__dict__) for model in get_free_models()]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    trace = create_trace(
        "rag_query",
        metadata={"company": request.company, "year": request.year},
    )

    nodes = retrieve_and_rerank(
        query=request.query,
        retrieval_top_k=request.retrieval_top_k,
        rerank_top_k=request.rerank_top_k,
        company=request.company,
        year=request.year,
        trace=trace,
    )

    if not nodes:
        raise HTTPException(status_code=404, detail="No relevant context found for this query.")

    result = generate(
        query=request.query,
        context_nodes=nodes,
        model=request.model or DEFAULT_MODEL,
        trace=trace,
    )
    trace.update(output=result["answer"])
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return ChatResponse(**result, latency_ms=latency_ms)


@app.get("/status/ingestion")
def ingestion_status() -> dict:
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=3)
        count_result = client.count(collection_name=QDRANT_COLLECTION, exact=True)
        indexed_chunks = count_result.count

        # Collect unique filenames via scroll
        filenames: set[str] = set()
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=256,
                offset=offset,
                with_payload=["filename"],
                with_vectors=False,
            )
            for rec in records:
                fn = (rec.payload or {}).get("filename")
                if fn:
                    filenames.add(fn)
            if offset is None:
                break

        return {
            "total_documents": TOTAL_DOCUMENTS,
            "indexed_documents": len(filenames),
            "indexed_chunks": indexed_chunks,
            "status": "idle",
        }
    except Exception as exc:
        return {
            "total_documents": TOTAL_DOCUMENTS,
            "indexed_documents": 0,
            "indexed_chunks": 0,
            "status": "error",
            "detail": str(exc),
        }


@app.get("/status/services")
def services_status() -> dict:
    result: dict = {}

    # Qdrant
    t0 = time.perf_counter()
    try:
        resp = httpx.get(f"{QDRANT_URL}/healthz", timeout=2)
        latency = int((time.perf_counter() - t0) * 1000)
        result["qdrant"] = {
            "status": "ok" if resp.status_code == 200 else "error",
            "latency_ms": latency,
            "detail": resp.text[:120],
        }
    except Exception as exc:
        result["qdrant"] = {"status": "error", "latency_ms": -1, "detail": str(exc)}

    # Langfuse
    t0 = time.perf_counter()
    langfuse_ok = False
    try:
        resp = httpx.get(f"{LANGFUSE_URL}/api/public/health", timeout=2)
        latency = int((time.perf_counter() - t0) * 1000)
        langfuse_ok = resp.status_code == 200
        result["langfuse"] = {
            "status": "ok" if langfuse_ok else "error",
            "latency_ms": latency,
            "detail": resp.text[:120],
        }
    except Exception as exc:
        result["langfuse"] = {"status": "error", "latency_ms": -1, "detail": str(exc)}

    # OpenRouter — key presence only
    key = os.getenv("OPENROUTER_API_KEY", "")
    result["openrouter"] = {
        "status": "ok" if key else "error",
        "latency_ms": -1,
        "detail": "API key configured" if key else "OPENROUTER_API_KEY not set",
    }

    # Postgres — inferred from Langfuse
    result["postgres"] = {
        "status": "ok" if langfuse_ok else "error",
        "latency_ms": -1,
        "detail": "via Langfuse" if langfuse_ok else "Langfuse unreachable",
    }

    return result
