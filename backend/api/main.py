from __future__ import annotations

"""
main.py — FastAPI application.

Routes:
  POST /chat                — query the RAG pipeline
  GET  /health              — liveness check
  GET  /status/ingestion    — Qdrant collection stats
  GET  /status/services     — dependency health checks
"""
import os
import json
import asyncio
import dataclasses
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
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
REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "ingestion_registry.json"
INGESTION_CACHE_TTL = 30  # seconds

_ingestion_cache: dict | None = None
_ingestion_cache_expires_at: float = 0.0


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    company: str | None = None
    year: str | None = None
    model: str | None = None
    retrieval_top_k: int = 20
    rerank_top_k: int = 5

    @field_validator("query", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    usage: dict
    model: str
    latency_ms: int
    trace_id: str
    reasoning: dict


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
    return [FreeModelResponse(**dataclasses.asdict(model)) for model in get_free_models()]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    trace = create_trace(
        "rag_query",
        metadata={"company": request.company, "year": request.year},
    )
    retrieval_ms_start = time.perf_counter()

    nodes, n_candidates = retrieve_and_rerank(
        query=request.query,
        retrieval_top_k=request.retrieval_top_k,
        rerank_top_k=request.rerank_top_k,
        company=request.company,
        year=request.year,
        trace=trace,
    )
    retrieval_ms = int((time.perf_counter() - retrieval_ms_start) * 1000)

    if not nodes:
        raise HTTPException(status_code=404, detail="No relevant context found for this query.")

    generation_ms_start = time.perf_counter()
    try:
        result = generate(
            query=request.query,
            context_nodes=nodes,
            model=request.model or DEFAULT_MODEL,
            trace=trace,
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    generation_ms = int((time.perf_counter() - generation_ms_start) * 1000)

    reasoning = _build_reasoning(
        query=request.query,
        trace_id=trace.id,
        request=request,
        context_nodes=nodes,
        n_candidates=n_candidates,
        generation_model=result["model"],
        usage=result["usage"],
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
    )

    trace.update(output=result["answer"])
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return ChatResponse(
        **result,
        latency_ms=latency_ms,
        trace_id=str(trace.id),
        reasoning=reasoning,
    )


def _build_reasoning(
    query: str,
    trace_id: str,
    request: ChatRequest,
    context_nodes,
    n_candidates: int,
    generation_model: str,
    usage: dict,
    retrieval_ms: int,
    generation_ms: int,
) -> dict:
    top_sources = []
    for node in context_nodes[:3]:
        top_sources.append(
            {
                "filename": node.metadata.get("filename"),
                "page_number": node.metadata.get("page_number"),
                "company": node.metadata.get("company"),
                "year": node.metadata.get("year"),
            }
        )

    return {
        "summary": (
            f"FinLens answered '{query}' using {len(context_nodes)} reranked chunks "
            f"from Langfuse trace {trace_id}."
        ),
        "retrieval": {
            "company_filter": request.company,
            "year_filter": request.year,
            "retrieval_top_k": request.retrieval_top_k,
            "retrieved_candidates": n_candidates,
            "latency_ms": retrieval_ms,
        },
        "rerank": {
            "requested_rerank_top_k": request.rerank_top_k,
            "selected_nodes": len(context_nodes),
            "top_sources": top_sources,
        },
        "generation": {
            "model": generation_model,
            "latency_ms": generation_ms,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost_usd": usage.get("cost_usd"),
        },
    }


@app.get("/status/ingestion")
def ingestion_status() -> dict:
    global _ingestion_cache, _ingestion_cache_expires_at

    now = time.time()
    if _ingestion_cache is not None and now < _ingestion_cache_expires_at:
        return _ingestion_cache

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
        total_documents = 0
        if REGISTRY_PATH.exists():
            try:
                registry = json.loads(REGISTRY_PATH.read_text())
                if isinstance(registry, dict):
                    total_documents = len(registry.get("ingested", []))
            except Exception:
                pass
        if total_documents == 0:
            total_documents = len(filenames)

        result = {
            "total_documents": total_documents,
            "indexed_documents": len(filenames),
            "indexed_chunks": indexed_chunks,
            "status": "idle",
        }
        _ingestion_cache = result
        _ingestion_cache_expires_at = now + INGESTION_CACHE_TTL
        return result
    except Exception as exc:
        try:
            total_documents = 0
            if REGISTRY_PATH.exists():
                registry = json.loads(REGISTRY_PATH.read_text())
                if isinstance(registry, dict):
                    total_documents = len(registry.get("ingested", []))
        except Exception:
            total_documents = 0

        result = {
            "total_documents": total_documents,
            "indexed_documents": 0,
            "indexed_chunks": 0,
            "status": "error",
            "detail": str(exc),
        }
        _ingestion_cache = result
        _ingestion_cache_expires_at = now + INGESTION_CACHE_TTL
        return result


@app.get("/status/services")
async def services_status() -> dict:
    async def _check(url: str) -> dict:
        start = time.perf_counter()
        try:
            resp = await client.get(url, timeout=2)
            detail = resp.text[:120]
            return {
                "status": "ok" if resp.status_code == 200 else "error",
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "detail": detail,
            }
        except Exception as exc:
            return {"status": "error", "latency_ms": -1, "detail": str(exc)}

    async with httpx.AsyncClient() as client:
        qdrant_task = _check(f"{QDRANT_URL}/healthz")
        langfuse_task = _check(f"{LANGFUSE_URL}/api/public/health")
        qdrant_status, langfuse_status = await asyncio.gather(qdrant_task, langfuse_task)

    result = {
        "qdrant": qdrant_status,
        "langfuse": langfuse_status,
        "openrouter": {
            "status": "ok" if os.getenv("OPENROUTER_API_KEY", "") else "error",
            "latency_ms": -1,
            "detail": "API key configured" if os.getenv("OPENROUTER_API_KEY", "") else "OPENROUTER_API_KEY not set",
        },
        "postgres": {
            "status": "ok" if langfuse_status["status"] == "ok" else "error",
            "latency_ms": -1,
            "detail": "via Langfuse" if langfuse_status["status"] == "ok" else "Langfuse unreachable",
        },
    }

    return result
