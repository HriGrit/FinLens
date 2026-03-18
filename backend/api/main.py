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
import logging
import dataclasses
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Final

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from generation.generate import (
    DEFAULT_MODEL,
    AllModelsTooHotError,
    MalformedGenerationResponseError,
    generate,
    register_langfuse_callbacks,
)
from generation.openrouter_models import get_free_models
from ingestion.discovery import discover_pdfs
from observability.tracing import create_trace
from observability.tracing import get_langfuse_host, get_langfuse_mode, has_langfuse_credentials
from retrieval.pipeline import retrieve_and_rerank
from shared.qdrant import (
    HOSTED_MODE,
    QDRANT_COLLECTION,
    get_qdrant_client,
    get_qdrant_config_error,
    get_qdrant_mode,
    get_qdrant_url,
)

@asynccontextmanager
async def _lifespan(app: FastAPI):
    register_langfuse_callbacks()
    yield


app = FastAPI(title="FinLens API", version="0.1.0", lifespan=_lifespan)
logger = logging.getLogger(__name__)

CORS_ALLOWED_ORIGINS_ENV: Final = "CORS_ALLOWED_ORIGINS"
STATIC_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def _load_allowed_origins() -> list[str]:
    raw = os.getenv(CORS_ALLOWED_ORIGINS_ENV, "").strip()
    if not raw:
        return ["*"]
    return [value.strip() for value in raw.split(",") if value.strip()]


def _is_frontend_route(path: str) -> bool:
    normalized = path.strip("/")
    if not normalized:
        return True
    if normalized.startswith(
        (
            "chat",
            "health",
            "models",
            "status",
            "ready",
            "openapi",
            "docs",
            "redoc",
        )
    ):
        return False
    return True


def _spa_index() -> FileResponse | None:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return None
    return FileResponse(index_file)


if STATIC_DIR.exists():
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_load_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "ingestion_manifest.json"
PDF_DIR = Path(__file__).resolve().parents[2] / "data" / "financebench" / "pdfs"
INGESTION_CACHE_TTL = 30  # seconds

_ingestion_cache: dict | None = None
_ingestion_cache_expires_at: float = 0.0


def _discover_ingestible_filenames() -> set[str]:
    discovered = {
        spec.path.name
        for spec in discover_pdfs(PDF_DIR)
        if spec.doc_type != "OTHER"
    }
    if discovered:
        return discovered
    manifest = _load_manifest()
    return {
        filename
        for filename, entry in manifest.items()
        if isinstance(filename, str) and isinstance(entry, dict)
    }


def _load_manifest() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
    except Exception:
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _manifest_filenames_with_status(
    manifest: dict[str, dict],
    status: str,
    ingestible_filenames: set[str],
) -> set[str]:
    return {
        filename
        for filename, entry in manifest.items()
        if filename in ingestible_filenames
        and isinstance(entry, dict)
        and entry.get("status") == status
    }


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    company: str | None = None
    year: str | None = None
    model: str | None = None
    retrieval_top_k: int = Field(default=20, gt=0)
    rerank_top_k: int = Field(default=5, gt=0)

    @field_validator("query", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    usage: dict
    model: str
    fallback: ChatFallbackState | None = None
    latency_ms: int
    trace_id: str
    reasoning: dict


class FallbackEvent(BaseModel):
    from_model: str
    to_model: str
    reason: str


class ChatFallbackState(BaseModel):
    requested_model: str
    active_model: str
    fallback_used: bool
    fallback_attempts: int
    attempted_models: list[str]
    events: list[FallbackEvent]


class ProvidersTooHotResponse(BaseModel):
    code: str
    message: str
    requested_model: str
    active_model: str
    attempted_models: list[str]
    fallback_attempts: int
    events: list[FallbackEvent]


class FreeModelResponse(BaseModel):
    id: str
    name: str
    context_length: int | None = None
    description: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    checks: dict[str, dict] = {}
    overall = "ok"

    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        checks["openrouter"] = {
            "status": "error",
            "detail": "OPENROUTER_API_KEY not set",
        }
        overall = "error"
    else:
        checks["openrouter"] = {"status": "ok", "detail": "API key configured"}

    if get_qdrant_config_error():
        checks["qdrant"] = {"status": "error", "detail": get_qdrant_config_error()}
        overall = "error"
    else:
        start = time.perf_counter()
        try:
            client = get_qdrant_client()
            collections = client.get_collections()
            checks["qdrant"] = {
                "status": "ok",
                "detail": f"connected to {get_qdrant_url()}",
                "collections": len(getattr(collections, "collections", [])),
                "latency_ms": int((time.perf_counter() - start) * 1000),
            }
        except Exception as exc:
            checks["qdrant"] = {"status": "error", "detail": str(exc)}
            overall = "error"

    langfuse_status = get_langfuse_mode()
    if langfuse_status == "hosted" and not has_langfuse_credentials():
        checks["langfuse"] = {
            "status": "error",
            "detail": "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY required in hosted mode",
        }
        overall = "error"
    else:
        checks["langfuse"] = {"status": "ok", "detail": f"mode={langfuse_status}"}

    return {"status": overall, "checks": checks}


@app.get("/models/free", response_model=list[FreeModelResponse])
def free_models() -> list[FreeModelResponse]:
    return [FreeModelResponse(**dataclasses.asdict(model)) for model in get_free_models()]


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={503: {"model": ProvidersTooHotResponse}},
)
def chat(request: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    trace = create_trace(
        "rag_query",
        metadata={"company": request.company, "year": request.year},
    )
    retrieval_ms_start = time.perf_counter()

    try:
        retrieval_result = retrieve_and_rerank(
            query=request.query,
            retrieval_top_k=request.retrieval_top_k,
            rerank_top_k=request.rerank_top_k,
            company=request.company,
            year=request.year,
            trace=trace,
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Retrieval backend failure.")
    nodes = retrieval_result.nodes
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
    except AllModelsTooHotError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "providers_too_hot",
                "message": "Generation is temporarily unavailable because model providers are too hot.",
                "requested_model": exc.requested_model,
                "active_model": exc.active_model,
                "attempted_models": exc.attempted_models,
                "fallback_attempts": exc.fallback_attempts,
                "events": [event.model_dump() for event in exc.events],
            },
        )
    except MalformedGenerationResponseError:
        raise HTTPException(status_code=502, detail="Malformed upstream generation response.")
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception:
        logger.exception("Generation failed for model=%s", request.model)
        raise HTTPException(status_code=502, detail="Generation backend failure.")
    generation_ms = int((time.perf_counter() - generation_ms_start) * 1000)

    reasoning = _build_reasoning(
        query=request.query,
        trace_id=trace.id,
        request=request,
        context_nodes=nodes,
        n_candidates=retrieval_result.candidate_count,
        generation_model=result["model"],
        usage=result["usage"],
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        model_reasoning=result.get("model_reasoning"),
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
    model_reasoning: str | None = None,
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

    result: dict = {
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
    if model_reasoning is not None:
        result["model_reasoning"] = model_reasoning
    return result


@app.get("/status/ingestion")
def ingestion_status() -> dict:
    global _ingestion_cache, _ingestion_cache_expires_at

    now = time.time()
    if _ingestion_cache is not None and now < _ingestion_cache_expires_at:
        return _ingestion_cache

    try:
        client = get_qdrant_client()
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

        ingestible_filenames = _discover_ingestible_filenames()
        manifest = _load_manifest()
        indexed_filenames = filenames & ingestible_filenames
        failed_filenames = _manifest_filenames_with_status(
            manifest,
            status="failed",
            ingestible_filenames=ingestible_filenames,
        ) - indexed_filenames
        pending_documents = len(ingestible_filenames - indexed_filenames - failed_filenames)

        result = {
            "total_documents": len(ingestible_filenames),
            "indexed_documents": len(indexed_filenames),
            "indexed_chunks": indexed_chunks,
            "pending_documents": pending_documents,
            "failed_documents": len(failed_filenames),
            "status": "idle",
        }
        _ingestion_cache = result
        _ingestion_cache_expires_at = now + INGESTION_CACHE_TTL
        return result
    except Exception as exc:
        try:
            ingestible_filenames = _discover_ingestible_filenames()
            manifest = _load_manifest()
            indexed_filenames = _manifest_filenames_with_status(
                manifest,
                status="success",
                ingestible_filenames=ingestible_filenames,
            )
            failed_filenames = _manifest_filenames_with_status(
                manifest,
                status="failed",
                ingestible_filenames=ingestible_filenames,
            )
        except Exception:
            ingestible_filenames = set()
            indexed_filenames = set()
            failed_filenames = set()

        result = {
            "total_documents": len(ingestible_filenames),
            "indexed_documents": len(indexed_filenames),
            "indexed_chunks": None,
            "pending_documents": len(ingestible_filenames - indexed_filenames - failed_filenames),
            "failed_documents": len(failed_filenames),
            "status": "error",
            "detail": f"Qdrant status unavailable: {exc}",
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

    async def _langfuse_health_status() -> dict:
        langfuse_url = get_langfuse_host()
        if get_langfuse_mode() == "hosted" and not has_langfuse_credentials():
            return {
                "status": "disabled",
                "latency_ms": -1,
                "detail": "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY required in hosted mode",
            }
        return await _check(f"{langfuse_url}/api/public/health")

    async def _check_qdrant() -> dict:
        start = time.perf_counter()
        try:
            if get_qdrant_mode() == HOSTED_MODE:
                config_error = get_qdrant_config_error()
                if config_error:
                    return {
                        "status": "disabled",
                        "latency_ms": -1,
                        "detail": config_error,
                    }
            await asyncio.to_thread(lambda: get_qdrant_client().get_collections())
            return {
                "status": "ok",
                "latency_ms": int((time.perf_counter() - start) * 1000),
                "detail": "collections endpoint reachable",
            }
        except RuntimeError as exc:
            return {"status": "error", "latency_ms": -1, "detail": str(exc)}
        except Exception as exc:
            return {"status": "error", "latency_ms": -1, "detail": str(exc)}

    async with httpx.AsyncClient() as client:
        qdrant_task = _check_qdrant()
        langfuse_task = _langfuse_health_status()
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
            "status": "ok" if langfuse_status["status"] == "ok" else "disabled",
            "latency_ms": -1,
            "detail": "via hosted Langfuse" if langfuse_status["status"] == "ok" else "Langfuse unavailable",
        },
    }

    return result


@app.get("/{path:path}", include_in_schema=False)
def serve_frontend(path: str) -> FileResponse:
    index_response = _spa_index()
    if not index_response:
        raise HTTPException(
            status_code=404,
            detail="Frontend assets are not available in this runtime.",
        )

    if not _is_frontend_route(path):
        raise HTTPException(status_code=404, detail="Not found")

    candidate = STATIC_DIR / path
    if candidate.is_file():
        return FileResponse(candidate)

    return index_response
