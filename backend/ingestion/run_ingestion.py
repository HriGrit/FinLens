"""
run_ingestion.py — CLI entrypoint for the offline ingestion pipeline.

Usage (from backend/):
  uv run python -m ingestion.run_ingestion --list         # show status (no ingestion)
  uv run python -m ingestion.run_ingestion --limit 10     # ingest next 10 unprocessed docs
  uv run python -m ingestion.run_ingestion                # ingest ALL remaining docs

Auto-discovers PDFs in data/financebench/pdfs/ and tracks per-document ingestion
state in data/ingestion_manifest.json.

Milestone coverage: M2.1, M2.2, M2.3, M2.4
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .chunk import split_paragraph_nodes
from .discovery import PDF_DIR, DocSpec, discover_pdfs, parse_pdf_filename
from .index import build_bm25_index, build_qdrant_index
from .parse import assert_node_metadata
from llama_index.core.schema import TextNode

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BM25_INDEX_PATH = REPO_ROOT / "data" / "bm25_index.pkl"
REGISTRY_PATH = REPO_ROOT / "data" / "ingestion_registry.json"
MANIFEST_PATH = REPO_ROOT / "data" / "ingestion_manifest.json"
CHUNKS_DIR = REPO_ROOT / "data" / "chunks"
_MAX_ERROR_LEN = 500
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "finlens_chunks_dev")


# ---------------------------------------------------------------------------
# Document discovery
# ---------------------------------------------------------------------------

def _parse_pdf_filename(pdf_path: Path) -> DocSpec | None:
    """Parse {COMPANY}_{YEAR}_{DOCTYPE}.pdf into a DocSpec. Returns None if unparseable."""
    spec = parse_pdf_filename(pdf_path)
    if spec is None:
        print(f"  WARN — skipping unparseable filename: {pdf_path.name}")
        return None
    return spec


def _discover_pdfs() -> list[DocSpec]:
    """Glob PDF_DIR for *.pdf files and parse each into a DocSpec."""
    known_specs = {spec.path.name: spec for spec in discover_pdfs(PDF_DIR)}
    if not PDF_DIR.exists():
        return []

    specs: list[DocSpec] = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        spec = known_specs.get(pdf_path.name)
        if spec is not None:
            specs.append(spec)
        else:
            _parse_pdf_filename(pdf_path)
    return specs


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _load_registry() -> set[str]:
    if not REGISTRY_PATH.exists():
        return set()
    with open(REGISTRY_PATH) as f:
        return set(json.load(f).get("ingested", []))


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _migrate_registry_to_manifest() -> dict[str, dict[str, Any]]:
    """Convert flat registry {"ingested":[...]} to per-doc manifest dict."""
    migrated: dict[str, dict[str, Any]] = {}
    for filename in _load_registry():
        migrated[filename] = {
            "status": "success",
            "attempts": 1,
            "last_error": None,
            "chunk_count": None,
            "updated_at": _utc_now_iso(),
        }
    return migrated


def _load_manifest() -> dict[str, dict[str, Any]]:
    """Load manifest; auto-migrate from legacy registry if manifest absent."""
    if MANIFEST_PATH.exists():
        try:
            payload = json.loads(MANIFEST_PATH.read_text())
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    if REGISTRY_PATH.exists():
        manifest = _migrate_registry_to_manifest()
        _save_manifest(manifest)
        return manifest
    return {}


def _save_manifest(manifest: dict[str, dict[str, Any]]) -> None:
    """Atomic write via tmp file + os.replace to prevent partial-write corruption."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MANIFEST_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp_path, MANIFEST_PATH)


def _mark_pending(manifest: dict[str, dict[str, Any]], filename: str) -> None:
    """Write pending status; increments attempts. Called before processing starts."""
    previous = manifest.get(filename, {})
    attempts = int(previous.get("attempts", 0) or 0) + 1
    manifest[filename] = {
        "status": "pending",
        "attempts": attempts,
        "last_error": None,
        "chunk_count": None,
        "updated_at": _utc_now_iso(),
    }


def _mark_success(manifest: dict[str, dict[str, Any]], filename: str, chunk_count: int) -> None:
    """Write success status with chunk count."""
    previous = manifest.get(filename, {})
    manifest[filename] = {
        "status": "success",
        "attempts": int(previous.get("attempts", 0) or 0),
        "last_error": None,
        "chunk_count": chunk_count,
        "updated_at": _utc_now_iso(),
    }


def _mark_failed(manifest: dict[str, dict[str, Any]], filename: str, error: str) -> None:
    """Write failed status with truncated error string."""
    previous = manifest.get(filename, {})
    manifest[filename] = {
        "status": "failed",
        "attempts": int(previous.get("attempts", 0) or 0),
        "last_error": (error or "")[:_MAX_ERROR_LEN],
        "chunk_count": None,
        "updated_at": _utc_now_iso(),
    }


def _save_chunk_artifact(filename: str, chunks: list[TextNode]) -> None:
    """Pickle chunks to data/chunks/{stem}.pkl. Creates dir if needed."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = CHUNKS_DIR / f"{Path(filename).stem}.pkl"
    payload = {"nodes": chunks, "filename": filename, "chunk_count": len(chunks)}
    with open(artifact_path, "wb") as f:
        pickle.dump(payload, f)


def _load_existing_bm25_nodes() -> list[TextNode]:
    """Return the raw node list from an existing BM25 pickle, or [] if none."""
    if not BM25_INDEX_PATH.exists():
        return []
    with open(BM25_INDEX_PATH, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        return []
    nodes = payload.get("nodes", [])
    return nodes if isinstance(nodes, list) else []


def _load_chunk_artifacts_for_successful_docs(
    manifest: dict[str, dict[str, Any]],
    include_filenames: set[str] | None = None,
) -> list[TextNode]:
    """Load nodes for successful docs plus explicit filenames.
    Falls back to the existing BM25 index when a migrated success entry has no artifact.
    """
    selected_filenames = {
        filename
        for filename, entry in manifest.items()
        if isinstance(entry, dict) and entry.get("status") == "success"
    }
    if include_filenames:
        selected_filenames.update(include_filenames)

    existing_nodes_by_filename: dict[str, list[TextNode]] = {}
    for node in _load_existing_bm25_nodes():
        filename = str(node.metadata.get("filename", ""))
        if filename:
            existing_nodes_by_filename.setdefault(filename, []).append(node)

    all_nodes: list[TextNode] = []
    for filename in sorted(selected_filenames):
        artifact_path = CHUNKS_DIR / f"{Path(filename).stem}.pkl"
        if not artifact_path.exists():
            fallback_nodes = existing_nodes_by_filename.get(filename)
            if fallback_nodes:
                all_nodes.extend(fallback_nodes)
                continue
            print(f"WARN: missing chunk artifact for successful doc: {filename}")
            continue
        with open(artifact_path, "rb") as f:
            payload = pickle.load(f)
        nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
        all_nodes.extend(nodes)
    return all_nodes


def _build_batch(
    all_specs: list[DocSpec],
    manifest: dict[str, dict[str, Any]],
    retry_failed: bool,
    limit: int | None,
) -> tuple[list[DocSpec], set[str], set[str]]:
    success_set = {
        filename
        for filename, entry in manifest.items()
        if isinstance(entry, dict) and entry.get("status") == "success"
    }
    failed_set = {
        filename
        for filename, entry in manifest.items()
        if isinstance(entry, dict) and entry.get("status") == "failed"
    }
    if retry_failed:
        queue = {s.path.name for s in all_specs} - success_set
    else:
        queue = {s.path.name for s in all_specs} - success_set - failed_set
    batch = [s for s in all_specs if s.path.name in queue]
    if limit is not None:
        batch = batch[:limit]
    return batch, success_set, failed_set


# ---------------------------------------------------------------------------
# Parallel parse worker (must be top-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _worker_init() -> None:
    """Ensure backend/ is on sys.path in each spawned worker process."""
    backend_dir = str(Path(__file__).resolve().parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)


def parse_document(spec: DocSpec) -> list:
    """Parse + enrich one document. Top-level function so ProcessPoolExecutor can pickle it."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from ingestion.parse import doc_to_nodes, inject_heading_context

    if not spec.path.exists():
        print(f"  SKIP — file not found: {spec.path}")
        return []

    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = False
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
    )

    print(f"  [parse] {spec.path.name} ...")
    dl_doc = converter.convert(str(spec.path)).document
    raw_nodes = doc_to_nodes(
        dl_doc,
        filename=spec.path.name,
        company=spec.company,
        year=spec.year,
        doc_type=spec.doc_type,
    )
    enriched = inject_heading_context(raw_nodes)
    print(f"  [parse] {spec.path.name} — {len(enriched)} nodes")
    return enriched


def _parse_batch(
    batch: list[DocSpec],
    manifest: dict[str, dict[str, Any]],
    workers: int,
    continue_on_error: bool,
) -> tuple[dict[str, list], bool]:
    parsed_by_filename: dict[str, list] = {}
    parse_failed = False

    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        future_to_spec = {}
        for spec in batch:
            _mark_pending(manifest, spec.path.name)
            _save_manifest(manifest)
            future_to_spec[pool.submit(parse_document, spec)] = spec

        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                parsed_by_filename[spec.path.name] = future.result()
            except Exception as exc:
                _mark_failed(manifest, spec.path.name, f"{type(exc).__name__}: {exc}")
                _save_manifest(manifest)
                parse_failed = True
                if not continue_on_error:
                    for pending_future in future_to_spec:
                        if pending_future is not future:
                            pending_future.cancel()
                    break

    return parsed_by_filename, parse_failed


def _rebuild_bm25_for_indexed_docs(
    manifest: dict[str, dict[str, Any]],
    indexed_chunk_counts: dict[str, int],
) -> list[TextNode]:
    all_bm25_nodes = _load_chunk_artifacts_for_successful_docs(
        manifest,
        include_filenames=set(indexed_chunk_counts),
    )
    if not all_bm25_nodes:
        if indexed_chunk_counts:
            raise RuntimeError("No BM25 nodes available for successfully indexed documents.")
        print("WARN: no successful docs with chunk artifacts — BM25 index not written.")
        return []

    print(f"Building BM25 index from {len(all_bm25_nodes)} artifact nodes -> {BM25_INDEX_PATH} ...")
    build_bm25_index(all_bm25_nodes, output_path=BM25_INDEX_PATH)

    for filename, chunk_count in indexed_chunk_counts.items():
        _mark_success(manifest, filename, chunk_count=chunk_count)
    if indexed_chunk_counts:
        _save_manifest(manifest)

    return all_bm25_nodes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FinLens incremental ingestion pipeline")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Ingest at most N unprocessed documents per run (default: all remaining)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print ingestion status and exit without ingesting",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of worker processes used for parsing (default: 4)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-queue documents marked failed in the manifest",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue the batch when a single document fails",
    )
    args = parser.parse_args()

    # Discover all PDFs and load manifest
    all_specs = _discover_pdfs()
    manifest = _load_manifest()
    batch, success_set, failed_set = _build_batch(
        all_specs=all_specs,
        manifest=manifest,
        retry_failed=args.retry_failed,
        limit=args.limit,
    )
    pending = [
        s for s in all_specs if s.path.name not in success_set and s.path.name not in failed_set
    ]

    if args.list:
        print(f"Ingestion status:")
        print(f"  Total PDFs found : {len(all_specs)}")
        print(f"  Successful       : {len(success_set)}")
        print(f"  Failed           : {len(failed_set)}")
        print(f"  Pending          : {len(pending)}")
        if batch:
            print(f"\nNext {len(batch)} doc(s) to ingest:")
            for spec in batch:
                print(f"  {spec.path.name}")
        return

    if not batch:
        print("Nothing to ingest — all discovered PDFs are already successful in the manifest.")
        return

    print(
        f"Ingesting {len(batch)} document(s) "
        f"({len(success_set)} successful, {len(failed_set)} failed, "
        f"{max(len(pending) - len(batch), 0)} remaining after this batch) ..."
    )

    # Ensure Qdrant collection exists (idempotent)
    try:
        from .setup_collection import setup_collection
    except ImportError:
        from setup_collection import setup_collection
    setup_collection()

    # Phase 1: parse all docs in batch in parallel (Docling is the biggest bottleneck)
    print("\nPhase 1/3 — Parsing documents in parallel (OCR disabled) ...")
    parsed_by_filename, parse_failed = _parse_batch(
        batch=batch,
        manifest=manifest,
        workers=args.workers,
        continue_on_error=args.continue_on_error,
    )
    if parse_failed and not args.continue_on_error:
        sys.exit(1)

    # Phase 2: chunk sequentially per document
    print("\nPhase 2/3 — Chunking ...")
    per_doc_chunks: dict[str, list[TextNode]] = {}
    for spec in batch:
        filename = spec.path.name
        if filename not in parsed_by_filename:
            continue
        enriched = parsed_by_filename[filename]

        if not enriched:
            _mark_failed(manifest, filename, "parse_document returned 0 nodes")
            _save_manifest(manifest)
            if not args.continue_on_error:
                sys.exit(1)
            continue
        try:
            print(f"  Chunking {filename} ({len(enriched)} nodes) ...")
            chunks = split_paragraph_nodes(enriched)
            assert_node_metadata(chunks)
            _save_chunk_artifact(filename, chunks)
            per_doc_chunks[filename] = chunks
            print(f"    -> {len(chunks)} chunks")
        except Exception as exc:
            _mark_failed(manifest, filename, str(exc))
            _save_manifest(manifest)
            if not args.continue_on_error:
                sys.exit(1)

    if not per_doc_chunks:
        print("\nNo chunks produced for this batch.")
        if not args.continue_on_error:
            sys.exit(1)

    print(f"\nDocs with chunks from this batch: {len(per_doc_chunks)}")

    # Phase 3: Qdrant upsert per document
    print("\nPhase 3/3 — Indexing ...")
    indexed_chunk_counts: dict[str, int] = {}
    for filename, chunks in per_doc_chunks.items():
        try:
            print(f"  Upserting {filename} -> Qdrant ({len(chunks)} chunks) ...")
            build_qdrant_index(chunks, collection_name=QDRANT_COLLECTION)
            indexed_chunk_counts[filename] = len(chunks)
        except Exception as exc:
            _mark_failed(manifest, filename, str(exc))
            _save_manifest(manifest)
            if not args.continue_on_error:
                raise

    all_bm25_nodes = _rebuild_bm25_for_indexed_docs(
        manifest=manifest,
        indexed_chunk_counts=indexed_chunk_counts,
    )

    print("\nIngestion complete.")
    print(f"  Qdrant collection : {QDRANT_COLLECTION}  (verify at http://localhost:6333/dashboard)")
    print(f"  BM25 index        : {BM25_INDEX_PATH}  ({len(all_bm25_nodes)} nodes total)")
    print(f"  Manifest          : {MANIFEST_PATH}  ({len(manifest)} tracked)")


if __name__ == "__main__":
    main()
