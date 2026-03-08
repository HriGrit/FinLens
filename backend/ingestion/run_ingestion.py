"""
run_ingestion.py — CLI entrypoint for the offline ingestion pipeline.

Usage (from backend/):
  uv run python -m ingestion.run_ingestion --list         # show status (no ingestion)
  uv run python -m ingestion.run_ingestion --limit 10     # ingest next 10 unprocessed docs
  uv run python -m ingestion.run_ingestion                # ingest ALL remaining docs

Auto-discovers PDFs in data/financebench/pdfs/ and tracks ingested files in
data/ingestion_registry.json to skip already-processed documents on future runs.

Milestone coverage: M2.1, M2.2, M2.3, M2.4
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

from .chunk import split_paragraph_nodes
from .index import build_bm25_index, build_qdrant_index
from .parse import assert_node_metadata, doc_to_nodes, inject_heading_context

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = REPO_ROOT / "data" / "financebench" / "pdfs"
BM25_INDEX_PATH = REPO_ROOT / "data" / "bm25_index.pkl"
REGISTRY_PATH = REPO_ROOT / "data" / "ingestion_registry.json"
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "finlens_chunks_dev")


@dataclass
class DocSpec:
    path: Path
    company: str
    year: str
    doc_type: str = "10-K"


# ---------------------------------------------------------------------------
# Document discovery
# ---------------------------------------------------------------------------

def _parse_pdf_filename(pdf_path: Path) -> DocSpec | None:
    """Parse {COMPANY}_{YEAR}_{DOCTYPE}.pdf into a DocSpec. Returns None if unparseable."""
    parts = pdf_path.stem.split("_")
    if len(parts) < 3:
        print(f"  WARN — skipping unparseable filename: {pdf_path.name}")
        return None
    doc_type_raw = parts[-1]                            # "10K" or "10Q"
    year = parts[-2]                                    # "2015" or "2022Q2"
    company = "_".join(parts[:-2])                      # "3M" or "ADOBE"
    # Insert hyphen after the digit prefix: "10K" -> "10-K", "10Q" -> "10-Q"
    if len(doc_type_raw) >= 3 and doc_type_raw[:-1].isdigit():
        doc_type = doc_type_raw[:-1] + "-" + doc_type_raw[-1]
    else:
        doc_type = doc_type_raw
    return DocSpec(path=pdf_path, company=company, year=year, doc_type=doc_type)


def _discover_pdfs() -> list[DocSpec]:
    """Glob PDF_DIR for *.pdf files and parse each into a DocSpec."""
    if not PDF_DIR.exists():
        return []
    specs = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        spec = _parse_pdf_filename(pdf_path)
        if spec is not None:
            specs.append(spec)
    return specs


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _load_registry() -> set[str]:
    if not REGISTRY_PATH.exists():
        return set()
    with open(REGISTRY_PATH) as f:
        return set(json.load(f).get("ingested", []))


def _save_registry(ingested: set[str]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump({"ingested": sorted(ingested)}, f, indent=2)


# ---------------------------------------------------------------------------
# BM25 incremental merge helper
# ---------------------------------------------------------------------------

def _load_existing_bm25_nodes() -> list:
    """Return the raw node list from an existing BM25 pickle, or [] if none."""
    if not BM25_INDEX_PATH.exists():
        return []
    with open(BM25_INDEX_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload.get("nodes", [])


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

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
    args = parser.parse_args()

    # Discover all PDFs and load registry
    all_specs = _discover_pdfs()
    already_ingested = _load_registry()

    pending = [s for s in all_specs if s.path.name not in already_ingested]
    batch = pending if args.limit is None else pending[: args.limit]

    if args.list:
        print(f"Ingestion status:")
        print(f"  Total PDFs found : {len(all_specs)}")
        print(f"  Already ingested : {len(already_ingested)}")
        print(f"  Pending          : {len(pending)}")
        if batch:
            print(f"\nNext {len(batch)} doc(s) to ingest:")
            for spec in batch:
                print(f"  {spec.path.name}")
        return

    if not batch:
        print("Nothing to ingest — all discovered PDFs are already in the registry.")
        return

    print(f"Ingesting {len(batch)} document(s) ({len(already_ingested)} already done, {len(pending) - len(batch)} remaining after this batch) ...")

    # Ensure Qdrant collection exists (idempotent)
    try:
        from .setup_collection import setup_collection
    except ImportError:
        from setup_collection import setup_collection
    setup_collection()

    # Phase 1: parse all docs in batch in parallel (Docling is the biggest bottleneck)
    print("\nPhase 1/3 — Parsing documents in parallel (OCR disabled) ...")
    with ProcessPoolExecutor(max_workers=4, initializer=_worker_init) as pool:
        all_enriched = list(pool.map(parse_document, batch))

    # Phase 2: chunk sequentially
    print("\nPhase 2/3 — Chunking ...")
    new_chunks: list = []
    for spec, enriched in zip(batch, all_enriched):
        if not enriched:
            continue
        print(f"  Chunking {spec.path.name} ({len(enriched)} nodes) ...")
        chunks = split_paragraph_nodes(enriched)
        assert_node_metadata(chunks)
        print(f"    -> {len(chunks)} chunks")
        new_chunks.extend(chunks)

    if not new_chunks:
        print("\nNo chunks produced — check that PDF files exist in the configured paths.")
        sys.exit(1)

    print(f"\nNew chunks from this batch: {len(new_chunks)}")

    # Phase 3: index
    print("\nPhase 3/3 — Indexing ...")

    print("Building Qdrant index ...")
    build_qdrant_index(new_chunks, collection_name=QDRANT_COLLECTION)

    existing_nodes = _load_existing_bm25_nodes()
    all_bm25_nodes = existing_nodes + new_chunks
    print(f"Building BM25 index ({len(existing_nodes)} existing + {len(new_chunks)} new = {len(all_bm25_nodes)} total) -> {BM25_INDEX_PATH} ...")
    build_bm25_index(all_bm25_nodes, output_path=BM25_INDEX_PATH)

    # Persist registry only after successful indexing
    newly_ingested = {spec.path.name for spec in batch}
    _save_registry(already_ingested | newly_ingested)

    print("\nIngestion complete.")
    print(f"  Qdrant collection : {QDRANT_COLLECTION}  (verify at http://localhost:6333/dashboard)")
    print(f"  BM25 index        : {BM25_INDEX_PATH}  ({len(all_bm25_nodes)} nodes total)")
    print(f"  Registry          : {REGISTRY_PATH}  ({len(already_ingested | newly_ingested)} ingested)")


if __name__ == "__main__":
    main()
