"""
run_ingestion.py — CLI entrypoint for the offline ingestion pipeline.

Usage (from backend/):
  uv run python -m ingestion.run_ingestion --list         # show status (no ingestion)
  uv run python -m ingestion.run_ingestion --pdf A.pdf    # ingest one named PDF
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

from qdrant_client.http.exceptions import ResponseHandlingException
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .chunk import split_paragraph_nodes
from .discovery import PDF_DIR, DocSpec, discover_pdfs
from .index import build_bm25_index, build_qdrant_index
from .parse import assert_node_metadata
from llama_index.core.schema import TextNode
from shared.qdrant import get_qdrant_client

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
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


@dataclass
class ManifestRecoverySummary:
    chunk_artifacts_read: int = 0
    chunk_artifacts_ignored: int = 0
    qdrant_available: bool = False
    qdrant_error: str | None = None
    qdrant_indexed_documents: int = 0
    recovered_success: int = 0
    recovered_pending: int = 0
    qdrant_only_reconstructed: int = 0
    ignored_qdrant_filenames: int = 0


@dataclass
class BM25RebuildSummary:
    source: str
    documents_included: int = 0
    documents_skipped: int = 0
    nodes_included: int = 0
    output_path: Path = BM25_INDEX_PATH
    skipped_reasons: list[str] | None = None


# ---------------------------------------------------------------------------
# Document discovery
# ---------------------------------------------------------------------------

def _discover_pdfs() -> list[DocSpec]:
    """Glob PDF_DIR for *.pdf files and return a DocSpec for every one found."""
    return discover_pdfs(PDF_DIR)


def _normalize_requested_specs(
    requested_pdfs: list[str] | None,
    all_specs: list[DocSpec],
) -> list[DocSpec]:
    """Validate explicit PDF selections and return the matching DocSpecs.

    Supports either bare filenames relative to PDF_DIR or explicit filesystem paths.
    Rejects missing files, non-PDF inputs, and duplicate/ambiguous basename collisions.
    """
    if not requested_pdfs:
        return all_specs

    discovered_by_name = {spec.path.name: spec for spec in all_specs}
    selected_by_name: dict[str, DocSpec] = {}

    for raw_value in requested_pdfs:
        requested = Path(raw_value)
        if requested.suffix.lower() != ".pdf":
            raise ValueError(f"Requested input is not a PDF: {raw_value}")

        if requested.parent == Path("."):
            spec = discovered_by_name.get(requested.name)
            if spec is None:
                raise ValueError(
                    f"Requested PDF was not found under {PDF_DIR}: {requested.name}"
                )
        else:
            resolved = requested.expanduser().resolve()
            if not resolved.exists():
                raise ValueError(f"Requested PDF does not exist: {raw_value}")
            spec = discover_pdfs(resolved.parent)
            matches = [candidate for candidate in spec if candidate.path.resolve() == resolved]
            if not matches:
                raise ValueError(f"Requested PDF could not be loaded: {raw_value}")
            spec = matches[0]

        existing = selected_by_name.get(spec.path.name)
        if existing is not None and existing.path.resolve() != spec.path.resolve():
            raise ValueError(
                f"Requested PDFs collide on filename '{spec.path.name}'. "
                "Use unique basenames for targeted ingestion."
            )
        selected_by_name[spec.path.name] = spec

    return list(selected_by_name.values())


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


def _invalid_manifest_error(reason: str) -> RuntimeError:
    return RuntimeError(
        f"Invalid ingestion manifest at {MANIFEST_PATH}: {reason}. "
        "Move the file aside or run with --recover-manifest --recover-force "
        "to rebuild it from durable artifacts."
    )


def _load_manifest(
    *,
    recover_if_missing: bool = False,
    all_specs: list[DocSpec] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load manifest; optionally recover from durable artifacts if absent."""
    if MANIFEST_PATH.exists():
        try:
            payload = json.loads(MANIFEST_PATH.read_text())
        except Exception as exc:
            raise _invalid_manifest_error(str(exc)) from exc
        if isinstance(payload, dict):
            return payload
        raise _invalid_manifest_error(f"expected JSON object, got {type(payload).__name__}")
    if REGISTRY_PATH.exists():
        manifest = _migrate_registry_to_manifest()
        _save_manifest(manifest)
        return manifest
    if recover_if_missing:
        specs = all_specs if all_specs is not None else _discover_pdfs()
        manifest, summary = _recover_manifest(specs, force=False)
        _print_recovery_summary(summary, MANIFEST_PATH)
        return manifest
    return {}


def _save_manifest(manifest: dict[str, dict[str, Any]]) -> None:
    """Atomic write via tmp file + os.replace to prevent partial-write corruption."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MANIFEST_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp_path, MANIFEST_PATH)


def _manifest_entry(
    status: str,
    *,
    chunk_count: int | None,
    attempts: int = 0,
    last_error: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "attempts": attempts,
        "last_error": last_error,
        "chunk_count": chunk_count,
        "updated_at": _utc_now_iso(),
    }


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


def _mark_success(
    manifest: dict[str, dict[str, Any]],
    filename: str,
    chunk_count: int,
    *,
    verify_artifact: bool = True,
) -> None:
    """Write success status with chunk count.

    I6: When verify_artifact=True (default), asserts that the chunk artifact file
    exists and is readable before recording success. Raises RuntimeError if not.
    """
    if verify_artifact:
        artifact_path = CHUNKS_DIR / f"{Path(filename).stem}.pkl"
        if not artifact_path.exists():
            raise RuntimeError(
                f"I6: Cannot mark {filename!r} as success — chunk artifact not found at "
                f"{artifact_path}. Ingestion may be incomplete."
            )
        try:
            with open(artifact_path, "rb") as _fh:
                pickle.load(_fh)
        except Exception as exc:
            raise RuntimeError(
                f"I6: Chunk artifact for {filename!r} exists but is not readable: {exc}"
            ) from exc

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


def _chunk_artifact_path(filename: str) -> Path:
    return CHUNKS_DIR / f"{Path(filename).stem}.pkl"


def _read_chunk_artifact(artifact_path: Path) -> tuple[str, list[TextNode]]:
    with open(artifact_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected dict payload, got {type(payload).__name__}")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("missing list payload key: nodes")
    if not nodes:
        raise ValueError("chunk artifact has no nodes")
    filename = payload.get("filename")
    if not isinstance(filename, str) or not filename:
        filename = f"{artifact_path.stem}.pdf"
    return filename, nodes


def _load_reusable_chunk_artifact(filename: str) -> list[TextNode] | None:
    artifact_path = _chunk_artifact_path(filename)
    if not artifact_path.exists():
        return None
    try:
        artifact_filename, nodes = _read_chunk_artifact(artifact_path)
    except Exception as exc:
        print(f"WARN: could not reuse chunk artifact for {filename}: {exc}")
        return None
    if artifact_filename != filename:
        print(
            f"WARN: chunk artifact filename mismatch for {filename}: "
            f"payload says {artifact_filename}; reparsing."
        )
        return None
    return nodes


_BM25_REQUIRED_METADATA_FIELDS = {
    "filename",
    "company",
    "year",
    "doc_type",
    "element_type",
    "chunk_index",
}


def _validate_nodes_for_bm25(filename: str, nodes: list[TextNode]) -> None:
    if not nodes:
        raise ValueError("no nodes")
    for index, node in enumerate(nodes):
        if not isinstance(node, TextNode):
            raise ValueError(f"node {index} is {type(node).__name__}, expected TextNode")
        metadata = node.metadata or {}
        missing = _BM25_REQUIRED_METADATA_FIELDS - set(metadata)
        if missing:
            raise ValueError(f"node {index} missing metadata fields: {sorted(missing)}")
        node_filename = metadata.get("filename")
        if node_filename != filename:
            raise ValueError(
                f"node {index} filename mismatch: expected {filename}, got {node_filename!r}"
            )


def _sort_nodes_for_artifact(nodes: list[TextNode]) -> list[TextNode]:
    def _sort_value(value: Any) -> tuple[int, str]:
        if value is None:
            return (1, "")
        try:
            return (0, f"{int(value):012d}")
        except (TypeError, ValueError):
            return (0, str(value))

    return sorted(
        nodes,
        key=lambda node: (
            str(node.metadata.get("filename", "")),
            _sort_value(node.metadata.get("chunk_index")),
            _sort_value(node.metadata.get("page_number")),
            _sort_value(node.metadata.get("reading_order")),
            node.text,
        ),
    )


def _load_bm25_nodes_from_chunk_artifacts(
    filenames: set[str],
) -> tuple[list[TextNode], int, list[str]]:
    all_nodes: list[TextNode] = []
    skipped_reasons: list[str] = []
    skipped_count = 0

    for filename in sorted(filenames):
        artifact_path = _chunk_artifact_path(filename)
        if not artifact_path.exists():
            skipped_count += 1
            skipped_reasons.append(f"{filename}: chunk artifact missing")
            continue
        try:
            artifact_filename, nodes = _read_chunk_artifact(artifact_path)
            if artifact_filename != filename:
                raise ValueError(
                    f"artifact filename mismatch: expected {filename}, got {artifact_filename}"
                )
            _validate_nodes_for_bm25(filename, nodes)
        except Exception as exc:
            skipped_count += 1
            skipped_reasons.append(f"{filename}: {exc}")
            continue
        all_nodes.extend(_sort_nodes_for_artifact(nodes))

    return all_nodes, skipped_count, skipped_reasons


def _bm25_source_filenames(
    source: str,
    all_specs: list[DocSpec],
    manifest: dict[str, dict[str, Any]],
) -> set[str]:
    discovered_filenames = {spec.path.name for spec in all_specs}
    if source == "success":
        return {
            filename
            for filename, entry in manifest.items()
            if filename in discovered_filenames
            and isinstance(entry, dict)
            and entry.get("status") == "success"
        }
    if source == "chunks":
        chunk_filenames: set[str] = set()
        if not CHUNKS_DIR.exists():
            return chunk_filenames
        for artifact_path in CHUNKS_DIR.glob("*.pkl"):
            filename = f"{artifact_path.stem}.pdf"
            if filename in discovered_filenames:
                chunk_filenames.add(filename)
        return chunk_filenames
    raise ValueError(f"Unknown BM25 source: {source}")


def _rebuild_bm25_from_artifacts(
    *,
    source: str,
    all_specs: list[DocSpec],
    manifest: dict[str, dict[str, Any]],
) -> BM25RebuildSummary:
    from .index import load_bm25_index

    selected_filenames = _bm25_source_filenames(source, all_specs, manifest)
    all_nodes, skipped_count, skipped_reasons = _load_bm25_nodes_from_chunk_artifacts(
        selected_filenames
    )
    if not all_nodes:
        raise RuntimeError(
            f"No valid chunk artifact nodes found for BM25 source={source!r}. "
            "Run ingestion or recover chunk artifacts first."
        )

    temp_path = BM25_INDEX_PATH.with_suffix(".tmp")
    build_bm25_index(all_nodes, output_path=temp_path)
    retriever = load_bm25_index(temp_path)
    corpus_size = getattr(getattr(retriever, "bm25", None), "corpus_size", None)
    if corpus_size is not None and corpus_size != len(all_nodes):
        raise RuntimeError(
            f"BM25 verification failed: expected {len(all_nodes)} nodes, got {corpus_size}."
        )
    os.replace(temp_path, BM25_INDEX_PATH)

    return BM25RebuildSummary(
        source=source,
        documents_included=len(selected_filenames) - skipped_count,
        documents_skipped=skipped_count,
        nodes_included=len(all_nodes),
        output_path=BM25_INDEX_PATH,
        skipped_reasons=skipped_reasons,
    )


def _print_bm25_rebuild_summary(summary: BM25RebuildSummary) -> None:
    print("Rebuilt BM25 index.")
    print(f"  Source            : {summary.source}")
    print(f"  Output            : {summary.output_path}")
    print(f"  Documents included: {summary.documents_included}")
    print(f"  Documents skipped : {summary.documents_skipped}")
    print(f"  Nodes included    : {summary.nodes_included}")
    if summary.skipped_reasons:
        print("  Skipped details   :")
        for reason in summary.skipped_reasons[:10]:
            print(f"    - {reason}")
        if len(summary.skipped_reasons) > 10:
            print(f"    - ... {len(summary.skipped_reasons) - 10} more")


def _scan_chunk_artifacts(
    discovered_by_name: dict[str, DocSpec],
    summary: ManifestRecoverySummary,
) -> dict[str, list[TextNode]]:
    chunks_by_filename: dict[str, list[TextNode]] = {}
    if not CHUNKS_DIR.exists():
        return chunks_by_filename

    for artifact_path in sorted(CHUNKS_DIR.glob("*.pkl")):
        try:
            filename, nodes = _read_chunk_artifact(artifact_path)
        except Exception as exc:
            summary.chunk_artifacts_ignored += 1
            print(f"WARN: ignoring unreadable chunk artifact {artifact_path}: {exc}")
            continue

        if filename not in discovered_by_name:
            fallback_filename = f"{artifact_path.stem}.pdf"
            if fallback_filename in discovered_by_name:
                filename = fallback_filename
            else:
                summary.chunk_artifacts_ignored += 1
                continue

        chunks_by_filename[filename] = nodes
        summary.chunk_artifacts_read += 1

    return chunks_by_filename


def _scan_qdrant_nodes(
    discovered_by_name: dict[str, DocSpec],
    summary: ManifestRecoverySummary,
) -> dict[str, list[TextNode]]:
    qdrant_nodes_by_filename: dict[str, list[TextNode]] = {}
    try:
        client = get_qdrant_client()
        if hasattr(client, "collection_exists") and not client.collection_exists(QDRANT_COLLECTION):
            summary.qdrant_available = True
            summary.qdrant_error = f"collection not found: {QDRANT_COLLECTION}"
            return qdrant_nodes_by_filename

        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                payload = rec.payload or {}
                filename = payload.get("filename")
                if not isinstance(filename, str) or not filename:
                    continue
                if filename not in discovered_by_name:
                    summary.ignored_qdrant_filenames += 1
                    continue
                text = payload.get("text")
                if not isinstance(text, str) or not text:
                    continue
                metadata = {key: value for key, value in payload.items() if key != "text"}
                qdrant_nodes_by_filename.setdefault(filename, []).append(
                    TextNode(text=text, metadata=metadata)
                )
            if offset is None:
                break

        summary.qdrant_available = True
        summary.qdrant_indexed_documents = len(qdrant_nodes_by_filename)
        return {
            filename: _sort_nodes_for_artifact(nodes)
            for filename, nodes in qdrant_nodes_by_filename.items()
        }
    except Exception as exc:
        summary.qdrant_available = False
        summary.qdrant_error = str(exc)
        return {}


def _recover_manifest(
    all_specs: list[DocSpec],
    *,
    force: bool,
) -> tuple[dict[str, dict[str, Any]], ManifestRecoverySummary]:
    if MANIFEST_PATH.exists() and not force:
        raise RuntimeError(
            f"Refusing to overwrite existing manifest at {MANIFEST_PATH}. "
            "Use --recover-force to rebuild it."
        )

    summary = ManifestRecoverySummary()
    discovered_by_name = {spec.path.name: spec for spec in all_specs}
    chunks_by_filename = _scan_chunk_artifacts(discovered_by_name, summary)
    qdrant_nodes_by_filename = _scan_qdrant_nodes(discovered_by_name, summary)

    manifest: dict[str, dict[str, Any]] = {}
    recovered_filenames = set(chunks_by_filename) | set(qdrant_nodes_by_filename)

    for filename in sorted(recovered_filenames):
        local_nodes = chunks_by_filename.get(filename)
        qdrant_nodes = qdrant_nodes_by_filename.get(filename)

        if local_nodes is not None and qdrant_nodes is not None and len(local_nodes) == len(qdrant_nodes):
            manifest[filename] = _manifest_entry("success", chunk_count=len(local_nodes))
            summary.recovered_success += 1
            continue

        if local_nodes is None and qdrant_nodes is not None:
            _save_chunk_artifact(filename, qdrant_nodes)
            manifest[filename] = _manifest_entry("success", chunk_count=len(qdrant_nodes))
            summary.recovered_success += 1
            summary.qdrant_only_reconstructed += 1
            continue

        manifest[filename] = _manifest_entry("pending", chunk_count=None)
        summary.recovered_pending += 1

    _save_manifest(manifest)
    return manifest, summary


def _print_recovery_summary(summary: ManifestRecoverySummary, manifest_path: Path) -> None:
    print("Recovered ingestion manifest.")
    print(f"  Manifest              : {manifest_path}")
    print(f"  Chunk artifacts read  : {summary.chunk_artifacts_read}")
    print(f"  Chunk artifacts ignored: {summary.chunk_artifacts_ignored}")
    print(f"  Qdrant reachable      : {summary.qdrant_available}")
    if summary.qdrant_error:
        print(f"  Qdrant detail         : {summary.qdrant_error}")
    print(f"  Qdrant documents      : {summary.qdrant_indexed_documents}")
    print(f"  Recovered success     : {summary.recovered_success}")
    print(f"  Recovered pending     : {summary.recovered_pending}")
    print(f"  Qdrant-only rebuilt   : {summary.qdrant_only_reconstructed}")
    print(f"  Stale Qdrant filenames: {summary.ignored_qdrant_filenames}")


def _load_existing_bm25_nodes() -> list[TextNode]:
    """Return the raw node list from an existing BM25 pickle, or [] if none.

    X1: Uses load_bm25_index() to respect the versioned serialization contract,
    then extracts the underlying node corpus for re-use.
    """
    if not BM25_INDEX_PATH.exists():
        return []
    try:
        from .index import load_bm25_index
        retriever = load_bm25_index(BM25_INDEX_PATH)
        # BM25Retriever stores the original nodes in .index.corpus or can be
        # accessed via the private _nodes attribute; fall back to pickle direct read
        # for the corpus only when needed.
        nodes = getattr(retriever, "_nodes", None)
        if nodes is not None and isinstance(nodes, list):
            return nodes
        # Fallback: reload the payload dict directly to extract the node corpus
        with open(BM25_INDEX_PATH, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return []
        raw_nodes = payload.get("nodes", [])
        return raw_nodes if isinstance(raw_nodes, list) else []
    except Exception:
        return []


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


def _reuse_chunk_artifacts_for_batch(
    batch: list[DocSpec],
    manifest: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[TextNode]], list[DocSpec]]:
    """Load reusable chunk artifacts and return the remaining docs that need parsing."""
    reusable_chunks: dict[str, list[TextNode]] = {}
    parse_needed: list[DocSpec] = []

    for spec in batch:
        filename = spec.path.name
        chunks = _load_reusable_chunk_artifact(filename)
        if chunks is None:
            parse_needed.append(spec)
            continue
        _mark_pending(manifest, filename)
        _save_manifest(manifest)
        reusable_chunks[filename] = chunks

    return reusable_chunks, parse_needed


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
        "--pdf",
        action="append",
        default=None,
        metavar="PDF",
        help="Target one or more PDFs by filename under the discovery directory or by path",
    )
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
    parser.add_argument(
        "--recover-manifest",
        action="store_true",
        help="Rebuild the ingestion manifest from chunk artifacts and Qdrant, then exit",
    )
    parser.add_argument(
        "--recover-force",
        action="store_true",
        help="Allow --recover-manifest to overwrite an existing manifest",
    )
    parser.add_argument(
        "--rebuild-bm25",
        action="store_true",
        help="Rebuild data/bm25_index.pkl from chunk artifacts, then exit",
    )
    parser.add_argument(
        "--bm25-source",
        choices=("success", "chunks"),
        default="success",
        help=(
            "Document set for --rebuild-bm25: manifest success entries or all "
            "valid discovered chunk artifacts (default: success)"
        ),
    )
    args = parser.parse_args()

    # Discover all PDFs and load manifest
    all_specs = _discover_pdfs()
    if args.recover_force and not args.recover_manifest:
        parser.error("--recover-force requires --recover-manifest")
    if args.recover_manifest and args.rebuild_bm25:
        parser.error("--recover-manifest and --rebuild-bm25 must be run separately")
    if args.recover_manifest:
        try:
            manifest, summary = _recover_manifest(all_specs, force=args.recover_force)
        except RuntimeError as exc:
            parser.error(str(exc))
        _print_recovery_summary(summary, MANIFEST_PATH)
        print(f"  Manifest entries      : {len(manifest)}")
        return
    if args.rebuild_bm25:
        try:
            manifest = _load_manifest() if args.bm25_source == "success" else {}
            summary = _rebuild_bm25_from_artifacts(
                source=args.bm25_source,
                all_specs=all_specs,
                manifest=manifest,
            )
        except RuntimeError as exc:
            parser.error(str(exc))
        _print_bm25_rebuild_summary(summary)
        return

    try:
        selected_specs = _normalize_requested_specs(args.pdf, all_specs)
        manifest = _load_manifest()
        if not manifest and not MANIFEST_PATH.exists() and not REGISTRY_PATH.exists():
            manifest, summary = _recover_manifest(all_specs, force=False)
            _print_recovery_summary(summary, MANIFEST_PATH)
    except RuntimeError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))
    batch, success_set, failed_set = _build_batch(
        all_specs=selected_specs,
        manifest=manifest,
        retry_failed=args.retry_failed,
        limit=args.limit,
    )
    pending = [
        s
        for s in selected_specs
        if s.path.name not in success_set and s.path.name not in failed_set
    ]

    if args.list:
        print(f"Ingestion status:")
        print(f"  Total PDFs found : {len(selected_specs)}")
        print(
            f"  Successful       : "
            f"{sum(1 for s in selected_specs if s.path.name in success_set)}"
        )
        print(
            f"  Failed           : "
            f"{sum(1 for s in selected_specs if s.path.name in failed_set)}"
        )
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

    per_doc_chunks, parse_needed = _reuse_chunk_artifacts_for_batch(batch, manifest)
    if per_doc_chunks:
        print(f"\nReusing chunk artifacts for {len(per_doc_chunks)} document(s).")
        for filename, chunks in sorted(per_doc_chunks.items()):
            print(f"  Reusing {filename} ({len(chunks)} chunks)")

    parsed_by_filename: dict[str, list] = {}
    if parse_needed:
        # Phase 1: parse docs without reusable chunks in parallel (Docling is the biggest bottleneck)
        print("\nPhase 1/3 — Parsing documents in parallel (OCR disabled) ...")
        parsed_by_filename, parse_failed = _parse_batch(
            batch=parse_needed,
            manifest=manifest,
            workers=args.workers,
            continue_on_error=args.continue_on_error,
        )
        if parse_failed and not args.continue_on_error:
            sys.exit(1)
    else:
        print("\nPhase 1/3 — No parsing needed; all queued docs have chunk artifacts.")

    # Phase 2: chunk sequentially per document
    print("\nPhase 2/3 — Chunking ...")
    for spec in parse_needed:
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
            is_timeout = False
            current = exc
            while current is not None:
                if isinstance(current, ResponseHandlingException):
                    is_timeout = True
                    break
                current_type = type(current)
                if current_type.__name__ == "ReadTimeout" and current_type.__module__.startswith(("httpx", "httpcore")):
                    is_timeout = True
                    break
                current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
            if is_timeout:
                print(
                    f"\n[ERROR] Qdrant upsert timed out for '{filename}' after all retries.\n"
                    f"  Cause : {exc}\n"
                    f"  Fix   : Increase QDRANT_TIMEOUT (current default: 60s) or reduce "
                    f"_UPSERT_BATCH_SIZE in index.py\n"
                    f"  State : manifest updated; re-run ingestion to retry this document."
                )
                sys.exit(1)
            if not args.continue_on_error:
                raise

    all_bm25_nodes = _rebuild_bm25_for_indexed_docs(
        manifest=manifest,
        indexed_chunk_counts=indexed_chunk_counts,
    )

    print("\nIngestion complete.")
    print(f"  Qdrant collection : {QDRANT_COLLECTION}  (verify at {QDRANT_URL}/dashboard)")
    print(f"  BM25 index        : {BM25_INDEX_PATH}  ({len(all_bm25_nodes)} nodes total)")
    print(f"  Manifest          : {MANIFEST_PATH}  ({len(manifest)} tracked)")


if __name__ == "__main__":
    main()
