# FinLens Codespaces Runbook

This runbook covers how to operate FinLens in GitHub Codespaces with the repository as it exists today.

## Current repo reality

- The repo does not currently ship a checked-in `.devcontainer/` or `scripts/codespaces/` bootstrap flow.
- Bring-up is manual: install dependencies, configure `.env`, run ingestion, then run API/frontend.
- Ingestion state is tracked in `data/ingestion_manifest.json` (active artifact), not `data/ingestion_registry.json` (legacy).

## 1. Prerequisites

Create Codespaces secrets (or set env vars manually):

- `OPENROUTER_API_KEY` (required for generation)
- `QDRANT_URL` (required; local Docker or managed endpoint)
- `QDRANT_API_KEY` (required only for managed/authenticated Qdrant)
- `QDRANT_COLLECTION` (optional; defaults to `finlens_chunks_dev`)
- Optional: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

## 2. Install dependencies

From repo root:

```bash
uv sync
cd backend
uv sync --group dev
cd ../frontend
npm ci
```

## 3. Data location

Place PDFs under:

- `data/financebench/pdfs`

The ingestion command discovers files from this directory.

## 4. Choose Qdrant mode

### Option A: Managed Qdrant

```env
QDRANT_URL=https://<your-cluster-url>
QDRANT_API_KEY=<api-key>
QDRANT_COLLECTION=finlens_chunks_dev
```

### Option B: Local Docker Qdrant

```env
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=finlens_chunks_dev
```

Start local dependencies if using Docker:

```bash
docker compose up -d qdrant postgres langfuse
```

## 5. Run ingestion

List current ingestion state:

```bash
cd backend
uv run python -m ingestion.run_ingestion --list
```

Ingest all pending documents:

```bash
uv run python -m ingestion.run_ingestion
```

Ingest a bounded batch:

```bash
uv run python -m ingestion.run_ingestion --limit 10
```

## 6. Runtime artifacts

Keep these artifacts available for query-time runtime:

- `data/bm25_index.pkl` (sparse retrieval artifact)
- `data/ingestion_manifest.json` (document-level ingestion state and status)
- Qdrant collection data in your configured Qdrant instance

## 7. Verify service status

Start API:

```bash
cd backend
uv run python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then check:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/status/services
curl http://localhost:8000/status/ingestion
```

## 8. Run frontend

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

## 9. End-to-end validation scripts

From `backend/`:

```bash
uv run python ../tests/validate_ingestion.py
uv run python ../tests/validate_qdrant.py
uv run python ../tests/validate_api.py
uv run python ../tests/validate_e2e.py
```

These scripts read shared runtime configuration for Qdrant URL/collection and align with current ingestion artifacts.

## 10. Troubleshooting

- No PDFs discovered: verify `data/financebench/pdfs` contains `.pdf` files.
- `/status/ingestion` shows pending docs: rerun ingestion, then re-check manifest and Qdrant status.
- No retrieval results: verify both Qdrant collection contents and `data/bm25_index.pkl` are present.
- Generation failures: verify `OPENROUTER_API_KEY` and upstream model availability.
