# Codespaces Runbook (FinLens)

This repository runs ingestion best on a GitHub Codespace and uses Qdrant as either:
- local Docker Qdrant (`QDRANT_URL=http://localhost:6333` with no API key), or
- managed Qdrant Cloud (`QDRANT_URL=https://...` with `QDRANT_API_KEY`).

## Prerequisites
- GitHub repository has Codespaces enabled.
- Optional managed Qdrant credentials:
  - `QDRANT_URL`
  - `QDRANT_API_KEY`
- Required local-ish runtime keys when needed:
  - `OPENROUTER_API_KEY`
  - `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` if tracing is enabled.

## What is provisioned automatically
On first create, Codespaces runs `scripts/codespaces/bootstrap.sh`, which:
1. generates `.env` from `.env.example`,
2. hydrates `data/financebench` from `patronus-ai/financebench` (no repo checkout),
3. prints active Qdrant mode,
4. runs `python -m ingestion.run_ingestion --list`.

## Recommended modes

### 1) Managed Qdrant (recommended for long ingestion runs)
Set these variables in your Codespaces secrets before opening the space:
- `QDRANT_URL=https://<your-cloud-url>`
- `QDRANT_API_KEY=<api-key>`
- `QDRANT_COLLECTION=finlens_chunks_dev` (or your preferred collection)

Then start with a non-destructive list:
```bash
cd backend
uv run python -m ingestion.run_ingestion --list
```

Run in controlled batches:
```bash
cd backend
uv run python -m ingestion.run_ingestion --limit 10
```

### 2) Local Qdrant via Docker
Keep `QDRANT_API_KEY` blank and use:
- `QDRANT_URL=http://localhost:6333`

Start a local Qdrant:
```bash
docker compose up -d qdrant
```

Run ingestion:
```bash
cd backend
uv run python -m ingestion.run_ingestion --limit 10
```

## Resume behavior
Ingestion is manifest/registry-driven. Use:
```bash
cd backend
uv run python -m ingestion.run_ingestion --list
```
to see pending/ingested docs before continuing.

## Quick API smoke
```bash
cd backend
uv run python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Then verify:
- `http://localhost:8000/health`
- `http://localhost:8000/status/services`
- `http://localhost:8000/status/ingestion`

## Export artifacts for downstream deployment
Keep these files available for your deployment runtime:
- `data/bm25_index.pkl`
- `data/ingestion_registry.json`
- `data/chunks/` (optional for rebuild/debug)
- Any local Qdrant collection snapshot exported by your managed/local Qdrant provider.
