# FinLens on GitHub Codespaces

This document explains how to run the FinLens ingestion pipeline in GitHub Codespaces, store embeddings in Qdrant, and query the indexed data from the backend and frontend.

The intended operating model is:
- use Codespaces as the ingestion machine,
- keep the FinanceBench dataset out of the main repository history,
- choose either managed Qdrant or local Docker Qdrant,
- keep the BM25 artifact on disk because retrieval is still hybrid.

## 1. What Codespaces does in this repo

The repository includes a dev container at [devcontainer.json](/Users/soam/Documents/Work/FinOps/.devcontainer/devcontainer.json). When a fresh Codespace is created, it:

1. installs backend dependencies with `uv`,
2. installs frontend dependencies with `npm`,
3. runs [bootstrap.sh](/Users/soam/Documents/Work/FinOps/scripts/codespaces/bootstrap.sh),
4. renders `.env` from `.env.example` plus Codespaces secrets,
5. clones `patronus-ai/financebench` into `data/financebench` if it is not already present,
6. runs `uv run python -m ingestion.run_ingestion --list` as a non-destructive smoke check.

The bootstrap scripts involved are:
- [bootstrap.sh](/Users/soam/Documents/Work/FinOps/scripts/codespaces/bootstrap.sh)
- [render_env.sh](/Users/soam/Documents/Work/FinOps/scripts/codespaces/render_env.sh)
- [fetch_financebench.sh](/Users/soam/Documents/Work/FinOps/scripts/codespaces/fetch_financebench.sh)
- [smoke.sh](/Users/soam/Documents/Work/FinOps/scripts/codespaces/smoke.sh)

## 2. Before creating the Codespace

Create Codespaces secrets in GitHub for the repository or for your user account.

### Required secrets for normal operation
- `OPENROUTER_API_KEY`

### Required secrets for managed Qdrant
- `QDRANT_URL`
- `QDRANT_API_KEY`

### Optional but recommended
- `QDRANT_COLLECTION`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST`

If you do not set `QDRANT_URL`, the repo defaults to local mode and expects `http://localhost:6333`.

## 3. Choose your Qdrant mode

FinLens supports two modes.

### Option A: Managed Qdrant

Use this for full-corpus ingestion. It avoids relying on the Codespace VM for vector persistence.

Set:
```env
QDRANT_URL=https://<your-cluster-url>
QDRANT_API_KEY=<your-api-key>
QDRANT_COLLECTION=finlens_chunks_dev
```

Behavior in this mode:
- the backend uses authenticated Qdrant clients,
- `/status/services` checks Qdrant through the client, not through anonymous `/healthz`,
- ingestion writes vectors to the managed cluster,
- you do not need to run local Docker Qdrant.

### Option B: Local Docker Qdrant

Use this for short experiments or local debugging inside Codespaces.

Set:
```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=finlens_chunks_dev
```

Then start Qdrant manually:
```bash
docker compose up -d qdrant
```

Behavior in this mode:
- vectors live inside the Codespace-attached Docker volume,
- if the Codespace is deleted, you should assume the local Qdrant state is gone unless you exported it.

## 4. Creating the Codespace

Create the Codespace from the branch that contains the Codespaces integration files.

On first boot:
- dependency installation runs from `updateContentCommand`,
- bootstrap runs from `postCreateCommand`,
- the repo generates a root `.env`,
- the dataset is cloned into `data/financebench`,
- a smoke check prints discovered PDF counts.

Expected locations after boot:
- dataset: `data/financebench/pdfs`
- sparse index: `data/bm25_index.pkl`
- ingestion registry: `data/ingestion_registry.json`

## 5. Verify bootstrap state

From the repo root:
```bash
ls data/financebench/pdfs | head
```

From the backend:
```bash
cd backend
uv run python -m ingestion.run_ingestion --list
```

You should see:
- total PDFs discovered,
- already ingested documents,
- pending documents,
- the next batch of documents if any remain.

If `--list` prints zero PDFs, the dataset clone step did not complete correctly.

## 6. Running the ingestion pipeline

The current ingestion CLI is defined in [run_ingestion.py](/Users/soam/Documents/Work/FinOps/backend/ingestion/run_ingestion.py). The supported flags today are:
- `--list`
- `--limit N`

### List current state
```bash
cd backend
uv run python -m ingestion.run_ingestion --list
```

### Ingest the next 10 documents
```bash
cd backend
uv run python -m ingestion.run_ingestion --limit 10
```

### Ingest all remaining documents
```bash
cd backend
uv run python -m ingestion.run_ingestion
```

### What the ingestion run does

For each batch, the pipeline:
1. discovers PDFs under `data/financebench/pdfs`,
2. skips files already recorded in `data/ingestion_registry.json`,
3. ensures the Qdrant collection exists,
4. parses PDFs with Docling,
5. converts parsed content into LlamaIndex nodes,
6. semantically splits paragraph nodes into chunks,
7. embeds and upserts chunks into Qdrant,
8. appends the new chunks into the local BM25 payload and writes `data/bm25_index.pkl`,
9. records successfully ingested filenames in `data/ingestion_registry.json`.

### Important operational detail

The current implementation is resumable only at the document-registry level, not at a per-document checkpoint level. That means:
- rerunning the command skips documents already in the registry,
- the Qdrant point IDs are deterministic, so repeated upserts for the same chunk are stable,
- the BM25 file is still rebuilt from existing BM25 nodes plus the new batch during each run.

If you stop after `--limit 10`, run the same command again to process the next 10.

## 7. Monitoring ingestion progress

### CLI status
```bash
cd backend
uv run python -m ingestion.run_ingestion --list
```

### API status
Start the backend:
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

`/status/services` reports:
- Qdrant connectivity,
- Langfuse connectivity,
- whether `OPENROUTER_API_KEY` is configured,
- a derived Postgres status via Langfuse.

`/status/ingestion` reports:
- `total_documents`
- `indexed_documents`
- `indexed_chunks`
- current status

## 8. Running the backend and frontend in Codespaces

### Backend only
```bash
cd backend
uv run python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Codespaces forwards port `8000` automatically because it is declared in the dev container config.

### Frontend only
```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Codespaces forwards port `5173` automatically.

### Query flow

The frontend talks to the backend. The backend retrieval path is hybrid:
- dense retrieval comes from Qdrant,
- sparse retrieval comes from the local file `data/bm25_index.pkl`.

That means remote Qdrant alone is not enough. The backend instance that answers queries must also have access to the BM25 artifact generated during ingestion.

## 9. Running the full stack with Docker Compose

You can still use Docker Compose, but in Codespaces it is usually better to run ingestion directly with `uv`.

Compose is defined in [docker-compose.yml](/Users/soam/Documents/Work/FinOps/docker-compose.yml).

### Start local Qdrant and Langfuse stack
```bash
docker compose up -d qdrant postgres langfuse
```

### Start backend and frontend in Compose
```bash
docker compose up -d backend frontend
```

Notes:
- the backend now respects `QDRANT_URL` and `QDRANT_API_KEY` from `.env`,
- the backend no longer requires the local `qdrant` service to be running if you are pointing it at managed Qdrant,
- `LANGFUSE_HOST` inside Compose is set to the internal service URL.

## 10. Common workflows

### Workflow A: Full ingestion to managed Qdrant
1. Set Codespaces secrets for `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, and `OPENROUTER_API_KEY`.
2. Create the Codespace.
3. Run `cd backend && uv run python -m ingestion.run_ingestion --list`.
4. Start with `cd backend && uv run python -m ingestion.run_ingestion --limit 10`.
5. Repeat until the pending count reaches zero.
6. Start the backend and frontend and verify queries.

### Workflow B: Quick test using local Qdrant
1. Leave `QDRANT_API_KEY` empty.
2. Set `QDRANT_URL=http://localhost:6333`.
3. Run `docker compose up -d qdrant`.
4. Run `cd backend && uv run python -m ingestion.run_ingestion --limit 5`.
5. Start backend and frontend and verify retrieval.

### Workflow C: Resume a stopped session
1. Reopen the same Codespace.
2. Run `cd backend && uv run python -m ingestion.run_ingestion --list`.
3. Continue with `--limit N` or run the full remaining ingestion.

## 11. Artifacts you must keep

For a working query runtime after ingestion, keep:
- `data/bm25_index.pkl`
- `data/ingestion_registry.json`
- Qdrant collection contents in your chosen Qdrant instance

`data/chunks/` is not currently the primary source of truth in the active implementation, so the key runtime artifact remains `data/bm25_index.pkl`.

## 12. Troubleshooting

### The Codespace starts but no PDFs are found

Check:
```bash
ls data/financebench
ls data/financebench/pdfs | head
```

If the folder is missing, rerun:
```bash
./scripts/codespaces/fetch_financebench.sh
```

### `.env` does not contain the expected secrets

Rerun:
```bash
./scripts/codespaces/render_env.sh
```

Remember that `.env` is regenerated from `.env.example` plus the active environment values seen by the bootstrap script.

### Managed Qdrant is configured but `/status/services` shows Qdrant errors

Check:
- `QDRANT_URL` is the full HTTPS cluster endpoint,
- `QDRANT_API_KEY` is present,
- the collection exists or can be created,
- your network policy allows Codespaces to reach the cluster.

### The frontend loads but chat returns no results

Check all three conditions:
1. Qdrant contains vectors for the target documents.
2. `data/bm25_index.pkl` exists in the runtime that serves the backend.
3. `OPENROUTER_API_KEY` is configured if generation is required.

### Langfuse is not needed

You can still ingest and retrieve without using Langfuse as the primary concern, but `/status/services` will show it as degraded if it is not reachable.

## 13. Recommended operating approach

For the full FinanceBench run, use managed Qdrant and incremental batches from Codespaces. That gives you:
- a reproducible environment,
- persistent vector storage outside the Codespace VM,
- the ability to stop and resume ingestion at the registry level,
- a backend/frontend path you can validate from the same environment.

For small debugging loops, local Docker Qdrant is fine, but it should not be the default for the full corpus.
