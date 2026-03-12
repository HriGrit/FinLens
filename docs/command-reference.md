# FinLens Command Reference

This document collects the commands and callable endpoints that FinLens currently exposes.

Use this page when you want to:

- start the stack
- run or resume ingestion
- check how many files are already ingested
- see how many files are pending or failed
- query the API directly
- validate that the main parts of the system still work

## Base Paths And URLs

Repository root:

```bash
/Users/soam/Documents/Work/FinOps
```

Common local URLs after the stack is running:

- Frontend: `http://localhost:5173`
- FastAPI: `http://localhost:8000`
- FastAPI Swagger docs: `http://localhost:8000/docs`
- FastAPI ReDoc: `http://localhost:8000/redoc`
- OpenAPI schema: `http://localhost:8000/openapi.json`
- Qdrant dashboard: `http://localhost:6333/dashboard`
- Langfuse: `http://localhost:3000`

## 1. Environment And Stack Commands

### Install dependencies

Root workspace:

```bash
uv sync
```

Backend workspace:

```bash
cd backend
uv sync
```

Frontend workspace:

```bash
cd frontend
npm ci
```

### Prepare environment

```bash
cp .env.example .env
```

Fill in `OPENROUTER_API_KEY` and any other required values before using generation features.

### Start the full local stack

```bash
docker compose up -d
```

This starts:

- `backend`
- `frontend`
- `qdrant`
- `postgres`
- `langfuse`

### Start only selected services

Qdrant only:

```bash
docker compose up -d qdrant
```

Infra only:

```bash
docker compose up -d qdrant postgres langfuse
```

App containers only:

```bash
docker compose up -d backend frontend
```

## 2. Backend Runtime Commands

Start the API manually from `backend/`:

```bash
cd backend
uv run python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Alternative command also used in repo docs:

```bash
cd backend
uv run uvicorn api.main:app --reload --port 8000
```

Create the Qdrant collection and payload indexes:

```bash
cd backend
uv run python ingestion/setup_collection.py
```

## 3. Ingestion Commands

All ingestion commands are run from `backend/`.

### Show ingestion status without ingesting anything

```bash
cd backend
uv run python -m ingestion.run_ingestion --list
```

This is the primary CLI command for checking:

- total PDFs found
- successful ingestions
- failed ingestions
- pending ingestions
- the next documents queued for ingestion

### Ingest the next N unprocessed files

```bash
cd backend
uv run python -m ingestion.run_ingestion --limit 10
```

Replace `10` with any batch size you want.

### Ingest all remaining files

```bash
cd backend
uv run python -m ingestion.run_ingestion
```

### Change the parse worker count

```bash
cd backend
uv run python -m ingestion.run_ingestion --workers 8
```

### Retry files previously marked as failed

```bash
cd backend
uv run python -m ingestion.run_ingestion --retry-failed
```

### Keep the batch running even if one document fails

```bash
cd backend
uv run python -m ingestion.run_ingestion --continue-on-error
```

### Combine ingestion flags

Retry failed files, use 8 workers, and continue past single-file failures:

```bash
cd backend
uv run python -m ingestion.run_ingestion --retry-failed --workers 8 --continue-on-error
```

Retry failed files but only process the next 5:

```bash
cd backend
uv run python -m ingestion.run_ingestion --retry-failed --limit 5
```

### Get CLI help

```bash
cd backend
uv run python -m ingestion.run_ingestion --help
```

## 4. Status Commands You Will Use Most Often

### CLI status: how many files are ingested or left

```bash
cd backend
uv run python -m ingestion.run_ingestion --list
```

### API status: document counts and indexed chunk count

```bash
curl http://localhost:8000/status/ingestion
```

This returns:

- `total_documents`
- `indexed_documents`
- `indexed_chunks`
- `pending_documents`
- `failed_documents`
- `status`

### Service health status

```bash
curl http://localhost:8000/status/services
```

This returns health info for:

- Qdrant
- Langfuse
- OpenRouter configuration
- Postgres reachability through Langfuse

### Basic API liveness check

```bash
curl http://localhost:8000/health
```

## 5. API Endpoints Exposed By The Application

These are the HTTP routes defined by `backend/api/main.py`, plus the standard FastAPI documentation routes.

### `GET /health`

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### `GET /models/free`

```bash
curl http://localhost:8000/models/free
```

Returns the free generation models surfaced by the backend.

### `GET /status/ingestion`

```bash
curl http://localhost:8000/status/ingestion
```

Best endpoint for operational document counts.

### `GET /status/services`

```bash
curl http://localhost:8000/status/services
```

Best endpoint for stack health.

### `POST /chat`

Example request:

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What were 3M'\''s total net sales in 2022?",
    "company": "3M",
    "year": "2022",
    "retrieval_top_k": 20,
    "rerank_top_k": 5
  }'
```

Accepted request fields:

- `query`
- `company`
- `year`
- `model`
- `retrieval_top_k`
- `rerank_top_k`

Returns:

- `answer`
- `citations`
- `usage`
- `model`
- `latency_ms`
- `trace_id`
- `reasoning`

### FastAPI built-in docs routes

Swagger UI:

```bash
open http://localhost:8000/docs
```

ReDoc:

```bash
open http://localhost:8000/redoc
```

OpenAPI JSON:

```bash
curl http://localhost:8000/openapi.json
```

If `open` is not available in your shell environment, paste the URL into your browser directly.

## 6. Frontend Commands

All frontend commands are run from `frontend/`.

Start the Vite dev server:

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Build the frontend:

```bash
cd frontend
npm run build
```

Preview the built frontend:

```bash
cd frontend
npm run preview
```

Run frontend tests:

```bash
cd frontend
npm test -- --run
```

## 7. Validation And Smoke-Test Commands

These are not runtime API routes, but they are executable checks the repo already provides.

### Validate Qdrant connectivity and round-trip behavior

```bash
cd backend
uv run python ../tests/validate_qdrant.py
```

### Validate LLM connectivity through OpenRouter

```bash
cd backend
uv run python ../tests/validate_llm.py
```

### Validate the reranker model

```bash
cd backend
uv run python ../tests/validate_reranker.py
```

### Validate the API endpoints

```bash
cd backend
uv run python ../tests/validate_api.py
```

### Validate post-ingestion artifacts

```bash
cd backend
uv run python ../tests/validate_ingestion.py
```

### Validate the full end-to-end retrieval and generation path

```bash
cd backend
uv run python ../tests/validate_e2e.py
```

## 8. Pytest Suite Commands

Run backend unit tests:

```bash
cd backend
uv run pytest tests/unit -q
```

Run backend integration tests:

```bash
cd backend
uv run pytest tests/integration -m integration -q
```

Run backend live tests:

```bash
cd backend
uv run pytest tests/live -m live
```

Run all backend tests except live:

```bash
cd backend
uv run pytest tests -m "not live"
```

## 9. Evaluation Commands

Run the full RAGAS evaluation:

```bash
cd backend
uv run python ../eval/ragas_eval.py
```

Run a smaller random sample:

```bash
cd backend
uv run python ../eval/ragas_eval.py --sample 5
```

Run against a specific dataset path:

```bash
cd backend
uv run python ../eval/ragas_eval.py --dataset ../eval/qa_dataset.json --sample 3
```

## 10. Files Backing The Main Status Commands

These are useful when you want to understand where the status numbers come from:

- Manifest file: `data/ingestion_manifest.json`
- Chunk artifacts: `data/chunks/`
- BM25 index: `data/bm25_index.pkl`
- PDF discovery directory: `data/financebench/pdfs/`

`uv run python -m ingestion.run_ingestion --list` reads discovered PDFs and manifest state.

`GET /status/ingestion` combines manifest state with Qdrant collection data when Qdrant is reachable.

## 11. Quick Answer For Daily Use

If you only need the most common operational commands, these are the ones to remember:

Start everything:

```bash
docker compose up -d
```

See how many files are done, failed, or left:

```bash
cd backend
uv run python -m ingestion.run_ingestion --list
```

See indexed document and chunk counts over HTTP:

```bash
curl http://localhost:8000/status/ingestion
```

Ingest the next batch:

```bash
cd backend
uv run python -m ingestion.run_ingestion --limit 10
```

Query the app manually:

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What were 3M'\''s total net sales in 2022?",
    "company": "3M",
    "year": "2022"
  }'
```
