# FinLens

FinLens is a Python workspace for document parsing, hybrid retrieval, and LLM-backed Q&A over SEC filings.

## Setup

```bash
cd backend
uv sync
cp ../.env.example .env
```

Run services:

```bash
docker compose up -d
```

Langfuse mode:

- Local mode (default): set `LANGFUSE_MODE=local` and keep `LANGFUSE_HOST=http://langfuse:3000` in Docker and `LANGFUSE_HOST=http://localhost:3000` when running backend directly. This uses the included local Langfuse stack.
- Hosted mode: set `LANGFUSE_MODE=hosted` and set `LANGFUSE_HOST` to your hosted Langfuse URL (for example `https://us.cloud.langfuse.com`). This sends traces to live Langfuse without local container requirements.

`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are required when `LANGFUSE_MODE=hosted`; when absent, tracing is disabled at runtime with a warning.

Qdrant mode:

- Local mode (default): set `QDRANT_MODE=local` (or omit). In this mode, the stack expects local, unauthenticated Qdrant at `QDRANT_URL`.
- Hosted mode: set `QDRANT_MODE=hosted`, `QDRANT_URL` to the managed cluster endpoint, and `QDRANT_API_KEY` to the cluster API key.

In hosted mode, missing `QDRANT_URL` or `QDRANT_API_KEY` keeps the backend running but disables Qdrant checks and ingestion/retrieval work with clear status details.

## CI & Validation

Pull requests are validated by `.github/workflows/pr-checks.yml` and are path-aware.

- Backend path changes trigger:
  - `backend-unit`: `uv run pytest tests/unit -q`
  - `backend-integration`: `uv run pytest tests/integration -m integration -q` with local Qdrant
- Frontend path changes trigger:
  - `npm test -- --run`
  - `npm run build`
- Docs-only changes are not run against backend/frontend gates.
- RAGAS eval is manual only (`workflow_dispatch`) and no longer runs automatically on push/PR.

## Testing Layout

Backend tests are split into:

- `backend/tests/unit/` fast, mocked tests.
- `backend/tests/integration/` local-service tests (Qdrant).
- `backend/tests/live/` external service tests (OpenRouter/Langfuse), not part of required PR checks.

Frontend tests are under:

- `frontend/src/**/*.__tests__/*` and Vitest component tests under `frontend/src/**/__tests__/*`.
- `frontend/vitest.config.ts` runs tests in jsdom with setup from `frontend/src/test/setup.ts`.

`backend/tests/conftest.py` provides fixtures for:

- sample nodes and document paths
- mocked embedding model (prevents local heavy model load)
- `qdrant_client` integration guard
- optional live service key guards

## Run Tests

Run backend unit tests in parity with CI:

```bash
cd backend
uv run pytest tests/unit -q
```

Run backend integration tests in parity with CI:

```bash
cd backend
uv run pytest tests/integration -m integration -q
```

Run backend live tests (requires secrets/services):

```bash
cd backend
uv run pytest tests/live -m live
```

Run frontend unit tests/build:

```bash
cd frontend
npm ci
npm test -- --run
npm run build
```

Run everything except live:

```bash
cd backend
uv run pytest tests -m "not live"
```

Manual eval workflow:

```bash
In GitHub, run workflow_dispatch on `.github/workflows/eval.yml`
```

## Useful one-off commands

```bash
cd backend
uv run python ingestion/setup_collection.py
uv run python -m ingestion.run_ingestion --pdf 3M_2022_10K.pdf
uv run python ingestion/run_ingestion.py
uv run uvicorn api.main:app --reload --port 8000
```
