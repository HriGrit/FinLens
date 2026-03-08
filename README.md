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

## Testing Layout

Tests are split into:

- `backend/tests/unit/` fast, fully mocked tests.
- `backend/tests/integration/` tests that use local services (Qdrant, local filesystem).
- `backend/tests/live/` tests that call external providers (OpenRouter).

`backend/tests/conftest.py` provides fixtures for:

- sample nodes and document paths
- mocked embedding model (prevents local heavy model load)
- integration `qdrant_client` guard
- optional live service requirement guards

## Run Tests

Run all unit tests:

```bash
cd backend
uv run pytest tests/unit -m "unit"
```

Run integration tests:

```bash
cd backend
uv run pytest tests/integration -m "integration"
```

Run live tests (requires API keys and external access):

```bash
cd backend
uv run pytest tests/live -m "live"

```

Run everything except live:

```bash
cd backend
uv run pytest tests -m "not live"
```

## Useful one-off commands

```bash
uv run python ingestion/setup_collection.py
uv run python ingestion/run_ingestion.py
uv run uvicorn api.main:app --reload --port 8000
```
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

## Testing Layout

Tests are split into:

- `backend/tests/unit/` fast, fully mocked tests.
- `backend/tests/integration/` tests that use local services (Qdrant, local filesystem).
- `backend/tests/live/` tests that call external providers (OpenRouter).

`backend/tests/conftest.py` provides fixtures for:

- sample nodes and document paths
- mocked embedding model (prevents local heavy model load)
- integration `qdrant_client` guard
- optional live service requirement guards

## Run Tests

Run all unit tests:

```bash
cd backend
uv run pytest tests/unit -m "unit"
```

Run integration tests:

```bash
cd backend
uv run pytest tests/integration -m "integration"
```

Run live tests (requires API keys and external access):

```bash
cd backend
uv run pytest tests/live -m "live"

```

Run everything except live:

```bash
cd backend
uv run pytest tests -m "not live"
```

## Useful one-off commands

```bash
uv run python ingestion/setup_collection.py
uv run python ingestion/run_ingestion.py
uv run uvicorn api.main:app --reload --port 8000
```
