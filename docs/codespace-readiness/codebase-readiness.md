# Codebase Readiness Plan For Codespaces

## Summary
Prepare the application so it can run correctly in a future Codespace, but do not add any Codespaces-specific config yet. The goal of this phase is to remove local-only assumptions, make ingestion/query behavior portable, and add tests that prove the code can run against either local Docker Qdrant or managed Qdrant.

---

## Feature Groups

The key changes below are organized into four self-contained groups. Each group is independently deliverable: all files it touches, the tests that prove it works, and the context needed to implement it are described within the group.

---

### Group A — Qdrant Client, Identity & Health

**Scope:** Everything that touches how the application connects to, authenticates with, and health-checks Qdrant. Completing this group makes the entire application able to target either a local Docker Qdrant or a managed cloud Qdrant purely through environment variables, with no code changes.

**Why it is its own group:** All three changes share the same root cause — the codebase assumes an unauthenticated, locally-reachable Qdrant. Fixing the client factory (change 1), the health check (change 6), and point identity (change 2) together makes Qdrant usage fully portable and idempotent. None of these changes require the manifest (Group B) or the status endpoint (Group C) to be done first.

#### Changes

##### 1. Centralize Qdrant configuration and client creation
- Add one shared helper module for Qdrant config/client creation, used by:
  - ingestion setup
  - ingestion indexing
  - retrieval
  - API ingestion status
- Standardize env inputs:
  - `QDRANT_URL`
  - `QDRANT_API_KEY` optional for local, required for cloud
  - `QDRANT_COLLECTION`
- Make every `QdrantClient(...)` call go through this helper so switching between local and cloud is env-only.
- Remove code paths that assume unauthenticated `http://localhost:6333` is always valid.

##### 2. Fix Qdrant point identity so reingestion is safe
- Replace the current point ID scheme based on `filename:page_number:chunk_index`.
- Use a collision-safe deterministic key that uniquely identifies each emitted chunk, including multiple nodes on the same page.
- The ID must remain stable across reruns for the same source content so upserts are idempotent.
- Keep this logic inside ingestion/indexing, not spread across callers.

##### 6. Make service health checks compatible with managed Qdrant
- Replace raw `GET {QDRANT_URL}/healthz` checks in API status code with checks that work through the authenticated Qdrant client.
- Keep Langfuse health separate.
- Ensure cloud Qdrant does not show as unhealthy just because HTTP healthz is not anonymously exposed.

#### Files touched
- New shared helper module (likely `backend/ingestion/qdrant_client.py` or `backend/shared/qdrant.py`)
- `backend/ingestion/index.py` — replace direct `QdrantClient(...)` calls
- `backend/retrieval/hybrid.py` — replace direct `QdrantClient(...)` calls
- `backend/api/main.py` — replace `/healthz` HTTP check and any inline `QdrantClient(...)` instantiation

#### Environment variables required
| Variable | Required | Notes |
|---|---|---|
| `QDRANT_URL` | Yes | e.g. `http://localhost:6333` for local, cloud URL for managed |
| `QDRANT_API_KEY` | Cloud only | Absent = unauthenticated (local Docker) |
| `QDRANT_COLLECTION` | Yes | Collection name |

#### Test coverage
- **Unit:** shared Qdrant helper returns correct client config for local and cloud env combinations.
- **Unit:** point ID generation produces stable IDs for the same chunk and different IDs for different chunks on the same page.
- **Integration:** reingesting the same document does not increase Qdrant point count.
- **Integration:** `/status/services` works with mocked authenticated Qdrant access and no anonymous `/healthz`.

#### Completion criteria
- No `QdrantClient(...)` instantiation exists outside the shared helper.
- `QDRANT_API_KEY` absent → unauthenticated client (local Docker works).
- `QDRANT_API_KEY` present → authenticated client (cloud works).
- `/status/services` does not make a raw HTTP call to `/healthz`; it uses the Qdrant client directly.
- Reingest of the same document produces zero new Qdrant points.

---

### Group B — Ingestion State & BM25 Durability

**Scope:** The ingestion pipeline's ability to track per-document progress durably and rebuild BM25 correctly from that state. Completing this group makes long-running batch ingestion resumable and prevents BM25 drift across reruns.

**Why it is its own group:** Changes 3 and 4 are tightly coupled — BM25 rebuild (change 4) must read from the manifest (change 3) to know which documents succeeded. Neither depends on the Qdrant client refactor (Group A) or the status endpoint (Group C), though Group C will benefit from the manifest once it exists.

#### Changes

##### 3. Make ingestion state durable enough for long-running batch work
- Replace the current registry shape `{"ingested": [...]}` with a per-document manifest keyed by filename.
- Track at least:
  - `status`
  - `attempts`
  - `last_error`
  - `chunk_count`
  - `updated_at`
- Update `run_ingestion.py` to:
  - mark successful docs individually
  - preserve failures without losing progress on successful docs
  - support rerunning failed/pending docs deterministically
- Add CLI flags needed for resumable operation:
  - `--workers`
  - `--retry-failed`
  - `--continue-on-error`
- Keep `--limit` and `--list`.

##### 4. Stop BM25 rebuild logic from drifting or duplicating
- Do not keep appending blindly to an existing BM25 payload loaded from pickle.
- Rebuild BM25 from the set of documents marked successful in the manifest, using deterministic chunk output for each successful document.
- If document-level cached chunk artifacts are introduced, make BM25 rebuild use them.
- Keep `data/bm25_index.pkl` as the runtime artifact for query-time sparse retrieval.

#### Files touched
- `backend/ingestion/run_ingestion.py` — manifest read/write, CLI flags, per-doc status tracking
- `backend/ingestion/index.py` — `build_bm25_index` rebuilds from manifest-confirmed docs only
- New manifest schema (JSON file at `data/ingestion_manifest.json` or similar, gitignored)

#### Manifest schema (minimum)
```
{
  "<filename>": {
    "status": "success" | "failed" | "pending",
    "attempts": <int>,
    "last_error": "<string or null>",
    "chunk_count": <int or null>,
    "updated_at": "<ISO 8601 timestamp>"
  }
}
```

#### CLI flags to add
| Flag | Behaviour |
|---|---|
| `--workers N` | Parallel ingestion workers |
| `--retry-failed` | Re-queue docs with `status: failed` |
| `--continue-on-error` | Do not abort batch on single-doc failure |
| `--limit N` | (keep existing) |
| `--list` | (keep existing) |

#### Test coverage
- **Unit:** manifest read/write round-trips document status, attempts, and errors.
- **Unit:** `run_ingestion` CLI selection logic handles pending vs failed vs completed docs correctly.
- **Integration:** a mixed batch with one failing document still records successful documents and preserves the failed one for retry.
- **Integration:** BM25 rebuild from successful manifest entries does not duplicate nodes across reruns.

#### Completion criteria
- Running ingestion twice on the same document updates the manifest entry in place; it does not append.
- A failed document is recorded with `status: failed` and `last_error` populated; successful docs in the same batch are unaffected.
- `--retry-failed` picks up and retries only `status: failed` entries.
- BM25 rebuilt from a clean manifest over two identical runs produces a pickle of the same byte length.
- `data/bm25_index.pkl` is never loaded and appended to; it is always fully rebuilt.

---

### Group C — API & Ingestion Status Reporting

**Scope:** The `/status/ingestion` endpoint accurately reflecting the discovered corpus and gracefully degrading when Qdrant is unavailable. Completing this group gives operators a reliable view of pipeline state from the API regardless of environment.

**Why it is its own group:** This is a single endpoint with a well-defined contract change. It reads from the filesystem (PDF discovery) and optionally from the manifest (Group B) and Qdrant. It can be implemented against a stub manifest and completed before or after Group B, but it will be more accurate once Group B's manifest exists.

#### Changes

##### 5. Fix ingestion/API status reporting
- Change `/status/ingestion` so `total_documents` reflects the actual discovered corpus under `data/financebench/pdfs`, not the number already ingested.
- Keep `indexed_documents` based on indexed filenames and `indexed_chunks` based on Qdrant count.
- If Qdrant is unavailable, still return a meaningful total/pending/error view from local manifest data.
- Remove localhost-specific health/report text from API responses where it implies local-only deployment.

#### Files touched
- `backend/api/main.py` — `/status/ingestion` handler

#### Response contract (minimum)
| Field | Source |
|---|---|
| `total_documents` | Count of PDFs discovered on disk under `data/financebench/pdfs` |
| `indexed_documents` | Count of filenames present in Qdrant (or manifest if Qdrant unavailable) |
| `indexed_chunks` | Qdrant point count for the collection (or `null` if Qdrant unavailable) |
| `pending_documents` | `total_documents − indexed_documents` |
| `failed_documents` | Count of manifest entries with `status: failed` (0 if no manifest yet) |

#### Test coverage
- **Integration:** `/status/ingestion` reports discovered corpus totals correctly even when manifest and Qdrant differ.

#### Completion criteria
- `total_documents` matches the file count under `data/financebench/pdfs` regardless of ingestion state.
- When Qdrant is unreachable, the endpoint returns HTTP 200 with `indexed_chunks: null` and a non-empty `total_documents`.
- No response field or message text contains `localhost` or any environment-specific URL.

---

### Group D — Query-Time Portability

**Scope:** Making the online retrieval path's assumptions about BM25 and Qdrant explicit and ensuring it fails clearly when prerequisites are missing. Completing this group ensures the query path behaves predictably in any environment.

**Why it is its own group:** This is a read-only audit and hardening of the retrieval path. It does not require the manifest (Group B) or the Qdrant client refactor (Group A) to be complete, though it will benefit from Group A's unified client. No new behavior is introduced — existing hybrid retrieval is preserved.

#### Changes

##### 7. Keep query-time behavior portable
- Preserve the current hybrid retrieval design, but make its assumptions explicit:
  - dense retrieval comes from Qdrant
  - sparse retrieval comes from local `bm25_index.pkl`
- Do not redesign retrieval in this phase.
- Ensure the backend fails clearly when BM25 is missing, except for the already-implemented corrupt-pickle dense-only fallback.

#### Files touched
- `backend/retrieval/hybrid.py` — explicit failure path when `bm25_index.pkl` is absent
- `backend/retrieval/pipeline.py` — surface clear error if BM25 unavailable (not a silent fallback for missing file)

#### Behaviour contract
| Condition | Expected behaviour |
|---|---|
| `bm25_index.pkl` present and valid | Hybrid BM25 + Qdrant retrieval |
| `bm25_index.pkl` corrupt | Existing dense-only fallback (keep as-is) |
| `bm25_index.pkl` absent entirely | Raise a clear, descriptive error; do not silently degrade |
| Qdrant reachable, BM25 present | Full hybrid pipeline |

#### Test coverage
- **Regression:** keep the currently passing retrieval/API tests and extend them to cover the new manifest and Qdrant helper paths.

#### Completion criteria
- Missing `bm25_index.pkl` raises a named exception with a message that tells the operator to run ingestion first.
- Corrupt `bm25_index.pkl` still falls back to dense-only (existing behaviour, do not break).
- All existing retrieval and API tests continue to pass after Group A's Qdrant client refactor is applied.

---

## Tests (Full List)

The tests below are distributed across the groups above. They are also listed here for completeness.

### Unit tests
- Shared Qdrant helper returns correct client config for local and cloud env combinations. → **Group A**
- Point ID generation produces stable IDs for the same chunk and different IDs for different chunks on the same page. → **Group A**
- Manifest read/write round-trips document status, attempts, and errors. → **Group B**
- `run_ingestion` CLI selection logic handles pending vs failed vs completed docs correctly. → **Group B**

### Integration tests
- Reingesting the same document does not increase Qdrant point count. → **Group A**
- A mixed batch with one failing document still records successful documents and preserves the failed one for retry. → **Group B**
- BM25 rebuild from successful manifest entries does not duplicate nodes across reruns. → **Group B**
- `/status/ingestion` reports discovered corpus totals correctly even when manifest and Qdrant differ. → **Group C**
- `/status/services` works with mocked authenticated Qdrant access and no anonymous `/healthz`. → **Group A**

### Regression tests
- Keep the currently passing retrieval/API tests and extend them to cover the new manifest and Qdrant helper paths. → **Group D**

---

## Assumptions
- Codespaces setup, devcontainer files, startup scripts, and dataset hydration are out of scope for this phase.
- The backend will continue to require `data/bm25_index.pkl` at query time.
- FinanceBench remains external to repo history and is discovered from `data/financebench/pdfs`.

---

## Delivery Order (Recommended)

The groups are independent but the following order minimizes rework:

1. **Group A** — Qdrant client, identity, health. Unblocks all other groups from targeting cloud Qdrant.
2. **Group B** — Ingestion manifest and BM25 rebuild. Produces the manifest that Group C reads.
3. **Group C** — API status endpoint. Best implemented after Group B's manifest exists.
4. **Group D** — Query-time portability. Can be done at any point; no upstream dependencies.
