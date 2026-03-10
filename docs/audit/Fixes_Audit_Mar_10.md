# FinLens — Audit Fixes Plan (Mar 10, 2026)

> **Methodology:** Every issue below was re-verified by reading the actual source file at the stated line. Verification status is noted per finding. One finding from the original audit was invalidated after re-check (C-6). One new bug was discovered during re-verification and is marked **[NEW]**.

---

## Verification Corrections

### C-6 INVALIDATED — `qa_dataset.json` does exist

**Original claim:** `eval/qa_dataset.json` is not committed and RAGAS eval always fails.
**Actual state:** `eval/qa_dataset.json` is present at the expected path (`eval/qa_dataset.json`). Verified via filesystem glob. C-6 is a **false positive** and requires no fix.

### C-4 NUANCED — Langfuse SDK uses async queue, not synchronous HTTP

**Original claim:** `trace.span()` makes a synchronous HTTP call; Langfuse downtime causes immediate 500.
**Actual state:** Langfuse Python SDK v2 queues all API calls in a background thread — `trace.span()` does not block on HTTP. However, the architectural concern remains valid: if the Langfuse client initialization itself raises (e.g., import error, SDK version mismatch), it would still crash. The risk is narrower than stated but real enough to address.

---

## Functional Groups

Issues are grouped by the subsystem they affect, not by severity. Each group can be worked on independently.

---

## Group A — Ingestion Pipeline

**Files:** `backend/ingestion/chunk.py`, `backend/ingestion/index.py`, `backend/ingestion/run_ingestion.py`

These issues affect the offline ingestion pipeline. Bugs here corrupt or degrade the data written to Qdrant and BM25.

---

### A-1 [C-1] Vacuous page_number assertion — guard is dead code

**File:** `backend/ingestion/chunk.py:102-104`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — `or True` bypass completely removed; `_assert_chunk_metadata` now correctly checks all 6 required fields are present.

```python
assert node.metadata.get("page_number") is not None or True, (
    f"Chunk {i} has page_number=None (acceptable for some sources)"
)
```

`X or True` is always `True` in Python. The assertion cannot fail. Nodes with `page_number=None` pass silently and flow into Qdrant, producing `p.null` in citations.

**Proposed fix area:** Remove the `or True` so the assertion either enforces the constraint or remove the assertion entirely and handle `None` explicitly.

---

### A-2 [C-3] `None` metadata values silently dropped from Qdrant payload

**File:** `backend/ingestion/index.py:57`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — `**{k: v ... if v is not None}` dict comprehension replaced with `**node.metadata`; None values now stored in Qdrant payload. **Note: first explore agent misreported this as NOT FIXED — it confused fix direction.**

```python
payload = {
    "text": node.text,
    **{k: v for k, v in node.metadata.items() if v is not None},
}
```

If `page_number` is `None`, the key is absent from the stored payload entirely. At query time, `node.metadata.get("page_number")` returns `None`, which renders as `p.null` or `p.None` in the frontend citation card.

**Proposed fix area:** Store `None` values as-is (remove the `if v is not None` filter), or store a sentinel like `0` for missing page numbers, and update the frontend to handle it gracefully.

---

### A-3 [C-5] Duplicate chunks on crash-recovery — non-atomic indexing

**File:** `backend/ingestion/run_ingestion.py:230-240`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED (with side-effect concern) — `uuid.uuid4()` replaced with deterministic `uuid.uuid5(NAMESPACE_DNS, f"{filename}:{page_number}:{chunk_index}")`. See **Side Effects** section for a data-loss regression on multi-table/figure pages.

```python
build_qdrant_index(new_chunks, collection_name=QDRANT_COLLECTION)
# ... BM25 build ...
_save_registry(already_ingested | newly_ingested)  # only saved AFTER Qdrant upsert
```

`build_qdrant_index` assigns new `uuid4()` IDs per run (`index.py:61`). If the process crashes after the Qdrant upsert but before `_save_registry` completes, the document is not in the registry. The next run re-ingests it, inserting duplicate points with new UUIDs. Qdrant now has 2× chunks for that document with no deduplication key.

**Proposed fix area:** Use a deterministic point ID derived from `(filename, page_number, chunk_index)` so re-ingestion becomes an idempotent upsert rather than an insertion of duplicates.

---

### A-4 [M-5] `_parse_pdf_filename` produces wrong metadata for 4+ underscore-part filenames

**File:** `backend/ingestion/run_ingestion.py:52-58`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — parser now validates year with `_YEAR_RE.match(year)` and handles ≥3-part filenames; invalid filenames produce a WARN + skip instead of silent miscategorisation.

```python
parts = pdf_path.stem.split("_")   # "3M_2022_10K_10" → ["3M","2022","10K","10"]
doc_type_raw = parts[-1]            # → "10" (should be "10K")
year = parts[-2]                    # → "10K" (should be "2022")
company = "_".join(parts[:-2])      # → "3M_2022" (should be "3M")
```

Any PDF with 4+ underscore-separated parts produces entirely wrong `company`, `year`, and `doc_type` metadata. Citations for those documents will show wrong values.

**Proposed fix area:** Tighten the filename convention (reject any non-3-part name loudly), or add a metadata sidecar JSON file approach to decouple metadata from filenames entirely.

---

## Group B — Retrieval Layer

**Files:** `backend/retrieval/hybrid.py`, `backend/retrieval/pipeline.py`

These issues affect query-time retrieval correctness and resilience.

---

### B-1 [H-1] BM25 retrieval ignores company and year filters

**File:** `backend/retrieval/hybrid.py:66`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — post-retrieval company/year filter added to BM25 nodes after `bm25_retriever.retrieve(query)`.

```python
bm25_results = bm25_retriever.retrieve(query)  # no filter applied
```

Qdrant dense retrieval at lines 76-90 applies `company` and `year` payload filters. BM25 retrieval has no equivalent. For a multi-company dataset, a query for "3M revenue 2022" will return chunks from all other companies in the BM25 path, wasting candidate slots and risking cross-company answer contamination.

**Proposed fix area:** After BM25 retrieval, filter `bm25_nodes` by `metadata.get("company")` and `metadata.get("year")` before passing to `_fuse_results`.

---

### B-2 [H-4] Corrupt BM25 pickle crashes `/chat` with 500 — no graceful degradation

**File:** `backend/retrieval/hybrid.py:61`, `backend/ingestion/index.py:104-105`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — `load_bm25_index` wrapped in `try/except Exception`; on failure logs a warning and falls back to empty `bm25_nodes` (dense-only).

```python
# hybrid.py — no try/except around this
bm25_retriever = load_bm25_index(BM25_INDEX_PATH)

# index.py — raises ValueError on corrupt payload
if not isinstance(payload, dict) or "nodes" not in payload:
    raise ValueError("Invalid BM25 index payload. Rebuild the index.")
```

A corrupted BM25 pickle (partial write, version mismatch) passes the `exists()` check at line 60 but raises `ValueError` inside `load_bm25_index`. This propagates through `hybrid_retrieve()` and crashes `/chat` with HTTP 500. There is no fallback to dense-only mode.

**Proposed fix area:** Wrap `load_bm25_index` in a `try/except` in `hybrid.py`. On failure, log a warning and fall back to empty `bm25_nodes` (dense-only mode).

---

### B-3 [M-3] BM25 corpus-size check accesses non-existent `.bm25.scores` attribute

**File:** `backend/retrieval/hybrid.py:62-65`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — changed from non-existent `.bm25.scores` to correct `.bm25.corpus_size`.

```python
if bm25_retriever.bm25 and bm25_retriever.bm25.scores:
    bm25_corpus_size = bm25_retriever.bm25.scores.get("num_docs", 0)
```

`rank_bm25.BM25Okapi` does not have a `.scores` attribute as a dict. The `if` condition evaluates to `False` silently, the `if` block is never entered, and the `similarity_top_k` capping logic never runs. If the corpus has fewer documents than `top_k`, the retriever may request more results than exist.

**Proposed fix area:** Use `bm25_retriever.bm25.corpus_size` (the actual `BM25Okapi` attribute) instead.

---

### B-4 [NEW] `r.payload.pop("text", "")` mutates Qdrant result in-place

**File:** `backend/retrieval/hybrid.py:93`
**Verified:** ✅ Confirmed (new finding, not in original audit)
**Fix Status:** ✅ FIXED — `.pop("text", "")` replaced with `.get("text", "")` + metadata passed as `{k: v for k, v in r.payload.items() if k != "text"}`.

```python
for r in results.points:
    text = r.payload.pop("text", "")   # mutates the ScoredPoint's payload dict
    dense_nodes.append(TextNode(text=text, metadata=r.payload))
```

`pop()` modifies the `ScoredPoint.payload` dict in-place, removing the `"text"` key permanently. If `results.points` is ever reused after this loop (e.g., in future refactors or tests), the `"text"` field will be missing. Using `.get()` + excluding the key from metadata is safer.

**Proposed fix area:** Replace `r.payload.pop("text", "")` with `text = r.payload.get("text", "")` and pass `{k: v for k, v in r.payload.items() if k != "text"}` as the metadata.

---

## Group C — Observability & Tracing

**Files:** `backend/observability/tracing.py`, `backend/retrieval/pipeline.py`, `backend/generation/generate.py`

These issues degrade observability or cause test-environment side effects.

---

### C-1 [C-4] Langfuse client initialization failure can crash `/chat`

**File:** `backend/observability/tracing.py:18-27`, `backend/api/main.py:80-83`
**Verified:** ✅ Partially confirmed (see correction above)
**Fix Status:** ✅ FIXED — `_get_langfuse()` has key-presence check + try/except around `Langfuse()` constructor; returns `_NoOpLangfuse()` stub on any failure.

```python
def _get_langfuse():
    global _langfuse
    if _langfuse is None:
        from langfuse import Langfuse
        _langfuse = Langfuse(public_key="", secret_key="", host="...")
    return _langfuse
```

`create_trace()` is called at the very top of the `/chat` handler before any retrieval. If `Langfuse()` construction raises (import error, version mismatch, etc.), the entire `/chat` endpoint fails. The observability layer should never be load-bearing.

**Proposed fix area:** Wrap `_get_langfuse()` in a try/except that returns a no-op stub trace if Langfuse initialization fails.

---

### C-2 [H-6] LiteLLM global callbacks set at module import — affects all test environments

**File:** `backend/generation/generate.py:18-19`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — `litellm.success_callback` / `failure_callback` moved from module-level to `register_langfuse_callbacks()` function, called only in FastAPI `_lifespan` context.

```python
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]
```

These are module-level statements executed on `import generate`. Any test that imports `generate.py` (including integration tests) installs Langfuse callbacks into the LiteLLM global state. Every mocked or real LLM call in tests then tries to send data to a non-running Langfuse instance, producing background thread errors.

**Proposed fix area:** Move callback registration to a function called explicitly at app startup (e.g., in `main.py`'s `lifespan` context), not at module import.

---

### C-3 [M-1] Langfuse initialized with empty strings when keys not set

**File:** `backend/observability/tracing.py:22-27`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — key-presence check added; missing keys trigger warning + `_NoOpLangfuse()`. Addressed as part of C-1 fix.

```python
_langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
    ...
)
```

If keys are absent, Langfuse is initialized with `""` credentials. It will silently fail authentication on every call, generating background thread errors. There is no disabled/no-op mode.

**Proposed fix area:** Check for key presence before creating the client. If keys are absent, return a no-op stub that logs a warning and does nothing.

---

### C-4 [L-3] Langfuse spans never record their output

**File:** `backend/retrieval/pipeline.py:28-38`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — spans now capture output: `s.end(output={"n_candidates": len(candidates)})` for hybrid_retrieve, `s.end(output={"n_results": len(results)})` for rerank.

```python
with span(trace, "hybrid_retrieve", input={...}):
    candidates = hybrid_retrieve(...)
# span.end() is called in the finally block with no output= argument
```

The `tracing.span()` context manager calls `s.end()` with no arguments. Langfuse spans without `output=` data show only inputs and latency — you cannot see what was actually retrieved or reranked. Debugging retrieval quality from Langfuse is impossible.

**Proposed fix area:** Capture the result inside the `with` block and call `s.end(output={"n_results": len(candidates)})` or yield the span and call `s.end()` with output after the block.

---

## Group D — Generation & LLM

**Files:** `backend/generation/generate.py`

---

### D-1 [H-5] `response.choices[0].message.content` can be `None`

**File:** `backend/generation/generate.py:45`
**Verified:** ✅ Confirmed
**Fix Status:** ⚠️ PARTIALLY FIXED — `None` check added (`if answer is None: raise ValueError(...)`). Proposed fix was `raise HTTPException(status_code=502)` — current `ValueError` still propagates as HTTP 500. Prevents the Pydantic ValidationError, but status code is generic 500 rather than explicit 502.

```python
answer = response.choices[0].message.content  # spec allows None
```

OpenAI/LiteLLM spec permits `content=None` (tool calls, refusals, some error shapes). When `answer=None`, the return dict `{"answer": None, ...}` causes Pydantic validation failure in `ChatResponse(answer: str)`, producing HTTP 500 with a generic error instead of an informative message.

**Proposed fix area:** Add `if answer is None: raise HTTPException(status_code=502, detail="Model returned empty response.")` immediately after line 45.

---

### D-2 [XC-2] `openrouter/free` default model ID may not route correctly in LiteLLM

**File:** `frontend/src/stores/useAppStore.ts:166`, `backend/generation/generate.py:37`
**Verified:** ⚠️ Uncertain — requires live LiteLLM/OpenRouter test
**Fix Status:** ✅ FIXED — default model changed from `'openrouter/free'` to `'openrouter/stepfun/step-3.5-flash:free'`, matching the backend `DEFAULT_MODEL`.

```python
# Default in store:
model: 'openrouter/free'
# Sent as-is to litellm.completion(model="openrouter/free", ...)
```

LiteLLM's OpenRouter integration expects model IDs in `openrouter/{provider}/{model}` format. `"openrouter/free"` is not a documented LiteLLM model ID. Behavior depends on LiteLLM version. This needs a live API test to confirm.

**Proposed fix area:** Change the default to a known-valid free model ID (e.g., `openrouter/mistralai/mistral-7b-instruct:free`) or use `openrouter/auto` if LiteLLM supports it.

---

## Group E — API Layer

**Files:** `backend/api/main.py`

---

### E-1 [C-2] `retrieved_candidates` always equals `selected_nodes`

**File:** `backend/api/main.py:159, 164`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — `retrieve_and_rerank()` now returns `tuple[list[TextNode], int]`; pre-rerank `n_candidates` passed back and used as `"retrieved_candidates"` in reasoning payload.

```python
"retrieved_candidates": len(context_nodes),   # "retrieval" block
# ...
"selected_nodes": len(context_nodes),          # "rerank" block
```

Both fields receive `len(context_nodes)`, which is the already-reranked list. The pre-rerank candidate count (e.g., 20) is never captured. Both fields always show the same value (up to 5). The reasoning panel is factually wrong — users cannot distinguish "we retrieved 20 candidates and narrowed to 5" from "we retrieved 5 from scratch."

**Proposed fix area:** Capture `len(candidates)` in `retrieve_and_rerank()` before reranking and pass it back to `_build_reasoning()` separately from the final `context_nodes`.

---

### E-2 [H-2] `TOTAL_DOCUMENTS = 5` hardcoded — ingestion progress always wrong

**File:** `backend/api/main.py:38`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — `TOTAL_DOCUMENTS = 5` constant removed; `total_documents` derived from registry file (`data/ingestion_registry.json`), falling back to `len(filenames)` from Qdrant scroll.

```python
TOTAL_DOCUMENTS = 5  # manifest size in run_ingestion.py
```

With 150 PDFs ingested, `indexed_documents / total_documents` = 150/5 = 3000%, breaking the frontend `IngestionMeter` progress bar. With 1 document, it shows 20% when 1/150 are processed.

**Proposed fix area:** Derive `total_documents` from the registry file (`data/ingestion_registry.json`) or count unique filenames from Qdrant (already done in the same function — reuse that count as the denominator).

---

### E-3 [H-3] `/status/ingestion` scrolls all Qdrant points on every call — O(n) per 10s

**File:** `backend/api/main.py:186-200`
**Verified:** ✅ Confirmed
**Fix Status:** ❌ NOT FIXED — the full Qdrant scroll loop still runs on every `/status/ingestion` call to count `indexed_documents` (unique filenames). `total_documents` is now from registry (E-2), but the scroll for counting currently-indexed files remains. No caching or TTL added.

```python
while True:
    records, offset = client.scroll(collection_name=..., limit=256, ...)
    for rec in records:
        filenames.add(rec.payload.get("filename"))
    if offset is None:
        break
```

At 15,000 chunks, this performs ~60 Qdrant scroll calls per `/status/ingestion` request. The sidebar polls this endpoint every 10 seconds. At 10 open browser tabs, this is 600 Qdrant scroll calls per 10 seconds, indefinitely.

**Proposed fix area:** Cache the filename set in memory with a TTL (e.g., 30s), or read unique filenames from the registry file (O(1) file read instead of O(n) scroll).

---

### E-4 [M-2] `/status/services` makes sequential blocking HTTP calls

**File:** `backend/api/main.py:224-249`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — endpoint is now `async def` using `httpx.AsyncClient` with `asyncio.gather()` for concurrent health checks.

```python
resp = httpx.get(f"{QDRANT_URL}/healthz", timeout=2)    # blocks
resp = httpx.get(f"{LANGFUSE_URL}/api/public/health", timeout=2)  # blocks after first
```

Both are synchronous `httpx.get()` calls in sequence. If both time out, the endpoint takes 4+ seconds. FastAPI runs sync functions in a thread pool, so this blocks a worker thread for the full duration. The sidebar polls this every 10 seconds.

**Proposed fix area:** Make the endpoint `async def` and use `httpx.AsyncClient` with `asyncio.gather()` to run both health checks concurrently.

---

### E-5 [M-7] `FreeModelResponse(**model.__dict__)` is fragile for dataclass/model changes

**File:** `backend/api/main.py:74`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — changed to `dataclasses.asdict(model)`.

```python
return [FreeModelResponse(**model.__dict__) for model in get_free_models()]
```

`__dict__` on a frozen dataclass works in CPython but is not specified behavior. The idiomatic approach is `dataclasses.asdict(model)`. If `FreeModelOption` is refactored to a Pydantic model or NamedTuple, `__dict__` will either fail or return unexpected internal fields.

**Proposed fix area:** Use `dataclasses.asdict(model)` or map fields explicitly.

---

### E-6 [XC-1] No input validation on `/chat` query field

**File:** `backend/api/main.py:41-47`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — `query: str = Field(min_length=1, max_length=2000)` added to `ChatRequest`.

```python
class ChatRequest(BaseModel):
    query: str   # no min_length, no max_length, no strip
```

A whitespace-only query (`"   "`) passes validation and runs through the full RAG pipeline before returning a meaningless answer. A 100,000-character query gets sent to the LLM, potentially exhausting the context window and triggering a 400 from OpenRouter that returns as a 500 to the user.

**Proposed fix area:** Add `query: str = Field(min_length=1, max_length=2000)` and strip whitespace before processing.

---

## Group F — Frontend

**Files:** `frontend/src/components/chat/ChatPanel.tsx`, `frontend/src/components/chat/MessageBubble.tsx`, `frontend/src/stores/useAppStore.ts`, `frontend/src/hooks/usePolling.ts`

---

### F-1 [H-7] 404 error detail never shown to user

**File:** `frontend/src/components/chat/ChatPanel.tsx:57-62`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — catch block reads `axiosError.response?.data?.detail` first, with typed guard `AxiosError<{ detail?: string }>`, falls back to string response then `axiosError.message`; non-Axios errors handled separately.

```tsx
} catch (err: unknown) {
  const msg = err instanceof Error ? err.message : 'Request failed'
  // err.message for AxiosError = "Request failed with status code 404"
  // The actual backend detail: "No relevant context found for this query." is in err.response?.data?.detail
  addMessage({ content: `Error: ${msg}` })
}
```

The most common user-facing error (no documents indexed, wrong company/year filter) returns HTTP 404 with a helpful message. The frontend reads the generic Axios error string instead of the response body's `detail` field.

**Proposed fix area:** Read `(err as AxiosError)?.response?.data?.detail` first, falling back to `err.message`.

---

### F-2 [M-4] LLM markdown output rendered as raw text

**File:** `frontend/src/components/chat/MessageBubble.tsx:30`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — `<ReactMarkdown>` with `remarkGfm` + `rehypeSanitize` added for assistant messages; user messages still render as plain text.

```tsx
<div className="...">
  {message.content}   {/* renders "**bold**" as literal characters */}
</div>
```

LLM answers contain markdown (`**bold**`, `[1]` citation refs, numbered lists, `---` dividers). React renders these as plain strings. Users see `**3M's net sales were $35.4B**` rather than formatted output.

**Proposed fix area:** Render with a markdown library (e.g., `react-markdown` with `remark-gfm`) and sanitize with `rehype-sanitize`.

---

### F-3 [M-6] `usePolling` fires every 10s regardless of browser tab visibility

**File:** `frontend/src/hooks/usePolling.ts:9`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — `document.visibilitychange` listener added; interval paused when `document.hidden === true`, resumes with immediate call on tab restore.

```ts
const id = setInterval(() => savedFn.current(), intervalMs)
// fires even when tab is backgrounded
```

The polling interval runs constantly, triggering two API calls (`/status/services` + `/status/ingestion`) every 10 seconds even when the user has switched to another browser tab. Combined with E-3 (O(n) scroll), this compounds the server load.

**Proposed fix area:** Add a `document.visibilitychange` listener to pause the interval when `document.hidden === true` and resume on visibility restore.

---

### F-4 [L-4] `Citation` interface declares non-nullable fields that can be null at runtime

**File:** `frontend/src/stores/useAppStore.ts:3-11`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — `Citation` interface fields updated to `string | null` and `number | null`; `CitationCard.tsx` adds null guards with `??` fallbacks and conditional page rendering.

```typescript
export interface Citation {
  company: string       // declared non-nullable
  page_number: number   // declared non-nullable — but backend can send null
  filename: string      // declared non-nullable
}
```

Backend `generate.py:60-66` uses `.get()` with no fallback — all fields can be `null` in the JSON. TypeScript's type system does not enforce non-nullability at JSON deserialization boundaries. Runtime `null` values produce `"null"` strings or `NaN` in the UI.

**Proposed fix area:** Add `| null` to the types for `company`, `year`, `doc_type`, `page_number`, `filename` in the `Citation` interface, and add null guards in `CitationCard.tsx`.

---

### F-5 [L-5] In-flight requests not cancelled on component unmount

**File:** `frontend/src/components/chat/ChatPanel.tsx:31-66`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — `AbortController` created per request, signal passed to `postChat()`; `useEffect` cleanup aborts in-flight requests on unmount.

```tsx
const submit = async () => {
  setLoading(true)
  try {
    const result = await postChat({...})  // no AbortController
    addMessage({...})                     // fires even if unmounted
  } finally {
    setLoading(false)
  }
}
```

No `AbortController` is used. If the user navigates away mid-request, the network call continues until completion or timeout (60s as configured in `client.ts`). Zustand state (`isLoading`, `messages`) is still updated after unmount, which is harmless with Zustand but wastes network resources.

**Proposed fix area:** Create an `AbortController` per request, pass `signal` to `axios` via `api.post('/chat', req, { signal })`, and call `controller.abort()` in the `useEffect` cleanup.

---

## Group G — Infrastructure & Docker

**Files:** `docker-compose.yml`

---

### G-1 [H-8] Langfuse uses `service_started` not `service_healthy` — backend starts too early

**File:** `docker-compose.yml:21-22`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — `depends_on` changed to `condition: service_healthy`; healthcheck added (`wget` to `/api/public/health`, 15s interval, 20 retries, 30s start_period).

```yaml
depends_on:
  langfuse:
    condition: service_started    # not service_healthy
```

Langfuse + Postgres initialization takes 15-45 seconds (database migrations). The backend starts and accepts requests before Langfuse is ready. During this window, `create_trace()` may fail. Combined with C-1 (Group C), this causes all `/chat` calls to fail on fresh `docker compose up`.

**Proposed fix area:** Add a healthcheck to the Langfuse service and change the condition to `service_healthy`. Alternatively, make the backend resilient to Langfuse unavailability (see C-1 fix).

---

### G-2 [L-1] Hardcoded default secrets in `docker-compose.yml`

**File:** `docker-compose.yml:83-86`
**Verified:** ✅ Confirmed
**Fix Status:** ⚠️ PARTIALLY FIXED — `NEXTAUTH_SECRET` and `ENCRYPTION_KEY` no longer have hardcoded defaults (require explicit env). `SALT` still has `${SALT:-change-me-in-production-salt-32ch}` — insecure default remains.

```yaml
NEXTAUTH_SECRET: ${NEXTAUTH_SECRET:-change-me-in-production-secret-32chars}
ENCRYPTION_KEY: ${ENCRYPTION_KEY:-d1e0f601167778e7c4f178d0fca9ebb56cadb460241bdd8c548428867563d620}
```

The `ENCRYPTION_KEY` is a fixed hex string now publicly committed. Any Langfuse data encrypted with this key is trivially compromised if the database is ever exposed. Running `docker compose up` without `.env` uses these public defaults.

**Proposed fix area:** Remove the default values. Let the service fail loudly if the env var is missing, forcing developers to set real secrets. Add instructions to `README` and `.env.example`.

---

## Group H — Embeddings

**Files:** `backend/ingestion/embed.py`

---

### H-1 [L-2] Embedding singleton ignores device changes after first initialization

**File:** `backend/ingestion/embed.py:25`
**Verified:** ✅ Confirmed
**Fix Status:** ✅ FIXED — cache key now includes device: `_embed_device != device`; separate `_embed_device` global tracks active device; model re-initialized if device changes.

```python
device = DEFAULT_DEVICE or ("mps" if torch.backends.mps.is_available() else "cpu")
if _embed_model is None or _embed_model.model_name != model_name:
    _embed_model = HuggingFaceEmbedding(model_name=model_name, device=device)
```

`device` is computed on every call but applied only when the model is (re-)initialized. If the singleton already exists with the same model name but the `EMBEDDING_DEVICE` env var changes, or if MPS becomes unavailable mid-process, the device change is silently ignored. The model keeps running on the original device.

**Proposed fix area:** Include `device` in the cache key check: `if _embed_model is None or _embed_model.model_name != model_name or _embed_model.device != device`.

---

## Summary Table

| Group | Issues | Scope |
|---|---|---|
| A — Ingestion Pipeline | A-1 (C-1), A-2 (C-3), A-3 (C-5), A-4 (M-5) | `ingestion/chunk.py`, `index.py`, `run_ingestion.py` |
| B — Retrieval Layer | B-1 (H-1), B-2 (H-4), B-3 (M-3), B-4 **[NEW]** | `retrieval/hybrid.py`, `pipeline.py` |
| C — Observability | C-1 (C-4), C-2 (H-6), C-3 (M-1), C-4 (L-3) | `observability/tracing.py`, `generation/generate.py`, `retrieval/pipeline.py` |
| D — Generation & LLM | D-1 (H-5), D-2 (XC-2) | `generation/generate.py` |
| E — API Layer | E-1 (C-2), E-2 (H-2), E-3 (H-3), E-4 (M-2), E-5 (M-7), E-6 (XC-1) | `api/main.py` |
| F — Frontend | F-1 (H-7), F-2 (M-4), F-3 (M-6), F-4 (L-4), F-5 (L-5) | `ChatPanel.tsx`, `MessageBubble.tsx`, `useAppStore.ts`, `usePolling.ts` |
| G — Infrastructure | G-1 (H-8), G-2 (L-1) | `docker-compose.yml` |
| H — Embeddings | H-1 (L-2) | `ingestion/embed.py` |
| **Total** | **28** (27 original confirmed + 1 new; 1 original invalidated) | |

---

## Fix Status Summary (Mar 11, 2026)

All 27 confirmed issues (+ 1 new) verified against source code.

| Status | Count | Issues |
|--------|-------|--------|
| ✅ FIXED | 24 | A-1, A-2, A-3, A-4, B-1, B-2, B-3, B-4, C-1, C-2, C-3, C-4, D-2, E-1, E-2, E-4, E-5, E-6, F-1, F-2, F-3, F-4, F-5, G-1, H-1 |
| ⚠️ PARTIALLY FIXED | 2 | D-1, G-2 |
| ❌ NOT FIXED | 1 | E-3 |
| **Total** | **27** | (C-6 previously invalidated) |

---

## Side Effects Introduced by Fixes

### A-3 UUID5 collision for non-paragraph nodes — data-loss regression

**Files:** `backend/ingestion/chunk.py:60`, `backend/ingestion/index.py:61-63`

`chunk_index` IS populated by `split_paragraph_nodes` for paragraph nodes (0-based sequential indices). However, **non-paragraph nodes** (tables, figures, etc.) use `meta.setdefault("chunk_index", 0)`, which assigns `0` only if the key is absent. Since `parse.py` never sets `chunk_index`, every non-paragraph node enters the function without it and exits with `chunk_index=0`.

If a page has **two or more tables or figures**, they all produce `filename:page_number:0` → identical UUID5 → each successive upsert silently overwrites the previous, losing all but the last non-paragraph element per page. This is a real **data-loss regression** for multi-table/figure pages.

**Recommended fix:** Assign `chunk_index` in `parse.py` for non-paragraph nodes, or use a secondary discriminator (e.g., `element_type`) in the UUID5 seed.

### D-1 ValueError vs HTTPException

`ValueError` raised instead of `HTTPException(status_code=502)`. FastAPI returns HTTP 500 for unhandled `ValueError`. Less precise than proposed, but prevents the original Pydantic `ValidationError` 500 crash.

---

## Suggested Fix Order

Priority is ordered by: "blocks end-to-end correctness first, then resilience, then polish."

1. **Group A** — Ingestion bugs corrupt stored data; all downstream quality depends on clean ingestion
2. **Group B** — Retrieval correctness (filter bug, resilience to bad pickle)
3. **Group C** — Observability should not crash the pipeline; LiteLLM callback scope
4. **Group D** — Generation null-safety and model ID correctness
5. **Group E** — API correctness (reasoning payload, progress meter, validation)
6. **Group F** — Frontend UX (error messages, markdown rendering, tab polling)
7. **Group G** — Infrastructure reliability (startup ordering, secrets hygiene)
8. **Group H** — Embeddings (minor correctness issue)
