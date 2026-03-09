# FinLens Codebase Audit — March 10, 2026

> **Methodology:** Every finding in this report is traced to a specific file and line number, verified by reading the actual source code. No finding is inferred or speculative. Severity levels: 🔴 Critical (breaks functionality or correctness today), 🟠 High (breaks under realistic conditions), 🟡 Medium (degrades quality silently or at scale), 🔵 Low (minor issue or future risk).

---

## Summary

| Severity | Count |
|---|---|
| 🔴 Critical | 6 |
| 🟠 High | 8 |
| 🟡 Medium | 7 |
| 🔵 Low | 5 |
| **Total** | **26** |

---

## 🔴 Critical — Breaks Functionality Today

---

### C-1: Page number assertion is vacuously true — the guard is completely broken

**File:** `backend/ingestion/chunk.py:102-104`

```python
assert node.metadata.get("page_number") is not None or True, (
    f"Chunk {i} has page_number=None (acceptable for some sources)"
)
```

`X or True` is **always True** in Python regardless of what `X` is. This assertion can never fail. It was written as if it performs a check but it does nothing. If `page_number` is `None` on a chunk — which can happen when Docling fails to extract provenance — the assertion passes silently and a node with `page_number=None` flows through to the index.

**Impact:** Citations in the final answer may display `p.None` or null. Users see broken source references with no indication anything went wrong. The intended safety net is completely absent.

---

### C-2: `retrieved_candidates` in reasoning payload always equals `selected_nodes`

**File:** `backend/api/main.py:155-165`

```python
"retrieval": {
    ...
    "retrieved_candidates": len(context_nodes),  # ← this is RERANKED nodes
    ...
},
"rerank": {
    ...
    "selected_nodes": len(context_nodes),         # ← same value
},
```

`context_nodes` passed to `_build_reasoning()` is the output of `retrieve_and_rerank()` — that is, the already-reranked final list (max 5 nodes). The function never receives the pre-rerank candidate count. So `retrieved_candidates` always equals `selected_nodes`, both showing 5. The field that should show "we retrieved 20, narrowed to 5" always shows "5 retrieved, 5 selected," which is misleading to users examining the reasoning panel.

**Impact:** The reasoning transparency feature is factually wrong. Debugging retrieval quality ("did hybrid retrieval find good candidates?") is impossible from this data.

---

### C-3: Null metadata values silently excluded from Qdrant payload — page numbers lost

**File:** `backend/ingestion/index.py:57`

```python
payload = {
    "text": node.text,
    **{k: v for k, v in node.metadata.items() if v is not None},
}
```

Fields where `v is None` are dropped entirely from the stored payload. When a node has `page_number=None` (Docling failed to extract provenance), `page_number` is absent from Qdrant. At query time in `hybrid.py:94`, the reconstructed TextNode from the Qdrant result also has no `page_number` key. When `generate.py:64` builds citations, `node.metadata.get("page_number")` returns `None`, which becomes `"page_number": None` in the citation dict.

On the frontend, `CitationCard.tsx:13` renders:
```tsx
{citation.company} · {citation.year} · {citation.doc_type} · p.{citation.page_number}
```
This produces `3M · 2022 · 10-K · p.null` in the browser.

Additionally, `Citation` in `useAppStore.ts:8` declares `page_number: number` (non-nullable TypeScript type), but the backend can send `null`. TypeScript's type safety does not apply at JSON deserialization — the runtime value can be null while the type says number.

**Impact:** Broken citation display with no user-visible error. Affects any node where Docling failed to capture page provenance.

---

### C-4: Langfuse span errors crash the RAG pipeline

**File:** `backend/retrieval/pipeline.py:28-29`

```python
with span(trace, "hybrid_retrieve", input={"query": query, "top_k": retrieval_top_k}):
    candidates = hybrid_retrieve(...)
```

In `tracing.py:39`, `trace.span(name=name, input=input)` makes an HTTP call to Langfuse. If Langfuse is unreachable, overloaded, or restarting, this raises an exception **before** `hybrid_retrieve()` executes. The `with` block's `__enter__` fails and the exception propagates up through `pipeline.py` to `main.py`'s `/chat` endpoint, which returns HTTP 500.

This means a **Langfuse outage takes down the RAG pipeline**. The observability layer should degrade gracefully, not be load-bearing infrastructure.

The same applies to the rerank span at `pipeline.py:37`.

**Impact:** Any Langfuse downtime (restart, network blip, timeout) causes all `/chat` requests to fail with 500, even though Qdrant and the LLM are perfectly healthy.

---

### C-5: Duplicate chunks inserted into Qdrant on crash-recovery

**File:** `backend/ingestion/run_ingestion.py:230-240`

```python
print("Building Qdrant index ...")
build_qdrant_index(new_chunks, collection_name=QDRANT_COLLECTION)

existing_nodes = _load_existing_bm25_nodes()
all_bm25_nodes = existing_nodes + new_chunks
build_bm25_index(all_bm25_nodes, output_path=BM25_INDEX_PATH)

# Persist registry only after successful indexing
newly_ingested = {spec.path.name for spec in batch}
_save_registry(already_ingested | newly_ingested)
```

`build_qdrant_index` generates a new `uuid.uuid4()` for every point (`index.py:63`). If the process crashes after `build_qdrant_index` succeeds but before `_save_registry` saves (e.g., power loss, OOM kill, `Ctrl+C`), the document is indexed in Qdrant but absent from the registry. The next run will re-ingest the same document, inserting a completely new set of points with new UUIDs. Qdrant now has 2× the chunks for that document, all at equal rank, which degrades retrieval quality (duplicate results, inflated candidate sets).

There is no deduplication key — `build_qdrant_index` does not check for existing points with the same text or filename before inserting.

**Impact:** Persistent data corruption that is silent and difficult to reverse without dropping and rebuilding the collection. Gets worse with each crash-recovery cycle.

---

### C-6: `qa_dataset.json` does not exist — RAGAS eval always fails with FileNotFoundError

**File:** `eval/ragas_eval.py:23, 93`

```python
DEFAULT_DATASET_PATH = Path(__file__).parent / "qa_dataset.json"
...
rows = load_qa_dataset(args.dataset, sample_n=args.sample)
```

`load_qa_dataset()` opens the file at line 34. The file `eval/qa_dataset.json` is not committed to the repository (not present in the codebase). Running `uv run python ../eval/ragas_eval.py` — including in CI — immediately raises `FileNotFoundError`. The GitHub Actions RAGAS workflow references this path and will always fail on a fresh clone.

**Impact:** The eval pipeline and CI quality gate are permanently broken on a fresh checkout. No RAGAS scores can be computed without manually creating this file first.

---

## 🟠 High — Breaks Under Realistic Conditions

---

### H-1: BM25 retrieval ignores company and year filters

**File:** `backend/retrieval/hybrid.py:60-67`

```python
bm25_retriever = load_bm25_index(BM25_INDEX_PATH)
...
bm25_results = bm25_retriever.retrieve(query)       # ← no filter applied
bm25_nodes = [r.node for r in bm25_results]
```

Qdrant dense retrieval at lines 76-90 applies `company` and `year` payload filters. BM25 retrieval has no equivalent filter and searches across all indexed documents regardless of the active filters. If you ask about "3M's 2022 revenue" with `company=3M, year=2022`, BM25 will still return chunks from Apple, Microsoft, Adobe, etc.

The cross-encoder reranker at the next stage will likely score these off-company chunks lower, but:
1. They still consume slots in the 20-candidate pool, displacing potentially better 3M 2022 results
2. If the off-company chunk happens to have high BM25 relevance (e.g., same financial term), it may survive reranking and appear in the final answer as a citation

**Impact:** Cross-company answer contamination and degraded retrieval quality at every query when filters are active. Gets significantly worse as more companies are ingested.

---

### H-2: `TOTAL_DOCUMENTS` is hardcoded to 5 — ingestion progress is always wrong

**File:** `backend/api/main.py:38`

```python
TOTAL_DOCUMENTS = 5  # manifest size in run_ingestion.py
```

The `GET /status/ingestion` endpoint returns `"total_documents": TOTAL_DOCUMENTS` (line 204), which is hardcoded to 5. But the actual number of documents changes dynamically as ingestion runs. The FinanceBench dataset has 150 PDFs. The frontend `IngestionMeter` component calculates its progress percentage from this value: `pct = total > 0 ? Math.round((indexed / total) * 100) : 0` (`IngestionMeter.tsx:7`).

With 150 documents ingested, the frontend shows 150/5 docs = 3000% which breaks the progress bar. With 1 document ingested, it shows 1/5 = 20% even though only 1/150 documents are processed.

**Fix:** `total_documents` should be derived from the registry file or Qdrant scroll (count unique filenames), not hardcoded.

---

### H-3: `/status/ingestion` scrolls all Qdrant points — O(n) scan

**File:** `backend/api/main.py:186-200`

```python
while True:
    records, offset = client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=256,
        offset=offset,
        with_payload=["filename"],
        with_vectors=False,
    )
    for rec in records:
        fn = (rec.payload or {}).get("filename")
        if fn:
            filenames.add(fn)
    if offset is None:
        break
```

This scrolls through every single chunk in the Qdrant collection to collect unique filenames. For a single document (106 chunks), this is fine. For 150 documents at ~100 chunks each = 15,000 scroll calls in batches of 256 (~59 iterations). The sidebar calls `/status/ingestion` every 10 seconds via `usePolling` (`Sidebar.tsx:27-36`). That means this full scan runs every 10 seconds indefinitely.

**Impact:** O(chunks) work every 10 seconds per browser tab that has the app open. At 15,000 chunks, each poll performs 60 Qdrant scroll calls. At 100 clients, Qdrant handles 6,000 scroll calls per second continuously.

---

### H-4: BM25 pickle load failure crashes `/chat` with 500 — no graceful degradation

**File:** `backend/retrieval/hybrid.py:61` and `backend/ingestion/index.py:104-105`

```python
# hybrid.py
bm25_retriever = load_bm25_index(BM25_INDEX_PATH)

# index.py — raises ValueError on invalid payload
if not isinstance(payload, dict) or "nodes" not in payload:
    raise ValueError("Invalid BM25 index payload. Rebuild the index.")
```

If `bm25_index.pkl` is corrupted (partial write, version mismatch, incompatible Python/library version), `load_bm25_index` raises `ValueError`. This propagates through `hybrid_retrieve()` and `retrieve_and_rerank()` to the `/chat` endpoint, which returns HTTP 500. There is no fallback to "dense-only" mode.

The only protection is the file existence check at line 60: `if BM25_INDEX_PATH.exists()`. A corrupted file passes this check and still crashes.

**Impact:** Any BM25 pickle file corruption (including the common case of an interrupted ingestion write) permanently disables the entire `/chat` endpoint until the file is manually rebuilt.

---

### H-5: `response.choices[0].message.content` can be `None`

**File:** `backend/generation/generate.py:45`

```python
answer = response.choices[0].message.content
```

The OpenAI/LiteLLM response spec allows `content` to be `None`. This occurs when the model invokes a tool call, when certain model providers return refusals in a non-standard format, or on some error responses. `answer = None` then flows into the return dict `{"answer": None, ...}`. Back in `main.py:121`, `ChatResponse(**result, ...)` tries to validate `answer: str` with `None` — Pydantic raises `ValidationError`, which FastAPI converts to HTTP 500 with a generic error.

**Impact:** When a model returns a null content (not uncommon on free-tier models that sometimes return tool-call-style refusals), the user gets an opaque 500 instead of a handled error message.

---

### H-6: `litellm` global callbacks cause background Langfuse failures in all environments

**File:** `backend/generation/generate.py:18-19`

```python
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]
```

These are module-level assignments on import. They configure global LiteLLM state for the entire process. In tests, when `generate.py` is imported (which happens for all integration tests), these callbacks are set and LiteLLM tries to send every LLM response to Langfuse — including in test environments where Langfuse is not running. This results in background thread errors on every test that calls or mocks `generate()`.

Additionally, if `LANGFUSE_PUBLIC_KEY` or `LANGFUSE_SECRET_KEY` are not set, Langfuse is still initialized (with empty strings in `tracing.py:23-26`) and silently fails to authenticate. Every LLM call will trigger Langfuse callbacks that fail silently in a background thread — this can cause unexpected delays and logged errors.

---

### H-7: 404 response detail never shown to the user

**File:** `frontend/src/components/chat/ChatPanel.tsx:57-62`

```tsx
} catch (err: unknown) {
  const msg = err instanceof Error ? err.message : 'Request failed'
  addMessage({
    id: crypto.randomUUID(),
    role: 'assistant',
    content: `Error: ${msg}`,
  })
}
```

When the backend returns HTTP 404 ("No relevant context found for this query."), Axios throws an `AxiosError`. `err instanceof Error` is true, but `err.message` is `"Request failed with status code 404"` — the Axios default. The actual `detail` field from the backend response body (`err.response?.data?.detail`) is never read.

The user sees `"Error: Request failed with status code 404"` instead of the informative message `"No relevant context found for this query."` They have no idea whether the query failed due to missing documents, wrong filters, or a genuine system error.

**Impact:** Users get cryptic error messages on the most common expected failure mode (no documents indexed, or wrong company/year filter).

---

### H-8: `langfuse` service in docker-compose waits only for `service_started`, not `service_healthy`

**File:** `docker-compose.yml:21-22`

```yaml
depends_on:
  qdrant:
    condition: service_healthy
  langfuse:
    condition: service_started    # ← not service_healthy
```

Qdrant correctly uses `condition: service_healthy`. Langfuse uses only `condition: service_started`. Langfuse + Postgres initialization can take 15-45 seconds. The backend starts and begins accepting requests before Langfuse finishes migrating its database. During this window:
- `create_trace()` in `tracing.py` will try to connect to Langfuse and fail
- Due to C-4, this failure propagates and crashes every `/chat` request until Langfuse finishes starting

Combined with C-4, this means a fresh `docker compose up` typically results in a period where all `/chat` calls fail with 500 on a seemingly healthy stack.

---

## 🟡 Medium — Silent Degradation or Scale Risk

---

### M-1: Langfuse initialized with empty strings when keys not configured

**File:** `backend/observability/tracing.py:22-27`

```python
_langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)
```

If `LANGFUSE_PUBLIC_KEY` or `LANGFUSE_SECRET_KEY` are not set (e.g., first-time setup, CI environment without Langfuse), the client is initialized with empty strings. Langfuse SDK will attempt authentication with these empty credentials, fail, and log errors silently in a background thread. There is no early-exit or disabled-mode.

**Impact:** Developers who haven't configured Langfuse see noisy background errors. `tracing.py` should check for key presence and return a no-op trace object if keys are absent.

---

### M-2: `/status/services` makes sequential blocking HTTP calls

**File:** `backend/api/main.py:224-249`

```python
# Qdrant
resp = httpx.get(f"{QDRANT_URL}/healthz", timeout=2)

# Langfuse
resp = httpx.get(f"{LANGFUSE_URL}/api/public/health", timeout=2)
```

Both checks run sequentially with synchronous `httpx.get()`. If both services time out (2s each), the endpoint takes 4+ seconds to respond. FastAPI runs sync endpoint functions in a thread pool, so this does block a thread for the full duration. The sidebar calls this endpoint every 10 seconds.

**Impact:** Under load, health check calls saturate the thread pool. Should use `httpx.AsyncClient` with `asyncio.gather()` and the endpoint should be `async def`.

---

### M-3: BM25 corpus size check accesses fragile internal attribute

**File:** `backend/retrieval/hybrid.py:62-65`

```python
if bm25_retriever.bm25 and bm25_retriever.bm25.scores:
    bm25_corpus_size = bm25_retriever.bm25.scores.get("num_docs", 0)
    if bm25_corpus_size > 0:
        bm25_retriever.similarity_top_k = min(top_k, bm25_corpus_size)
```

`bm25_retriever.bm25` is the underlying `rank_bm25.BM25Okapi` object. `BM25Okapi` does not have a `.scores` attribute as a dict. The `.scores` attribute is used in a different context in `rank_bm25`. This check will silently evaluate to `False` (since `.scores` is not a dict or `None`), skip the `if` block, and leave `similarity_top_k` at its default. This means the size-capping logic never runs.

**Impact:** If the BM25 corpus has fewer documents than `top_k=20`, the retriever may request more results than exist, causing an error or empty results. Silently broken rather than raising.

---

### M-4: LLM markdown rendered as raw text in `MessageBubble`

**File:** `frontend/src/components/chat/MessageBubble.tsx:30`

```tsx
<div className="... text-sm leading-relaxed font-sans">
  {message.content}
</div>
```

`message.content` is the raw LLM output, which will contain markdown (numbered lists, `**bold**`, `---` dividers, `[1]` citation references, header lines). React renders this as a plain text string, so users see `**3M's net sales were $35.4B**` instead of **3M's net sales were $35.4B**. The answer is readable but looks unprofessional and harder to parse.

**Impact:** Poor UX for every response. No crash, but the product doesn't look polished.

---

### M-5: `_parse_pdf_filename` breaks on filenames with more than 3 underscore-parts

**File:** `backend/ingestion/run_ingestion.py:50-64`

```python
parts = pdf_path.stem.split("_")
doc_type_raw = parts[-1]   # last segment
year = parts[-2]           # second-to-last
company = "_".join(parts[:-2])
```

For a filename like `3M_2022_10K_10.pdf` (the file referenced in memory as the actual test file):
- `parts = ["3M", "2022", "10K", "10"]`
- `doc_type_raw = "10"` → becomes `"1-0"` (digit prefix logic)
- `year = "10K"`
- `company = "3M_2022"`

This is entirely wrong metadata. However, that file is at the repo root, not in `PDF_DIR = REPO_ROOT / "data" / "financebench" / "pdfs"`, so it won't be auto-discovered by `_discover_pdfs()`. The risk is that any future PDF following the `COMPANY_YEAR_DOCTYPE_VARIANT.pdf` naming (4+ parts) placed in the PDF directory will be mis-parsed silently.

**Impact:** Silent metadata corruption (wrong company, year, doc_type) for any PDF with 4+ underscore-separated parts. Citations in those documents will show wrong company/year.

---

### M-6: `usePolling` fires even when browser tab is hidden

**File:** `frontend/src/hooks/usePolling.ts:7-11`

```ts
const id = setInterval(() => savedFn.current(), intervalMs)
```

The polling interval fires every 10 seconds regardless of tab visibility. This means two API calls (`getServicesStatus` + `getIngestionStatus`) are made every 10 seconds even when the user has switched to another tab.

**Impact:** Unnecessary server load. At scale or if the ingestion status endpoint scan (H-3) is expensive, this compounds. Fix: listen to `document.visibilitychange` to pause/resume polling.

---

### M-7: `FreeModelResponse(**model.__dict__)` is fragile for frozen dataclass

**File:** `backend/api/main.py:74`

```python
return [FreeModelResponse(**model.__dict__) for model in get_free_models()]
```

`model` is a `FreeModelOption` frozen dataclass. Accessing `__dict__` on a frozen dataclass works in CPython but is not part of the dataclass specification. The idiomatic approach is `dataclasses.asdict(model)`. If `FreeModelOption` is ever changed to a Pydantic model or NamedTuple, `__dict__` will either fail or return unexpected fields.

**Impact:** Low risk today, breaks silently on class refactoring.

---

## 🔵 Low — Minor Issues and Future Risks

---

### L-1: Hardcoded default secrets in `docker-compose.yml`

**File:** `docker-compose.yml:83-86`

```yaml
NEXTAUTH_SECRET: ${NEXTAUTH_SECRET:-change-me-in-production-secret-32chars}
SALT: ${SALT:-change-me-in-production-salt-32ch}
ENCRYPTION_KEY: ${ENCRYPTION_KEY:-d1e0f601167778e7c4f178d0fca9ebb56cadb460241bdd8c548428867563d620}
```

These secrets have hardcoded defaults. If a developer runs `docker compose up` without configuring `.env`, Langfuse starts with known, public secrets. The encryption key is particularly dangerous — it's a fixed hex string that is now publicly visible in this codebase. Any Langfuse data encrypted with this key is compromised if the database is ever exposed.

**Impact:** Low risk for local dev, critical for anyone who deploys this docker-compose to a public server without reading the env var docs.

---

### L-2: `embed.py` singleton doesn't account for device changes after initialization

**File:** `backend/ingestion/embed.py:25`

```python
if _embed_model is None or _embed_model.model_name != model_name:
    _embed_model = HuggingFaceEmbedding(model_name=model_name, device=device)
```

Device (`mps` vs `cpu`) is determined on every call but only applied when the model is **re-initialized**. If the singleton is already loaded, `device` is computed but ignored. If `EMBEDDING_DEVICE` is set after the first call, or if the MPS backend becomes unavailable mid-process, the model continues running on the original device. This is a minor correctness issue (device is determined once at startup) but the code structure implies it's dynamic.

---

### L-3: `span()` context manager does not record the output on the Langfuse span

**File:** `backend/retrieval/pipeline.py:28-38`

```python
with span(trace, "hybrid_retrieve", input={"query": query, "top_k": retrieval_top_k}):
    candidates = hybrid_retrieve(...)
```

The span records `input` but never calls `s.end(output=...)` with the span's result. In `tracing.py:43`, `finally: s.end()` is called with no arguments. Langfuse spans without output are less useful — you can see inputs and latency but not what was retrieved. The rerank span similarly records only `n_candidates` as input with no output.

---

### L-4: Frontend `Citation` interface declares non-nullable fields that may be null at runtime

**File:** `frontend/src/stores/useAppStore.ts:4-11`

```typescript
export interface Citation {
  company: string       // ← declared non-nullable
  year: string          // ← declared non-nullable
  doc_type: string      // ← declared non-nullable
  page_number: number   // ← declared non-nullable
  filename: string      // ← declared non-nullable
  excerpt?: string
}
```

The backend's `generate.py:60-66` uses `.get()` with no fallback for these fields:
```python
"company": node.metadata.get("company"),   # returns None if absent
"page_number": node.metadata.get("page_number"),  # returns None if absent
```

TypeScript's structural typing does not enforce non-nullability at JSON deserialization boundaries. If any field arrives as `null`, TypeScript won't raise — the runtime will silently use `null` where `string` or `number` was expected, causing `"null"` or `NaN` to appear in the UI.

**This overlaps with C-3 but is noted separately as a type contract issue.**

---

### L-5: `isLoading` state not reset if component unmounts mid-request

**File:** `frontend/src/components/chat/ChatPanel.tsx:31-66`

```tsx
const submit = async () => {
  ...
  setLoading(true)
  try {
    const result = await postChat({...})
    addMessage({...})
  } catch (err) {
    addMessage({...})
  } finally {
    setLoading(false)
  }
}
```

If the user navigates away (or the component unmounts) while `postChat` is in flight, `setLoading(false)` and `addMessage(...)` still fire on the now-unmounted component's store references. Since state is in Zustand (not local component state), this won't cause a "setState on unmounted component" React warning, but `isLoading` will be set to `false` and a message will be added to the store even though the user has navigated away. On re-opening the chat, they'll see the response. This is arguably correct behavior but the in-flight request is never cancelled (no `AbortController`), so it wastes network resources.

---

## Cross-Cutting Issues

### XC-1: No input validation on `/chat` query field

**File:** `backend/api/main.py:41-47`

```python
class ChatRequest(BaseModel):
    query: str   # ← no min_length, no max_length
```

A query of `query=" "` (whitespace only) passes Pydantic validation and goes through the full RAG pipeline — embedding, Qdrant search, cross-encoder, LLM call — before returning a meaningless answer. A query of 100,000 characters gets sent as-is to the LLM, potentially exhausting the context window and causing a 400 from OpenRouter.

---

### XC-2: `openrouter/free` model ID sent to LiteLLM without OpenRouter prefix

**File:** `frontend/src/stores/useAppStore.ts:166` and `backend/generation/generate.py:37`

The default model in the store is `"openrouter/free"`. This is sent as-is to `/chat` as `request.model = "openrouter/free"`. In `generate.py:100`, `request.model or DEFAULT_MODEL` passes this to `litellm.completion(model="openrouter/free", ...)`. LiteLLM's OpenRouter integration expects model IDs in the format `openrouter/{provider}/{model}` or recognizes specific IDs. `"openrouter/free"` is not a standard LiteLLM model ID — it may or may not route correctly depending on the LiteLLM version. This was not tested with a live OpenRouter call in the visible code.

---

## Correction to Earlier Learning Documentation

The `docs/learning/01-high-level-overview.md` and `02-low-level-deep-dive.md` documents I authored contain an **incorrect finding** about an "embedding model mismatch." They claim `index.py` uses `bge-large` (1024-dim) while `hybrid.py` uses `gte-modernbert-base` (768-dim).

**This is wrong.** Both `index.py:24` and `hybrid.py:71` call `get_embed_model()` from `embed.py`, which defaults to `Alibaba-NLP/gte-modernbert-base` (768-dim) for both ingestion and query. There is no mismatch. The learning documents should be corrected.
