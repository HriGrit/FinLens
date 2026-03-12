# FinLens Codebase Audit — 12 March 2026

**Scope:** Full codebase audit covering ingestion pipeline, retrieval, generation, API, tests, eval, and config.
**Purpose:** Identify bugs, incorrect logic, and test-passing workarounds. No code solutions — discovery only.
**Output target:** `docs/audit/2026-03-12_codebase_audit.md`

Each issue carries a priority label: **P0** (silent data loss / wrong results), **P1** (runtime crash or behavioral bug), **P2** (reliability / drift / maintainability).

---

## 1. Qdrant / Vector Store

### [P0] Upsert Result Never Checked
**File:** `backend/ingestion/index.py:83`

`client.upsert(collection_name=..., points=batch)` returns an `UpdateResult` with a status field. The return value is discarded entirely. If Qdrant rejects points (dimension mismatch, payload size limit, collection misconfiguration), the upsert silently fails and the ingestion run reports success. The vectors are never written to the store, but no exception is raised.

**Root cause:** Return value of upsert is not captured or inspected.

---

### [P0] Embedding Batch Failure Is Undetected
**File:** `backend/ingestion/index.py:59`

`embed_model.get_text_embedding_batch(texts, show_progress=True)` is called without error handling. If the model returns a partial result or raises mid-batch (rate limit, OOM, model not loaded), the subsequent `zip(nodes, embeddings)` at line 61 silently truncates to the shorter list. Nodes with no corresponding embedding are dropped from the index with no log or exception. The resulting Qdrant collection is incomplete.

**Root cause:** No length validation between input nodes and returned embeddings before constructing points.

---

### [P1] Point ID UUID Collision on None Metadata Fields
**File:** `backend/ingestion/index.py:15–23`

`_make_point_id()` generates a UUID5 from a concatenated key of `filename`, `page_number`, `element_type`, `chunk_index`. When any field is `None`, Python's `str(None)` produces the literal string `"None"`. Two nodes from different documents that both have `page_number=None` and identical `element_type` will produce the same UUID if their other fields match.

Since Qdrant upsert semantics are "insert-or-overwrite", the second node silently replaces the first. No deduplication check exists. This is not a theoretical edge case — `page_number=None` is an accepted value (see §2, `page_number=None` passes assertion).

**Root cause:** No sentinel or error for None-valued fields before UUID construction.

---

### [P2] Qdrant Collection Names Drift Between Test, Dev, and Validation
**Files:** `backend/tests/conftest.py:68`, `.env.example:11`, `tests/validate_ingestion.py:16`

Three different collection names are in use:
- `conftest.py`: `finlens_chunks_test`
- `.env.example`: `finlens_chunks_dev`
- `validate_ingestion.py`: `finlens_chunks_dev`

Unit tests that mock Qdrant use `finlens_chunks_test`, but validation scripts hit the live Qdrant using `finlens_chunks_dev`. The test environment and the "dev" validation environment are invisibly disconnected.

**Root cause:** No single source of truth for collection name configuration.

---

## 2. Ingestion Pipeline

### [P1] Docling Parse Failure Not Detected at Conversion Time
**File:** `backend/ingestion/run_ingestion.py:271–278`

`converter.convert(str(spec.path)).document` is called with no check on whether conversion succeeded. If Docling encounters a corrupt PDF, an encrypted file, or a layout it cannot parse, `convert()` may return a result with an empty or partial `.document`. The error only surfaces much later during node extraction in `parse.py`, far from the actual failure point. There is no check on `result.status`, no try/except, and no validation that the resulting document contains any items.

**Root cause:** Docling's `ConversionResult` has a status field that is never read.

---

### [P1] `page_number=None` Passes Required Metadata Assertion
**File:** `backend/ingestion/parse.py:112–121`

`assert_node_metadata()` uses a set-difference check (`REQUIRED_METADATA_FIELDS - node.metadata.keys()`) to confirm all required keys exist. This validates key presence, not value presence. A node with `{"page_number": None, ...}` passes the assertion. This `None` propagates into Qdrant payloads, BM25 nodes, citation assembly, and the API response — silently producing citations with `page_number: null`. It also feeds the UUID collision issue above.

**Root cause:** Assertion checks key existence, not value truthiness.

---

### [P1] SemanticSplitter `_source_idx` Field May Be Silently Dropped
**File:** `backend/ingestion/chunk.py:46–53`

Before passing paragraph nodes to `SemanticSplitterNodeParser`, the code injects `_source_idx` into metadata (line 46) to track which original node each chunk came from. After splitting, line 52 reads it back with a default of `0`: `src_idx = chunk_node.metadata.get("_source_idx", 0)`.

If the splitter drops `_source_idx` (not guaranteed to be preserved), all chunks are silently attributed to the first source node. The grouping logic then collapses all chunks into a single parent, corrupting the chunk-to-source mapping for every subsequent node. No assertion or log confirms the field survived.

**Root cause:** Implicit assumption that SemanticSplitterNodeParser preserves arbitrary metadata keys — not documented or tested.

---

### [P1] Chunk Metadata Overwritten During Semantic Splitting
**File:** `backend/ingestion/chunk.py:76–79`

After splitting, the code merges chunk metadata into the parent node's metadata:
```python
meta = dict(node.metadata)
meta.update({k: v for k, v in chunk_node.metadata.items() if not k.startswith("_")})
```
The `update()` call means any metadata key produced by the SemanticSplitter (e.g., if it sets its own `page_number` or `element_type`) overwrites the original ingestion-time values, which were populated from the authoritative Docling parse. The original values are lost with no log.

**Root cause:** Merge strategy unconditionally favors the splitter's output over the original node metadata.

---

### [P1] BM25 Pickle Reconstruction Is Structurally Fragile
**File:** `backend/ingestion/index.py:88–107, 120–125`

`build_bm25_index()` manually extracts attributes (`similarity_top_k`, `skip_stemming`, `token_pattern`) and pickles them as a plain dict. `load_bm25_index()` reconstructs the retriever via `BM25Retriever.from_defaults()` using those extracted attributes. The internal BM25 scoring state (fitted vocabulary, IDF weights, document frequencies) is not persisted — it is recomputed from the pickled `nodes` on every load.

If `llama-index-retrievers-bm25` changes `from_defaults()` parameter defaults between versions, the reconstructed retriever silently behaves differently. No checksum, version tag, or round-trip test verifies that results before and after pickling are identical.

**Root cause:** Manual attribute extraction instead of direct object pickling.

---

### [P2] Manifest/Artifact State Can Silently Desync
**File:** `backend/ingestion/run_ingestion.py:185–220`

`_load_chunk_artifacts_for_successful_docs()` loads chunk artifacts based on manifest status (`status == "success"`). If the artifact file is missing or corrupt, the code silently falls back to an empty list (lines 210–213) and continues as if no chunks exist for that document. The manifest still records the document as successfully ingested. The downstream BM25 and Qdrant indexes are therefore missing data, with no warning to the operator.

**Root cause:** Artifact existence is assumed from manifest status without independent verification.

---

### [P2] `doc_type` Filename Parsing Has an Off-By-One Edge Case
**File:** `backend/ingestion/discovery.py:33–36`

The logic to normalize `"10K"` → `"10-K"` checks `doc_type_raw[:-1].isdigit()`. For a filename containing `"123K"`, this produces `"12-3K"` instead of `"123-K"`. While current documents only use `"10K"` and `"8K"`, any expansion to other SEC form types (e.g., `"20F"`, `"40F"`) would silently produce malformed `doc_type` values.

**Root cause:** Regex would be more appropriate than manual string slicing for this pattern.

---

## 3. Retrieval — Hybrid Search & Reranking

### [P0] RRF Fusion Correctness Depends on Implicit Processing Order
**File:** `backend/retrieval/hybrid.py:22–41`

`_fuse_results()` deduplicates by text key and unconditionally overwrites the node reference (`nodes_by_key[key]`) on each occurrence — last write wins. When the same text appears in both BM25 and dense results, the final node stored is whichever source was processed last.

The code works today only because BM25 hits are iterated first (lines 30–33) and dense hits second (lines 35–38), so dense metadata always survives. There is no comment, assertion, or documented contract enforcing this order. If a refactor swaps the loops, dense metadata is silently lost on all deduplicated nodes — invisible to all existing tests.

The test `test_fuse_results_deduplicates_by_text_and_prefers_dense_metadata_when_duplicate` verifies the desired outcome but does not enforce the order dependency.

**Root cause:** No explicit merge strategy — just last-write-wins via dict assignment.

---

### [P2] `retrieve_and_rerank` Returns a Tuple, Forcing All Callers to Unpack
**File:** `backend/retrieval/pipeline.py:24, 40–41`

The function signature is `-> tuple[list[TextNode], int]`. The second element is the pre-rerank candidate count, used only for tracing. Every caller must unpack the tuple even if the count is irrelevant. The API endpoint discards the count; `ragas_eval.py` could break silently if written expecting a plain list.

The function also has three return sites (`([], 0)`, `(results, len(candidates))`, `(rerank(...), len(candidates))`) with no inline documentation of the tuple contract.

**Root cause:** Tracing side-channel data was embedded in the public return type rather than passed through the trace object.

---

## 4. Generation & LLM

### [P1] LiteLLM Response Structure Not Validated Before Attribute Access
**File:** `backend/generation/generate.py:48–62`

The code accesses nested response fields without structural guards:
- Line 48: `response.choices[0].message.content` — no check that `choices` is non-empty
- Line 52: `response.usage` — no check that the usage object exists
- Lines 59–62: `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens` — no check these attributes exist

If LiteLLM returns a response with an empty `choices` list (can occur on certain error states), line 48 raises an unguarded `IndexError`. The API caller catches `ValueError` and returns HTTP 502, but `IndexError` propagates as an unhandled 500. The single check at line 49 (`if answer is None`) only covers the value, not the structure.

**Root cause:** Structural validation absent; only null-value check exists.

---

### [P2] Cost Calculation Silently Falls Back to `0.0` on Failure
**File:** `backend/generation/generate.py:53–56`

```python
try:
    cost = litellm.completion_cost(completion_response=response)
except Exception:
    cost = 0.0
```

`0.0` is a semantically valid cost value, indistinguishable from a genuinely zero-cost call. If `litellm.completion_cost` fails (unknown model, changed API), the response reports `"cost_usd": 0.0` rather than `null`. Downstream analytics will under-report costs with no indication that the calculation failed.

**Root cause:** Bare `except Exception` discards the failure signal and substitutes a misleading value.

---

## 5. Testing Infrastructure

### [P0] Global Test Fixture Mocks Wrong Embedding Dimension (1024 vs 768)
**Files:** `backend/tests/conftest.py:91–99`, `backend/tests/conftest.py:17`

The autouse fixture `_patch_embedding_model` replaces `get_embed_model` globally for every test. The fake model returns a constant **1024-dimensional** zero vector (line 17). The real production model (`Alibaba-NLP/gte-modernbert-base`) produces **768-dimensional** embeddings.

Consequences:
- Tests that build a Qdrant index through the mock validate against 1024-dim vectors — a collection that cannot exist in production (which uses 768-dim).
- `validate_qdrant.py:25` reads the dimension dynamically from the model; in the test context the fixture would return 1024, yielding a different collection size than production.
- Any integration or live test using this fixture silently exercises behavior incompatible with the real Qdrant collection.

**Root cause:** Fake model dimension is hard-coded to 1024 with no connection to the actual model's output size.

---

### [P2] `validate_ingestion.py` Accesses Private BM25 Attribute
**File:** `tests/validate_ingestion.py:65`

```python
len(bm25.index._index) if hasattr(bm25, 'index') else 'n/a'
```

`._index` is a private internal attribute. The `hasattr` check guards against `.index` not existing, but not against `._index` not existing on the `.index` object. If `llama-index-retrievers-bm25` changes its internal structure, this line raises `AttributeError` at validation time for a cosmetic reporting reason unrelated to actual index health.

**Root cause:** Using private implementation details instead of the public retriever API.

---

## 6. Evaluation (RAGAS)

### [P2] No Error Handling for Failed Queries
**File:** `eval/ragas_eval.py:40–64`

`run_pipeline_on_dataset()` iterates over rows and calls `retrieve_and_rerank()` and `generate()` in a bare loop with no try/except. A single API rate limit (OpenRouter 429, which has occurred per project history), network timeout, or LLM error aborts the entire eval run. Partial results are not saved. RAGAS thresholds in `thresholds.yaml` have no data to compare against, and the CI check cannot run.

**Root cause:** No per-query error isolation or partial result persistence.

---

### [P2] Evaluation Uses Only 3 of Available Metrics
**File:** `eval/ragas_eval.py:104`

The eval pipeline runs only `faithfulness`, `context_recall`, and `answer_relevancy`. Absent metrics:
- `answer_correctness` / `answer_similarity` — whether the generated answer matches ground truth
- `context_precision` — whether retrieved chunks are relevant, not just recalled

A pipeline can score above all three thresholds while still generating factually incorrect answers if the LLM "faithfully" repeats misleading context. Without correctness metrics, this failure mode is undetectable by the eval.

**Root cause:** Incomplete metric selection at initial eval setup.

---

## 7. Configuration & Infrastructure

### [P2] `docs/codespaces.md` References Deleted File `ingestion_registry.json`
**File:** `docs/codespaces.md:102, 153, 160, 278`

The ingestion system migrated from `data/ingestion_registry.json` (a flat filenames list) to `data/ingestion_manifest.json` (per-document status records with chunk counts and timestamps). The code reflects this migration, but `docs/codespaces.md` still describes the old registry file in four places, instructing developers to inspect a file that no longer exists.

**Root cause:** Documentation not updated after the manifest migration.

---

### [P2] Docker Healthchecks Are Fragile
**File:** `docker-compose.yml:26, 73, 94`

Three issues:
1. **Backend** (line 26): Uses an inline Python heredoc calling `urllib.request.urlopen`. Requires Python in PATH inside the container; 2-second timeout is too short for cold starts; no error handling inside the heredoc.
2. **Langfuse** (line 73): Uses `hostname -i | awk '{print $1}'` to resolve the container's own IP. `hostname -i` can return multiple IPs; awk picks the first, which may not be the correct interface. Using `127.0.0.1` directly would be unambiguous.
3. **Postgres** (line 94): `pg_isready -U ${POSTGRES_USER:-langfuse}` — variable expansion occurs at docker-compose parse time. If the `.env` file is not loaded at invocation, the fallback `langfuse` may not match the actual Postgres user.

**Root cause:** Healthchecks written with unnecessary complexity instead of standard `curl` or native container health endpoints.

---

## Summary Table

| ID | Priority | File | Issue |
|----|----------|------|-------|
| Q1 | P0 | `ingestion/index.py:83` | Upsert result never checked |
| Q2 | P0 | `ingestion/index.py:59` | Embedding batch failure undetected — zip truncates |
| R1 | P0 | `retrieval/hybrid.py:22–41` | RRF fusion correct only by accident (loop order) |
| T1 | P0 | `tests/conftest.py:17` | Fixture embeds 1024-dim vectors; production uses 768 |
| Q3 | P1 | `ingestion/index.py:15–23` | UUID collision when metadata fields are `None` |
| I1 | P1 | `ingestion/run_ingestion.py:271–278` | Docling parse failure not detected at conversion time |
| I2 | P1 | `ingestion/parse.py:112–121` | `page_number=None` passes required metadata assertion |
| I3 | P1 | `ingestion/chunk.py:46–53` | `_source_idx` may be silently dropped by SemanticSplitter |
| I4 | P1 | `ingestion/chunk.py:76–79` | Chunk metadata overwritten with splitter output |
| I5 | P1 | `ingestion/index.py:88–107` | BM25 pickle reconstruction fragile across versions |
| G1 | P1 | `generation/generate.py:48–62` | LiteLLM response structure not validated before access |
| Q4 | P2 | `tests/conftest.py:68` | Qdrant collection names drift between test/dev/validation |
| I6 | P2 | `ingestion/run_ingestion.py:185–220` | Manifest/artifact state can silently desync |
| I7 | P2 | `ingestion/discovery.py:33–36` | `doc_type` parsing off-by-one for multi-digit form types |
| R2 | P2 | `retrieval/pipeline.py:24, 40–41` | `retrieve_and_rerank` tuple return pollutes public API |
| G2 | P2 | `generation/generate.py:53–56` | Cost fallback to `0.0` masks calculation failures |
| T2 | P2 | `tests/validate_ingestion.py:65` | Private BM25 attribute access in validation script |
| E1 | P2 | `eval/ragas_eval.py:40–64` | No per-query error handling; single failure aborts full run |
| E2 | P2 | `eval/ragas_eval.py:104` | Eval uses only 3 of 5 available RAGAS metrics |
| C1 | P2 | `docs/codespaces.md:102, 153, 160, 278` | References deleted `ingestion_registry.json` |
| C2 | P2 | `docker-compose.yml:26, 73, 94` | Docker healthchecks fragile (Python heredoc, hostname -i, env expansion) |

---

## Appendix: Items Investigated and Confirmed Correct

The following were audited and found to have no issues:
- Qdrant filter application with `FieldCondition` / `MatchValue` (`hybrid.py:87–102`)
- Qdrant payload extraction — text separated from metadata without mutation (`hybrid.py:105–109`)
- BM25 corpus-size cap on `similarity_top_k` (`hybrid.py:61–64`)
- BM25 missing-index fallback to dense-only (`hybrid.py:56–80`)
- Cross-encoder lazy-loading singleton pattern (`rerank.py:15–20`)
- LiteLLM callback registration in lifespan handler, not at import (`generate.py:19–22`, `main.py:34`)
- FastAPI request validation with Pydantic field bounds (`main.py:84–96`)
- Langfuse no-op fallback when keys are absent (`tracing.py:19–44`)
- Citation assembly using `.get()` with safe defaults (`generate.py:64–74`)
- Prompt construction with indexed context blocks (`prompt.py:25–44`)
