# FinLens — Fixes Audit Plan (Mar 12, 2026)

> Purpose: turn the March 12 audit into validated, track-based implementation guidance for Ralph agents.
>
> Constraint: this document is planning only. Do not implement fixes directly from this file without opening a separate execution task.

---

## Verification Method

- Re-verified the audit against the current source under `backend/`, `eval/`, `tests/`, `docs/`, and `docker-compose.yml`.
- Checked existing test coverage to see which claims are already guarded and which are still untested.
- Ran a representative backend unit slice successfully with:
  - `uv run --project backend --group dev python -m pytest backend/tests/unit/test_hybrid_retrieve.py backend/tests/unit/test_generate.py backend/tests/unit/test_run_ingestion.py backend/tests/unit/test_parse.py backend/tests/unit/test_chunk.py -q`
- Added independently found issues with the marker **[FOUND SEPARATELY]**.

---

## Verification Corrections

These items from `Audit_Mar_12.md` need wording adjustments before work starts.

### Q4 Nuanced — collection-name drift is a config consistency problem, not a direct correctness bug

The mismatch between test, dev, and validation collection names is real, but some separation is intentional for isolation. Treat this as a single-source-of-truth/config hygiene problem.

### I6 Nuanced — artifact/manifest desync is not fully silent anymore

`run_ingestion.py` now warns when a successful doc is missing a chunk artifact and falls back to BM25 nodes when possible. The integrity issue still exists because manifest success can remain out of sync with actual chunk artifacts.

### I7 Nuanced — the cited example is weak, but the parser is still fragile

The current `doc_type` slicing does not fail exactly the way the audit describes for `123K`, but the manual slicing approach is still overly permissive and should be replaced with an explicit parser or regex.

### C1 Nuanced — `ingestion_registry.json` is legacy, not deleted

The active system uses `ingestion_manifest.json`, but `ingestion_registry.json` still exists as a migration source. The docs are still wrong because they describe the legacy artifact as primary.

### C2 Nuanced — the healthcheck concern is valid, but the backend issue is not “Python missing”

The backend container already has Python. The real problem is that the healthchecks are more brittle and complex than necessary.

---

## Track Index

| Track | Scope | Audit IDs | Extra IDs | Priority |
|---|---|---|---|---|
| A | Ingestion data integrity and indexing contracts | Q1, Q2, Q3, I1, I2, I3, I4, I5, I6, I7 | X1 | P0/P1 heavy |
| B | Retrieval contract and caller alignment | R1, R2 | X2 | P0/P2 |
| C | Generation and API resilience | G1, G2 | X3, X4 | P1/P2 |
| D | Test harness and validation script correctness | T1, T2, Q4 | X5 | P0/P2 |
| E | Eval, docs, and operational hygiene | E1, E2, C1, C2 | X6, X7 | P2 |

**Recommended order**

1. Track A first. It protects the ingestion source of truth and prevents silent data corruption.
2. Track B and Track C can run in parallel after Track A is scoped.
3. Track D should land once Track A/B contracts are stable.
4. Track E can run in parallel with D, but its eval fixes should consume the final retrieval contract from Track B.

---

## Track A — Ingestion Data Integrity And Indexing Contracts

### Why this is one track

These issues all affect whether a parsed document becomes a complete, trustworthy, idempotent indexed artifact. They should be solved together so ingestion has one explicit failure model.

### Included validated issues

- **Q1** Upsert result is ignored in `backend/ingestion/index.py`.
- **Q2** Embedding batch length is never validated before `zip(nodes, embeddings)`.
- **Q3** Deterministic point IDs can collide when required metadata fields are `None`.
- **I1** Docling conversion status is not checked before using `.document`.
- **I2** `assert_node_metadata()` checks key presence, not non-null values.
- **I3** `_source_idx` is trusted without proving the splitter preserved it.
- **I4** Splitter metadata can overwrite parse-time metadata silently.
- **I5** BM25 persistence reconstructs a retriever from a handcrafted payload instead of preserving an explicit serialization contract.
- **I6** Manifest success can drift from chunk artifacts.
- **I7** `doc_type` parsing is too loose and should be explicit.
- **X1 [FOUND SEPARATELY]** `tests/validate_ingestion.py` assumes `bm25_index.pkl` loads as a retriever object, but the current file format is a dict payload.

### Fix direction

- Define ingestion invariants first:
  - a document is not “successful” unless parsing, chunking, Qdrant indexing, and BM25 rebuild all complete and validate.
  - all required metadata fields must be both present and semantically valid.
  - idempotent re-ingestion must never create duplicates or overwrite unrelated points.
- Make failures fail fast at the boundary where they occur:
  - validate Docling conversion result immediately.
  - validate embedding count before building points.
  - validate Qdrant upsert status after each batch.
- Tighten metadata handling:
  - either reject `None` for fields used in point identity, or separate “required for storage” from “optional for citation” metadata explicitly.
  - stop using blanket `dict.update()` from splitter output; use an allowlist or explicit merge precedence.
  - make `_source_idx` preservation an asserted contract, not a best-effort fallback to `0`.
- Formalize BM25 persistence:
  - either persist only the node corpus plus a versioned serialization contract and test round-trip equivalence, or persist a fully supported artifact format that is intentionally reconstructed.
  - add a format version and a rebuild path for incompatible payloads.
- Treat manifest and chunk artifacts as coupled state:
  - a “success” manifest entry with no readable artifact should either fail validation loudly or be marked for rebuild.
- Replace ad hoc filename parsing with an explicit grammar:
  - only accept supported filename shapes, or move metadata to a sidecar manifest if filename flexibility is needed.

### Ralph agent work plan

1. Define the ingestion invariants in code comments and tests before changing logic.
2. Harden parse and chunk boundaries.
3. Harden Qdrant write and point-ID behavior.
4. Harden BM25 serialization/loading behavior.
5. Add repair logic for manifest/artifact mismatch.

### Testing gate for Track A

Add or extend tests in:

- `backend/tests/unit/test_parse.py`
- `backend/tests/unit/test_chunk.py`
- `backend/tests/unit/test_run_ingestion.py`
- `backend/tests/integration/test_ingestion_to_qdrant.py`
- `backend/tests/integration/test_bm25_build_and_load.py`
- New: `backend/tests/unit/test_index.py`

Required cases:

- Docling conversion with non-success status is rejected immediately.
- Embedding batch shorter than node list raises and does not truncate silently.
- Non-success Qdrant upsert status fails the run.
- Point-ID creation rejects ambiguous metadata or proves deterministic uniqueness.
- `page_number=None` is either rejected upstream or handled by an explicit contract.
- Splitter dropping `_source_idx` fails predictably.
- Splitter metadata cannot overwrite authoritative parse metadata unintentionally.
- BM25 save/load round-trips preserve ranking for a fixed corpus snapshot.
- Missing chunk artifact for a manifest-success doc is surfaced as repair/failure, not quietly tolerated.

### Exit criteria

- Ingestion can no longer produce silent partial indexes.
- Re-running ingestion on the same document is idempotent.
- BM25 and Qdrant artifacts are validated, versioned, and rebuildable.

---

## Track B — Retrieval Contract And Caller Alignment

### Why this is one track

These issues are about the contract between retrieval internals and every downstream consumer. The fix should make retrieval output explicit and stable.

### Included validated issues

- **R1** RRF fusion keeps duplicate-node metadata only because dense results are processed second.
- **R2** `retrieve_and_rerank()` returns a tuple that leaks trace-side data into the public function contract.
- **X2 [FOUND SEPARATELY]** `eval/ragas_eval.py` and `tests/validate_e2e.py` currently call `retrieve_and_rerank()` as if it returns only a node list.

### Fix direction

- Replace order-dependent overwrite behavior with an explicit duplicate-merge rule.
  - Preferred direction: keep dense metadata intentionally when duplicate text appears in both sources, but encode that as a merge policy rather than loop-order luck.
- Decide the retrieval public API once:
  - either return only `list[TextNode]` and push tracing counts into the tracing layer, or
  - return a small result object/dataclass with named fields like `nodes` and `candidate_count`.
- Update all non-API callers together.
  - `ragas_eval.py` and `tests/validate_e2e.py` are already mismatched and should be corrected in the same change.

### Ralph agent work plan

1. Lock the return contract.
2. Refactor fusion to use explicit merge semantics.
3. Update all consumers in one sweep.
4. Add regression tests for both metadata selection and public return shape.

### Testing gate for Track B

Add or extend tests in:

- `backend/tests/unit/test_hybrid_fusion.py`
- `backend/tests/unit/test_tracing.py`
- New: `backend/tests/unit/test_pipeline_contract.py`
- `tests/validate_e2e.py`
- `eval/ragas_eval.py` smoke coverage if eval tests are introduced

Required cases:

- Duplicate text from BM25 and dense retrieval resolves by explicit policy, not iteration order.
- Retrieval return type is named and stable.
- All callers consume the new contract correctly.
- Eval and validation scripts do not break on contract changes.

### Exit criteria

- Retrieval has one documented return contract.
- No downstream consumer relies on tuple positional knowledge.
- Fusion metadata preference is deliberate and regression-tested.

---

## Track C — Generation And API Resilience

### Why this is one track

Generation and `/chat` currently share one failure surface. The fix should turn malformed upstream responses into controlled, observable API behavior.

### Included validated issues

- **G1** LiteLLM response structure is accessed without guarding `choices`, `message`, or `usage`.
- **G2** Cost calculation failure is reported as `0.0`, which is misleading.
- **X3 [FOUND SEPARATELY]** `ChatRequest` does not bound `retrieval_top_k` and `rerank_top_k` to positive values.
- **X4 [FOUND SEPARATELY]** `/chat` still leaks non-`ValueError` retrieval/generation failures as raw 500s.

### Fix direction

- Introduce one internal generation validation layer:
  - validate `choices` exists and is non-empty.
  - validate `message.content`.
  - validate `usage` presence and fields.
  - raise one domain-specific error type for malformed upstream responses.
- Represent unknown cost honestly:
  - return `null` or a separate status field when cost calculation fails.
- Harden `/chat` error mapping:
  - retrieval backend failures and malformed LLM responses should become controlled upstream-failure responses, not framework 500s.
- Tighten request validation:
  - enforce positive bounds for `retrieval_top_k` and `rerank_top_k`.

### Ralph agent work plan

1. Define the generation error model.
2. Harden `generate()`.
3. Harden API request validation and exception mapping.
4. Update response tests and API reasoning payload expectations.

### Testing gate for Track C

Add or extend tests in:

- `backend/tests/unit/test_generate.py`
- `backend/tests/integration/test_api.py`
- New: `backend/tests/unit/test_api_validation.py`

Required cases:

- Empty `choices` raises a controlled generation error.
- Missing `usage` or partial token counts raises a controlled generation error.
- `completion_cost` failure yields `cost_usd = null` or equivalent explicit failure state.
- Invalid `retrieval_top_k` and `rerank_top_k` are rejected at request validation time.
- Retrieval and generation upstream failures map to expected HTTP status codes.

### Exit criteria

- No malformed LiteLLM response can crash `/chat` with an unhandled 500.
- Cost reporting no longer falsifies telemetry.
- API request bounds are explicit and enforced.

---

## Track D — Test Harness And Validation Script Correctness

### Why this is one track

The current harness mixes real-production assumptions with test-only defaults. This can hide breakage or validate the wrong system shape.

### Included validated issues

- **T1** The pytest-wide fake embedding model returns 1024-dim vectors while production uses a different shape.
- **T2** `tests/validate_ingestion.py` reaches into a private BM25 attribute.
- **Q4** Collection names drift between tests, app defaults, and validation scripts.
- **X5 [FOUND SEPARATELY]** `tests/validate_ingestion.py` expects `pickle.load()` to return a retriever, but the current BM25 artifact is a dict payload.

### Fix direction

- Centralize runtime/test config:
  - define collection names in one shared config module or environment-loading helper.
  - validation scripts should consume the same config source as runtime code.
- Fix embedding-test drift:
  - either align the fake model dimension with the production default, or make tests declare the dimension they target and assert collection creation accordingly.
- Treat validation scripts as first-class code:
  - stop reading private retriever internals.
  - load BM25 through `load_bm25_index()` or a public validation helper instead of `pickle.load()` directly.

### Ralph agent work plan

1. Introduce shared config for collection name and embedding expectations.
2. Align fixtures with production contracts.
3. Repair standalone validation scripts to use public loaders/APIs.

### Testing gate for Track D

Add or extend tests in:

- `backend/tests/conftest.py`
- `backend/tests/integration/test_setup_collection.py`
- New: `backend/tests/unit/test_validation_scripts.py`

Required cases:

- Test fixture embedding dimension matches the intended runtime contract.
- Validation scripts resolve collection names from the shared config source.
- Validation logic does not use private BM25 attributes.
- BM25 validation uses the current public artifact format.

### Exit criteria

- Test infrastructure validates the real system shape.
- Validation scripts do not break when BM25 internals change.
- Collection naming becomes consistent and explicit.

---

## Track E — Eval, Docs, And Operational Hygiene

### Why this is one track

These issues do not usually corrupt user answers directly, but they weaken regression detection, onboarding, and day-2 operations.

### Included validated issues

- **E1** RAGAS eval aborts on the first failed query.
- **E2** Eval only measures three metrics and omits correctness-oriented coverage.
- **C1** Codespaces docs still describe the legacy ingestion registry as the main state artifact.
- **C2** Docker healthchecks are brittle and unnecessarily complex.
- **X6 [FOUND SEPARATELY]** `docs/codespace-readiness/codespaces.md` references setup paths that do not exist in this repo.
- **X7 [FOUND SEPARATELY]** `eval/ragas_eval.py --sample N` is not reproducible because sampling is unseeded.

### Fix direction

- Make eval resilient and useful:
  - isolate failures per question.
  - persist partial outputs.
  - record failed/skipped rows explicitly.
  - make sampling reproducible.
- Expand eval coverage only after the retrieval contract in Track B is fixed.
  - add correctness-oriented metrics and matching thresholds.
- Repair docs to match the repo as it exists now.
  - manifest, not registry, is the active operator-facing artifact.
  - remove or fix references to missing Codespaces files/scripts.
- Simplify healthchecks.
  - prefer direct localhost endpoints and standard commands.
  - remove fragile shell tricks where a simpler check exists.

### Ralph agent work plan

1. Stabilize eval execution and output persistence.
2. Expand metrics and thresholds.
3. Repair codespaces/readiness docs.
4. Simplify Compose healthchecks and re-verify with `docker compose config`.

### Testing gate for Track E

Add or extend tests in:

- New: `backend/tests/unit/test_ragas_eval.py` or `eval/tests/test_ragas_eval.py`
- Existing docs verification workflow if introduced
- Manual operational validation for Compose healthchecks

Required cases:

- One failed query does not abort the entire eval run.
- Eval partial results are persisted.
- Sampling with a fixed seed is reproducible.
- Threshold file and enabled metric list stay in sync.
- Docs do not reference missing operational files.
- `docker compose config` remains valid after healthcheck simplification.

### Exit criteria

- Eval becomes reliable for regression tracking.
- Operator docs reflect the current repository.
- Healthchecks become maintainable and predictable.

---

## Coverage Matrix

Every item from `docs/audit/Audit_Mar_12.md` is mapped below.

| Audit ID | Status | Track | Notes |
|---|---|---|---|
| Q1 | Confirmed | A | Upsert status is ignored |
| Q2 | Confirmed | A | Batch embedding length mismatch can truncate silently |
| Q3 | Confirmed | A | Ambiguous metadata can collide in point IDs |
| Q4 | Confirmed, nuanced | D | Config inconsistency more than direct correctness bug |
| I1 | Confirmed | A | Docling conversion result not validated early |
| I2 | Confirmed | A | Required metadata allows `None` values |
| I3 | Confirmed | A | `_source_idx` contract is implicit and unsafe |
| I4 | Confirmed | A | Splitter metadata can overwrite parse metadata |
| I5 | Confirmed | A | BM25 serialization contract is fragile |
| I6 | Confirmed, nuanced | A | Warns today, but integrity drift remains |
| I7 | Confirmed, nuanced | A | Parser is fragile even if the example is imperfect |
| R1 | Confirmed | B | Fusion behavior depends on loop order |
| R2 | Confirmed | B | Public retrieval contract is overloaded |
| G1 | Confirmed | C | LiteLLM response shape is not validated |
| G2 | Confirmed | C | Cost failures are misreported as zero |
| T1 | Confirmed | D | Fixture embedding shape drifts from runtime |
| T2 | Confirmed | D | Validation script uses private BM25 internals |
| E1 | Confirmed | E | Eval has no per-query fault isolation |
| E2 | Confirmed | E | Eval metric coverage is incomplete |
| C1 | Confirmed, nuanced | E | Docs describe legacy state artifact as primary |
| C2 | Confirmed, nuanced | E | Healthchecks are brittle, though not for the exact stated reason |

### Additional issues found during verification

| Extra ID | Status | Track | Issue |
|---|---|---|---|
| X1 | Confirmed | A | `tests/validate_ingestion.py` expects the wrong BM25 artifact shape |
| X2 | Confirmed | B | `eval/ragas_eval.py` and `tests/validate_e2e.py` do not unpack retrieval results |
| X3 | Confirmed | C | `ChatRequest` lacks positive bounds for retrieval/rerank values |
| X4 | Confirmed | C | `/chat` still leaks non-`ValueError` upstream failures as 500s |
| X5 | Confirmed | D | Validation logic mixes public and private BM25 access patterns incorrectly |
| X6 | Confirmed | E | Codespaces docs reference missing repo paths |
| X7 | Confirmed | E | Eval sampling is not reproducible |

---

## Final Direction

- Do not assign Ralph agents by audit priority alone. Assign them by track boundary so each agent can own one contract surface and its tests.
- The highest-value starting point is Track A because silent ingestion corruption invalidates everything downstream.
- No track should be considered done unless its testing gate is added or updated in the same change.
