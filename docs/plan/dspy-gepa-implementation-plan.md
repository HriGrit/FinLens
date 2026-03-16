# DSPy + GEPA Implementation Plan for FinLens

## Purpose

This document turns the ideas in `docs/plan/dspy-gepa-integration.md` into an execution plan for the current FinLens repository. It is written to be executable by Ralph-style agent loops, meaning the work is split into bounded tracks with clear inputs, outputs, dependencies, stop conditions, validation steps, and handoff points.

This document stays at the implementation-planning level. It does not include code. It defines what needs to be built, in what order, with what acceptance criteria, and how the work should be governed so prompt optimization becomes a real operating capability rather than an isolated experiment.

## What This Plan Is Optimizing For

The target outcome is not merely “DSPy runs.” The target outcome is:

- FinLens can optimize prompts offline without adding complexity to the live request path.
- Candidate prompts can be evaluated against a trusted dataset and a stable metric policy.
- Prompt promotion is explicit, reviewable, reversible, and governed.
- The implementation can be delivered incrementally by agent loops without requiring one large risky change.

## Current Codebase Reality

The implementation must be anchored to the current repo, not to older assumptions.

- The active application lives under `backend/`.
- The live answer path is:
  `backend/api/main.py` -> `backend/retrieval/pipeline.py` -> `backend/generation/generate.py` -> `backend/generation/prompt.py`
- Prompt construction is centralized in `backend/generation/prompt.py`.
- Evaluation exists in `eval/ragas_eval.py` and uses `eval/qa_dataset.json` and `eval/thresholds.yaml`.
- Langfuse tracing exists in `backend/observability/tracing.py`, but there is not yet a real prompt-optimization review loop built around it.
- RAGAS evaluation exists in `.github/workflows/eval.yml`, but prompt optimization is not currently operationalized as its own governed workflow.

## Core Design Principles

These principles should constrain every implementation track.

- DSPy and GEPA run offline only.
- Retrieval, reranking, indexing, and API contracts remain outside the optimizer’s control.
- The production request path must keep working without DSPy installed.
- The context format used for optimization must match the context format used in production.
- Prompt changes must be reviewable as artifacts, not hidden in optimizer internals.
- Promotion must require explicit approval and a rollback path.

## Ralph Agent Loop Execution Model

This plan assumes Ralph agent loops execute work in repeated bounded cycles. Each loop should operate as follows:

1. Load one track and one scope-limited objective.
2. Inspect only the files and documents needed for that track.
3. Produce the stated deliverables for that track.
4. Validate against the listed acceptance criteria.
5. Stop at the track gate instead of spilling into unrelated work.
6. Hand off a concise status summary to the next loop.

Each track below is designed so an agent can complete it independently, provided dependencies are already satisfied.

## Global Constraints for All Tracks

- Do not move DSPy logic into the live `/chat` runtime path.
- Do not let prompt optimization modify retrieval, reranking, Qdrant schema, or API schemas.
- Do not rely on Langfuse-derived training data in the first implementation pass.
- Do not treat the current QA dataset as sufficient without explicit expansion and split policy.
- Do not allow promotion without both evaluation and human review.

## Track Overview

The implementation should be executed through seven tracks. The tracks are ordered, but some sub-work inside later tracks can begin once dependencies are satisfied.

| Track | Goal | Primary Outcome |
|---|---|---|
| Track 1 | Stabilize the prompt boundary | Production prompt rendering becomes reusable and optimizer-safe |
| Track 2 | Create the offline optimization workspace | DSPy/GEPA can exist without affecting runtime |
| Track 3 | Build the optimization dataset and dataset policy | Optimization has trustworthy training and holdout inputs |
| Track 4 | Define metrics, thresholds, and promotion gates | “Better” is measurable and stable |
| Track 5 | Wrap FinLens as an optimizer-controlled program | DSPy optimizes only the intended generation layer |
| Track 6 | Add candidate artifacts, review, promotion, and rollback | Prompt changes become governed operational assets |
| Track 7 | Operationalize the workflow for repeat use | The system becomes a repeatable capability rather than a one-off effort |

## Track 1: Stabilize the Prompt Boundary

### Objective

Create a clean, reusable prompt boundary so both production and offline optimization can rely on the same context representation and prompt contract.

### Why this track must go first

The optimizer cannot safely target the prompt if prompt logic is only embedded in production-only code paths. FinLens currently has the right starting point in `backend/generation/prompt.py`, but the prompt layer still needs to become a stable boundary rather than a single runtime helper.

### Scope

This track is limited to prompt structure, context formatting responsibilities, and prompt artifact shape. It is not allowed to redesign retrieval or generation policy.

### Inputs

- `backend/generation/prompt.py`
- `backend/generation/generate.py`
- `backend/tests/unit/test_prompt.py`
- `backend/tests/unit/test_generate.py`
- `docs/plan/dspy-gepa-integration.md`

### Work to perform

- Separate prompt concerns into explicit conceptual responsibilities:
  - render retrieved nodes into stable numbered context blocks
  - assemble messages using the system prompt and rendered context
- Preserve the existing numbered citation context format because downstream expectations and tests already depend on it.
- Define a canonical representation for a prompt artifact that an optimizer could produce and that a reviewer could inspect.
- Document what parts of the prompt are mutable and what parts are fixed:
  - mutable: instruction wording and possibly optional demonstrations
  - fixed: context block structure, citation numbering expectations, message contract
- Confirm that production prompt rendering can be invoked outside the live API path without copying logic.

### Deliverables

- A documented prompt boundary for FinLens.
- A clear statement of which prompt parts DSPy may optimize.
- A clear statement of which prompt parts are not negotiable because other system behavior depends on them.

### Acceptance criteria

- A future optimizer can reuse the same context rendering behavior as production.
- Existing prompt behavior remains understandable and testable.
- There is no ambiguity about what “the prompt” means in this repository.

### Stop condition

Stop when the prompt boundary is explicit and reusable. Do not proceed into DSPy workspace setup inside the same loop unless Track 1 is complete and validated.

### Handoff to next track

The next loop should inherit:

- the exact prompt boundary definition
- the list of mutable versus fixed prompt parts
- any tests or contracts that cannot be broken by later tracks

## Track 2: Create the Offline Optimization Workspace

### Objective

Create an isolated optimization area where DSPy and GEPA can run without becoming part of the production serving path.

### Why this track matters

FinLens should treat prompt optimization as an offline subsystem. If DSPy becomes a runtime dependency of `backend/api/main.py` or `backend/generation/generate.py`, the architecture will have regressed.

### Scope

This track covers packaging, directory layout, artifact locations, and operator-facing organization of the optimization workspace.

### Inputs

- `backend/pyproject.toml`
- `README.md`
- current repo layout under `backend/`, `eval/`, and `docs/`

### Work to perform

- Define a dedicated optimization workspace location.
- Add DSPy as an optional dependency group rather than a default dependency.
- Define where the following will live:
  - optimization entrypoints
  - optimization reports
  - candidate prompt artifacts
  - promotion review outputs
  - rollback references
- Decide the ownership boundary between the optimization workspace and the production backend.
- Document the rule that the optimizer imports backend logic, but backend runtime code does not import optimizer logic.

### Deliverables

- An agreed workspace layout for offline optimization.
- An agreed dependency policy showing that DSPy is optional and offline-only.
- A documented artifact storage plan.

### Acceptance criteria

- FinLens can conceptually support DSPy installation only when optimization work is being performed.
- The live API path can remain DSPy-free.
- Future agent loops know exactly where optimization assets belong.

### Stop condition

Stop when the workspace and dependency plan are defined clearly enough that implementation can proceed without structural uncertainty.

### Handoff to next track

The next loop should inherit:

- the chosen workspace location
- the dependency-group policy
- the agreed artifact locations

## Track 3: Build the Optimization Dataset and Dataset Policy

### Objective

Create a trustworthy dataset strategy so GEPA is optimizing against meaningful signals rather than a narrow or accidental sample.

### Why this track matters

The current `eval/qa_dataset.json` is a useful seed, but it is not sufficient as the sole optimization dataset. It appears narrow, heavily 3M-oriented, and primarily useful for smoke or early evaluation. Prompt optimization will overfit unless the dataset policy is made explicit.

### Scope

This track covers dataset composition, dataset split policy, data quality rules, and future Langfuse usage policy. It does not require implementing Langfuse export in the first pass.

### Inputs

- `eval/qa_dataset.json`
- `eval/ragas_eval.py`
- `backend/observability/tracing.py`
- any existing product expectations for FinLens answer behavior

### Work to perform

- Define dataset tiers:
  - optimization training set
  - validation set used during candidate selection
  - promotion set used to approve or reject a candidate
- Expand beyond the current narrow QA seed set.
- Ensure the dataset includes:
  - exact numerical queries
  - qualitative questions
  - table-heavy evidence questions
  - company-filtered and year-filtered questions
  - refusal cases where the correct answer is that context is insufficient
  - multi-claim answers that require citation discipline
- Define dataset quality rules for every example:
  - verified answer
  - known source document
  - expected answer style
  - expected citation availability
  - whether refusal is expected
- Define how future Langfuse data may be admitted:
  - only after filtering policy exists
  - only when real traces can be separated into reliable and unreliable examples
  - only when feedback or review provenance is attached

### Deliverables

- A dataset policy for optimization, validation, and promotion.
- A documented quality standard for labeled examples.
- A rule for when Langfuse data can be used later.

### Acceptance criteria

- The dataset is broad enough to represent real FinLens behavior, not just one narrow question family.
- Holdout data exists that the optimizer never sees during search.
- Refusal behavior is explicitly part of the dataset policy.

### Stop condition

Stop when the dataset policy is documented and strong enough that metric design can proceed on top of it.

### Handoff to next track

The next loop should inherit:

- the split policy
- the data quality rules
- the explicit limitations of the current seed dataset

## Track 4: Define Metrics, Thresholds, and Promotion Gates

### Objective

Define the quality model that tells GEPA what to optimize and tells reviewers what can be promoted.

### Why this track matters

Without a metric policy, GEPA will search blindly. Without a promotion gate, FinLens will have no trustworthy rule for deciding whether a candidate prompt is actually better.

### Scope

This track covers optimization metrics, evaluation metrics, threshold cleanup, regression rules, and promotion blocking criteria.

### Inputs

- `eval/ragas_eval.py`
- `eval/thresholds.yaml`
- dataset policy from Track 3
- prompt boundary definition from Track 1

### Work to perform

- Decide the relationship between:
  - the inner-loop optimization metric
  - the holdout evaluation metric
  - the final promotion gate
- The recommended policy for this repo is:
  - use a lightweight metric during DSPy/GEPA search
  - use RAGAS-aligned evaluation for promotion decisions
- Define non-negotiable answer qualities:
  - groundedness
  - answer relevance
  - citation presence or citation discipline
  - refusal correctness
  - answer length and structure sanity
- Clean up `eval/thresholds.yaml` so it becomes trustworthy as part of the promotion gate. The duplicated `answer_correctness` entry should be resolved before this file is treated as governance-critical.
- Define what counts as a blocking regression:
  - threshold failure
  - citation collapse
  - unsafe increase in hallucination behavior
  - unacceptable degradation in refusal handling

### Deliverables

- A documented optimization metric policy.
- A documented promotion gate policy.
- A documented regression-blocking policy.

### Acceptance criteria

- There is a stable and explicit answer to “what does better mean?”
- There is a stable and explicit answer to “when is a candidate blocked?”
- Threshold configuration is clean enough to be trusted.

### Stop condition

Stop when the repo has a metric policy strong enough to govern candidate evaluation and promotion.

### Handoff to next track

The next loop should inherit:

- the optimization metric definition
- the promotion gate definition
- the regression-blocking rules

## Track 5: Wrap FinLens as an Optimizer-Controlled Program

### Objective

Expose the current FinLens retrieval-plus-generation flow to DSPy in a way that lets the optimizer tune only the intended generation instruction layer.

### Why this track matters

This is where the theoretical plan becomes a real integration. The wrapper is the boundary that determines what DSPy can and cannot mutate.

### Scope

This track covers task signature design, wrapper boundaries, retrieval reuse, and optimizer ownership of the prompt layer. It does not cover promotion workflow or operational governance beyond what is needed to make the wrapper safe.

### Inputs

- prompt boundary output from Track 1
- workspace design from Track 2
- dataset policy from Track 3
- metric policy from Track 4
- `backend/retrieval/pipeline.py`
- `backend/generation/prompt.py`
- `backend/generation/generate.py`

### Work to perform

- Define the FinLens answering task in optimizer terms:
  - what inputs are provided
  - what output is optimized
  - what context representation is fixed
- Reuse the existing retrieval pipeline rather than inventing a parallel retrieval path.
- Reuse the production context format rather than synthesizing a separate optimization-only format.
- Limit the optimizer’s mutable surface area to:
  - system instruction wording
  - optional output guidance
  - optional few-shot demonstrations if explicitly allowed by policy
- Explicitly forbid the optimizer from owning:
  - retrieval selection logic
  - reranking behavior
  - citation payload structure
  - Qdrant contracts
  - API request and response models
  - provider fallback policy in `backend/generation/generate.py`

### Deliverables

- A defined optimizer-facing FinLens task boundary.
- A clear mapping between production logic and optimizer-controlled logic.
- An ownership statement for what DSPy may mutate.

### Acceptance criteria

- The wrapper uses real FinLens retrieval behavior.
- The wrapper uses real FinLens context formatting.
- The optimizer cannot accidentally drift into owning unrelated system behavior.

### Stop condition

Stop when the optimizer integration point is fully bounded and aligned with the repo’s live architecture.

### Handoff to next track

The next loop should inherit:

- the optimizer task definition
- the mutable surface area policy
- the list of explicitly forbidden optimization targets

## Track 6: Add Candidate Artifacts, Review, Promotion, and Rollback

### Objective

Turn prompt optimization outputs into governed engineering artifacts that can be reviewed, promoted, rejected, and rolled back safely.

### Why this track matters

Without this track, the project would only know how to generate prompt candidates, not how to safely operationalize them.

### Scope

This track covers artifact design, baseline comparison, review output, promotion flow, and rollback rules.

### Inputs

- metric and gate policy from Track 4
- optimizer wrapper output from Track 5
- current production prompt state in `backend/generation/prompt.py`
- existing documentation and workflow expectations under `docs/` and `.github/workflows/`

### Work to perform

- Define the canonical candidate artifact content:
  - candidate prompt contents
  - baseline prompt reference
  - optimizer model used
  - dataset version used
  - train and validation sizes
  - summary scores
  - timestamp and run identity
- Define the human-review artifact:
  - what changed in the prompt
  - why the candidate won
  - how it scored versus baseline
  - whether any metrics worsened
  - what risks remain
- Define the promotion action:
  - how an approved candidate becomes the production prompt
  - what baseline reference must be preserved
  - how to record promotion rationale
- Define rollback action:
  - restore previous known-good prompt
  - restore prior baseline reference
  - document why rollback happened

### Deliverables

- A candidate artifact policy.
- A review artifact policy.
- A promotion policy.
- A rollback policy.

### Acceptance criteria

- A reviewer can understand a candidate without reading DSPy internals.
- Promotion requires an explicit step and cannot happen accidentally.
- Rollback is simple and well-defined.

### Stop condition

Stop when prompt candidates are operational artifacts rather than opaque optimizer output.

### Handoff to next track

The next loop should inherit:

- the artifact schema expectations
- the promotion rules
- the rollback rules

## Track 7: Operationalize the Workflow for Repeat Use

### Objective

Make prompt optimization a repeatable FinLens workflow that can be run again later with confidence.

### Why this track matters

A one-time integration is not enough. The real value appears only when the team can re-run optimization as the dataset grows, new failure modes appear, or answer-quality priorities change.

### Scope

This track covers operator workflow, review cadence, run triggers, documentation standards, and ongoing governance.

### Inputs

- outputs from Tracks 1 through 6
- current CI and workflow setup in `.github/workflows/pr-checks.yml` and `.github/workflows/eval.yml`

### Work to perform

- Define when optimization runs should occur:
  - manual only
  - after enough new labeled examples exist
  - before prompt promotion
  - on a scheduled cadence if the team eventually wants that
- Define a standard runbook:
  - prepare dataset
  - run optimization
  - inspect candidate artifacts
  - run promotion-gate evaluation
  - approve or reject
  - promote or rollback
- Define operator expectations:
  - who can run optimization
  - who can approve promotion
  - who can trigger rollback
- Align the workflow with existing repo reality:
  - backend tests run on backend changes
  - RAGAS evaluation is currently a separate workflow
  - prompt optimization should hook into that reality rather than assume a fully automated gate already exists

### Deliverables

- A repeatable operator workflow.
- A governance model for rerunning prompt optimization later.
- A clear review and approval path.

### Acceptance criteria

- The workflow can be repeated without rediscovering process every time.
- Different operators or agent loops can understand their responsibilities.
- Prompt optimization becomes a maintained capability, not a one-off branch experiment.

### Stop condition

Stop when the repo has a documented, repeatable, governable optimization lifecycle.

## Track Dependencies

The tracks should be executed in this dependency order:

- Track 1 has no dependencies.
- Track 2 depends on Track 1.
- Track 3 depends on Track 1 and benefits from Track 2.
- Track 4 depends on Track 3.
- Track 5 depends on Tracks 1 through 4.
- Track 6 depends on Tracks 4 and 5.
- Track 7 depends on all previous tracks.

If an agent loop starts a track before dependencies are satisfied, it should stop and hand back a dependency-blocked status rather than invent missing policy.

## Loop-Level Execution Template

Each Ralph loop should use the following execution template when working one track:

### Loop input

- track identifier
- objective
- dependencies already completed
- files and docs to inspect
- explicit out-of-scope items

### Loop action

- inspect only the required files
- update or create only the artifacts needed for that track
- validate against acceptance criteria
- produce a handoff summary

### Loop output

- completed work summary
- unresolved risks
- validation result
- next-track recommendation
- explicit blocked items if any remain

## Recommended Detailed Sequence for Ralph Agent Loops

The following sequence is the recommended execution order.

### Loop 1

- Execute Track 1.
- Do not begin workspace or dataset work yet.
- End only when the prompt boundary and mutable-versus-fixed prompt policy are clear.

### Loop 2

- Execute Track 2.
- Use Track 1 outputs to isolate optimization ownership from runtime ownership.
- End only when the offline workspace and dependency-group model are defined.

### Loop 3

- Execute Track 3.
- Treat `eval/qa_dataset.json` as the seed, not the end state.
- End only when train, validation, and promotion dataset policy is explicit.

### Loop 4

- Execute Track 4.
- Clean up threshold-policy ambiguity before defining promotion rules.
- End only when optimization and promotion metrics are clearly separated and documented.

### Loop 5

- Execute Track 5.
- Build the optimizer-facing task boundary around the real FinLens pipeline.
- End only when DSPy ownership is tightly bounded.

### Loop 6

- Execute Track 6.
- Convert candidate prompts into reviewable artifacts with promotion and rollback policy.
- End only when human review can be performed without reading optimizer internals.

### Loop 7

- Execute Track 7.
- Turn the implementation into a repeatable workflow that future contributors or agents can operate.
- End only when the overall lifecycle is documented end to end.

## Definition of True Completion

This integration should be considered truly complete only when all of the following are true:

- FinLens can run prompt optimization offline without affecting the live request path.
- The optimizer uses the same prompt context contract that production uses.
- The optimization dataset is curated, split, and governed.
- “Better” is defined by explicit metrics and thresholds.
- Candidate prompts are stored as reviewable artifacts.
- Promotion is explicit and reversible.
- The workflow is repeatable by future operators or agent loops.

## Main Risks and Required Mitigations

### Risk: overfitting to a narrow dataset

Mitigation:

- expand the dataset before trusting optimization results
- preserve a promotion holdout set the optimizer never sees

### Risk: optimization weakens citation discipline

Mitigation:

- keep citation behavior in both the metric policy and the review policy
- preserve context formatting as a fixed contract

### Risk: DSPy leaks into runtime architecture

Mitigation:

- keep DSPy optional and offline-only
- enforce one-way ownership where optimization code may import backend logic but backend runtime does not import optimization logic

### Risk: prompt promotion becomes subjective

Mitigation:

- define candidate artifacts, review artifacts, promotion gates, and rollback rules before promotion begins

### Risk: Langfuse data is incorporated too early

Mitigation:

- do not use Langfuse traces for optimization until trace quality, feedback provenance, and privacy/noise filtering are all defined

## Final Recommendation

The correct way to integrate DSPy + GEPA into FinLens is to build an offline prompt-optimization subsystem executed through bounded tracks. The first real milestone is not “wire up DSPy.” The first real milestone is “stabilize the prompt, dataset, metric, and governance boundaries so DSPy has something trustworthy to optimize.”

If Ralph agent loops execute the tracks in order, stop at each gate, and preserve the fixed architectural boundaries listed in this document, the result will be a real FinLens capability. If those controls are skipped, the system will produce prompt variants but not an operational prompt-optimization workflow.
