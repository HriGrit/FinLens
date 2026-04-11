# DSPy Offline Evaluation & Prompt Optimisation Pipeline

> **Goal:** Evaluate and improve the FinLens system prompt using real, expert-verified financial QA data — without touching the production pipeline or needing manual financial knowledge to write questions.

This folder documents a completely **offline, separate** pipeline. Nothing here runs at query time. You run these scripts on your laptop (or a CI job), review the outputs, and then manually decide whether to update `backend/generation/prompt.py`.

---

## Why a Separate Pipeline?

The production pipeline (`backend/`) is optimised for latency and correctness at query time. Optimisation work is expensive (many LLM calls, batch RAGAS scoring) and should never run in the hot path. Keeping it separate also means:

- You can run it without a live Qdrant instance (it will still need one for retrieval, but it is a deliberate, scheduled job — not an always-on concern)
- Prompt changes are a human decision, not an automated deployment
- You can iterate on the eval dataset independently of the application code

---

## The Three Stages

```
Stage 1: Dataset
  Download FinanceBench → normalise format → split into train / dev / test
  Output: eval/financebench_qa.json (test set used everywhere else)

Stage 2: Baseline Evaluation
  Run existing prompt against FinanceBench test set
  Output: RAGAS scores (faithfulness, context_recall, answer_relevancy, answer_correctness)
  Purpose: Establish a hard number to beat before touching anything

Stage 3: DSPy Prompt Optimisation
  Wrap FinLens pipeline as a DSPy Module
  Run an Optimizer (BootstrapFewShot → MIPROv2 → GRPO) on the train set
  Output: one or more candidate prompt strings + their RAGAS scores
  Decision: human reviews candidate prompts, picks the best, updates prompt.py manually
```

Each stage is documented in its own file:

| File | What it covers |
|---|---|
| `01-financebench-dataset.md` | What FinanceBench is, how to obtain and prepare it |
| `02-baseline-evaluation.md` | Running the existing prompt against the full dataset |
| `03-prompt-optimisation.md` | DSPy module design, optimizer selection, GRPO |
| `04-model-selection.md` | How to compare models for cost vs. accuracy |

---

## Dependency on Production Components

Even though this is an offline pipeline, it does call the **same retrieval and generation functions** as production. This is intentional — you are evaluating the real system, not a simulation.

```
FinanceBench QA pair
      ↓
retrieve_and_rerank()      ← same function as backend/retrieval/pipeline.py
      ↓
generate()                 ← same function as backend/generation/generate.py
      ↓
RAGAS scoring              ← offline, uses a separate judge LLM
      ↓
Score recorded
```

This means you need:
- A running Qdrant instance with documents already ingested
- `OPENROUTER_API_KEY` set in `.env`
- The FinanceBench PDFs ingested (the same companies the QA pairs reference)

---

## What Changes After This Pipeline

Only one file should ever change as a result of this pipeline:

```
backend/generation/prompt.py → SYSTEM_PROMPT (the string constant)
```

Everything else — retrieval logic, reranking, the API contract — is not touched. The prompt is the single lever.

---

## Quick Reference: Run Order

```
1. Prepare dataset     → dspy/scripts/01_prepare_financebench.py
2. Baseline eval       → eval/ragas_eval.py --dataset eval/financebench_qa.json
3. DSPy optimise       → dspy/scripts/03_run_optimiser.py
4. Compare scores      → dspy/scripts/04_compare_results.py
5. Human review        → pick a candidate prompt, update backend/generation/prompt.py
6. Re-run baseline     → eval/ragas_eval.py --dataset eval/financebench_qa.json
   (verify improvement)
```
