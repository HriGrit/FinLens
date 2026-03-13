# DSPy + GEPA Integration Plan for FinLens

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background: What Is DSPy and GEPA?](#2-background-what-is-dspy-and-gepa)
3. [Why This Matters for FinLens](#3-why-this-matters-for-finlens)
4. [Architecture Overview](#4-architecture-overview)
5. [What DSPy Will Optimise (and What It Won't Touch)](#5-what-dspy-will-optimise-and-what-it-wont-touch)
6. [Step-by-Step Integration Plan](#6-step-by-step-integration-plan)
   - 6.1 [Dependencies](#61-dependencies)
   - 6.2 [Defining the DSPy Signature](#62-defining-the-dspy-signature)
   - 6.3 [Wrapping the FinLens Pipeline as a DSPy Module](#63-wrapping-the-finlens-pipeline-as-a-dspy-module)
   - 6.4 [Building the Training Set from RAGAS Logs](#64-building-the-training-set-from-ragas-logs)
   - 6.5 [Configuring the GEPA Teleprompter](#65-configuring-the-gepa-teleprompter)
   - 6.6 [Running Optimisation and Persisting the Result](#66-running-optimisation-and-persisting-the-result)
   - 6.7 [Serving the Optimised Prompt](#67-serving-the-optimised-prompt)
7. [Control & Safety Mechanisms](#7-control--safety-mechanisms)
   - 7.1 [Human-in-the-Loop Gate](#71-human-in-the-loop-gate)
   - 7.2 [Diff Review Before Promotion](#72-diff-review-before-promotion)
   - 7.3 [Regression Guard](#73-regression-guard)
   - 7.4 [Rollback Path](#74-rollback-path)
   - 7.5 [Audit Trail in Langfuse](#75-audit-trail-in-langfuse)
8. [Key Decisions You Must Make](#8-key-decisions-you-must-make)
   - 8.1 [What Metric Drives Optimisation?](#81-what-metric-drives-optimisation)
   - 8.2 [Training Set Size and Quality](#82-training-set-size-and-quality)
   - 8.3 [Optimisation Frequency](#83-optimisation-frequency)
   - 8.4 [Scope: Which Prompts Does DSPy Own?](#84-scope-which-prompts-does-dspy-own)
   - 8.5 [Promotion Policy](#85-promotion-policy)
   - 8.6 [LLM Used During Optimisation vs Production](#86-llm-used-during-optimisation-vs-production)
   - 8.7 [Few-Shot vs Instruction-Only Optimisation](#87-few-shot-vs-instruction-only-optimisation)
8. [File and Module Layout](#9-file-and-module-layout)
9. [Environment Variables](#10-environment-variables)
10. [RAGAS Metric Reference Points](#11-ragas-metric-reference-points)
11. [Risks and Mitigations](#12-risks-and-mitigations)
12. [Worked Example: One Optimisation Cycle End-to-End](#13-worked-example-one-optimisation-cycle-end-to-end)
13. [Glossary](#14-glossary)

---

## 1. Executive Summary

This document describes how to integrate **DSPy** (Declarative Self-improving Python) with the **GEPA** (Generative Evolutionary Prompt Assembler) teleprompter into the FinLens RAG chatbot pipeline. The goal is to let the system **iteratively discover better prompts automatically**, while keeping you firmly in control of what goes into production.

The integration is deliberately **opt-in and offline**: DSPy optimisation runs as a separate script, produces a candidate prompt file, and only reaches production after you review a human-readable diff and explicitly approve promotion. Nothing changes at query-time unless you decide it should.

---

## 2. Background: What Is DSPy and GEPA?

### DSPy

DSPy is a framework that treats LLM prompts as **learnable parameters** rather than hard-coded strings. Instead of writing:

```
"Answer questions using ONLY the provided context passages…"
```

you write a **Signature** (an abstract input→output schema) and a **Module** (a pipeline of LLM calls). DSPy's **teleprompters** then search over the space of possible prompts — including the instruction text, the few-shot demonstrations, and the chain-of-thought prefix — to find a combination that maximises a metric you define.

The key insight: **you control the metric, DSPy controls the search**. Your application logic (retrieval, reranking, citation assembly) is untouched.

### GEPA

GEPA (Generative Evolutionary Prompt Assembler) is a DSPy teleprompter that uses an LLM to *generate* candidate instruction mutations, evaluates them against your training set, and keeps the survivors. It is well-suited for:

- **Instruction-level improvement**: refining the wording, ordering, and constraints of your system prompt.
- **Avoiding over-fitting** on few-shot examples (it works in instruction-space, not just example-space).
- **Financial/domain-specific tasks** where the right instruction framing matters enormously (e.g., "cite the exact line item" vs. "provide the figure").

Compared to older teleprompters (MIPRO, BootstrapFewShot), GEPA tends to produce shorter, more stable instructions that generalise better to unseen queries.

---

## 3. Why This Matters for FinLens

The current FinLens system prompt in `backend/generation/prompt.py` was written by hand. It is good but static. As you ingest more documents, observe new failure modes in Langfuse, or update your RAGAS thresholds, the hand-written prompt will drift further from optimal.

Specific FinLens pain points that DSPy/GEPA can address:

| Pain Point | How DSPy Helps |
|---|---|
| LLM adds figures not in context ("hallucination") | GEPA can strengthen the grounding constraint in the instruction |
| Citations missing or using wrong page number | GEPA can add an explicit formatting rule it discovers works best |
| Verbose answers to simple numerical queries | GEPA can discover a conciseness instruction |
| Poor faithfulness on table-heavy chunks | GEPA can add a table-reading instruction |
| Answer relevancy drops when company filter is used | GEPA can discover a scoping phrase |

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  OFFLINE (optimisation loop — run manually or on schedule)       │
│                                                                   │
│  eval/qa_dataset.json                                            │
│       │                                                          │
│       ▼                                                          │
│  dspy_opt/build_trainset.py ──► dspy_opt/trainset.json          │
│       │                                                          │
│       ▼                                                          │
│  dspy_opt/optimise.py                                            │
│    ├── DSPy Signature: FinLensAnswer                             │
│    ├── DSPy Module: FinLensRAG (wraps retrieve + generate)       │
│    ├── Teleprompter: GEPA                                        │
│    └── Metric: composite(faithfulness, answer_relevancy)         │
│       │                                                          │
│       ▼                                                          │
│  dspy_opt/candidate_prompt.json  ◄── serialised optimised prompt │
│                                                                   │
│  dspy_opt/review.py  ◄── diff vs current prompt.py              │
│       │                                                          │
│  YOU REVIEW ──► approve? ──► dspy_opt/promote.py                │
│                      │                                           │
│                      ▼                                           │
│            backend/generation/prompt.py  (updated)              │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│  ONLINE (query-time — unchanged)                  │
│                                                   │
│  POST /chat                                       │
│    → hybrid_retrieve()                            │
│    → cross_encoder_rerank()                       │
│    → build_prompt()  ◄── uses prompt.py           │
│    → litellm.completion()                         │
│    → citations + usage                            │
└──────────────────────────────────────────────────┘
```

**Critical design principle:** The online query path imports `prompt.py` exactly as it does today. DSPy adds zero runtime latency, zero new dependencies at query time, and zero new failure modes at query time. It is a purely offline improvement process.

---

## 5. What DSPy Will Optimise (and What It Won't Touch)

### In scope for DSPy/GEPA

| Component | What changes |
|---|---|
| `SYSTEM_PROMPT` in `prompt.py` | The instruction text (wording, rules, formatting directives) |
| Few-shot demonstrations (optional) | 2–4 worked Q&A examples prepended to the prompt |
| Chain-of-thought prefix (optional) | A reasoning scaffold e.g. "Let me check each claim…" |

### Out of scope — DSPy never touches

| Component | Reason |
|---|---|
| Retrieval logic (`hybrid.py`) | Retrieval is not a DSPy Module output; it's a fixed pipeline step |
| Reranker (`rerank.py`) | Same — deterministic model, not LLM |
| Embedding model | Fixed; changes require full re-ingestion |
| `build_prompt()` context formatting | The `[1] (Company Year …)` block format must stay stable for citation extraction to work |
| Citation extraction logic | Downstream of LLM call; tied to structured metadata |
| API contract | `ChatRequest`/`ChatResponse` schemas are unchanged |
| Qdrant schema | Completely separate from prompt optimisation |
| RAGAS thresholds | You set these; DSPy tries to meet them |

---

## 6. Step-by-Step Integration Plan

### 6.1 Dependencies

Add to `backend/pyproject.toml` under a new optional group so DSPy is **not installed in production**:

```toml
[dependency-groups.dspy]
dspy-ai>=2.5.0
```

Install:

```bash
cd backend
uv sync --group dspy
```

> **Why a separate group?** Production containers (`docker compose`) only run `uv sync` without `--group dspy`. This keeps the production image smaller and prevents any DSPy import from accidentally running at query time.

### 6.2 Defining the DSPy Signature

Create `dspy_opt/signatures.py`:

```python
import dspy

class FinLensAnswer(dspy.Signature):
    """
    Answer a financial question using ONLY the provided context passages
    from SEC filings. Cite company, year, and page for every factual claim.
    Do not use prior knowledge.
    """
    # --- Inputs ---
    question: str = dspy.InputField(
        desc="The user's financial question"
    )
    context: str = dspy.InputField(
        desc=(
            "Numbered context passages from SEC filings, each prefixed with "
            "[N] (Company Year DocType p.Page). Example:\n"
            "[1] (Acme Corp 2023 10-K p.42)\n"
            "Revenue for FY2023 was $4.2B…\n"
            "[2] (Acme Corp 2023 10-K p.43)\n"
            "Operating income was $800M…"
        )
    )
    # --- Output ---
    answer: str = dspy.OutputField(
        desc=(
            "Concise answer citing document references inline using [N] notation. "
            "If the answer cannot be found in the context, say so explicitly."
        )
    )
```

**Key decisions baked into the Signature:**
- The `context` field description fixes the `[N] (Company Year …)` format that the citation extractor in `generate.py` relies on.
- The docstring becomes the *seed instruction* that GEPA mutates. Keep it accurate but not over-specified — let GEPA find the refinements.

### 6.3 Wrapping the FinLens Pipeline as a DSPy Module

Create `dspy_opt/module.py`:

```python
import dspy
from dspy_opt.signatures import FinLensAnswer
from retrieval.pipeline import retrieve_and_rerank
from generation.prompt import format_context_blocks

class FinLensRAG(dspy.Module):
    """
    DSPy module wrapping the FinLens retrieval + generation pipeline.

    The retrieval half is a deterministic side-effect (not a DSPy predictor)
    so it does not participate in optimisation. Only the generation call
    (ChainOfThought over FinLensAnswer) is tunable by GEPA.
    """

    def __init__(self, rerank_top_k: int = 5) -> None:
        super().__init__()
        self.rerank_top_k = rerank_top_k
        # The one LLM call DSPy will optimise
        self.generate = dspy.ChainOfThought(FinLensAnswer)

    def forward(
        self,
        question: str,
        company: str | None = None,
        year: str | None = None,
    ) -> dspy.Prediction:
        # --- Retrieval (not optimised by DSPy) ---
        nodes = retrieve_and_rerank(
            query=question,
            company=company,
            year=year,
            rerank_top_k=self.rerank_top_k,
        )
        if not nodes:
            return dspy.Prediction(
                answer="I don't have enough information to answer this question "
                       "from the provided documents.",
                context="",
            )

        # --- Format context (same format as production build_prompt) ---
        context = format_context_blocks(nodes)

        # --- Generation (optimised by DSPy) ---
        prediction = self.generate(question=question, context=context)
        return prediction
```

> **Important:** `format_context_blocks` must be extracted from `generation/prompt.py` as a standalone function (it currently lives inside `build_prompt`). This extraction is the only change to existing production code required for the DSPy integration.

**Minimal refactor to `generation/prompt.py`:**

```python
# Add this standalone function (extract from build_prompt):
def format_context_blocks(nodes: list) -> str:
    """Format retrieved nodes into numbered context blocks for the prompt."""
    lines = []
    for i, node in enumerate(nodes, 1):
        m = node.metadata
        header = f"[{i}] ({m['company']} {m['year']} {m['doc_type']} p.{m.get('page_number', '?')})"
        lines.append(f"{header}\n{node.text}")
    return "\n\n".join(lines)
```

This function is then called from both the existing `build_prompt()` and the new DSPy module. **No change to the API or retrieval path.**

### 6.4 Building the Training Set from RAGAS Logs

Create `dspy_opt/build_trainset.py`:

```python
"""
Build a DSPy training set from:
  - eval/qa_dataset.json  (existing RAGAS Q&A pairs)
  - Optional: Langfuse trace exports (real user queries with thumbs-up)

Output: dspy_opt/trainset.json
"""
import json
import random
from pathlib import Path
import dspy

QA_DATASET_PATH = Path("../eval/qa_dataset.json")
TRAINSET_PATH = Path("dspy_opt/trainset.json")
DEVSET_FRACTION = 0.2  # 20% held out for validation

def load_from_ragas_dataset(path: Path) -> list[dspy.Example]:
    rows = json.loads(path.read_text())
    examples = []
    for row in rows:
        ex = dspy.Example(
            question=row["question"],
            company=row.get("company"),
            year=row.get("year"),
            answer=row["ground_truth"],  # gold answer for metric evaluation
        ).with_inputs("question", "company", "year")
        examples.append(ex)
    return examples

def split(examples: list, dev_fraction: float = DEVSET_FRACTION):
    random.shuffle(examples)
    n_dev = max(1, int(len(examples) * dev_fraction))
    return examples[n_dev:], examples[:n_dev]  # train, dev

if __name__ == "__main__":
    examples = load_from_ragas_dataset(QA_DATASET_PATH)
    train, dev = split(examples)
    print(f"Train: {len(train)}  Dev: {len(dev)}")
    TRAINSET_PATH.parent.mkdir(exist_ok=True)
    TRAINSET_PATH.write_text(json.dumps({
        "train": [e.toDict() for e in train],
        "dev": [e.toDict() for e in dev],
    }, indent=2))
    print(f"Saved to {TRAINSET_PATH}")
```

**Training set quality checklist:**

- [ ] At least 20 examples (GEPA needs enough signal; 50+ is better)
- [ ] Cover multiple companies and years
- [ ] Include both numerical and qualitative questions
- [ ] Include questions that should return "I don't have enough information" (negative examples)
- [ ] Ground-truth answers were verified against the actual documents

### 6.5 Configuring the GEPA Teleprompter

Create `dspy_opt/optimise.py`:

```python
"""
Run GEPA optimisation over the FinLensRAG module.

Usage:
    cd backend
    uv run --group dspy python ../dspy_opt/optimise.py

Output:
    dspy_opt/candidate_prompt.json   ← serialised optimised program
    dspy_opt/optimisation_report.json ← per-example scores, summary stats
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import dspy
from dspy.teleprompt import GEPA

from dspy_opt.module import FinLensRAG
from dspy_opt.metrics import composite_metric
from dspy_opt.build_trainset import load_from_ragas_dataset, split

# ── LLM for the GEPA *optimiser* (can differ from production LLM) ────────────
# GEPA uses this model to generate candidate instruction mutations.
# Use a more capable model here if budget allows; the cost is paid once offline.
OPTIMISER_LLM = os.getenv("DSPY_OPTIMISER_MODEL", "openrouter/mistralai/mistral-small-3.1-24b:free")
PRODUCTION_LLM = os.getenv("OPENROUTER_MODEL", "openrouter/stepfun/step-3.5-flash:free")

def main() -> None:
    # ── 1. Configure DSPy LM ────────────────────────────────────────────────
    lm = dspy.LM(
        model=OPTIMISER_LLM,
        api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_tokens=1024,
    )
    dspy.configure(lm=lm)

    # ── 2. Load training + dev sets ─────────────────────────────────────────
    qa_path = Path("../eval/qa_dataset.json")
    examples = load_from_ragas_dataset(qa_path)
    trainset, devset = split(examples)
    print(f"Loaded {len(trainset)} train / {len(devset)} dev examples")

    # ── 3. Instantiate the module ────────────────────────────────────────────
    module = FinLensRAG(rerank_top_k=5)

    # ── 4. Configure GEPA teleprompter ──────────────────────────────────────
    teleprompter = GEPA(
        metric=composite_metric,
        # How many candidate instruction mutations to generate per generation
        num_candidates=10,
        # Number of evolutionary generations
        num_iterations=3,
        # Evaluate candidates on this many training examples per iteration
        # (smaller = faster but noisier; larger = slower but more reliable)
        num_threads=4,
        # Hold-out set used to select the winning candidate after optimisation
        # (prevents over-fitting to the training set)
        valset=devset,
    )

    # ── 5. Run optimisation ──────────────────────────────────────────────────
    print("Starting GEPA optimisation…")
    optimised = teleprompter.compile(module, trainset=trainset)

    # ── 6. Save candidate ───────────────────────────────────────────────────
    out_dir = Path("dspy_opt")
    out_dir.mkdir(exist_ok=True)

    candidate_path = out_dir / "candidate_prompt.json"
    optimised.save(str(candidate_path))
    print(f"Saved candidate prompt to {candidate_path}")

    # ── 7. Evaluate on dev set and save report ──────────────────────────────
    scores = []
    for ex in devset:
        pred = optimised(question=ex.question, company=ex.company, year=ex.year)
        score = composite_metric(ex, pred)
        scores.append({"question": ex.question, "score": score})

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "optimiser_model": OPTIMISER_LLM,
        "production_model": PRODUCTION_LLM,
        "train_size": len(trainset),
        "dev_size": len(devset),
        "dev_mean_score": sum(s["score"] for s in scores) / len(scores),
        "per_example": scores,
    }
    report_path = out_dir / "optimisation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nDev score: {report['dev_mean_score']:.3f}")
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
```

### 6.6 Running Optimisation and Persisting the Result

Create `dspy_opt/metrics.py` to define what "better" means:

```python
"""
Composite metric for DSPy optimisation.

DSPy metrics must return a float in [0.0, 1.0] or a bool.
This metric blends:
  - Groundedness: does the answer avoid claims not in the context?
  - Relevancy:    does the answer address the question?
  - Citation:     does the answer include at least one [N] reference?

For production-quality evaluation, plug in your RAGAS scores here.
The lightweight version below works without external APIs and is fast
enough for GEPA's inner loop.
"""
import re
import dspy

# Weight for each sub-metric
W_GROUNDED  = 0.5
W_RELEVANCY = 0.3
W_CITATION  = 0.2

def _has_citation(answer: str) -> float:
    """Returns 1.0 if answer contains at least one [N] citation."""
    return 1.0 if re.search(r'\[\d+\]', answer) else 0.0

def _not_refusal(answer: str) -> float:
    """
    Returns 0.0 if answer is a blanket refusal with no content.
    Penalises the model for refusing to answer when context was available.
    """
    refusal_phrases = [
        "i don't have enough information",
        "cannot be found in the context",
        "no information available",
    ]
    lower = answer.lower()
    return 0.0 if any(p in lower for p in refusal_phrases) else 1.0

def _answer_length_score(answer: str) -> float:
    """
    Penalise extremely short (< 20 words) or extremely long (> 300 words) answers.
    Financial answers should be concise but not telegraphic.
    """
    words = len(answer.split())
    if words < 20:
        return max(0.0, words / 20.0)
    if words > 300:
        return max(0.0, 1.0 - (words - 300) / 300.0)
    return 1.0

def composite_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """
    Lightweight composite metric used by GEPA during optimisation.

    For a richer metric, replace with RAGAS faithfulness + answer_relevancy
    evaluated via the ragas library. That is slower but more accurate.
    """
    answer = prediction.answer if hasattr(prediction, "answer") else str(prediction)

    citation_score   = _has_citation(answer)
    not_refusal      = _not_refusal(answer)
    length_score     = _answer_length_score(answer)

    # Simple weighted blend
    # Grounded = not_refusal * citation (if it cites, it's more likely grounded)
    grounded = not_refusal * citation_score
    relevancy = length_score  # proxy; replace with semantic similarity if available

    score = (
        W_GROUNDED  * grounded
        + W_RELEVANCY * relevancy
        + W_CITATION  * citation_score
    )
    return round(score, 4)
```

> **Decision point:** The lightweight metric above is a proxy. See [Section 8.1](#81-what-metric-drives-optimisation) for the decision between a fast proxy vs. full RAGAS scoring.

### 6.7 Serving the Optimised Prompt

Create `dspy_opt/promote.py`:

```python
"""
Promote a candidate prompt to production after human review.

Steps:
  1. Load dspy_opt/candidate_prompt.json
  2. Extract the optimised instruction from the FinLensAnswer predictor
  3. Show a diff against the current SYSTEM_PROMPT in prompt.py
  4. Ask for confirmation
  5. Write the new SYSTEM_PROMPT into prompt.py

Usage:
    cd backend
    uv run --group dspy python ../dspy_opt/promote.py
"""
import json
import re
import sys
from pathlib import Path

import dspy
from dspy_opt.module import FinLensRAG

PROMPT_FILE = Path("generation/prompt.py")
CANDIDATE_FILE = Path("../dspy_opt/candidate_prompt.json")


def extract_instruction(optimised: FinLensRAG) -> str:
    """Pull the optimised instruction text from the DSPy program."""
    predictor = optimised.generate
    # DSPy stores the compiled instruction here after optimisation
    return predictor.signature.instructions


def read_current_system_prompt() -> str:
    source = PROMPT_FILE.read_text()
    match = re.search(
        r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', source, re.DOTALL
    )
    if not match:
        raise ValueError("Could not find SYSTEM_PROMPT in prompt.py")
    return match.group(1).strip()


def show_diff(old: str, new: str) -> None:
    import difflib
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="current SYSTEM_PROMPT",
        tofile="candidate SYSTEM_PROMPT",
    )
    print("".join(diff) or "(no changes)")


def write_new_system_prompt(new_instruction: str) -> None:
    source = PROMPT_FILE.read_text()
    new_source = re.sub(
        r'(SYSTEM_PROMPT\s*=\s*""").*?(""")',
        lambda m: m.group(1) + "\n" + new_instruction + "\n" + m.group(2),
        source,
        flags=re.DOTALL,
    )
    PROMPT_FILE.write_text(new_source)
    print(f"Updated {PROMPT_FILE}")


def main() -> None:
    # Load optimised program
    module = FinLensRAG()
    module.load(str(CANDIDATE_FILE))

    new_instruction = extract_instruction(module)
    current_instruction = read_current_system_prompt()

    print("=" * 70)
    print("DIFF: current vs. candidate SYSTEM_PROMPT")
    print("=" * 70)
    show_diff(current_instruction, new_instruction)

    print("\n" + "=" * 70)
    print("Optimisation report:")
    report_path = Path("../dspy_opt/optimisation_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        print(f"  Dev score:      {report['dev_mean_score']:.3f}")
        print(f"  Train size:     {report['train_size']}")
        print(f"  Dev size:       {report['dev_size']}")
        print(f"  Optimiser LLM:  {report['optimiser_model']}")
        print(f"  Timestamp:      {report['timestamp']}")
    print("=" * 70)

    answer = input("\nPromote this candidate to production? [y/N] ").strip().lower()
    if answer != "y":
        print("Promotion aborted. Candidate saved at dspy_opt/candidate_prompt.json")
        sys.exit(0)

    write_new_system_prompt(new_instruction)
    print("\nPromotion complete. Run tests to verify:")
    print("  cd backend && uv run pytest tests/unit -q")
    print("  uv run python ../eval/ragas_eval.py --sample 10")


if __name__ == "__main__":
    main()
```

---

## 7. Control & Safety Mechanisms

This section addresses the single most important concern: **you must never lose control of what goes into production**.

### 7.1 Human-in-the-Loop Gate

The promotion script (`promote.py`) **always requires an explicit `y` confirmation**. There is no `--force` or `--auto` flag. This is intentional. Even if you eventually automate optimisation runs (e.g. via a weekly cron), the promotion step stays manual.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│ optimise.py  │───►│ candidate    │───►│ promote.py       │
│ (automated   │    │ _prompt.json │    │ (ALWAYS manual   │
│  OK)         │    │              │    │  confirmation)   │
└──────────────┘    └──────────────┘    └──────────────────┘
```

### 7.2 Diff Review Before Promotion

`promote.py` always shows a **unified diff** of the old and new system prompt before asking for confirmation. You read exactly what changed before approving. This is not optional.

### 7.3 Regression Guard

Before promoting, always run the eval suite:

```bash
# Quick sanity check (5 examples, ~2 min)
cd backend && uv run python ../eval/ragas_eval.py --sample 5

# Full evaluation (all examples)
cd backend && uv run python ../eval/ragas_eval.py
```

The RAGAS thresholds in `eval/thresholds.yaml` act as a hard gate:

```yaml
faithfulness: 0.80
context_recall: 0.75
answer_relevancy: 0.80
```

If the optimised prompt causes any metric to fall below threshold, do not promote.

Add a pre-promotion checklist to your workflow:

- [ ] `dev_mean_score` in `optimisation_report.json` is higher than the baseline
- [ ] RAGAS faithfulness ≥ 0.80
- [ ] RAGAS answer_relevancy ≥ 0.80
- [ ] Unit tests pass: `uv run pytest tests/unit -q`
- [ ] The diff looks semantically reasonable (no hallucinated rules)

### 7.4 Rollback Path

Because `promote.py` modifies `generation/prompt.py` in-place and you are working in a git repo, rollback is a single command:

```bash
git diff backend/generation/prompt.py   # Inspect the change
git checkout -- backend/generation/prompt.py  # Revert
```

**Recommendation:** commit `prompt.py` before every promotion so the rollback is clean:

```bash
git add backend/generation/prompt.py
git commit -m "chore: pre-optimisation snapshot of SYSTEM_PROMPT"
# … run promote.py …
git add backend/generation/prompt.py
git commit -m "feat: GEPA-optimised SYSTEM_PROMPT (dev_score=0.87)"
```

### 7.5 Audit Trail in Langfuse

Tag every Langfuse trace with the prompt version so you can correlate metric changes with prompt changes:

```python
# In generation/generate.py — add after building the prompt
trace.update(metadata={
    "prompt_version": read_prompt_version(),  # e.g. git short SHA
})
```

Add a `PROMPT_VERSION` constant to `prompt.py`:

```python
# Bump this manually or via promote.py after each promotion
PROMPT_VERSION = "2024-01-15-gepa-v1"
```

---

## 8. Key Decisions You Must Make

These are the decisions that will most affect the quality and stability of the integration. Think through each one before running the first optimisation.

### 8.1 What Metric Drives Optimisation?

This is the most important decision. The metric you pass to GEPA completely determines what "better" means.

| Option | Pros | Cons | Recommended for |
|---|---|---|---|
| **Lightweight proxy** (citation presence, length, non-refusal) | Fast (~5 s/example), no external API calls, free | Coarse signal; may not correlate with actual quality | First iteration, rapid prototyping |
| **RAGAS faithfulness** | Directly measures hallucination, proven metric | Needs an LLM judge, ~10 s/example, costs tokens | Production-quality optimisation |
| **RAGAS answer_relevancy** | Measures if answer addresses the question | Same as above | Combined with faithfulness |
| **RAGAS composite** (faithfulness × answer_relevancy) | Best signal quality | Slowest; ~15–20 s/example | Final polishing runs |
| **Human label** (thumbs up/down from Langfuse) | Ground truth from real users | Requires collecting labels first | Once you have ≥50 labelled examples |

**Recommendation:** Start with the lightweight proxy for the first 2–3 optimisation rounds to verify the pipeline works. Then switch to RAGAS composite once you trust the setup.

To switch to RAGAS in `metrics.py`:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

def ragas_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    dataset = Dataset.from_dict({
        "question": [example.question],
        "answer": [prediction.answer],
        "contexts": [[]],          # supply retrieved chunks here for faithfulness
        "ground_truth": [example.answer],
    })
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    f = result["faithfulness"]
    r = result["answer_relevancy"]
    return (f + r) / 2
```

### 8.2 Training Set Size and Quality

| Size | Expected outcome |
|---|---|
| < 15 examples | High variance; GEPA may not converge |
| 15–30 examples | Usable for instruction-level improvements |
| 30–100 examples | Good convergence; covers most financial question types |
| > 100 examples | Diminishing returns; use the extras as a held-out test set |

**Quality beats quantity.** Five well-verified Q&A pairs with exact ground-truth answers are worth more than fifty pairs with approximate answers.

**Negative examples matter.** Include 10–20% questions where the ground truth is "the document does not contain this information." This teaches GEPA to preserve the refusal instruction.

### 8.3 Optimisation Frequency

| Trigger | Rationale | Risk |
|---|---|---|
| **On demand** (manual, first approach) | Maximum control; no surprises | May drift if documents change significantly |
| **After every major ingestion batch** | Aligns prompt with new document types | Needs robust regression gate |
| **Weekly scheduled** | Continuous improvement | Risk of silent degradation if metric is misconfigured |
| **When RAGAS score drops below threshold** | Reactive, need-based | Requires monitoring infrastructure |

**Recommendation:** Start with on-demand. Run manually after each significant expansion of your document corpus. Graduate to weekly automated runs only after you have at least 3 successful on-demand cycles and trust the metric.

### 8.4 Scope: Which Prompts Does DSPy Own?

FinLens currently has one prompt: the `SYSTEM_PROMPT` in `prompt.py`. But you could expand scope later:

| Scope | Description | Risk |
|---|---|---|
| **System prompt only** (recommended) | GEPA changes the instruction text only | Low; citation format untouched |
| **System prompt + few-shot examples** | GEPA adds 2–4 Q&A demonstrations | Medium; demos could be misleading if low quality |
| **Query rewriting** | Add a DSPy step that rewrites the user query before retrieval | High; changes what gets retrieved |
| **Chain-of-thought prefix** | GEPA adds a reasoning scaffold | Low-medium; can improve numerical reasoning |

**Recommendation:** Start with system prompt only. Add few-shot demonstrations in round 2 if faithfulness is still below 0.85 after instruction optimisation.

### 8.5 Promotion Policy

Define a written policy (add it to your `CONTRIBUTING.md` or this document):

```
Promotion Policy v1:
  1. optimise.py must complete without error.
  2. dev_mean_score must be ≥ current_baseline + 0.03 (3% improvement).
  3. RAGAS faithfulness ≥ 0.80, answer_relevancy ≥ 0.80 on full devset.
  4. Unit tests pass.
  5. At least one human reviewer reads the diff.
  6. promote.py must be run interactively (not piped).
  7. Commit before and after promotion with descriptive messages.
```

The 3% improvement threshold prevents promoting negligibly different prompts and avoids thrashing.

### 8.6 LLM Used During Optimisation vs Production

GEPA needs an LLM both to **generate candidate instructions** and to **evaluate them**. This does not need to be the same model as production.

| Configuration | Cost | Quality |
|---|---|---|
| Same free model for both | Cheapest | Lower optimisation quality |
| Larger model for optimisation, free for production | Moderate | Better instruction discovery |
| GPT-4/Claude Sonnet for optimisation, Mistral for production | Higher | Best quality improvements |

The key insight: **optimisation happens once (offline), production runs forever**. Spending more on the optimisation LLM is usually justified.

Set `DSPY_OPTIMISER_MODEL` separately from the production `OPENROUTER_MODEL`:

```bash
# .env
DSPY_OPTIMISER_MODEL=openrouter/anthropic/claude-3.5-haiku:beta
OPENROUTER_MODEL=openrouter/stepfun/step-3.5-flash:free
```

### 8.7 Few-Shot vs Instruction-Only Optimisation

GEPA primarily works in instruction-space, but you can also bootstrap few-shot demonstrations:

| Mode | DSPy config | Effect |
|---|---|---|
| **Instruction-only** | Default GEPA | Rewrites the system prompt text |
| **Instruction + few-shot** | GEPA + BootstrapFewShot | Adds Q&A examples before the user message |
| **Few-shot only** | BootstrapFewShot | Adds examples, keeps instruction fixed |

Instruction-only is safer because:
- No retrieved context is baked into the prompt (privacy-safe for financial docs)
- Prompt stays shorter (lower token cost per query)
- Fewer things to audit in the diff

Start with instruction-only.

---

## 9. File and Module Layout

The DSPy integration lives entirely in a new top-level directory `dspy_opt/` to keep it cleanly separated from production code:

```
dspy_opt/
  __init__.py
  signatures.py         ← DSPy Signature definition (FinLensAnswer)
  module.py             ← DSPy Module (FinLensRAG wrapping the pipeline)
  metrics.py            ← Composite metric function for GEPA
  build_trainset.py     ← Loads eval/qa_dataset.json → trainset.json
  optimise.py           ← Main GEPA optimisation script
  promote.py            ← Human-gated promotion to prompt.py
  trainset.json         ← Generated (gitignore this if it has PII)
  candidate_prompt.json ← Generated (commit this for audit trail)
  optimisation_report.json  ← Generated (commit this)

backend/generation/
  prompt.py             ← MODIFIED: add format_context_blocks(), PROMPT_VERSION
  generate.py           ← UNCHANGED

backend/pyproject.toml  ← MODIFIED: add [dependency-groups.dspy]
```

**What to gitignore vs commit:**

```gitignore
# .gitignore additions
dspy_opt/trainset.json         # May contain sensitive Q&A pairs
dspy_opt/__pycache__/

# Commit these (they are audit artifacts):
# dspy_opt/candidate_prompt.json
# dspy_opt/optimisation_report.json
```

---

## 10. Environment Variables

Add to `.env.example`:

```bash
# DSPy / GEPA optimisation (offline only — not needed for API server)
DSPY_OPTIMISER_MODEL=openrouter/anthropic/claude-3.5-haiku:beta
# If unset, defaults to OPENROUTER_MODEL
```

The existing `OPENROUTER_API_KEY` and `OPENROUTER_API_BASE` are reused by DSPy — no new secrets required.

---

## 11. RAGAS Metric Reference Points

Current thresholds in `eval/thresholds.yaml`:

| Metric | Threshold | Meaning |
|---|---|---|
| faithfulness | 0.80 | 80% of claims in the answer are supported by context |
| context_recall | 0.75 | 75% of ground-truth facts are present in retrieved chunks |
| answer_relevancy | 0.80 | Answer is 80% relevant to the question |

These thresholds are your **promotion gate**. After optimisation, run:

```bash
cd backend
uv run python ../eval/ragas_eval.py --sample 20
```

and verify all three metrics are at or above threshold before promoting.

Track these values in `optimisation_report.json` for each cycle so you can see the trend over time.

---

## 12. Risks and Mitigations

| Risk | Probability | Severity | Mitigation |
|---|---|---|---|
| GEPA produces a prompt that hallucinates more | Medium | High | RAGAS faithfulness gate; git rollback |
| Citation format broken by optimised prompt | Low | High | Citation regex test in `test_prompt.py` |
| Optimisation over-fits to training set | Medium | Medium | Use dev set score, not train score, for decisions |
| LLM API cost during optimisation exceeds budget | Low | Medium | Use free-tier models; set `num_candidates=5` initially |
| Optimised prompt is too long (high token cost per query) | Low | Medium | Check token count in `optimisation_report.json` |
| DSPy dependency conflicts with production deps | Low | High | Separate `[dependency-groups.dspy]` |
| Optimisation run takes too long (> 2 hours) | Medium | Low | Start with `num_iterations=2, num_candidates=5` |
| Metric is misconfigured (optimises wrong thing) | Low | High | Manual spot-check of 5 answers before any promotion |

---

## 13. Worked Example: One Optimisation Cycle End-to-End

This section walks through a complete optimisation cycle from start to finish.

### Step 0: Prerequisites

```bash
# Services running
docker compose up -d

# BM25 index and Qdrant populated
cd backend && uv run python -m ingestion.run_ingestion --list

# DSPy installed
uv sync --group dspy

# Baseline RAGAS score recorded
uv run python ../eval/ragas_eval.py --sample 10
# e.g. faithfulness=0.82, answer_relevancy=0.81
```

### Step 1: Build Training Set

```bash
cd backend
uv run --group dspy python ../dspy_opt/build_trainset.py
# Output: Train: 40  Dev: 10
# Saved to dspy_opt/trainset.json
```

### Step 2: Run Optimisation (takes 20–90 minutes)

```bash
uv run --group dspy python ../dspy_opt/optimise.py
# Starting GEPA optimisation…
# [GEPA] Generation 1/3: evaluating 10 candidates…
# [GEPA] Generation 2/3: evaluating 10 candidates…
# [GEPA] Generation 3/3: evaluating 10 candidates…
# Saved candidate prompt to dspy_opt/candidate_prompt.json
# Dev score: 0.847
# Report saved to dspy_opt/optimisation_report.json
```

### Step 3: Review the Diff

```bash
uv run --group dspy python ../dspy_opt/promote.py
```

Output (example):

```diff
======================================================================
DIFF: current vs. candidate SYSTEM_PROMPT
======================================================================
--- current SYSTEM_PROMPT
+++ candidate SYSTEM_PROMPT
@@ -1,10 +1,12 @@
 You are FinLens, a financial document analysis assistant.

 Answer questions using ONLY the provided context passages from SEC filings.
-If the answer cannot be found in the context, say "I don't have enough information to answer this question from the provided documents."
+If the answer cannot be found in the provided context passages, explicitly state that the information is unavailable in the given documents.

 Rules:
 - Cite the specific document (company, year, page) for every factual claim.
 - Use exact figures when available; do not round or estimate.
+- When citing tables, include the table title and row label alongside the figure.
 - Do not use prior knowledge outside the provided context.
 - Keep answers concise and structured.

Dev score:      0.847   (was: ~0.78 from proxy metric baseline)
Optimiser LLM:  openrouter/anthropic/claude-3.5-haiku:beta
Timestamp:      2024-01-15T14:23:11+00:00
======================================================================

Promote this candidate to production? [y/N]
```

### Step 4: Run Regression Tests

```bash
uv run pytest tests/unit -q
uv run python ../eval/ragas_eval.py --sample 10
# faithfulness=0.85, answer_relevancy=0.84  ← above thresholds ✓
```

### Step 5: Approve and Commit

```bash
# At the prompt.py in promote.py:
# Promote this candidate to production? [y/N] y
# Updated backend/generation/prompt.py

git add backend/generation/prompt.py dspy_opt/candidate_prompt.json dspy_opt/optimisation_report.json
git commit -m "feat: GEPA-optimised SYSTEM_PROMPT — dev_score=0.847, faithfulness=0.85"
```

---

## 14. Glossary

| Term | Definition |
|---|---|
| **DSPy** | Declarative Self-improving Python; a framework for optimising LLM prompts and pipelines |
| **GEPA** | Generative Evolutionary Prompt Assembler; a DSPy teleprompter that uses an LLM to mutate instructions |
| **Teleprompter** | DSPy's name for an optimisation algorithm (analogous to a compiler for prompts) |
| **Signature** | A DSPy abstraction that defines the input and output fields of an LLM call |
| **Module** | A DSPy class that composes one or more LLM calls (Predictors) into a pipeline |
| **Predictor** | A DSPy component that wraps a single LLM call; its instruction is what GEPA optimises |
| **ChainOfThought** | A DSPy Predictor that elicits step-by-step reasoning before the final answer |
| **Trainset** | Labelled Q&A examples used by GEPA to evaluate candidate instructions |
| **Devset** | Held-out Q&A examples used to select the winning candidate (prevents over-fitting) |
| **Composite metric** | The scalar score function GEPA maximises; defined in `dspy_opt/metrics.py` |
| **Promotion** | The act of copying the optimised instruction from `candidate_prompt.json` into production `prompt.py` |
| **RAGAS** | Retrieval-Augmented Generation Assessment; evaluation framework measuring faithfulness, recall, relevancy |
| **RRF** | Reciprocal Rank Fusion; the algorithm FinLens uses to merge BM25 and dense retrieval results |
