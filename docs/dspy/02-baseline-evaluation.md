# Stage 2: Baseline Evaluation (Testing the Existing Prompt)

> **Question you are answering here:** "How good is the current prompt, right now, on a real dataset I did not design myself?"

This is the most important stage to run first. Without a baseline, you cannot know whether DSPy optimisation actually helped.

---

## Good News: The Pipeline Already Exists

You do not need to write a new evaluation script. `eval/ragas_eval.py` already does exactly this. It accepts a `--dataset` flag that points to any JSON file in `qa_dataset.json` format.

To test the existing prompt against FinanceBench:

```
cd backend
uv run python ../eval/ragas_eval.py --dataset ../eval/financebench_test.json
```

That is the entire baseline run. The script will:
1. Load the FinanceBench test set
2. For each question, call `retrieve_and_rerank()` then `generate()` with the current prompt
3. Score with RAGAS (faithfulness, context_recall, answer_relevancy, answer_correctness)
4. Write results to `eval/ragas_eval_results.json`
5. Print a pass/fail table against `eval/thresholds.yaml`

The only prerequisite is that the FinanceBench test set exists at the path you point to (output of Stage 1) and the relevant documents are ingested into Qdrant.

---

## What the Four RAGAS Metrics Mean in Practice

Understanding what each metric is actually measuring matters when you interpret results and decide what to fix.

### Faithfulness (target: ≥ 0.80)
**What it measures:** Are the claims in the answer supported by the retrieved context? A faithfulness score of 0.80 means 80% of the atomic statements in the answer can be traced back to a context passage.

**What causes low faithfulness:** The LLM is hallucinating — making up figures or facts not present in the retrieved chunks. This is a generation problem, not a retrieval problem. A better system prompt with stricter instructions fixes this.

**What does NOT fix it:** Retrieval improvements. If faithfulness is low, the prompt is telling the model it can use prior knowledge. Fix the prompt.

---

### Context Recall (target: ≥ 0.75)
**What it measures:** Is the ground truth answer covered by the retrieved context? A score of 0.75 means 75% of the information needed to answer the question is present in what was retrieved.

**What causes low context recall:** Retrieval is missing the relevant chunk. This is a retrieval problem — BM25 or dense search is not returning the right passages. The prompt cannot fix this.

**What does NOT fix it:** Prompt changes. If context recall is low, improve chunking, embedding quality, or increase `retrieval_top_k`.

---

### Answer Relevancy (target: ≥ 0.80)
**What it measures:** Is the answer actually responsive to the question? A score of 0.80 means the answer addresses what was asked 80% of the time.

**What causes low answer relevancy:** The model is answering a different question, adding irrelevant caveats, or failing to extract the specific figure requested. This is a prompt instruction quality problem.

**What does NOT fix it:** Better retrieval. If relevancy is low, the prompt is not being specific enough about the answer format.

---

### Answer Correctness (target: ≥ 0.70)
**What it measures:** How closely does the answer match the ground truth? Combines factual correctness and semantic similarity. This is the hardest metric to achieve because it requires both good retrieval AND good generation.

**What causes low answer correctness:** Any combination of hallucination, wrong retrieval, or poor phrasing. This is the aggregate outcome metric — if everything else is high but correctness is low, look at the ground truth format (e.g., the model says "$32.8B" but the ground truth is "$32.8 billion").

---

## Reading the Baseline Results Diagnostically

Do not just look at the overall pass/fail. Break down results by question type (if you tagged rows in Stage 1):

| Pattern | Diagnosis |
|---|---|
| Faithfulness low across all types | Prompt needs stricter "use only context" instruction |
| Context recall low for arithmetic questions | Tables are not being retrieved — chunking issue |
| Answer relevancy low for risk-factor questions | Prompt does not handle open-ended questions well |
| Correctness low only for arithmetic questions | Model cannot do arithmetic in context — consider showing worked examples in the prompt |
| All metrics high for factual, all low for multi-hop | Retrieval is fine; prompt does not handle synthesis |

This breakdown is far more useful than a single aggregate number. It tells you exactly which prompt change to attempt.

---

## Storing the Baseline

Save the raw results file with a meaningful name before running any optimisation:

```
eval/ragas_eval_results.json  →  eval/results/baseline_financebench_YYYY-MM-DD.json
```

Every subsequent run (optimised prompt, different model, different dataset) should be saved with a similar timestamp. This gives you a history of what changed and what the effect was.

---

## Smoke Test First

Before running all 80–150 rows (which may take 30–60 minutes and cost API credits), use the `--sample` flag to validate the pipeline is working end-to-end:

```
uv run python ../eval/ragas_eval.py --dataset ../eval/financebench_test.json --sample 5
```

5 rows should complete in 2–3 minutes. If it fails, fix the issue before committing to a full run.

---

## Expected Baseline Scores (Rough Expectation)

Based on the current hand-written prompt in `backend/generation/prompt.py`, reasonable baseline expectations on FinanceBench:

| Metric | Expected range | Why |
|---|---|---|
| Faithfulness | 0.70 – 0.85 | The prompt says "only use provided context" which helps, but free models hallucinate |
| Context Recall | 0.55 – 0.75 | Retrieval is good but FinanceBench has harder questions than the current eval set |
| Answer Relevancy | 0.65 – 0.80 | The prompt is terse on format — free models may add caveats |
| Answer Correctness | 0.50 – 0.70 | Correctness is the hardest — arithmetic and format matching will hurt |

If you score above 0.80 on correctness out of the box, that is a strong pipeline. If you score below 0.50, something structural is wrong (likely retrieval, not the prompt).

---

## What to Do With This Information Before Moving to Stage 3

Before running DSPy optimisation, make a list of:
1. Which metrics are below threshold?
2. Which question types are most problematic?
3. Is the issue retrieval (context recall) or generation (faithfulness / relevancy)?

If context recall is the primary problem, DSPy prompt optimisation will not help much — you would need to address retrieval first. DSPy is a prompt optimiser, not a retrieval optimiser. Only move to Stage 3 if you believe the prompt is the bottleneck.
