# Stage 3: DSPy Prompt Optimisation

> **What DSPy does:** Given a pipeline definition (inputs → outputs), a set of training examples, and a metric function, DSPy automatically rewrites the prompt to maximise the metric. You do not write the new prompt. You review it.

This document covers how to wrap FinLens as a DSPy program, which optimiser to use, and how to interpret the outputs.

---

## Mental Model: What DSPy Actually Changes

In FinLens, `backend/generation/prompt.py` contains a single string constant: `SYSTEM_PROMPT`. Every call to `generate()` uses this string. DSPy's job is to find a better version of that string by running experiments.

It does this by:
1. Taking your training examples (question + ground truth)
2. Running the pipeline many times with different prompt candidates
3. Scoring each candidate with your metric (RAGAS scores)
4. Using the results to propose better candidates
5. Returning the best-found candidate at the end

**What DSPy does not change:**
- The retrieval logic (`hybrid.py`, `rerank.py`)
- The citation format
- The API contract
- Any code — only a string

---

## The FinLens DSPy Program Structure

A DSPy program has three components:

### 1. The Signature
A signature is a typed declaration of what the program does: what goes in, what comes out, and what the intent is. For FinLens:

```
Inputs:
  question      — the user's financial query
  context       — the retrieved SEC filing passages, with citation markers already applied

Output:
  answer        — a concise, cited answer using only the provided context
```

The instruction text on the signature (the docstring) is what DSPy will rewrite. The current version in `prompt.py` is the starting instruction.

### 2. The Module
The module wraps the FinLens pipeline. It calls `retrieve_and_rerank()` to get context, formats the context nodes into a single string (same as `build_prompt()` does today), and then calls the DSPy LLM with the signature.

The module does **not** use the existing `generate()` function directly — because `generate()` already has the old prompt baked in via `build_prompt()`. Instead, the module builds the context string and passes it to a DSPy predictor, which is where the optimised prompt gets injected.

This is the key architectural point: **DSPy replaces the LLM call inside generate(), not the retrieval step.** Retrieval happens outside DSPy's control.

### 3. The Metric Function
The metric function takes one training example and one model prediction, and returns a score between 0 and 1. For FinLens, the natural metric is RAGAS faithfulness (the most directly influenced by prompt quality). You can also use a combined score:

```
metric = 0.4 * faithfulness + 0.3 * answer_relevancy + 0.3 * answer_correctness
```

The exact weights are a choice you make based on what you care about most. Faithfulness should have the highest weight because it is the one the prompt most directly controls. Context recall is retrieval-controlled and should not be in the metric (the prompt cannot fix it).

---

## The Three Optimisers: When to Use Each

DSPy has a family of optimisers. They differ in how many LLM calls they make, how sophisticated the search is, and how well they work with small vs. large training sets.

### BootstrapFewShot (Start Here)

**How it works:** Runs your training examples through the pipeline. When the pipeline produces a correct answer (judged by the metric), it saves that full trace — the question, the context, and the answer — as a few-shot example. It then constructs a prompt that includes these successful traces as demonstrations.

**Why start here:**
- Fastest: only runs each training example once (plus some validation)
- Cheapest: 1x training set size in API calls
- No instruction rewriting: it does not touch the instruction text, only adds examples
- Diagnostic: if even this does not improve scores, something more fundamental is wrong

**Limitation:** It only adds examples to the prompt — it cannot discover a better instruction. If the core instruction is wrong, few-shot examples cannot compensate.

**When to use:** First run, always. Treat it as a sanity check.

---

### MIPROv2 (Main Optimiser)

**How it works:** Uses a multi-step process:
1. Generates many candidate instruction strings (using a separate "proposer" LLM call)
2. Evaluates each candidate on a subset of the training set
3. Uses Bayesian optimisation (a smart search strategy) to pick which candidates to try next
4. Returns the instruction that scored highest on the dev set

**Why this is the main choice:**
- Actually rewrites the instruction, not just adds examples
- Bayesian search is efficient — finds a good instruction without trying all combinations
- Works well with 30–60 training examples
- Produces human-readable candidate prompts you can inspect

**Cost:** Roughly 5–10x more API calls than BootstrapFewShot. On free-tier OpenRouter models, this might mean 30–60 minutes of wall clock time and moderate API usage.

**What the output looks like:** A set of candidate prompts with scores, like:
```
Candidate 1 (score: 0.81):
  "You are a financial analyst assistant. Answer questions using ONLY 
   the numbered passages below. For each fact, cite [n] where n is the 
   passage number. If a figure requires arithmetic, show the calculation.
   Do not round; use exact numbers from the text."

Candidate 2 (score: 0.78):
  "Answer the question from the provided SEC filing extracts only.
   ..."
```

You read these, pick the one that makes the most sense, and update `prompt.py` manually.

---

### GRPO (Advanced: Reinforcement Learning-Based)

**What GRPO stands for:** Group Relative Policy Optimisation. This is the same technique DeepSeek used to train their reasoning models, adapted by DSPy to work at the prompt level.

**How it works:** Instead of a Bayesian search over discrete candidates, GRPO treats prompt optimisation as a reinforcement learning problem:
1. For each training question, generate K different responses (e.g., K=8) using the current prompt
2. Score each response with the metric function
3. Calculate a "group relative reward": responses above the average for that question get positive reward, below get negative reward
4. Update the prompt to reinforce the patterns that led to above-average responses

**Why this matters for FinLens:**
- GRPO discovers non-obvious patterns — e.g., it might find that asking the model to "extract the exact figure as it appears in the text, then state it" consistently improves correctness scores
- It handles the arithmetic question type much better than instruction rewriting alone, because it can reward answers that show calculation steps
- It requires a reward-shaping metric, which is a slightly different formulation than BootstrapFewShot/MIPRO's binary pass/fail

**Cost:** Significantly more expensive than MIPROv2. Each training step runs K=8 completions per question. On 36 training examples with K=8, that is 288 completions per iteration, multiplied by the number of iterations (typically 20–50). This is appropriate for a weekly optimisation job, not a daily one.

**When to use GRPO:**
- After MIPROv2 has plateaued (you have already run it and the marginal improvement is <0.02)
- When arithmetic questions are your specific weak point (GRPO's reward shaping handles this)
- When you have budget for a longer overnight run

**GRPO is not a replacement for MIPROv2.** Start with BootstrapFewShot to validate the setup, then MIPROv2 for the main improvement, then optionally GRPO for the last mile.

---

## The Optimisation Run: What to Expect

### Input
- `eval/financebench_train.json` — 36 rows (or whatever your 60% split gives you)
- `eval/financebench_dev.json` — 12 rows (used by MIPROv2's Bayesian selector)
- The current `SYSTEM_PROMPT` as the starting point

### Process
The optimiser will print progress. For MIPROv2 you will see it generating instruction candidates ("Proposing instruction 1 of 10...") and then evaluating them ("Evaluating candidate 3..."). Expect this to take 20–60 minutes.

### Output
The script should write all candidate prompts and their scores to:
```
eval/results/dspy_candidates_YYYY-MM-DD.json
```

Format:
```json
[
  {
    "rank": 1,
    "score": 0.81,
    "faithfulness": 0.88,
    "answer_relevancy": 0.79,
    "answer_correctness": 0.74,
    "optimiser": "MIPROv2",
    "prompt": "You are FinLens..."
  },
  ...
]
```

### What You Do With the Output

1. Read each candidate prompt
2. Ask yourself: Does this instruction make sense? Is it coherent? Could a human follow it?
3. Check if it introduces any instructions you would not want (e.g., "use prior knowledge if needed" — reject this)
4. Pick the top-scoring candidate that you are comfortable with
5. Update `backend/generation/prompt.py` manually with that string
6. Run the baseline eval again on the test set to confirm the improvement:
   ```
   uv run python ../eval/ragas_eval.py --dataset ../eval/financebench_test.json
   ```
7. Compare against the saved baseline results

---

## The "DSPy Does Not Know Your Data" Problem

DSPy's optimiser generates candidate prompts using a meta-LLM (a separate LLM call that proposes new instructions). This meta-LLM has no knowledge of what FinanceBench questions look like, what SEC filings are, or what financial analysis means.

This means some candidates will be generic or poorly targeted. **That is why human review is non-negotiable.** DSPy is a search tool, not a domain expert. It finds prompts that score well on your metric, but it cannot guarantee those prompts are financially accurate, safe, or coherent to a human reader.

**Red flags to reject a candidate:**
- Any instruction that says "use your general knowledge" or "estimate if the document is unclear"
- Instructions with made-up citation formats that conflict with the existing `[n]` format
- Instructions that are unusually short (just "Answer the question") — these often overfit to the metric on the train set but fail on unseen questions

---

## Connecting DSPy Output Back to the Application

After you update `prompt.py`, the change flows through the entire stack automatically:

```
prompt.py → build_prompt() → generate() → API → Frontend
```

No other file needs to change. The `build_prompt()` function reads `SYSTEM_PROMPT` at call time, so the update is live immediately on the next server start.

Run the full RAGAS eval on the test set (not train or dev) to get the final confirmed improvement number. This number is what you would report in a model card or evaluation summary.
