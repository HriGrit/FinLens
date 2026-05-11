# Stage 1: FinanceBench Dataset

> **Why FinanceBench?** It is the standard public benchmark for exactly this kind of system — financial question answering over SEC filings. Every question was written by domain experts and verified against source documents. You do not need financial knowledge to use it.

---

## What FinanceBench Is

FinanceBench (PatronusAI, 2023) is a public benchmark dataset for financial document QA. The key properties:

- **~150 public QA pairs** in the released version (a private extended set of ~10 000 exists for research)
- **Source:** Real SEC filings — 10-K annual reports from companies like Apple, Microsoft, Nvidia, 3M, Boeing, Coca-Cola, and ~15 others
- **Each row contains:** question, verified answer, company name, fiscal year, document name, and a question type label
- **Question types included:**
  - Single-document factual: "What was Apple's net income in FY2022?"
  - Arithmetic / derived: "By what percentage did revenue grow from 2021 to 2022?"
  - Multi-document: "Which of these two companies had higher operating margin?"
  - Risk-factor: "What litigation risks did Boeing disclose?"
  - Table-extraction: Figures that live only inside a financial table

Compare this to the 20 hand-written pairs currently in `eval/qa_dataset.json` — those cover only 3M, only factual lookups, and only 2015–2019. FinanceBench covers a broader set of companies, question types, and years.

**HuggingFace identifier:** `PatronusAI/financebench`

---

## Structure of the Raw Data

When you download FinanceBench from HuggingFace, each row looks like this:

```
question:        "What was Apple's total net sales for fiscal year 2022?"
answer:          "$394.3 billion"
company:         "AAPL"          ← ticker symbol, needs mapping to full name
doc_name:        "AAPL_2022_10K.pdf"
fiscal_year:     "2022"
question_type:   "single_doc_factual"
```

Your `eval/qa_dataset.json` uses this format:

```
question:        "What were 3M's total net sales in fiscal year 2015?"
ground_truth:    "$30.3 billion"
company:         "3M"
year:            "2015"
```

The differences are:
- `answer` → rename to `ground_truth`
- `fiscal_year` → rename to `year`
- `company` → ticker to full name (e.g. "AAPL" → "Apple") — needs a small mapping table
- `doc_name` → optional to include; `ragas_eval.py` does not use it today

---

## What You Need to Ingest First

FinanceBench QA pairs are only useful if the PDFs they reference have been ingested into Qdrant. Before running any evaluation, confirm the overlap between:

- The companies referenced in FinanceBench
- The PDFs you have in `data/financebench/pdfs/`

**Strategy:** Filter the FinanceBench rows to only companies + years where you have the corresponding PDF already ingested. This avoids retrieval failures that would corrupt your RAGAS scores.

For example, if you only have 3M PDFs ingested, your usable FinanceBench subset is ~5–10 rows (the 3M entries). If you ingest Apple and Nvidia PDFs too, you unlock those rows as well. The public FinanceBench PDFs are all freely downloadable from SEC EDGAR.

---

## Dataset Splits

For DSPy to work correctly, you need three separate splits with no overlap:

| Split | Size | Purpose |
|---|---|---|
| Train | ~60% | DSPy optimizer sees these examples to generate prompt candidates |
| Dev / Val | ~20% | Optimizer uses this to pick the best candidate (held-out from training) |
| Test | ~20% | Final RAGAS eval — never seen by the optimizer |

If you have 150 rows and filter to companies you have ingested, you may end up with 50–80 usable rows. A rough 60/20/20 split on 60 rows gives 36 train / 12 dev / 12 test — that is enough for BootstrapFewShot and MIPROv2 to work well.

**Critical rule:** the test set is the same set used to report baseline performance (Stage 2). Once you fix the test set, never change it. All comparisons — baseline vs. optimised prompt, Model A vs. Model B — must use the same fixed test set.

---

## Preparation Steps (Offline Script: `01_prepare_financebench.py`)

The preparation script does the following in order. This is what the script you will write needs to accomplish:

**Step 1 — Download**
Pull the dataset from HuggingFace Datasets. This requires no API key; FinanceBench is publicly available. The download is ~2 MB and takes a few seconds.

**Step 2 — Filter to ingested companies**
Read the `data/ingestion_manifest.json` to know which (company, year) pairs are successfully ingested. Keep only FinanceBench rows where the (company, year) pair has a matching successful entry in the manifest.

**Step 3 — Normalise format**
Rename fields to match `qa_dataset.json`. Map ticker symbols to company names using a small hardcoded lookup table (AAPL → Apple, MSFT → Microsoft, etc.).

**Step 4 — Split**
Stratify by `question_type` so that all three splits contain a representative mix of factual, arithmetic, and risk-factor questions. A purely random split on 60–80 rows risks putting all arithmetic questions in one split.

**Step 5 — Write outputs**
Write three JSON files:
- `eval/financebench_train.json` (used by DSPy optimizer)
- `eval/financebench_dev.json` (used by DSPy optimizer for selection)
- `eval/financebench_test.json` (used for all RAGAS evals — never change this after first write)

**Step 6 — Print a summary**
Print the count of rows per split, per question type, and per company. Inspect this to make sure no split is dominated by a single company or question type.

---

## Notes on Question Quality

Not all FinanceBench questions are equally useful for prompt optimisation. Some guidance:

- **Arithmetic questions** (percentage change, ratio calculations) are the hardest and most valuable. If your prompt cannot handle these, it is a real signal.
- **Single-document factual** questions are easier and what your existing 20-row dataset covers. They are good for baseline comparison.
- **Multi-document questions** are only usable if you have both referenced documents ingested. Skip these unless you have broad ingestion coverage.
- **Table-extraction questions** test whether your retrieval pipeline is surfacing table nodes correctly. These are high-value for diagnosing retrieval quality separately from prompt quality.

Consider tagging each row in your output JSON with `question_type` from FinanceBench — this lets you break down RAGAS scores by type later, which is far more diagnostic than a single aggregate score.
