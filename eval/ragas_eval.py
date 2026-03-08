"""
ragas_eval.py — Evaluate the FinLens pipeline with RAGAS on ingested 3M 10-K documents.

Usage (from backend/):
  uv run python ../eval/ragas_eval.py               # full eval (all 20 questions)
  uv run python ../eval/ragas_eval.py --sample 5    # quick smoke test (5 random questions)
  uv run python ../eval/ragas_eval.py --dataset ../eval/qa_dataset.json --sample 3

Outputs a results table and exits non-zero if any metric is below threshold.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_PATH = Path(__file__).parent / "thresholds.yaml"
DEFAULT_DATASET_PATH = Path(__file__).parent / "qa_dataset.json"


def load_thresholds() -> dict[str, float]:
    with open(THRESHOLDS_PATH) as f:
        return yaml.safe_load(f)


def load_qa_dataset(path: Path, sample_n: int | None = None) -> list[dict]:
    """Load QA pairs from JSON, optionally random-sample N rows."""
    with open(path) as f:
        data = json.load(f)
    if sample_n is not None and sample_n < len(data):
        data = random.sample(data, sample_n)
    return data


def run_pipeline_on_dataset(rows: list[dict]) -> list[dict]:
    """Run retrieve_and_rerank + generate for each question, using metadata filters."""
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from generation.generate import generate
    from retrieval.pipeline import retrieve_and_rerank

    results = []
    for i, row in enumerate(rows, 1):
        question = row["question"]
        company = row.get("company")
        year = row.get("year")
        print(f"  [{i}/{len(rows)}] {question[:70]}...")

        nodes = retrieve_and_rerank(query=question, company=company, year=year)
        result = generate(query=question, context_nodes=nodes)

        results.append(
            {
                "question": question,
                "answer": result["answer"],
                "contexts": [n.text for n in nodes],
                "ground_truth": row["ground_truth"],
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS eval on FinLens pipeline.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N questions from the dataset (default: all).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        metavar="PATH",
        help=f"Path to QA dataset JSON (default: {DEFAULT_DATASET_PATH}).",
    )
    args = parser.parse_args()

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_recall, faithfulness

    thresholds = load_thresholds()
    print(f"Thresholds: {thresholds}")

    print(f"Loading dataset from {args.dataset} ...")
    rows = load_qa_dataset(args.dataset, sample_n=args.sample)
    print(f"Evaluating {len(rows)} question(s).")

    print("Running pipeline ...")
    evaluated_rows = run_pipeline_on_dataset(rows)

    ragas_ds = Dataset.from_list(evaluated_rows)

    print("\nScoring with RAGAS ...")
    results = evaluate(
        dataset=ragas_ds,
        metrics=[faithfulness, context_recall, answer_relevancy],
    )

    print("\n--- RAGAS Results ---")
    scores = results.to_pandas().mean().to_dict()
    failed = []
    for metric, threshold in thresholds.items():
        score = scores.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"  {metric:<25} {score:.4f}  (threshold={threshold})  [{status}]")
        if score < threshold:
            failed.append(metric)

    if failed:
        print(f"\nFAIL: {len(failed)} metric(s) below threshold: {failed}")
        sys.exit(1)
    else:
        print("\nAll metrics above threshold. PASS.")


if __name__ == "__main__":
    main()
