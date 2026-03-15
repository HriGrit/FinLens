"""
ragas_eval.py — Evaluate the FinLens pipeline with RAGAS on ingested 3M 10-K documents.

Usage (from backend/):
  uv run python ../eval/ragas_eval.py               # full eval (all questions)
  uv run python ../eval/ragas_eval.py --sample 5    # quick smoke test
  uv run python ../eval/ragas_eval.py --sample 5 --sample-seed 7

Outputs a results table, persists partial/final run artifacts, and exits non-zero
if there are query failures or metrics below threshold.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_PATH = Path(__file__).parent / "thresholds.yaml"
DEFAULT_DATASET_PATH = Path(__file__).parent / "qa_dataset.json"
DEFAULT_RESULTS_PATH = Path(__file__).parent / "ragas_eval_results.json"
ENABLED_METRIC_NAMES = (
    "faithfulness",
    "context_recall",
    "answer_relevancy",
    "answer_correctness",
)


def load_thresholds() -> dict[str, float]:
    with open(THRESHOLDS_PATH) as f:
        return yaml.safe_load(f)


def load_qa_dataset(
    path: Path,
    sample_n: int | None = None,
    sample_seed: int = 42,
) -> list[dict]:
    """Load QA pairs from JSON, optionally random-sample N rows."""
    with open(path) as f:
        data = json.load(f)
    if sample_n is not None and sample_n < len(data):
        data = random.Random(sample_seed).sample(data, sample_n)
    return data


def _load_pipeline_callables():
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from generation.generate import generate
    from retrieval.pipeline import retrieve_and_rerank

    return retrieve_and_rerank, generate


def run_pipeline_on_dataset(
    rows: list[dict],
    *,
    retrieve_and_rerank_fn=None,
    generate_fn=None,
) -> tuple[list[dict], list[dict]]:
    """Run retrieve_and_rerank + generate with per-row failure isolation."""
    if retrieve_and_rerank_fn is None or generate_fn is None:
        retrieve_and_rerank_fn, generate_fn = _load_pipeline_callables()

    successful_rows: list[dict] = []
    failed_rows: list[dict] = []
    for i, row in enumerate(rows, 1):
        question = row["question"]
        company = row.get("company")
        year = row.get("year")
        print(f"  [{i}/{len(rows)}] {question[:70]}...")
        try:
            retrieval = retrieve_and_rerank_fn(query=question, company=company, year=year)
            nodes = retrieval.nodes
            if not nodes:
                raise ValueError("No retrieval context nodes returned.")
            result = generate_fn(query=question, context_nodes=nodes)
        except Exception as exc:
            failed_rows.append(
                {
                    "question": question,
                    "company": company,
                    "year": year,
                    "ground_truth": row["ground_truth"],
                    "error": str(exc),
                }
            )
            continue

        successful_rows.append(
            {
                "question": question,
                "answer": result["answer"],
                "contexts": [n.text for n in nodes],
                "ground_truth": row["ground_truth"],
            }
        )
    return successful_rows, failed_rows


def validate_metric_threshold_sync(thresholds: dict[str, float]) -> None:
    expected = set(ENABLED_METRIC_NAMES)
    actual = set(thresholds)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"Threshold config mismatch. missing={missing} extra={extra}. "
            f"Expected exactly: {sorted(expected)}"
        )


def _resolve_metrics():
    from ragas import metrics as ragas_metrics

    metrics = []
    for name in ENABLED_METRIC_NAMES:
        metric = getattr(ragas_metrics, name, None)
        if metric is None:
            raise ValueError(f"RAGAS metric '{name}' is not available in the current environment.")
        metrics.append(metric)
    return metrics


def persist_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
        "--sample-seed",
        type=int,
        default=42,
        metavar="N",
        help="Seed used for reproducible --sample selection (default: 42).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        metavar="PATH",
        help=f"Path to QA dataset JSON (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        metavar="PATH",
        help=f"Path to write partial/final eval results JSON (default: {DEFAULT_RESULTS_PATH}).",
    )
    args = parser.parse_args()

    from datasets import Dataset
    from ragas import evaluate

    thresholds = load_thresholds()
    validate_metric_threshold_sync(thresholds)
    metrics = _resolve_metrics()
    print(f"Thresholds: {thresholds}")

    print(f"Loading dataset from {args.dataset} ...")
    rows = load_qa_dataset(args.dataset, sample_n=args.sample, sample_seed=args.sample_seed)
    print(f"Evaluating {len(rows)} question(s).")

    print("Running pipeline ...")
    evaluated_rows, failed_rows = run_pipeline_on_dataset(rows)

    payload: dict[str, Any] = {
        "dataset": str(args.dataset),
        "sample_size": len(rows),
        "sample_seed": args.sample_seed,
        "successful_rows": len(evaluated_rows),
        "failed_rows": failed_rows,
        "scores": {},
        "failed_metrics": [],
    }
    persist_results(args.output, payload)

    if not evaluated_rows:
        print("FAIL: no successful rows to score. See persisted output for per-row failures.")
        sys.exit(1)

    ragas_ds = Dataset.from_list(evaluated_rows)
    print("\nScoring with RAGAS ...")
    results = evaluate(dataset=ragas_ds, metrics=metrics)

    print("\n--- RAGAS Results ---")
    scores = results.to_pandas().mean().to_dict()
    failed_metrics = []
    for metric, threshold in thresholds.items():
        score = scores.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"  {metric:<25} {score:.4f}  (threshold={threshold})  [{status}]")
        if score < threshold:
            failed_metrics.append(metric)

    payload["scores"] = scores
    payload["failed_metrics"] = failed_metrics
    persist_results(args.output, payload)

    if failed_rows:
        print(f"\nWARN: {len(failed_rows)} query row(s) failed during pipeline execution.")
    if failed_metrics:
        print(f"\nFAIL: {len(failed_metrics)} metric(s) below threshold: {failed_metrics}")
        sys.exit(1)
    if failed_rows:
        print("FAIL: query failures detected. See persisted output for details.")
        sys.exit(1)
    print("\nAll metrics above threshold. PASS.")


if __name__ == "__main__":
    main()
