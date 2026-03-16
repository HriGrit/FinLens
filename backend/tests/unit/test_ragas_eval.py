from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RAGAS_EVAL_PATH = REPO_ROOT / "eval" / "ragas_eval.py"


def _load_ragas_eval_module():
    spec = importlib.util.spec_from_file_location("ragas_eval_under_test", RAGAS_EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_qa_dataset_sampling_is_reproducible(tmp_path: Path) -> None:
    module = _load_ragas_eval_module()
    dataset_path = tmp_path / "qa.json"
    rows = [{"question": f"q{i}", "ground_truth": f"a{i}"} for i in range(10)]
    dataset_path.write_text(json.dumps(rows), encoding="utf-8")

    sample_a = module.load_qa_dataset(dataset_path, sample_n=4, sample_seed=123)
    sample_b = module.load_qa_dataset(dataset_path, sample_n=4, sample_seed=123)
    sample_c = module.load_qa_dataset(dataset_path, sample_n=4, sample_seed=321)

    assert sample_a == sample_b
    assert sample_a != sample_c


def test_run_pipeline_on_dataset_isolates_per_row_failures() -> None:
    module = _load_ragas_eval_module()
    rows = [
        {"question": "ok", "ground_truth": "g1"},
        {"question": "boom", "ground_truth": "g2"},
    ]

    def _retrieve(**kwargs):
        if kwargs["query"] == "boom":
            raise RuntimeError("retrieval failed")
        return SimpleNamespace(nodes=[SimpleNamespace(text="ctx")])

    def _generate(**kwargs):
        return {"answer": "answer"}

    success, failed = module.run_pipeline_on_dataset(
        rows,
        retrieve_and_rerank_fn=_retrieve,
        generate_fn=_generate,
    )

    assert len(success) == 1
    assert success[0]["question"] == "ok"
    assert len(failed) == 1
    assert failed[0]["question"] == "boom"
    assert "retrieval failed" in failed[0]["error"]


def test_validate_metric_threshold_sync_rejects_mismatch() -> None:
    module = _load_ragas_eval_module()
    thresholds = {
        "faithfulness": 0.8,
        "context_recall": 0.75,
        "answer_relevancy": 0.8,
    }

    with pytest.raises(ValueError, match="Threshold config mismatch"):
        module.validate_metric_threshold_sync(thresholds)
