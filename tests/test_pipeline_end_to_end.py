"""End-to-end check: scripts/run_pipeline.py produces a valid results.json.

Runs on the tiny synthetic dataset, so it only proves the code works -- not the
paper's numbers. The federated stage is skipped here because random Dirichlet
shards of such a tiny dataset can miss whole classes; it runs on the real data
in the CI "train" job.
"""
from __future__ import annotations

import json

from scripts import run_pipeline


def _no_nan(constant):
    raise ValueError(f"results.json contains invalid JSON constant {constant}")


def test_run_pipeline_writes_valid_results(synthetic_csv_gz, tmp_path):
    out = tmp_path / "results.json"
    code = run_pipeline.main(["--data", str(synthetic_csv_gz), "--output", str(out), "--no-federated"])
    assert code == 0

    results = json.loads(out.read_text(), parse_constant=_no_nan)

    for key in ["build", "settings", "paper", "dataset", "metrics", "paper_comparison",
                "efficiency", "confusion_matrix", "per_class", "feature_importance_rf",
                "shap_top_features", "stages"]:
        assert key in results, key

    names = [m["model"] for m in results["metrics"]]
    assert names == ["Random Forest", "CatBoost", "XGBoost", "LGBM", "Hybrid XGB-LGBM",
                     "Hybrid XGB-LGBM (calibrated)"]
    for row in results["metrics"]:
        for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]:
            assert 0.0 <= row[metric] <= 1.0

    assert results["dataset"]["raw_columns"] == 63
    assert results["dataset"]["n_features"] == 45
    assert len(results["dataset"]["classes"]) == 13
    matrix = results["confusion_matrix"]["matrix"]
    assert len(matrix) == 13 and all(len(r) == 13 for r in matrix)
    assert sum(map(sum, matrix)) == results["dataset"]["test_rows"]
    assert set(results["shap_top_features"]) == {"XGBoost", "LGBM"}
    assert results["federated"] is None
    assert results["settings"]["xgboost_params"]["n_estimators"] == 215


def test_quality_gate_fails_below_threshold(synthetic_csv_gz, tmp_path):
    out = tmp_path / "results.json"
    code = run_pipeline.main(["--data", str(synthetic_csv_gz), "--output", str(out),
                              "--no-federated", "--no-shap", "--no-calibration",
                              "--min-accuracy", "1.01"])
    assert code == 2
