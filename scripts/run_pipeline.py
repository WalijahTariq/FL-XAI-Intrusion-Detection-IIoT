"""Run the full FL-XAI pipeline and export the results for the frontend page.

This is the single entry point used by CI (GitHub Actions) and by you locally:

    python scripts/run_pipeline.py

It trains every model with the settings in ``src/config.py``, evaluates them on
the held-out test set, and writes ``frontend/public/results.json``. The
frontend page reads that file, so whatever the code produced is exactly what
the page shows -- together with the commit and settings that produced it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402

from src import config, data_loader, evaluation, federated  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

DEFAULT_OUTPUT = ROOT / "frontend" / "public" / "results.json"
PACKAGES = ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost",
            "lightgbm", "catboost", "shap", "optuna"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clean(value):
    """Make a value JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, Path):
        return str(value)
    return value


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                             text=True, check=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def build_build_info(duration_s: float) -> dict:
    """Commit + run information so the page shows which change produced it."""
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    sha = os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_seconds": round(duration_s, 1),
        "source": "github-actions" if os.environ.get("GITHUB_ACTIONS") == "true" else "local",
        "commit_sha": sha,
        "commit_message": _git("log", "-1", "--pretty=%B"),
        "commit_author": _git("log", "-1", "--pretty=%an"),
        "commit_time": _git("log", "-1", "--pretty=%cI"),
        "branch": os.environ.get("GITHUB_REF_NAME") or _git("rev-parse", "--abbrev-ref", "HEAD"),
        "repository": repo,
        "commit_url": f"{server}/{repo}/commit/{sha}" if server and repo and sha else None,
        "run_url": f"{server}/{repo}/actions/runs/{run_id}" if server and repo and run_id else None,
        "python": platform.python_version(),
        "packages": {p: _version(p) for p in PACKAGES},
    }


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def build_settings() -> dict:
    """The configuration values used for this run (from src/config.py)."""
    return {
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "stratify": config.STRATIFY,
        "smote_k_neighbors": config.SMOTE_K_NEIGHBORS,
        "dropped_classes": config.DROPPED_CLASSES,
        "zero_day_source_classes": config.ZERO_DAY_SOURCE_CLASSES,
        "zero_day_label": config.ZERO_DAY_LABEL,
        "dropped_columns": config.DROP_COLUMNS,
        "xgboost_params": config.XGB_PARAMS,
        "lightgbm_params": config.LGBM_PARAMS,
        "calibration_method": config.CALIBRATION_METHOD,
        "calibration_cv": config.CALIBRATION_CV,
        "fl_clients": config.FL_N_CLIENTS,
        "fl_non_iid_alpha": config.FL_NON_IID_ALPHA,
    }


def _metric_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for name, row in df.iterrows():
        rows.append({"model": str(name), **{k: float(v) for k, v in row.items()}})
    return rows


def build_results(result, dataset_path: Path, duration_s: float, stages: dict) -> dict:
    """Assemble everything the frontend needs from a finished PipelineResult."""
    class_names = list(result.class_names)
    report = result.preprocessed.report

    # Raw class counts straight from the CSV (only the label column is read).
    raw_labels = pd.read_csv(dataset_path, usecols=[config.ATTACK_TYPE_COLUMN],
                             low_memory=False)[config.ATTACK_TYPE_COLUMN]
    raw_columns = len(pd.read_csv(dataset_path, nrows=0).columns)

    y_all = np.concatenate([result.y_train, result.y_test])
    values, counts = np.unique(y_all, return_counts=True)
    after_smote = {class_names[int(v)]: int(c) for v, c in zip(values, counts)}

    # Best available final model: calibrated hybrid if it was trained.
    if result.calibrated_hybrid is not None:
        final_name, final_model = "Hybrid XGB-LGBM (calibrated)", result.calibrated_hybrid
    else:
        final_name, final_model = "Hybrid XGB-LGBM", result.trained_models["Hybrid XGB-LGBM"]
    y_pred = final_model.predict(result.X_test)
    cm = evaluation.build_confusion_matrix(result.y_test, y_pred,
                                           labels=list(range(len(class_names))))
    per_class = classification_report(result.y_test, y_pred,
                                      labels=list(range(len(class_names))),
                                      target_names=class_names, output_dict=True,
                                      zero_division=0)

    final_metrics = result.metrics_df.loc[final_name].to_dict()
    paper = config.BENCHMARK_TARGETS
    comparison = [
        {"metric": k, "paper": float(paper[k]), "reproduction": float(final_metrics[k]),
         "difference": float(final_metrics[k]) - float(paper[k])}
        for k in config.METRIC_NAMES if k in paper
    ]

    efficiency = []
    for name, row in result.efficiency_df.iterrows():
        efficiency.append({"model": str(name), **{k: row[k] for k in row.index}})

    rf_importance = result.importance_df.head(20)[["feature", "importance"]].to_dict("records")

    shap_section = {}
    for name, payload in (result.shap or {}).items():
        shap_section[name] = payload["importance"].head(20).to_dict("records")

    fed_section = None
    if result.federated:
        fed = result.federated
        fed_pred = federated.federated_predict(fed["client_models"], result.X_test)
        imps = fed["client_importances"]
        pairs = [spearmanr(imps[i], imps[j]).statistic
                 for i, j in combinations(range(len(imps)), 2)]
        fed_section = {
            "n_clients": len(fed["client_indices"]),
            "non_iid_alpha": config.FL_NON_IID_ALPHA,
            "shard_sizes": fed["shard_sizes"],
            "metrics": evaluation.compute_metrics(result.y_test, fed_pred),
            "mean_pairwise_spearman": float(np.mean(pairs)) if pairs else None,
        }

    return _clean({
        "schema_version": 1,
        "build": build_build_info(duration_s),
        "settings": build_settings(),
        "stages": stages,
        "paper": {
            "title": ("A Federated and Explainable Machine Learning Framework for Robust "
                      "Intrusion Detection and Network Security Enhancement in Industrial "
                      "Internet of Things (IIoT) Environments"),
            "venue": "ICECTE 2026",
            "benchmark_targets": paper,
        },
        "dataset": {
            "file": dataset_path.name,
            "raw_rows": int(len(raw_labels)),
            "raw_columns": raw_columns,
            "raw_class_counts": raw_labels.value_counts().to_dict(),
            "rows_after_cleaning": report["n_rows_after_cleaning"],
            "class_counts_after_cleaning": report["classes_after_cleaning"],
            "class_counts_after_smote": after_smote,
            "n_features": len(result.feature_names),
            "feature_names": list(result.feature_names),
            "train_rows": int(len(result.y_train)),
            "test_rows": int(len(result.y_test)),
            "classes": class_names,
        },
        "metrics": _metric_rows(result.metrics_df),
        "final_model": final_name,
        "paper_comparison": comparison,
        "efficiency": efficiency,
        "confusion_matrix": {"model": final_name, "labels": class_names, "matrix": cm},
        "per_class": [
            {"class": c, "precision": per_class[c]["precision"], "recall": per_class[c]["recall"],
             "f1": per_class[c]["f1-score"], "support": per_class[c]["support"]}
            for c in class_names
        ],
        "feature_importance_rf": rf_importance,
        "shap_top_features": shap_section,
        "federated": fed_section,
    })


def write_step_summary(results: dict) -> None:
    """Show the metrics table on the GitHub Actions run page."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    lines = ["## Model results (test set)", "",
             "| Model | Accuracy | Balanced acc. | Precision | Recall | F1 |",
             "|---|---|---|---|---|---|"]
    for r in results["metrics"]:
        lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} | "
                     f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
    lines += ["", f"Paper target accuracy: {results['paper']['benchmark_targets']['accuracy']}"]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=None,
                        help="Dataset path (default: data/ML-EdgeIIoT-dataset.csv or .csv.gz)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to write results.json (default: frontend/public/results.json)")
    parser.add_argument("--no-calibration", action="store_true", help="Skip isotonic calibration")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP explainability")
    parser.add_argument("--no-federated", action="store_true", help="Skip the federated simulation")
    parser.add_argument("--save-models", action="store_true", help="Also save trained models to outputs/models")
    parser.add_argument("--min-accuracy", type=float, default=None,
                        help="Quality gate: exit with an error if the final model's accuracy is below this")
    args = parser.parse_args(argv)

    dataset_path = data_loader.resolve_dataset_path(args.data)
    print(f"Dataset: {dataset_path}")

    stages = {"calibration": not args.no_calibration, "shap": not args.no_shap,
              "federated": not args.no_federated}
    start = time.perf_counter()
    result = Pipeline(dataset_path=dataset_path).run(
        calibrated=stages["calibration"], shap=stages["shap"],
        federated=stages["federated"], save_models=args.save_models,
    )
    duration = time.perf_counter() - start

    results = build_results(result, dataset_path, duration, stages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nFinished in {duration:.1f}s. Results written to {args.output}")
    print(result.metrics_df.round(4).to_string())
    write_step_summary(results)

    if args.min_accuracy is not None:
        final_acc = next(r["accuracy"] for r in results["metrics"] if r["model"] == results["final_model"])
        if final_acc < args.min_accuracy:
            print(f"QUALITY GATE FAILED: {results['final_model']} accuracy {final_acc:.4f} "
                  f"< required {args.min_accuracy}")
            return 2
        print(f"Quality gate passed: accuracy {final_acc:.4f} >= {args.min_accuracy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
