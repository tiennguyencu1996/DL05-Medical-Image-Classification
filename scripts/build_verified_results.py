#!/usr/bin/env python3
"""Rebuild DL-05 result tables from saved predictions and project metadata.

Run from the project root:
    python scripts/build_verified_results.py

Or place this file at scripts/build_verified_results.py and run from any directory.

The script does NOT train models or run inference. It recomputes evaluation metrics
from saved prediction CSVs, checks locked-threshold labels using float32 semantics,
checks prediction inventories against the saved split CSVs, and exports CSV tables.
Requires: numpy, scikit-learn.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def rel(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


def metric_bundle(y_true: list[int], y_score: list[float], y_pred: list[int]) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "n_images": len(y_true),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def check(condition: bool, name: str, detail: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
    if not condition:
        raise AssertionError(f"AUDIT CHECK FAILED: {name}: {detail}")


def load_predictions(path: Path, y_pred_col: str, threshold: float | None = None) -> tuple[list[dict[str, str]], list[int], list[float], list[int]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No prediction rows in {path}")
    y_true = [int(r["y_true"]) for r in rows]
    y_score = [float(r["y_score"]) for r in rows]
    y_pred = [int(r[y_pred_col]) for r in rows]
    if threshold is not None:
        recomputed = (np.asarray(y_score, dtype=np.float32) >= np.float32(threshold)).astype(int).tolist()
        mismatch = sum(a != b for a, b in zip(recomputed, y_pred))
        if mismatch:
            raise AssertionError(f"Float32 locked-threshold mismatch in {path}: {mismatch} rows")
    return rows, y_true, y_score, y_pred


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None,
                        help="Project root. Defaults to parent of this script's scripts/ directory.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output folder. Defaults to PROJECT_ROOT/outputs/reports/verified_results.")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    root = args.project_root.resolve() if args.project_root else script_path.parent.parent
    out_dir = args.output_dir.resolve() if args.output_dir else root / "outputs" / "reports" / "verified_results"
    checks: list[dict[str, Any]] = []

    # Dataset split inventories
    split_paths = {
        "Train": root / "data/processed/splits/train_v3.csv",
        "Validation": root / "data/processed/splits/validation_v3.csv",
        "Test": root / "data/processed/splits/test_v3.csv",
    }
    splits = {name: read_csv(path) for name, path in split_paths.items()}
    split_summary = read_json(root / "data/processed/splits/split_summary_v3.json")
    dedup_summary = read_json(root / "data/processed/deduplication_summary_v3.json")
    assignments = read_csv(root / "data/processed/splits/split_assignments_v3.csv")
    metadata_dedup = read_csv(root / "data/processed/metadata_deduplicated_v3.csv")

    split_rows = [r for rows in splits.values() for r in rows]
    split_paths_all = [r["relative_path"] for r in split_rows]
    assignment_paths = [r["relative_path"] for r in assignments]
    metadata_paths = [r["relative_path"] for r in metadata_dedup]
    check(len(split_paths_all) == len(set(split_paths_all)) == 5827,
          "Split relative paths are unique", f"{len(split_paths_all)} rows; {len(set(split_paths_all))} unique", checks)
    check(set(split_paths_all) == set(assignment_paths) == set(metadata_paths),
          "Split assignments cover deduplicated metadata exactly",
          f"splits={len(split_paths_all)}, assignments={len(assignment_paths)}, deduplicated metadata={len(metadata_paths)}", checks)

    patient_sets = {name: {r["patient_id"] for r in rows} for name, rows in splits.items()}
    sha_sets = {name: {r["sha256"] for r in rows} for name, rows in splits.items()}
    for a, b in [("Train", "Validation"), ("Train", "Test"), ("Validation", "Test")]:
        check(not (patient_sets[a] & patient_sets[b]), f"{a}-{b} patient overlap is zero",
              f"overlap={len(patient_sets[a] & patient_sets[b])}", checks)
        check(not (sha_sets[a] & sha_sets[b]), f"{a}-{b} SHA-256 overlap is zero",
              f"overlap={len(sha_sets[a] & sha_sets[b])}", checks)

    dedup_ok = (
        len(read_csv(root / "data/processed/metadata_v3.csv")) == dedup_summary["original_total_images"]
        and len(metadata_dedup) == dedup_summary["deduplicated_total_images"]
        and len(read_csv(root / "data/processed/duplicate_exclusions_v3.csv")) == dedup_summary["excluded_duplicate_files"]
        and dedup_summary["cross_split_duplicate_groups"] == 0
    )
    check(dedup_ok, "Deduplication counts match summary",
          f"original={dedup_summary['original_total_images']}; retained={len(metadata_dedup)}; excluded={dedup_summary['excluded_duplicate_files']}; groups={dedup_summary['duplicate_groups']}", checks)

    split_rows_out = []
    for name, rows in splits.items():
        normals = sum(int(r["binary_label"]) == 0 for r in rows)
        pneumonias = sum(int(r["binary_label"]) == 1 for r in rows)
        patients = len({r["patient_id"] for r in rows})
        split_rows_out.append({
            "split": name, "images": len(rows), "patients": patients,
            "normal_0": normals, "pneumonia_1": pneumonias,
            "pneumonia_share": pneumonias / len(rows),
            "relative_source": rel(root, split_paths[name]),
        })

    # Helper to validate prediction inventory order against split CSV.
    def check_inventory(pred_rows: list[dict[str, str]], split_name: str, source_name: str) -> None:
        expected = splits[split_name]
        if "relative_path" in pred_rows[0]:
            observed_paths = [r["relative_path"] for r in pred_rows]
            expected_paths = [r["relative_path"] for r in expected]
            check(observed_paths == expected_paths, f"{source_name} matches {split_name} inventory/order",
                  f"prediction rows={len(observed_paths)}; split rows={len(expected_paths)}", checks)
            check([int(r["y_true"]) for r in pred_rows] == [int(r["binary_label"]) for r in expected],
                  f"{source_name} labels match {split_name}", "Row-level labels compared in order", checks)
        else:
            # Baseline prediction artifact uses split-local row_index; verify the same ordered labels.
            check(len(pred_rows) == len(expected), f"{source_name} row count matches {split_name}",
                  f"prediction rows={len(pred_rows)}; split rows={len(expected)}", checks)
            if "row_index" in pred_rows[0]:
                check([int(r["row_index"]) for r in pred_rows] == list(range(len(expected))),
                      f"{source_name} row_index is contiguous", "Expected 0..N-1", checks)
            check([int(r["y_true"]) for r in pred_rows] == [int(r["binary_label"]) for r in expected],
                  f"{source_name} labels match {split_name}", "Row-level labels compared in order", checks)

    # Test metrics for required baselines and all DL experiments with Test outputs.
    test_rows: list[dict[str, Any]] = []
    baseline_pred_path = root / "outputs/metrics/predictions.csv"
    baseline_rows = read_csv(baseline_pred_path)
    baseline_specs = [
        ("Majority Classifier", "Majority Class", "majority class prediction; no threshold"),
        ("HOG + Logistic Regression", "HOG Logistic Regression", "Validation Sensitivity >= 0.85; maximize Specificity"),
    ]
    baseline_results = read_json(root / "outputs/metrics/baseline_results_v3.json")
    for display, source_model, policy in baseline_specs:
        rows = [r for r in baseline_rows if r["model"] == source_model and r["split"] == "test"]
        check_inventory(rows, "Test", f"{display} predictions")
        y_true = [int(r["y_true"]) for r in rows]
        y_score = [float(r["y_score"]) for r in rows]
        y_pred = [int(r["y_pred"]) for r in rows]
        m = metric_bundle(y_true, y_score, y_pred)
        baseline_threshold = ""
        if source_model == "HOG Logistic Regression":
            baseline_threshold = baseline_results["hog_logistic_regression"]["threshold_selection"]["threshold"]
            policy = "Validation-only threshold; Sensitivity >= 0.85, maximize Specificity"
        test_rows.append({"model": display, "training_fraction": "N/A", "threshold": baseline_threshold,
                          "threshold_policy": policy, **m,
                          "prediction_source": rel(root, baseline_pred_path),
                          "metrics_source": rel(root, root / "outputs/metrics/baseline_results_v3.json")})

    dl_specs = [
        ("CNN V2", "10%", "cnn_from_scratch_v2_010pct_full_20261119", "test_locked_threshold", "metrics.json", "predictions.csv"),
        ("CNN V2", "25%", "cnn_from_scratch_v2_025pct_full_20261120", "test_locked_threshold", "metrics.json", "predictions.csv"),
        ("CNN V2", "50%", "cnn_from_scratch_v2_050pct_full_20261121", "test_locked_threshold", "metrics.json", "predictions.csv"),
        ("CNN V2", "100%", "cnn_from_scratch_v2_100pct_full_20261121", "test_locked_threshold_100pct", "metrics.json", "predictions.csv"),
        ("MobileNetV2", "100%", "mobilenetv2_100pct_imagenet_20261122", "test_locked_threshold", "metrics.json", "predictions.csv"),
    ]
    for model, fraction, run, eval_dir, metrics_file, preds_file in dl_specs:
        eval_path = root / "outputs/experiments" / run / "evaluation" / eval_dir
        metrics_path = eval_path / metrics_file
        pred_path = eval_path / preds_file
        raw = read_json(metrics_path)
        threshold = float(raw.get("locked_threshold", raw.get("threshold")))
        pred_rows, y_true, y_score, y_pred = load_predictions(pred_path, "y_pred_at_locked_threshold", threshold)
        check_inventory(pred_rows, "Test", f"{model} {fraction} Test predictions")
        m = metric_bundle(y_true, y_score, y_pred)
        check(raw.get("test_split_loaded") is True and raw.get("test_split_evaluated") is True,
              f"{model} {fraction} Test evaluation flags", "Saved metrics JSON declares Test split loaded/evaluated", checks)
        if raw.get("test_split_csv_sha256"):
            import hashlib
            actual_hash = hashlib.sha256((root / "data/processed/splits/test_v3.csv").read_bytes()).hexdigest()
            check(actual_hash == raw["test_split_csv_sha256"], f"{model} {fraction} Test split hash matches",
                  f"saved={raw['test_split_csv_sha256']}; actual={actual_hash}", checks)
        metrics_json = raw.get("metrics", raw)
        cm = raw.get("confusion_matrix", {})
        saved_cm = cm.get("matrix") if isinstance(cm, dict) else cm
        check(saved_cm == [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
              f"{model} {fraction} confusion matrix matches predictions",
              f"recomputed={[[m['tn'],m['fp']],[m['fn'],m['tp']] ]}; saved={saved_cm}", checks)
        for k in ["auroc", "sensitivity", "specificity", "accuracy", "macro_f1"]:
            if k in metrics_json:
                check(math.isclose(m[k], float(metrics_json[k]), rel_tol=1e-10, abs_tol=1e-10),
                      f"{model} {fraction} {k} matches metrics.json",
                      f"recomputed={m[k]:.12f}; saved={float(metrics_json[k]):.12f}", checks)
        policy = raw.get("threshold_source", raw.get("threshold_selection_source", ""))
        target = raw.get("threshold_target_sensitivity")
        if target is not None and (not policy or policy == "Validation predictions only"):
            policy = f"Validation-only; target Sensitivity >= {float(target):.2f}; highest observed score meeting target"
        elif not policy:
            policy = "Validation-only threshold; see linked threshold-selection artifact"
        test_rows.append({"model": model, "training_fraction": fraction, "threshold": threshold,
                          "threshold_policy": policy, **m,
                          "prediction_source": rel(root, pred_path),
                          "metrics_source": rel(root, metrics_path)})

    test_rows.sort(key=lambda r: ({"Majority Classifier": 0, "HOG + Logistic Regression": 1, "CNN V2": 2, "MobileNetV2": 3}[r["model"]],
                                  {"N/A": 0, "10%": 1, "25%": 2, "50%": 3, "100%": 4}.get(r["training_fraction"], 0)))

    # Validation AUROC label-efficiency results from stored best-checkpoint predictions.
    le_rows: list[dict[str, Any]] = []
    for model, run_prefix in [("CNN V2", "cnn_from_scratch_v2"), ("MobileNetV2", "mobilenetv2")]:
        for fraction, level in [("10%", "010pct"), ("25%", "025pct"), ("50%", "050pct"), ("100%", "100pct")]:
            matches = sorted((root / "outputs/experiments").glob(f"{run_prefix}_{level}_*/evaluation/validation_best/predictions.csv"))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected one Validation predictions CSV for {model} {fraction}; got {matches}")
            pred_path = matches[0]
            rows = read_csv(pred_path)
            check_inventory(rows, "Validation", f"{model} {fraction} Validation predictions")
            y_true = [int(r["y_true"]) for r in rows]
            y_score = [float(r["y_score"]) for r in rows]
            auc = float(roc_auc_score(y_true, y_score))
            manifest_path = root / "data/processed/label_efficiency/manifests_v1" / f"train_{level}.csv"
            manifest_rows = read_csv(manifest_path)
            le_rows.append({"model": model, "training_fraction": fraction,
                            "actual_image_fraction": len(manifest_rows) / len(splits["Train"]),
                            "train_images": len(manifest_rows),
                            "validation_images": len(rows), "validation_auroc": auc,
                            "prediction_source": rel(root, pred_path)})

    le_rows.sort(key=lambda r: (0 if r["model"] == "CNN V2" else 1,
                                {"10%": 1, "25%": 2, "50%": 3, "100%": 4}[r["training_fraction"]]))
    le_wide = []
    for fraction in ["10%", "25%", "50%", "100%"]:
        cnn = next(r for r in le_rows if r["model"] == "CNN V2" and r["training_fraction"] == fraction)
        mob = next(r for r in le_rows if r["model"] == "MobileNetV2" and r["training_fraction"] == fraction)
        le_wide.append({"training_fraction": fraction, "actual_image_fraction": cnn["actual_image_fraction"],
                        "train_images": cnn["train_images"],
                        "cnn_v2_validation_auroc": cnn["validation_auroc"],
                        "mobilenetv2_validation_auroc": mob["validation_auroc"],
                        "mobilenet_minus_cnn": mob["validation_auroc"] - cnn["validation_auroc"],
                        "cnn_predictions_source": cnn["prediction_source"],
                        "mobilenet_predictions_source": mob["prediction_source"]})

    # Thresholds chosen only using Validation.
    threshold_rows = [
        {"model": "Majority Classifier", "training_fraction": "N/A", "threshold": "",
         "target_sensitivity": "N/A", "selection_policy": "Always predict PNEUMONIA", "evidence": rel(root, root / "outputs/metrics/baseline_results_v3.json")},
        {"model": "HOG + Logistic Regression", "training_fraction": "100%", "threshold": baseline_results["hog_logistic_regression"]["threshold_selection"]["threshold"],
         "target_sensitivity": baseline_results["hog_logistic_regression"]["threshold_selection"]["target_sensitivity"],
         "selection_policy": "Maximize Validation Specificity subject to Sensitivity >= target",
         "evidence": rel(root, root / "outputs/metrics/baseline_results_v3.json")},
    ]
    threshold_spec_paths = [
        ("CNN V2", "10%", "outputs/experiments/cnn_from_scratch_v2_010pct_full_20261119/evaluation/validation_threshold_selection/operating_threshold.json", "selected_threshold", "constraint"),
        ("CNN V2", "25%", "outputs/experiments/cnn_from_scratch_v2_025pct_full_20261120/evaluation/validation_threshold_locked/threshold_selection.json", "locked_validation_metrics.threshold", "selection_policy.minimum_sensitivity"),
        ("CNN V2", "50%", "outputs/experiments/cnn_from_scratch_v2_050pct_full_20261121/evaluation/validation_threshold_locked/threshold_selection.json", "locked_validation_metrics.threshold", "selection_policy.minimum_sensitivity"),
        ("CNN V2", "100%", "outputs/experiments/cnn_from_scratch_v2_100pct_full_20261121/evaluation/threshold_selection.json", "threshold_selection.threshold", "threshold_selection.target_sensitivity"),
        ("MobileNetV2", "10%", "outputs/experiments/mobilenetv2_010pct_imagenet_20261122/evaluation/validation_threshold_locked/threshold_selection.json", "threshold", "target_sensitivity"),
        ("MobileNetV2", "25%", "outputs/experiments/mobilenetv2_025pct_imagenet_20261122/evaluation/validation_threshold_locked/threshold_selection.json", "threshold", "target_sensitivity"),
        ("MobileNetV2", "50%", "outputs/experiments/mobilenetv2_050pct_imagenet_20261122/evaluation/validation_threshold_locked/threshold_selection.json", "threshold", "target_sensitivity"),
        ("MobileNetV2", "100%", "outputs/experiments/mobilenetv2_100pct_imagenet_20261122/evaluation/validation_threshold_locked/threshold_selection.json", "threshold", "target_sensitivity"),
    ]
    for model, frac, file_rel, t_key, s_key in threshold_spec_paths:
        obj = read_json(root / file_rel)
        def get_nested(d: dict[str, Any], key: str) -> Any:
            for part in key.split("."):
                d = d[part]
            return d
        target_value = get_nested(obj, s_key)
        if isinstance(target_value, str) and "0.85" in target_value:
            target_value = 0.85
        elif isinstance(target_value, str) and "0.99" in target_value:
            target_value = 0.99
        policy_obj = obj.get("selection_policy")
        if policy_obj:
            policy_text = json.dumps(policy_obj, ensure_ascii=False)
        elif "selection_method" in obj:
            policy_text = (f"{obj.get('selection_method')}; objective={obj.get('objective', 'not stated')}; "
                           f"constraint={obj.get('constraint', 'not stated')}; tie-break={obj.get('tie_break', 'not stated')}")
        elif "threshold_selection" in obj and isinstance(obj["threshold_selection"], dict):
            policy_text = obj["threshold_selection"].get("rule", json.dumps(obj["threshold_selection"], ensure_ascii=False))
        elif "threshold_selection_method" in obj:
            policy_text = f"{obj['threshold_selection_method']}; target sensitivity={target_value}"
        else:
            policy_text = "Validation-only threshold selection; see evidence artifact"
        threshold_rows.append({"model": model, "training_fraction": frac,
                               "threshold": get_nested(obj, t_key),
                               "target_sensitivity": target_value,
                               "selection_policy": policy_text,
                               "evidence": rel(root, root / file_rel)})

    # Manifests audit (identical Train subset inventories across model families).
    manifest_dir = root / "data/processed/label_efficiency/manifests_v1"
    manifest_paths = {frac: manifest_dir / f"train_{lvl}.csv" for frac, lvl in [("10%", "010pct"), ("25%", "025pct"), ("50%", "050pct"), ("100%", "100pct")]}
    manifests = {k: read_csv(v) for k, v in manifest_paths.items()}
    train_paths = {r["relative_path"] for r in splits["Train"]}
    manifest_sets = {k: {r["relative_path"] for r in rows} for k, rows in manifests.items()}
    for frac, paths in manifest_sets.items():
        check(paths <= train_paths, f"{frac} manifest is subset of Train", f"{len(paths)} manifest paths", checks)
    check(manifest_sets["10%"] <= manifest_sets["25%"] <= manifest_sets["50%"] <= manifest_sets["100%"],
          "Label-efficiency manifests are nested", "10% subset 25% subset 50% subset 100%", checks)

    # Output CSV tables
    write_csv(out_dir / "verified_test_metrics.csv", test_rows,
              ["model", "training_fraction", "n_images", "threshold", "threshold_policy", "auroc", "sensitivity", "specificity", "precision", "accuracy", "macro_f1", "tn", "fp", "fn", "tp", "prediction_source", "metrics_source"])
    write_csv(out_dir / "verified_label_efficiency.csv", le_wide,
              ["training_fraction", "actual_image_fraction", "train_images", "cnn_v2_validation_auroc", "mobilenetv2_validation_auroc", "mobilenet_minus_cnn", "cnn_predictions_source", "mobilenet_predictions_source"])
    write_csv(out_dir / "verified_thresholds.csv", threshold_rows,
              ["model", "training_fraction", "threshold", "target_sensitivity", "selection_policy", "evidence"])
    write_csv(out_dir / "verified_dataset_splits.csv", split_rows_out,
              ["split", "images", "patients", "normal_0", "pneumonia_1", "pneumonia_share", "relative_source"])
    write_csv(out_dir / "audit_checks.csv", checks, ["check", "status", "detail"])

    summary = {
        "project": "DL-05 Medical Image Classification",
        "operation": "recompute metrics from saved predictions; no training/inference",
        "project_root": str(root),
        "output_directory": str(out_dir),
        "test_metric_rows": len(test_rows),
        "label_efficiency_rows": len(le_wide),
        "threshold_rows": len(threshold_rows),
        "audit_checks_passed": sum(c["status"] == "PASS" for c in checks),
        "audit_checks_total": len(checks),
        "all_checks_passed": all(c["status"] == "PASS" for c in checks),
        "notes": [
            "AUROC and threshold-dependent metrics are recomputed from stored prediction CSVs.",
            "For deep-learning Test rows, saved predicted labels are checked against float32 score >= float32 locked threshold.",
            "The Test metrics are results on the same fixed Test split and should be interpreted with the report's repeated-evaluation limitation.",
            "This script does not verify or regenerate trained model checkpoints."
        ]
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote results to: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
