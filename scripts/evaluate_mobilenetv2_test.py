#!/usr/bin/env python3
"""One-time, locked-threshold Test evaluator for DL-05 MobileNetV2 100%.

Run from the project root with the project's .venv active:
    PYTHONPATH="$PWD" python scripts/evaluate_mobilenetv2_test.py

Important:
- Uses the already-trained MobileNetV2 100% best.keras checkpoint.
- Verifies the Validation-only threshold from stored Validation predictions.
- Reuses the exact MobileNetV2 preprocessing/dataset builder from the training
  module; it does not use the CNN-from-scratch preprocessing pipeline.
- Creates a one-time guard before opening the Test split metadata. If anything
  fails after the guard is created, do not rerun automatically. Investigate
  the partial output and preserve the Test evidence.
- Performs one model.predict pass over Test, never tunes threshold on Test.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.input_pipeline import load_split_metadata, validate_all_splits
from src.training.train_mobilenetv2_label_efficiency import (
    IMAGE_SIZE,
    build_dataset as build_mobilenetv2_dataset,
    configure_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "mobilenetv2_100pct_imagenet_20261122"
MODEL_NAME = "MobileNetV2_ImageNet_Transfer_Learning"
LEVEL = "100pct"

RUN_DIR = PROJECT_ROOT / "outputs" / "experiments" / RUN_NAME
SUMMARY_PATH = RUN_DIR / "training_summary.json"
CHECKPOINT_PATH = RUN_DIR / "checkpoints" / "best.keras"
FINE_TUNE_CHECKPOINT_PATH = RUN_DIR / "checkpoints" / "best_fine_tune.keras"
THRESHOLD_PATH = (
    RUN_DIR / "evaluation" / "validation_threshold_locked" / "threshold_selection.json"
)
VALIDATION_PREDICTIONS_PATH = (
    RUN_DIR / "evaluation" / "validation_best" / "predictions.csv"
)
VALIDATION_LOCKED_PREDICTIONS_PATH = (
    RUN_DIR
    / "evaluation"
    / "validation_threshold_locked"
    / "predictions_at_locked_threshold.csv"
)

TEST_SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "test_v3.csv"
OUT_DIR = RUN_DIR / "evaluation" / "test_locked_threshold"
GUARD_PATH = OUT_DIR / "test_evaluation_started.json"
PREDICTIONS_PATH = OUT_DIR / "predictions.csv"
METRICS_PATH = OUT_DIR / "metrics.json"
COMPLETE_PATH = OUT_DIR / "test_evaluation_complete.json"

EXPECTED_BEST_VALIDATION_AUROC = 0.9979898292074457
EXPECTED_BEST_STAGE = "fine_tune"
EXPECTED_BEST_EPOCH = 12
EXPECTED_CHECKPOINT_SHA256 = (
    "37aec3769feef68c40ead0bd88dfe870"
    "5872d7c6b1cd5d0e4072b69fe581325c"
)
EXPECTED_FINE_TUNE_CHECKPOINT_SHA256 = EXPECTED_CHECKPOINT_SHA256
EXPECTED_THRESHOLD = 0.33037638664245605
EXPECTED_THRESHOLD_HEX = EXPECTED_THRESHOLD.hex()
EXPECTED_TARGET_SENSITIVITY = 0.99
EXPECTED_MODEL_PARAMS = 2_259_265
EXPECTED_TRAIN_MANIFEST_SHA256 = (
    "e6635a99a819421c95ad79396a28848d"
    "7defe970d64fdb8445192537be1148f0"
)
EXPECTED_SOURCE_TRAIN_SHA256 = (
    "a016befaa42bf8dcdfd95628859ca6edf"
    "0a81c553aec96edac2574e755200f93"
)
EXPECTED_VALIDATION_SHA256 = (
    "cacb82bb6f50d14a5f09cadec8a602db"
    "8f61dabbd45761e184e15d504ffe1b7c"
)
EXPECTED_TEST_IMAGES = 621
EXPECTED_TEST_PATIENTS = 431
EXPECTED_TEST_COUNTS = {"NORMAL": 234, "PNEUMONIA": 387}


# ---------------------------------------------------------------------------
# Small, deterministic I/O helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV has no header: {path}")
        return list(reader)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temp_path, path)


def atomic_write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    temp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    with temp_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_close(actual: float, expected: float, name: str, atol: float = 1e-12) -> None:
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=0.0, abs_tol=atol
    ):
        raise RuntimeError(f"Unexpected {name}: {actual!r}; expected {expected!r}")


# ---------------------------------------------------------------------------
# Verify the locked Validation threshold without using Test labels/scores
# ---------------------------------------------------------------------------

def verify_validation_artifacts(
    validation_records: list[dict[str, str]],
    validation_labels: list[int],
    summary: dict[str, Any],
    threshold_doc: dict[str, Any],
) -> tuple[float, dict[str, int], dict[str, str]]:
    rows = read_csv(VALIDATION_PREDICTIONS_PATH)
    locked_rows = read_csv(VALIDATION_LOCKED_PREDICTIONS_PATH)

    required_prediction_columns = {
        "relative_path", "y_true", "y_score", "y_pred_at_0.5"
    }
    require(bool(rows), "Validation predictions CSV is empty.")
    require(
        required_prediction_columns.issubset(rows[0]),
        "Validation predictions CSV has unexpected columns.",
    )
    require(
        len(rows) == len(validation_labels) == 1042,
        f"Unexpected Validation prediction count: {len(rows)}",
    )

    expected_paths = [record["relative_path"] for record in validation_records]
    saved_paths = [row["relative_path"] for row in rows]
    require(saved_paths == expected_paths, "Validation prediction order/path mismatch.")
    require(
        len(locked_rows) == len(rows),
        "Locked Validation predictions have a different row count.",
    )
    require(
        [row.get("relative_path") for row in locked_rows] == expected_paths,
        "Locked Validation prediction order/path mismatch.",
    )

    labels: list[int] = []
    scores: list[float] = []
    for index, row in enumerate(rows):
        label = int(row["y_true"])
        score = float(row["y_score"])
        pred05 = int(row["y_pred_at_0.5"])
        require(label in (0, 1), f"Invalid Validation label at row {index + 1}.")
        require(
            math.isfinite(score) and 0.0 <= score <= 1.0,
            f"Invalid Validation score at row {index + 1}.",
        )
        require(
            pred05 == int(score >= 0.5),
            f"Validation 0.5 prediction mismatch at row {index + 1}.",
        )
        labels.append(label)
        scores.append(score)

        locked_row = locked_rows[index]
        require(int(locked_row["y_true"]) == label, "Locked Validation label mismatch.")
        require(
            float(locked_row["y_score"]).hex() == score.hex(),
            "Locked Validation score differs from best-checkpoint predictions.",
        )
        require(
            int(locked_row["y_pred_at_0.5"]) == pred05,
            "Locked Validation 0.5 prediction mismatch.",
        )

    require(labels == [int(value) for value in validation_labels],
            "Validation prediction labels differ from split metadata.")
    class_counts = Counter(labels)
    require(
        class_counts == Counter({0: 270, 1: 772}),
        f"Unexpected Validation class counts: {dict(class_counts)}",
    )

    require(
        threshold_doc.get("status") == "locked_on_validation_before_test",
        "Threshold artifact has an unexpected status.",
    )
    require(
        threshold_doc.get("split_used_for_selection") == "validation",
        "Threshold artifact does not identify Validation as selection source.",
    )
    require(
        float(threshold_doc.get("target_sensitivity", -1.0))
        == EXPECTED_TARGET_SENSITIVITY,
        "Threshold target is not 99% Sensitivity.",
    )
    require(threshold_doc.get("threshold_tuned_on_test") is False,
            "Threshold artifact says it was tuned on Test.")
    require(threshold_doc.get("test_split_loaded") is False,
            "Threshold artifact indicates Test was already loaded.")
    require(threshold_doc.get("test_split_evaluated") is False,
            "Threshold artifact indicates Test was already evaluated.")

    saved_threshold = float(threshold_doc.get("threshold", float("nan")))
    require(math.isfinite(saved_threshold), "Invalid threshold in threshold artifact.")
    require(
        saved_threshold.hex() == EXPECTED_THRESHOLD_HEX,
        "Saved threshold differs from the verified MobileNetV2 100% threshold.",
    )

    # Recompute the exact selection rule from Validation scores only.
    selected_threshold: float | None = None
    for candidate in sorted(set(scores), reverse=True):
        candidate_pred = np.asarray(scores, dtype=np.float64) >= candidate
        candidate_sensitivity = float(
            np.sum((np.asarray(labels) == 1) & candidate_pred)
            / np.sum(np.asarray(labels) == 1)
        )
        if candidate_sensitivity >= EXPECTED_TARGET_SENSITIVITY:
            selected_threshold = float(candidate)
            break

    require(selected_threshold is not None, "No Validation threshold meets target Sensitivity.")
    require(
        selected_threshold.hex() == saved_threshold.hex(),
        "Stored threshold is not the highest observed Validation score meeting target.",
    )

    y_true = np.asarray(labels, dtype=np.int32)
    y_pred = (np.asarray(scores, dtype=np.float64) >= saved_threshold).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    confusion = [[tn, fp], [fn, tp]]
    require(
        confusion == [[258, 12], [7, 765]],
        f"Recomputed locked Validation confusion matrix mismatch: {confusion}",
    )
    require(
        confusion == threshold_doc.get("confusion_matrix"),
        "Threshold artifact confusion matrix does not match Validation predictions.",
    )

    for index, row in enumerate(locked_rows):
        expected_prediction = int(y_pred[index])
        require(
            int(row["y_pred_at_locked_threshold"]) == expected_prediction,
            f"Locked Validation prediction mismatch at row {index + 1}.",
        )

    sensitivity = float(tp / (tp + fn))
    specificity = float(tn / (tn + fp))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0))

    for key, computed in {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }.items():
        verify_close(float(threshold_doc[key]), computed, f"Validation {key}")

    summary_threshold = summary.get("validation_operating_threshold", {})
    require(
        float(summary_threshold.get("threshold", float("nan"))).hex()
        == saved_threshold.hex(),
        "Training summary threshold differs from threshold artifact.",
    )

    return saved_threshold, {"TN": tn, "FP": fp, "FN": fn, "TP": tp}, {
        "validation_predictions_sha256": sha256_file(VALIDATION_PREDICTIONS_PATH),
        "validation_locked_predictions_sha256": sha256_file(
            VALIDATION_LOCKED_PREDICTIONS_PATH
        ),
        "threshold_artifact_sha256": sha256_file(THRESHOLD_PATH),
    }


def main() -> int:
    print("=" * 76)
    print("DL-05 MobileNetV2 100% — ONE-TIME TEST EVALUATION")
    print("=" * 76)

    # Refuse repeated attempts, including attempts after an interrupted run.
    if OUT_DIR.exists():
        raise RuntimeError(
            f"Test evaluation output directory already exists: {OUT_DIR}. "
            "Do not rerun or overwrite Test evidence; investigate the existing directory."
        )

    # Validate non-Test artifacts first. No Test CSV contents or Test images are read here.
    for path in (
        SUMMARY_PATH,
        CHECKPOINT_PATH,
        FINE_TUNE_CHECKPOINT_PATH,
        THRESHOLD_PATH,
        VALIDATION_PREDICTIONS_PATH,
        VALIDATION_LOCKED_PREDICTIONS_PATH,
        TEST_SPLIT_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    threshold_doc = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    config = summary.get("experiment_config", {})

    require(summary.get("training_status") == "completed", "Training run is not completed.")
    require(config.get("experiment_name") == RUN_NAME, "Unexpected experiment name.")
    require(config.get("model_name") == MODEL_NAME, "Unexpected model name.")
    require(config.get("label_efficiency_level") == LEVEL, "Not the 100pct experiment.")
    require(summary.get("best_stage") == EXPECTED_BEST_STAGE, "Unexpected best stage.")
    require(int(summary.get("best_epoch_within_stage", -1)) == EXPECTED_BEST_EPOCH,
            "Unexpected best Fine-tuning epoch.")
    require(summary.get("best_checkpoint_path") == str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "Training summary points to a different checkpoint.")
    verify_close(float(summary.get("best_validation_sklearn_auroc", float("nan"))),
                 EXPECTED_BEST_VALIDATION_AUROC, "best Validation sklearn AUROC")

    for flag_name in ("test_split_loaded", "test_split_evaluated"):
        require(summary.get(flag_name) is False, f"Training summary {flag_name} is not false.")
        require(config.get(flag_name) is False, f"Experiment config {flag_name} is not false.")

    checkpoint_hash = sha256_file(CHECKPOINT_PATH)
    fine_tune_hash = sha256_file(FINE_TUNE_CHECKPOINT_PATH)
    require(checkpoint_hash == EXPECTED_CHECKPOINT_SHA256,
            "best.keras SHA256 differs from the verified checkpoint.")
    require(fine_tune_hash == EXPECTED_FINE_TUNE_CHECKPOINT_SHA256,
            "best_fine_tune.keras SHA256 differs from the verified checkpoint.")
    require(checkpoint_hash == fine_tune_hash,
            "best.keras and best_fine_tune.keras are not byte-identical.")
    require(summary.get("best_checkpoint_sha256") == checkpoint_hash,
            "Training summary checkpoint SHA256 mismatch.")
    require(summary.get("best_validation_metrics", {}).get("checkpoint_sha256") == checkpoint_hash,
            "Best Validation metrics checkpoint SHA256 mismatch.")

    manifest_path = PROJECT_ROOT / config["train_manifest"]
    source_train_path = PROJECT_ROOT / "data/processed/splits/train_v3.csv"
    validation_path = PROJECT_ROOT / "data/processed/splits/validation_v3.csv"
    require(sha256_file(manifest_path) == EXPECTED_TRAIN_MANIFEST_SHA256 == config.get("train_manifest_sha256"),
            "Train manifest SHA256 mismatch.")
    require(sha256_file(source_train_path) == EXPECTED_SOURCE_TRAIN_SHA256 == config.get("source_train_sha256"),
            "Source Train CSV SHA256 mismatch.")
    require(sha256_file(validation_path) == EXPECTED_VALIDATION_SHA256 == config.get("validation_sha256"),
            "Validation CSV SHA256 mismatch.")

    # Load only Train and Validation metadata before the one-time Test guard.
    train_paths, train_labels, train_records = load_split_metadata("train")
    val_paths, val_labels, val_records = load_split_metadata("validation")
    require(len(train_paths) == 4164 and len(train_labels) == 4164,
            "Unexpected Train inventory.")
    require(len(val_paths) == 1042 and len(val_labels) == 1042,
            "Unexpected Validation inventory.")

    threshold, validation_cm, validation_hashes = verify_validation_artifacts(
        validation_records=val_records,
        validation_labels=val_labels,
        summary=summary,
        threshold_doc=threshold_doc,
    )

    # Configure runtime in the same way as the training pipeline and load/check
    # the checkpoint before creating the one-time Test guard.
    runtime_info = configure_runtime()
    model = tf.keras.models.load_model(CHECKPOINT_PATH, compile=False)
    require(tuple(model.input_shape) == (None, IMAGE_SIZE, IMAGE_SIZE, 3),
            f"Unexpected model input shape: {model.input_shape}")
    require(tuple(model.output_shape) == (None, 1),
            f"Unexpected model output shape: {model.output_shape}")
    require(model.count_params() == EXPECTED_MODEL_PARAMS,
            f"Unexpected model parameter count: {model.count_params():,}")

    # One-time guard is created BEFORE reading Test metadata. Any failure after
    # this point must be investigated; do not rerun this evaluator.
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    guard: dict[str, Any] = {
        "run_name": RUN_NAME,
        "model_name": MODEL_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "started_before_test_metadata_load",
        "checkpoint_sha256": checkpoint_hash,
        "threshold_artifact_sha256": validation_hashes["threshold_artifact_sha256"],
        "validation_predictions_sha256": validation_hashes["validation_predictions_sha256"],
        "validation_locked_predictions_sha256": validation_hashes["validation_locked_predictions_sha256"],
        "locked_validation_threshold": format(threshold, ".17g"),
        "threshold_hex": threshold.hex(),
        "test_split_csv_path": str(TEST_SPLIT_PATH.relative_to(PROJECT_ROOT)),
        "test_split_loaded": False,
        "test_split_evaluated": False,
        "note": "One-time Test evaluation started. Do not rerun after this guard exists.",
    }
    with GUARD_PATH.open("x", encoding="utf-8") as handle:
        json.dump(guard, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    # Test data is first loaded after the one-time guard has been persisted.
    test_paths, test_labels, test_records = load_split_metadata("test")
    validate_all_splits({
        "train": train_records,
        "validation": val_records,
        "test": test_records,
    })

    test_class_counts = Counter(row["label"] for row in test_records)
    test_patient_count = len({row["patient_id"] for row in test_records})
    require(len(test_paths) == EXPECTED_TEST_IMAGES,
            f"Unexpected Test image count: {len(test_paths)}")
    require(dict(test_class_counts) == EXPECTED_TEST_COUNTS,
            f"Unexpected Test class counts: {dict(test_class_counts)}")
    require(test_patient_count == EXPECTED_TEST_PATIENTS,
            f"Unexpected Test patient count: {test_patient_count}")
    require(set(test_labels) == {0, 1}, "Test must contain both labels 0 and 1.")

    test_hashes = [row["sha256"] for row in test_records]
    require(all(test_hashes), "Missing recorded SHA256 in Test metadata.")
    require(len(test_hashes) == len(set(test_hashes)),
            "Duplicate recorded SHA256 values found within Test.")
    test_csv_hash = sha256_file(TEST_SPLIT_PATH)

    guard.update({
        "state": "test_metadata_validated_inference_pending",
        "test_split_csv_sha256": test_csv_hash,
        "test_images": len(test_paths),
        "test_patients": test_patient_count,
        "test_class_counts": {
            "NORMAL": int(test_class_counts["NORMAL"]),
            "PNEUMONIA": int(test_class_counts["PNEUMONIA"]),
        },
        "train_validation_test_patient_and_sha_separation_validated": True,
        "test_split_loaded": True,
        "test_split_evaluated": False,
    })
    atomic_write_json(GUARD_PATH, guard)

    # Exact same deterministic resize/pad + MobileNetV2 preprocess_input builder
    # used during training. Dataset iteration begins only with the one predict call.
    test_dataset = build_mobilenetv2_dataset(
        test_paths,
        test_labels,
        training=False,
    )
    guard["state"] = "test_inference_started"
    guard["inference_started_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(GUARD_PATH, guard)

    # The one and only Test inference pass.
    scores = np.asarray(model.predict(test_dataset, verbose=1), dtype=np.float64).reshape(-1)
    y_true = np.asarray(test_labels, dtype=np.int32)
    require(len(scores) == len(y_true),
            f"Prediction count mismatch: {len(scores)} vs {len(y_true)}")
    require(np.isfinite(scores).all() and np.all(scores >= 0.0) and np.all(scores <= 1.0),
            "Invalid sigmoid scores returned for Test.")

    y_pred = (scores >= threshold).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    sensitivity = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    metrics = {
        "auroc": float(roc_auc_score(y_true, scores)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
    }

    prediction_rows = []
    for record, truth, score, prediction in zip(test_records, y_true, scores, y_pred):
        prediction_rows.append({
            "relative_path": record["relative_path"],
            "patient_id": record["patient_id"],
            "sha256": record["sha256"],
            "y_true": int(truth),
            "y_score": format(float(score), ".17g"),
            "y_pred_at_locked_threshold": int(prediction),
            "locked_threshold": format(threshold, ".17g"),
        })

    result: dict[str, Any] = {
        "run_name": RUN_NAME,
        "model_name": MODEL_NAME,
        "label_efficiency_level": LEVEL,
        "training_status_at_evaluation": summary["training_status"],
        "best_stage": EXPECTED_BEST_STAGE,
        "best_epoch_within_stage": EXPECTED_BEST_EPOCH,
        "best_validation_sklearn_auroc": EXPECTED_BEST_VALIDATION_AUROC,
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "threshold_artifact_sha256": validation_hashes["threshold_artifact_sha256"],
        "validation_predictions_sha256": validation_hashes["validation_predictions_sha256"],
        "validation_locked_predictions_sha256": validation_hashes["validation_locked_predictions_sha256"],
        "test_split_csv_sha256": test_csv_hash,
        "test_images": int(len(y_true)),
        "test_patients": int(test_patient_count),
        "test_class_counts": {
            "NORMAL": int(test_class_counts["NORMAL"]),
            "PNEUMONIA": int(test_class_counts["PNEUMONIA"]),
        },
        "class_mapping": {"NORMAL": 0, "PNEUMONIA": 1},
        "positive_class": "PNEUMONIA",
        "locked_threshold": float(threshold),
        "locked_threshold_hex": threshold.hex(),
        "prediction_rule": "y_score >= locked_threshold",
        "threshold_selection_source": "Validation predictions only",
        "threshold_target_sensitivity": EXPECTED_TARGET_SENSITIVITY,
        "validation_confusion_matrix_at_threshold": [
            [validation_cm["TN"], validation_cm["FP"]],
            [validation_cm["FN"], validation_cm["TP"]],
        ],
        "threshold_tuned_on_test": False,
        "test_split_used_for_training": False,
        "test_split_used_for_checkpoint_selection": False,
        "metrics": metrics,
        "confusion_matrix": {
            "label_order": ["NORMAL (0)", "PNEUMONIA (1)"],
            "matrix": cm.astype(int).tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "runtime_configuration": runtime_info,
        "test_split_loaded": True,
        "test_split_evaluated": True,
        "evaluation_policy": (
            "One Test prediction pass with the operating threshold verified on "
            "Validation; no Test-based threshold tuning, model selection, or retraining."
        ),
    }

    atomic_write_csv(
        PREDICTIONS_PATH,
        [
            "relative_path",
            "patient_id",
            "sha256",
            "y_true",
            "y_score",
            "y_pred_at_locked_threshold",
            "locked_threshold",
        ],
        prediction_rows,
    )
    atomic_write_json(METRICS_PATH, result)

    guard["state"] = "completed"
    guard["test_split_evaluated"] = True
    guard["completed_utc"] = datetime.now(timezone.utc).isoformat()
    guard["metrics_path"] = str(METRICS_PATH.relative_to(PROJECT_ROOT))
    guard["predictions_path"] = str(PREDICTIONS_PATH.relative_to(PROJECT_ROOT))
    atomic_write_json(COMPLETE_PATH, {
        "status": "completed",
        "completed_utc": guard["completed_utc"],
        "run_name": RUN_NAME,
        "metrics_path": guard["metrics_path"],
        "predictions_path": guard["predictions_path"],
        "test_split_evaluated": True,
    })
    # Replace the guard atomically with its final completed state.
    guard_temp = GUARD_PATH.with_name(GUARD_PATH.name + ".final.tmp")
    with guard_temp.open("x", encoding="utf-8") as handle:
        json.dump(guard, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(guard_temp, GUARD_PATH)

    print("\n" + "=" * 76)
    print("DL-05 MobileNetV2 100% — ONE-TIME TEST EVALUATION COMPLETE")
    print("=" * 76)
    print("Run:", RUN_NAME)
    print("Checkpoint SHA256:", checkpoint_hash)
    print("Test images / patients:", len(y_true), "/", test_patient_count)
    print("Test class counts:", dict(test_class_counts))
    print("Locked Validation threshold:", format(threshold, ".17g"))
    print()
    for key in ("auroc", "sensitivity", "specificity", "precision", "accuracy", "macro_f1"):
        print(f"{key:12s}: {metrics[key]:.6f}")
    print("\nConfusion matrix [NORMAL, PNEUMONIA]:")
    print(cm)
    print("\nMetrics:", METRICS_PATH)
    print("Predictions:", PREDICTIONS_PATH)
    print("Test evaluation guard:", GUARD_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
