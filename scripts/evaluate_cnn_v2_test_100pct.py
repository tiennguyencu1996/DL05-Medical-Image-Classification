"""Single-pass Test evaluator for the DL-05 CNN V2 100% run.

Run from the project root:
    PYTHONPATH="$PWD" python scripts/evaluate_cnn_v2_test_100pct.py

This evaluator:
- verifies the 100% run, Epoch 33 checkpoint, and checkpoint SHA-256;
- verifies the Validation-only threshold artifact against Validation predictions;
- validates split metadata and patient / recorded-SHA separation;
- performs exactly one Test prediction pass;
- never selects a model or tunes a threshold using Test.

Do not rerun after the Test guard is created. A failed run after that point
must be treated as an exposed Test evaluation and investigated, not rerun.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

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

from src.data.input_pipeline import (
    IMAGE_SIZE,
    build_dataset,
    load_split_metadata,
    normalize_for_scratch,
    validate_all_splits,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "cnn_from_scratch_v2_100pct_full_20261121"
MODEL_NAME = "CNNFromScratch_V2_Controlled"
LEVEL = "100pct"
EXPECTED_BEST_EPOCH = 33
EXPECTED_VALIDATION_AUROC = 0.9959556706966033
EXPECTED_CHECKPOINT_SHA256 = (
    "7b0b066fcde0b053cc5580c5de2e9ccb"
    "83e7a453a64a39a138705de6115bb292"
)
EXPECTED_THRESHOLD = 0.5910010933876038
EXPECTED_THRESHOLD_HEX = "0x1.2e97b20000000p-1"
EXPECTED_TEST_COUNTS = {"NORMAL": 234, "PNEUMONIA": 387}
EXPECTED_TEST_IMAGES = 621
EXPECTED_TEST_PATIENTS = 431
EXPECTED_PARAMS = 1_208_161


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def prepare_batch(images, labels):
    """Mirror the CNN V2 input normalization: divide raw pixels by 255 once."""
    images, labels = normalize_for_scratch(images, labels)
    images = tf.cast(images, tf.float32)
    labels = tf.cast(tf.reshape(labels, (-1, 1)), tf.float32)
    tf.debugging.assert_all_finite(images, "Non-finite Test image")
    tf.debugging.assert_all_finite(labels, "Non-finite Test label")
    tf.debugging.assert_greater_equal(tf.reduce_min(images), 0.0)
    tf.debugging.assert_less_equal(tf.reduce_max(images), 1.0 + 1e-6)
    return images, labels


def verify_validation_threshold(
    validation_rows: list[dict[str, str]],
    validation_records: list[dict[str, str]],
    validation_labels: list[int],
    threshold_doc: dict,
) -> tuple[float, dict[str, int]]:
    """Verify the recorded threshold is the highest observed score meeting
    Validation Sensitivity >= 99%, using the stored Validation predictions.
    """
    required = {"relative_path", "y_true", "y_score", "y_pred_at_0.5"}
    if not validation_rows:
        raise RuntimeError("Validation predictions CSV is empty.")
    if not required.issubset(validation_rows[0].keys()):
        raise RuntimeError("Validation predictions CSV has unexpected columns.")

    expected_paths = [r["relative_path"] for r in validation_records]
    saved_paths = [r["relative_path"] for r in validation_rows]
    if saved_paths != expected_paths:
        raise RuntimeError("Validation prediction paths/order differ from metadata.")

    labels: list[int] = []
    scores: list[float] = []
    for i, row in enumerate(validation_rows, start=1):
        label = int(row["y_true"])
        score = float(row["y_score"])
        pred05 = int(row["y_pred_at_0.5"])
        if label not in (0, 1):
            raise RuntimeError(f"Invalid Validation label at row {i}: {label}")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError(f"Invalid Validation score at row {i}: {score}")
        if pred05 != int(score >= 0.5):
            raise RuntimeError(f"Saved Validation 0.5 prediction mismatch at row {i}")
        labels.append(label)
        scores.append(score)

    if labels != [int(x) for x in validation_labels]:
        raise RuntimeError("Saved Validation labels differ from split metadata.")
    counts = Counter(labels)
    if len(labels) != 1042 or counts != Counter({0: 270, 1: 772}):
        raise RuntimeError(f"Unexpected Validation counts: {dict(counts)}")

    selection = threshold_doc.get("threshold_selection", {})
    if selection.get("split") != "validation":
        raise RuntimeError("Threshold artifact does not identify Validation as source.")
    if selection.get("positive_class") != "PNEUMONIA":
        raise RuntimeError("Threshold artifact has an unexpected positive class.")
    if selection.get("positive_label") != 1:
        raise RuntimeError("Threshold artifact has an unexpected positive label.")
    if float(selection.get("target_sensitivity", -1)) != 0.99:
        raise RuntimeError("Threshold target is not the expected 99% Sensitivity.")

    saved_threshold = float(selection.get("threshold", float("nan")))
    if not math.isfinite(saved_threshold):
        raise RuntimeError("Threshold artifact contains an invalid threshold.")
    if saved_threshold.hex() != EXPECTED_THRESHOLD.hex():
        raise RuntimeError("Threshold artifact differs from the verified 100% threshold.")
    if saved_threshold.hex() != EXPECTED_THRESHOLD_HEX:
        raise RuntimeError("Threshold float hexadecimal representation mismatch.")

    candidates = sorted(set(scores))
    feasible = []
    for candidate in candidates:
        tp = sum(y == 1 and s >= candidate for y, s in zip(labels, scores))
        fn = sum(y == 1 and s < candidate for y, s in zip(labels, scores))
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        if sensitivity >= 0.99:
            feasible.append(candidate)
    if not feasible:
        raise RuntimeError("No Validation threshold meets Sensitivity >= 99%.")

    recomputed_threshold = max(feasible)
    if recomputed_threshold.hex() != saved_threshold.hex():
        raise RuntimeError(
            "Saved threshold is not the highest observed Validation score "
            "that meets Sensitivity >= 99%."
        )

    tn = fp = fn = tp = 0
    for y, score in zip(labels, scores):
        pred = int(score >= saved_threshold)
        if y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 0:
            fn += 1
        else:
            tp += 1

    expected_cm = [[254, 16], [7, 765]]
    if [[tn, fp], [fn, tp]] != expected_cm:
        raise RuntimeError(
            f"Validation confusion matrix mismatch: {[[tn, fp], [fn, tp]]}"
        )

    return saved_threshold, {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def main() -> int:
    run_dir = PROJECT_ROOT / "outputs" / "experiments" / RUN_NAME
    summary_path = run_dir / "training_summary.json"
    checkpoint_path = run_dir / "checkpoints" / "best.keras"
    threshold_path = run_dir / "evaluation" / "threshold_selection.json"
    validation_predictions_path = (
        run_dir / "evaluation" / "validation_best" / "predictions.csv"
    )
    test_csv = PROJECT_ROOT / "data" / "processed" / "splits" / "test_v3.csv"
    out_dir = run_dir / "evaluation" / "test_locked_threshold_100pct"
    guard_path = out_dir / "test_evaluation_started.json"
    predictions_path = out_dir / "predictions.csv"
    metrics_path = out_dir / "metrics.json"

    # Refuse to run if this output location has already been used.
    if out_dir.exists():
        raise RuntimeError(
            f"Test evaluation output directory already exists: {out_dir}. "
            "Do not rerun or overwrite Test evidence."
        )

    for path in (
        summary_path,
        checkpoint_path,
        threshold_path,
        validation_predictions_path,
        test_csv,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary.get("experiment_config", {})
    if summary.get("training_status") != "early_stopped":
        raise RuntimeError("Unexpected training status in summary.")
    if int(summary.get("best_epoch", -1)) != EXPECTED_BEST_EPOCH:
        raise RuntimeError(f"Unexpected best epoch: {summary.get('best_epoch')}")
    if config.get("model_name") != MODEL_NAME:
        raise RuntimeError("Unexpected model name in training summary.")
    if config.get("label_efficiency_level") != LEVEL:
        raise RuntimeError("This evaluator is restricted to the 100pct run.")
    if config.get("runtime_disable_meta_optimizer") is not True:
        raise RuntimeError("Meta Optimizer was not disabled in the training run.")
    if summary.get("test_split_loaded") is True:
        raise RuntimeError("Training summary indicates Test was already loaded.")
    if summary.get("test_split_evaluated") is True:
        raise RuntimeError("Training summary indicates Test was already evaluated.")

    val_auc = float(summary.get("best_validation_sklearn_auroc", float("nan")))
    if not math.isclose(
        val_auc, EXPECTED_VALIDATION_AUROC, rel_tol=0.0, abs_tol=1e-10
    ):
        raise RuntimeError(f"Unexpected best Validation AUROC: {val_auc:.12f}")

    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint SHA-256 differs from the verified Epoch 33 checkpoint.")
    if summary.get("best_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Checkpoint SHA-256 differs from training_summary.json.")

    threshold_doc = json.loads(threshold_path.read_text(encoding="utf-8"))
    if threshold_doc.get("experiment") != RUN_NAME:
        raise RuntimeError("Threshold artifact belongs to a different experiment.")
    if threshold_doc.get("model") != MODEL_NAME:
        raise RuntimeError("Threshold artifact belongs to a different model.")
    if int(threshold_doc.get("best_epoch", -1)) != EXPECTED_BEST_EPOCH:
        raise RuntimeError("Threshold artifact refers to a different best epoch.")
    if threshold_doc.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Threshold artifact checkpoint SHA-256 mismatch.")

    # Configure TensorFlow in the same way as the V2 verification/training run.
    options = tf.config.optimizer.get_experimental_options()
    options["disable_meta_optimizer"] = True
    tf.config.optimizer.set_experimental_options(options)

    # Load split metadata and confirm patient / recorded SHA separation.
    train_paths, train_labels, train_records = load_split_metadata("train")
    val_paths, val_labels, val_records = load_split_metadata("validation")
    test_paths, test_labels, test_records = load_split_metadata("test")
    validate_all_splits({
        "train": train_records,
        "validation": val_records,
        "test": test_records,
    })

    # Confirm the selected threshold against Validation predictions only.
    validation_rows = read_csv(validation_predictions_path)
    locked_threshold, validation_cm = verify_validation_threshold(
        validation_rows, val_records, val_labels, threshold_doc
    )

    # Confirm expected Test split inventory before inference.
    class_counts = Counter(row["label"] for row in test_records)
    patient_count = len({row["patient_id"] for row in test_records})
    if len(test_paths) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"Unexpected Test image count: {len(test_paths)}")
    if dict(class_counts) != EXPECTED_TEST_COUNTS:
        raise RuntimeError(f"Unexpected Test class counts: {dict(class_counts)}")
    if patient_count != EXPECTED_TEST_PATIENTS:
        raise RuntimeError(f"Unexpected Test patient count: {patient_count}")
    test_hashes = [row["sha256"] for row in test_records]
    if len(test_hashes) != len(set(test_hashes)):
        raise RuntimeError("Duplicate recorded SHA-256 values found within Test.")
    y_true = np.asarray(test_labels, dtype=np.int32)
    if set(np.unique(y_true).tolist()) != {0, 1}:
        raise RuntimeError("Test must contain both labels 0 and 1.")
    test_csv_hash = sha256_file(test_csv)

    # Load and validate the checkpoint before creating the Test inference guard.
    test_ds = build_dataset(test_paths, test_labels, training=False).map(
        prepare_batch, num_parallel_calls=1, deterministic=True
    )
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    if tuple(model.input_shape) != (None, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise RuntimeError(f"Unexpected input shape: {model.input_shape}")
    if tuple(model.output_shape) != (None, 1):
        raise RuntimeError(f"Unexpected output shape: {model.output_shape}")
    if model.count_params() != EXPECTED_PARAMS:
        raise RuntimeError(f"Unexpected parameter count: {model.count_params():,}")

    # Create the one-time guard immediately before the one Test prediction pass.
    out_dir.mkdir(parents=True, exist_ok=False)
    guard = {
        "run_name": RUN_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": checkpoint_hash,
        "threshold_artifact_sha256": sha256_file(threshold_path),
        "validation_predictions_sha256": sha256_file(validation_predictions_path),
        "test_split_csv_sha256": test_csv_hash,
        "best_epoch": EXPECTED_BEST_EPOCH,
        "locked_validation_threshold": format(locked_threshold, ".17g"),
        "threshold_hex": locked_threshold.hex(),
        "note": "Single Test inference pass initiated; do not rerun or tune using Test.",
    }
    with guard_path.open("x", encoding="utf-8") as f:
        json.dump(guard, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # The one and only prediction pass over Test.
    scores = np.asarray(model.predict(test_ds, verbose=0), dtype=np.float64).reshape(-1)
    if len(scores) != len(y_true):
        raise RuntimeError(f"Prediction count mismatch: {len(scores)} vs {len(y_true)}")
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise RuntimeError("Invalid sigmoid scores returned for Test.")

    y_pred = (scores >= locked_threshold).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    sensitivity = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0

    result = {
        "run_name": RUN_NAME,
        "model_name": MODEL_NAME,
        "label_efficiency_level": LEVEL,
        "best_epoch": EXPECTED_BEST_EPOCH,
        "best_validation_sklearn_auroc": val_auc,
        "checkpoint_sha256": checkpoint_hash,
        "threshold_artifact_sha256": sha256_file(threshold_path),
        "validation_predictions_sha256": sha256_file(validation_predictions_path),
        "test_split_csv_sha256": test_csv_hash,
        "test_images": int(len(y_true)),
        "test_patients": int(patient_count),
        "test_class_counts": {
            "NORMAL": int(class_counts["NORMAL"]),
            "PNEUMONIA": int(class_counts["PNEUMONIA"]),
        },
        "class_mapping": {"NORMAL": 0, "PNEUMONIA": 1},
        "positive_class": "PNEUMONIA",
        "locked_threshold": float(locked_threshold),
        "locked_threshold_hex": locked_threshold.hex(),
        "prediction_rule": "y_score >= locked_threshold",
        "threshold_selection_source": "Validation predictions only",
        "threshold_target_sensitivity": 0.99,
        "validation_confusion_matrix_at_threshold": [
            [validation_cm["TN"], validation_cm["FP"]],
            [validation_cm["FN"], validation_cm["TP"]],
        ],
        "threshold_tuned_on_test": False,
        "metrics": {
            "auroc": float(roc_auc_score(y_true, scores)),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": float(precision_score(
                y_true, y_pred, pos_label=1, zero_division=0
            )),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(
                y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
            )),
        },
        "confusion_matrix": {
            "label_order": ["NORMAL (0)", "PNEUMONIA (1)"],
            "matrix": cm.astype(int).tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "test_split_loaded": True,
        "test_split_evaluated": True,
        "evaluation_policy": (
            "One Test prediction pass with the operating threshold verified on "
            "Validation; no Test-based threshold tuning, model selection, or retraining."
        ),
    }

    with predictions_path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "relative_path",
            "patient_id",
            "sha256",
            "y_true",
            "y_score",
            "y_pred_at_locked_threshold",
            "locked_threshold",
        ])
        writer.writeheader()
        for row, truth, score, pred in zip(test_records, y_true, scores, y_pred):
            writer.writerow({
                "relative_path": row["relative_path"],
                "patient_id": row["patient_id"],
                "sha256": row["sha256"],
                "y_true": int(truth),
                "y_score": format(float(score), ".17g"),
                "y_pred_at_locked_threshold": int(pred),
                "locked_threshold": format(locked_threshold, ".17g"),
            })

    with metrics_path.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("=" * 72)
    print("DL-05 CNN V2 100% — ONE-TIME TEST EVALUATION")
    print("=" * 72)
    print("Run:", RUN_NAME)
    print("Best epoch:", EXPECTED_BEST_EPOCH)
    print("Checkpoint SHA256:", checkpoint_hash)
    print("Test images / patients:", len(y_true), "/", patient_count)
    print("Test class counts:", dict(class_counts))
    print("Locked Validation threshold:", format(locked_threshold, ".17g"))
    print()
    for key in ("auroc", "sensitivity", "specificity", "precision", "accuracy", "macro_f1"):
        print(f"{key:12s}: {result['metrics'][key]:.6f}")
    print()
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print("Predictions:", predictions_path)
    print("Metrics:", metrics_path)
    print("One-time guard:", guard_path)
    print("Threshold tuned on Test: NO")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
