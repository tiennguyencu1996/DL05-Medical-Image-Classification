"""Single-pass locked-threshold Test evaluator for the DL-05 CNN V2 10% run.

Run from the project root:
    PYTHONPATH="$PWD" python scripts/evaluate_cnn_v2_test.py

The script does not train, select a model, or tune a threshold on Test.
It creates a one-time guard before the first Test prediction pass.
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
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from src.data.input_pipeline import (
    IMAGE_SIZE, build_dataset, load_split_metadata,
    normalize_for_scratch, validate_all_splits,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "cnn_from_scratch_v2_010pct_full_20261119"
MODEL_NAME = "CNNFromScratch_V2_Controlled"
LEVEL = "010pct"

# Locked using Validation only. Do not change after looking at Test.
LOCKED_THRESHOLD = 0.98173642158508301
THRESHOLD_RULE = (
    "Validation only: maximize specificity subject to sensitivity >= 0.85; "
    "tie-break by higher sensitivity, then higher threshold."
)

EXPECTED_CHECKPOINT_SHA256 = (
    "fb5648e0127b0282b4b32fbe44b2b4eb232a42b5fcf7d9553af97cbd2e7b8dd9"
)
EXPECTED_BEST_EPOCH = 43
EXPECTED_VALIDATION_AUROC = 0.9844439646900787
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


def prepare_batch(images, labels):
    """Same preprocessing as the V2 trainer: divide raw pixels by 255 once."""
    images, labels = normalize_for_scratch(images, labels)
    images = tf.cast(images, tf.float32)
    labels = tf.cast(tf.reshape(labels, (-1, 1)), tf.float32)
    tf.debugging.assert_all_finite(images, "Non-finite Test image")
    tf.debugging.assert_all_finite(labels, "Non-finite Test label")
    tf.debugging.assert_greater_equal(tf.reduce_min(images), 0.0)
    tf.debugging.assert_less_equal(tf.reduce_max(images), 1.0 + 1e-6)
    return images, labels


def main() -> int:
    run_dir = ROOT / "outputs" / "experiments" / RUN_NAME
    summary_path = run_dir / "training_summary.json"
    checkpoint_path = run_dir / "checkpoints" / "best.keras"
    test_csv = ROOT / "data" / "processed" / "splits" / "test_v3.csv"
    out_dir = run_dir / "evaluation" / "test_locked_threshold"
    guard_path = out_dir / "test_evaluation_started.json"
    predictions_path = out_dir / "predictions.csv"
    metrics_path = out_dir / "metrics.json"

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if guard_path.exists() or predictions_path.exists() or metrics_path.exists():
        raise RuntimeError(
            "A Test evaluation marker/output already exists. This is a "
            "single-pass evaluator; do not rerun it after Test has been exposed."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary.get("experiment_config", {})

    if summary.get("test_split_loaded") is True:
        raise RuntimeError("Training summary indicates Test was already loaded.")
    if summary.get("test_split_evaluated") is True:
        raise RuntimeError("Training summary indicates Test was already evaluated.")
    if config.get("model_name") != MODEL_NAME:
        raise RuntimeError("Unexpected model name in training summary.")
    if config.get("label_efficiency_level") != LEVEL:
        raise RuntimeError("This script is restricted to the 010pct run.")
    if config.get("runtime_disable_meta_optimizer") is not True:
        raise RuntimeError("Meta Optimizer was not disabled in the training run.")
    if int(summary.get("best_epoch", -1)) != EXPECTED_BEST_EPOCH:
        raise RuntimeError(f"Unexpected best epoch: {summary.get('best_epoch')}")

    val_auc = float(summary.get("best_validation_sklearn_auroc", float("nan")))
    if not math.isclose(val_auc, EXPECTED_VALIDATION_AUROC,
                        rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError(f"Unexpected best Validation AUROC: {val_auc:.12f}")

    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint SHA256 differs from the verified V2 checkpoint.")
    if summary.get("best_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Checkpoint SHA256 differs from training_summary.json.")

    # Match the V2 trainer and its checkpoint-verification script.
    options = tf.config.optimizer.get_experimental_options()
    options["disable_meta_optimizer"] = True
    tf.config.optimizer.set_experimental_options(options)

    # Read split metadata and revalidate patient/SHA separation.
    train_paths, train_labels, train_records = load_split_metadata("train")
    val_paths, val_labels, val_records = load_split_metadata("validation")
    test_paths, test_labels, test_records = load_split_metadata("test")
    validate_all_splits({
        "train": train_records,
        "validation": val_records,
        "test": test_records,
    })

    class_counts = Counter(row["label"] for row in test_records)
    patient_count = len({row["patient_id"] for row in test_records})
    if len(test_paths) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"Unexpected Test image count: {len(test_paths)}")
    if dict(class_counts) != EXPECTED_TEST_COUNTS:
        raise RuntimeError(f"Unexpected Test class counts: {dict(class_counts)}")
    if patient_count != EXPECTED_TEST_PATIENTS:
        raise RuntimeError(f"Unexpected Test patient count: {patient_count}")
    hashes = [row["sha256"] for row in test_records]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("Duplicate image SHA256 values found within Test.")
    y_true = np.asarray(test_labels, dtype=np.int32)
    if set(np.unique(y_true).tolist()) != {0, 1}:
        raise RuntimeError("Test must contain both labels 0 and 1.")

    test_csv_hash = sha256_file(test_csv)

    # Same deterministic decode/resize/batch pipeline and normalization as V2.
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

    out_dir.mkdir(parents=True, exist_ok=True)

    # Mark the start before inference. A later failure must not lead to an
    # unintentional second look at Test results.
    guard = {
        "run_name": RUN_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": checkpoint_hash,
        "test_split_csv_sha256": test_csv_hash,
        "locked_threshold": LOCKED_THRESHOLD,
        "threshold_source": THRESHOLD_RULE,
        "note": "Test inference initiated; do not rerun or tune using Test.",
    }
    with guard_path.open("x", encoding="utf-8") as f:
        json.dump(guard, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # The one and only prediction pass over Test.
    scores = np.asarray(model.predict(test_ds, verbose=0),
                        dtype=np.float64).reshape(-1)
    if len(scores) != len(y_true):
        raise RuntimeError(f"Prediction count mismatch: {len(scores)} vs {len(y_true)}")
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise RuntimeError("Invalid sigmoid scores returned for Test.")

    # Threshold was selected on Validation, not on this Test set.
    y_pred = (scores >= LOCKED_THRESHOLD).astype(np.int32)
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
        "test_split_csv_sha256": test_csv_hash,
        "test_images": int(len(y_true)),
        "test_patients": int(patient_count),
        "test_class_counts": {
            "NORMAL": int(class_counts["NORMAL"]),
            "PNEUMONIA": int(class_counts["PNEUMONIA"]),
        },
        "class_mapping": {"NORMAL": 0, "PNEUMONIA": 1},
        "positive_class": "PNEUMONIA",
        "locked_threshold": LOCKED_THRESHOLD,
        "prediction_rule": "y_score >= locked_threshold",
        "threshold_source": THRESHOLD_RULE,
        "threshold_tuned_on_test": False,
        "metrics": {
            "auroc": float(roc_auc_score(y_true, scores)),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": float(precision_score(
                y_true, y_pred, pos_label=1, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(
                y_true, y_pred, labels=[0, 1], average="macro",
                zero_division=0)),
        },
        "confusion_matrix": {
            "label_order": ["NORMAL (0)", "PNEUMONIA (1)"],
            "matrix": cm.astype(int).tolist(),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        },
        "test_split_loaded": True,
        "test_split_evaluated": True,
        "evaluation_policy": (
            "One Test pass with the operating threshold locked on Validation; "
            "no Test-based threshold tuning, model selection, or retraining."
        ),
    }

    # Preserve fixed manifest row order and include patient/image traceability.
    with predictions_path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "relative_path", "patient_id", "sha256", "y_true", "y_score",
            "y_pred_at_locked_threshold", "locked_threshold",
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
                "locked_threshold": format(LOCKED_THRESHOLD, ".17g"),
            })

    metrics_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("DL-05 CNN V2 — ONE-TIME TEST EVALUATION")
    print("=" * 72)
    print("Run:", RUN_NAME)
    print("Best epoch:", EXPECTED_BEST_EPOCH)
    print("Checkpoint SHA256:", checkpoint_hash)
    print("Test images / patients:", len(y_true), "/", patient_count)
    print("Test class counts:", dict(class_counts))
    print("Locked Validation threshold:", format(LOCKED_THRESHOLD, ".17g"))
    print()
    for key in ("auroc", "sensitivity", "specificity", "precision",
                "accuracy", "macro_f1"):
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
