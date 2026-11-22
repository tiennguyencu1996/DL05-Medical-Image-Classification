"""Re-evaluate the existing CNN V2 10% pilot checkpoint on Validation.

This is an inference-only correction. It does NOT retrain the model, modify
its checkpoint, overwrite the original pilot predictions, or load Test.
The Validation dataset is normalized exactly once before inference.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    build_dataset,
    load_split_metadata,
    normalize_for_scratch,
)

PILOT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "cnn_from_scratch_v2_010pct_pilot_20261119"
)
CHECKPOINT_PATH = PILOT_DIR / "checkpoints" / "pilot_epoch_001.keras"
SUMMARY_PATH = PILOT_DIR / "training_summary.json"
OLD_PREDICTIONS_PATH = (
    PILOT_DIR / "evaluation" / "validation" / "predictions.csv"
)
OUTPUT_DIR = (
    PILOT_DIR
    / "evaluation"
    / "validation_corrected_single_normalization"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "1fd16048fa3a91fafba3fc48e8292ae9a3ace68fd149b73769d5dded9c516be6"
)
DIAGNOSTIC_THRESHOLD = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_runtime() -> dict[str, Any]:
    """Apply controlled TensorFlow runtime settings before model loading."""
    options = tf.config.optimizer.get_experimental_options()
    options["disable_meta_optimizer"] = True
    tf.config.optimizer.set_experimental_options(options)

    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        threading_status = "configured intra_op=1, inter_op=2"
    except RuntimeError as exc:
        threading_status = f"unchanged because runtime was already initialized: {exc}"

    actual = tf.config.optimizer.get_experimental_options()
    if actual.get("disable_meta_optimizer") is not True:
        raise RuntimeError("TensorFlow Meta Optimizer is not disabled")

    return {"optimizer_options": actual, "threading": threading_status}


def prepare_batch(images: tf.Tensor, labels: tf.Tensor):
    """Decode pipeline emits [0,255]; normalize once to [0,1]."""
    images, labels = normalize_for_scratch(images, labels)
    labels = tf.cast(tf.reshape(labels, (-1, 1)), tf.float32)
    tf.debugging.assert_type(images, tf.float32)
    tf.debugging.assert_type(labels, tf.float32)
    tf.debugging.assert_all_finite(images, "Non-finite Validation image")
    tf.debugging.assert_all_finite(labels, "Non-finite Validation label")
    tf.debugging.assert_greater_equal(
        tf.reduce_min(images), tf.constant(0.0, tf.float32)
    )
    tf.debugging.assert_less_equal(
        tf.reduce_max(images), tf.constant(1.0 + 1e-6, tf.float32)
    )
    return images, labels


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def main() -> int:
    print("=" * 75)
    print("CNN V2 PILOT — CORRECTED VALIDATION INFERENCE (NO RETRAINING)")
    print("=" * 75)

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}. "
            "Refusing to overwrite existing evidence."
        )
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(SUMMARY_PATH)

    runtime = configure_runtime()
    print("TensorFlow:", tf.__version__)
    print("Keras:", tf.keras.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
    print("Threading:", runtime["threading"])
    print("Optimizer options:", json.dumps(runtime["optimizer_options"], default=str))

    checkpoint_hash = sha256_file(CHECKPOINT_PATH)
    print("Checkpoint SHA256:", checkpoint_hash)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Pilot checkpoint SHA256 differs from the expected checkpoint. "
            "Inspect the checkpoint before proceeding."
        )

    with SUMMARY_PATH.open("r", encoding="utf-8") as file:
        training_summary = json.load(file)

    # Read Validation only. Do not call load_split_metadata("test").
    val_paths, val_labels, val_records = load_split_metadata("validation")
    y_true_expected = np.asarray(val_labels, dtype=np.int32)

    raw_val_dataset = build_dataset(
        val_paths,
        val_labels,
        training=False,
    )
    # This is the ONLY normalization pass in this inference script.
    val_dataset = raw_val_dataset.map(
        prepare_batch,
        num_parallel_calls=1,
        deterministic=True,
    )

    # Ensure the checkpoint used for the correction is the pilot checkpoint.
    model = tf.keras.models.load_model(CHECKPOINT_PATH, compile=False)
    if tuple(model.input_shape) != (None, 224, 224, 3):
        raise RuntimeError(f"Unexpected model input shape: {model.input_shape}")
    if tuple(model.output_shape) != (None, 1):
        raise RuntimeError(f"Unexpected model output shape: {model.output_shape}")

    # Keras predict uses the prepared Dataset as-is. No second preprocessing.
    y_score = model.predict(val_dataset, verbose=0).reshape(-1).astype(np.float32)

    if len(y_score) != len(val_records):
        raise RuntimeError(
            f"Prediction count mismatch: {len(y_score)} vs {len(val_records)}"
        )
    if not np.all(np.isfinite(y_score)):
        raise RuntimeError("Non-finite Validation predictions detected")
    if np.any(y_score < 0.0) or np.any(y_score > 1.0):
        raise RuntimeError("Sigmoid predictions must be within [0,1]")

    # Verify the inference order/labels match the Validation metadata.
    y_true_from_dataset = []
    for _, batch_labels in val_dataset:
        y_true_from_dataset.extend(
            batch_labels.numpy().reshape(-1).astype(np.int32).tolist()
        )
    y_true = np.asarray(y_true_from_dataset, dtype=np.int32)
    if not np.array_equal(y_true, y_true_expected):
        raise RuntimeError("Validation labels/order differ from Validation metadata")

    # Fixed 0.5 is diagnostic only; this script does not lock a threshold.
    y_pred = (y_score >= DIAGNOSTIC_THRESHOLD).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Recompute Keras evaluate metrics from the same preprocessed dataset.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.AUC(name="auroc", curve="ROC", num_thresholds=1000),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        ],
        run_eagerly=False,
    )
    evaluation = model.evaluate(val_dataset, verbose=0, return_dict=True)

    keras_auc_from_predictions = tf.keras.metrics.AUC(
        curve="ROC", num_thresholds=1000
    )
    keras_auc_from_predictions.update_state(
        y_true.reshape(-1, 1), y_score.reshape(-1, 1)
    )

    sklearn_auc = float(roc_auc_score(y_true, y_score))
    history_auroc_values = training_summary.get("keras_history", {}).get("val_auroc", [])
    history_val_auroc = (
        float(history_auroc_values[-1]) if history_auroc_values else None
    )

    metrics = {
        "status": "corrected_pilot_validation_diagnostic",
        "experiment_name": training_summary.get("experiment_name"),
        "split": "validation",
        "num_images": int(len(y_true)),
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "inference_method": "Keras model.predict on tf.data Dataset",
        "runtime_device_policy": "TensorFlow default device placement; GPU was detected",
        "meta_optimizer_disabled": True,
        "input_preprocessing": "JPEG RGB decode/resize to [0,255], normalize_for_scratch exactly once to [0,1]",
        "threshold": DIAGNOSTIC_THRESHOLD,
        "threshold_locked": False,
        "threshold_optimization_performed": False,
        "auroc_sklearn": sklearn_auc,
        "auroc_keras_from_saved_scores": float(keras_auc_from_predictions.result().numpy()),
        "auroc_keras_model_evaluate": float(evaluation["auroc"]),
        "training_history_val_auroc": history_val_auroc,
        "accuracy_at_0.5": float(accuracy_score(y_true, y_pred)),
        "precision_at_0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_0.5": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1_at_0.5": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix_at_0.5": cm.tolist(),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "score_min": float(np.min(y_score)),
        "score_max": float(np.max(y_score)),
        "keras_model_evaluate": {key: float(value) for key, value in evaluation.items()},
        "test_split_loaded": False,
        "test_split_evaluated": False,
        "notes": [
            "Inference-only correction; model was not retrained.",
            "Validation images were normalized exactly once.",
            "Original pilot Validation predictions.csv was not overwritten.",
            "Threshold 0.5 is diagnostic only and is not locked.",
            "Test split was not loaded or evaluated.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    predictions_path = OUTPUT_DIR / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["relative_path", "y_true", "y_score", "y_pred_at_0.5"])
        for record, label, score, prediction in zip(
            val_records, y_true, y_score, y_pred
        ):
            writer.writerow([
                record["relative_path"],
                int(label),
                float(score),
                int(prediction),
            ])

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Compare historical CSV for diagnosis only; do not overwrite it.
    if OLD_PREDICTIONS_PATH.is_file():
        old_rows = read_csv(OLD_PREDICTIONS_PATH)
        old_paths = [row["relative_path"] for row in old_rows]
        old_labels = np.asarray([int(row["y_true"]) for row in old_rows], dtype=np.int32)
        old_scores = np.asarray([float(row["y_score"]) for row in old_rows], dtype=np.float32)
        print("\nHistorical CSV path order identical:", old_paths == [r["relative_path"] for r in val_records])
        print("Historical CSV labels identical:", np.array_equal(old_labels, y_true))
        if len(old_scores) == len(y_score):
            print("Historical CSV max score difference:", float(np.max(np.abs(old_scores - y_score))))
            print("Historical CSV sklearn AUROC:", f"{roc_auc_score(old_labels, old_scores):.10f}")

    print("\n" + "=" * 75)
    print("CORRECTED CNN V2 PILOT VALIDATION")
    print("=" * 75)
    print("Validation images:", len(y_true))
    print("Keras history val_auroc:", history_val_auroc)
    print("Reloaded model.evaluate AUROC:", f"{float(evaluation['auroc']):.10f}")
    print("Keras AUC from corrected scores:", f"{float(keras_auc_from_predictions.result().numpy()):.10f}")
    print("Scikit-learn AUROC:", f"{sklearn_auc:.10f}")
    print("Threshold:", DIAGNOSTIC_THRESHOLD, "(diagnostic only; not locked)")
    print("Predictions:", predictions_path)
    print("Metrics:", metrics_path)
    print("Test split loaded/evaluated: NO")
    print("No training occurred; original checkpoint and CSV were not modified.")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
