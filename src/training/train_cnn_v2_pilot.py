"""Run a one-epoch CNN V2 pilot on the frozen 10% Train manifest.

This is a pipeline pilot, not the final 10%-label-efficiency experiment.
It creates one new, fail-safe experiment directory and writes its own
checkpoint, history, Validation predictions and diagnostic metrics.
It never loads or evaluates Test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
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

from src.data.input_pipeline import (
    BATCH_SIZE,
    DATASET_ROOT,
    IMAGE_SIZE,
    LABEL_TO_ID,
    RANDOM_SEED,
    build_dataset,
    load_split_metadata,
    normalize_for_scratch,
)
from src.models.cnn_from_scratch_v2 import build_cnn_from_scratch_v2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = PROJECT_ROOT / "data" / "processed" / "label_efficiency" / "manifests_v1"
MANIFEST_PATH = MANIFEST_DIR / "train_010pct.csv"
METADATA_PATH = MANIFEST_DIR / "metadata.json"
SOURCE_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "train_v3.csv"
DEFAULT_RUN_NAME = "cnn_from_scratch_v2_010pct_pilot_20261119"

SEED = RANDOM_SEED
LEARNING_RATE = 1e-3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_runtime() -> dict[str, Any]:
    # Set this before model construction, tracing, or training.
    options = tf.config.optimizer.get_experimental_options()
    options["disable_meta_optimizer"] = True
    tf.config.optimizer.set_experimental_options(options)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        threading = "configured intra_op=1, inter_op=2"
    except RuntimeError as exc:
        threading = f"unchanged because runtime was already initialized: {exc}"
    actual = tf.config.optimizer.get_experimental_options()
    if actual.get("disable_meta_optimizer") is not True:
        raise RuntimeError("TensorFlow Meta Optimizer is not disabled")
    return {"optimizer_options": actual, "threading": threading}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def validate_and_load_train_manifest() -> tuple[list[str], list[int], list[dict[str, str]], str]:
    for path in (MANIFEST_PATH, METADATA_PATH, SOURCE_TRAIN_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest_hash = sha256_file(MANIFEST_PATH)
    source_hash = sha256_file(SOURCE_TRAIN_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("source_csv_sha256") != source_hash:
        raise RuntimeError("Source Train CSV SHA256 differs from frozen manifest metadata")

    rows = read_csv(MANIFEST_PATH)
    source_rows = read_csv(SOURCE_TRAIN_PATH)
    source_by_path = {row["relative_path"]: row for row in source_rows}
    source_groups: dict[str, set[str]] = {}
    for row in source_rows:
        source_groups.setdefault(row["patient_id"], set()).add(row["relative_path"])

    required = {"relative_path", "label", "patient_id", "split", "source_split", "sha256"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("Manifest is empty or lacks required columns")

    seen_paths: set[str] = set()
    observed_groups: dict[str, set[str]] = {}
    paths: list[str] = []
    labels: list[int] = []
    data_root = DATASET_ROOT.resolve()

    for row in rows:
        relative_path = row["relative_path"]
        patient_id = row["patient_id"]
        label_name = row["label"]
        if relative_path in seen_paths:
            raise RuntimeError(f"Duplicate path in manifest: {relative_path}")
        seen_paths.add(relative_path)
        if row["split"] != "train" or row["source_split"] != "train":
            raise RuntimeError(f"Non-Train row in manifest: {relative_path}")
        if label_name not in LABEL_TO_ID:
            raise RuntimeError(f"Unknown label {label_name!r}")
        if relative_path not in source_by_path:
            raise RuntimeError(f"Manifest path not present in original Train: {relative_path}")
        original = source_by_path[relative_path]
        if original["patient_id"] != patient_id or original["label"] != label_name:
            raise RuntimeError(f"Manifest metadata differs from original Train: {relative_path}")
        observed_groups.setdefault(patient_id, set()).add(relative_path)
        image_path = (PROJECT_ROOT / relative_path).resolve()
        try:
            image_path.relative_to(data_root)
        except ValueError as exc:
            raise RuntimeError(f"Image path escapes dataset root: {relative_path}") from exc
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        paths.append(str(image_path))
        labels.append(LABEL_TO_ID[label_name])

    # Verify every selected patient contributes their complete original Train group.
    for patient_id, selected_paths in observed_groups.items():
        if selected_paths != source_groups.get(patient_id, set()):
            raise RuntimeError(f"Incomplete patient group in manifest: {patient_id}")

    # Check the frozen manifest metadata records the same count.
    level_metadata = metadata.get("levels", {}).get("010pct", {})
    if int(level_metadata.get("images", -1)) != len(rows):
        raise RuntimeError("Manifest image count differs from metadata.json")

    if len(rows) != 416:
        raise RuntimeError(f"Expected 416 images for the 010pct pilot, found {len(rows)}")

    label_counts = {name: labels.count(value) for name, value in LABEL_TO_ID.items()}
    if label_counts != {"NORMAL": 108, "PNEUMONIA": 308}:
        raise RuntimeError(f"Unexpected 10% class distribution: {label_counts}")

    return paths, labels, rows, manifest_hash


def prepare_batch(images: tf.Tensor, labels: tf.Tensor):
    images, labels = normalize_for_scratch(images, labels)
    labels = tf.cast(tf.reshape(labels, (-1, 1)), tf.float32)
    tf.debugging.assert_all_finite(images, "Non-finite input image")
    tf.debugging.assert_all_finite(labels, "Non-finite label")
    return images, labels


def infer_validation(model: tf.keras.Model, dataset, records):
    """Generate Validation predictions from an ALREADY normalized dataset.

    The caller must pass a dataset that has already been mapped through
    ``prepare_batch`` exactly once. Do not normalize images again here.
    This explicit contract prevents accidental double normalization
    ([0, 1] -> [0, 1/255]) during post-checkpoint inference.
    """
    @tf.function(reduce_retracing=True)
    def infer_batch(images):
        return model(images, training=False)

    all_scores: list[float] = []
    all_labels: list[int] = []

    for images, labels in dataset:
        # Validate the preprocessing contract without modifying the images.
        tf.debugging.assert_type(images, tf.float32)
        tf.debugging.assert_greater_equal(
            tf.reduce_min(images),
            tf.constant(0.0, dtype=tf.float32),
            message="Validation images contain values below 0.0",
        )
        tf.debugging.assert_less_equal(
            tf.reduce_max(images),
            tf.constant(1.0 + 1e-6, dtype=tf.float32),
            message=(
                "Validation dataset must already be normalized to [0, 1]. "
                "Check for missing or repeated normalization."
            ),
        )
        tf.debugging.assert_type(labels, tf.float32)
        tf.debugging.assert_equal(
            tf.shape(labels)[-1],
            1,
            message="Validation labels must have shape (batch, 1)",
        )

        # IMPORTANT: do not call prepare_batch() here. The dataset has
        # already been normalized once in main() before model.fit().
        scores = infer_batch(images)
        all_scores.extend(scores.numpy().reshape(-1).astype(float).tolist())
        all_labels.extend(labels.numpy().reshape(-1).astype(int).tolist())

    y_true = np.asarray(all_labels, dtype=np.int32)
    y_score = np.asarray(all_scores, dtype=np.float32)

    if len(y_score) != len(records):
        raise RuntimeError(
            f"Validation prediction count mismatch: "
            f"got {len(y_score)}, expected {len(records)}"
        )
    if not np.all(np.isfinite(y_score)):
        raise RuntimeError("Non-finite Validation score")
    if np.any(y_score < 0.0) or np.any(y_score > 1.0):
        raise RuntimeError("Sigmoid Validation scores must be in [0, 1]")

    return y_true, y_score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--epochs", type=int, default=1,
                        help="Pilot epoch count; default is deliberately 1")
    args = parser.parse_args()
    if args.epochs != 1:
        parser.error("This script is pilot-only and accepts exactly --epochs 1")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("--run-name may contain only letters, numbers, underscore, and hyphen")

    output_dir = PROJECT_ROOT / "outputs" / "experiments" / args.run_name
    if output_dir.exists():
        raise FileExistsError(
            f"Experiment directory already exists: {output_dir}. "
            "Refusing to overwrite. Choose a new --run-name."
        )

    print("=" * 75)
    print("DL-05 CNN V2 — 10% ONE-EPOCH PILOT")
    print("=" * 75)

    runtime = configure_runtime()
    tf.keras.utils.set_random_seed(SEED)
    print("TensorFlow:", tf.__version__)
    print("Keras:", tf.keras.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
    print("Threading:", runtime["threading"])
    print("Optimizer options:", json.dumps(runtime["optimizer_options"], default=str))

    # Validate Train manifest first. This reads Train CSV and image paths only.
    train_paths, train_labels, train_rows, manifest_hash = validate_and_load_train_manifest()
    print("\nTrain manifest:", MANIFEST_PATH.relative_to(PROJECT_ROOT))
    print("Train images:", len(train_paths))
    print("Train patients:", len({r["patient_id"] for r in train_rows}))
    print("Train labels:", {"NORMAL": train_labels.count(0), "PNEUMONIA": train_labels.count(1)})
    print("Manifest SHA256:", manifest_hash)

    # Validation is used for monitoring/diagnostics. Test is never loaded.
    val_paths, val_labels, val_records = load_split_metadata("validation")
    train_ds = build_dataset(
        train_paths, train_labels, training=True, shuffle_seed=SEED
    ).map(prepare_batch, num_parallel_calls=1, deterministic=True)
    val_ds = build_dataset(
        val_paths, val_labels, training=False
    ).map(prepare_batch, num_parallel_calls=1, deterministic=True)

    # Construct only after runtime options have been applied.
    model = build_cnn_from_scratch_v2()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.AUC(name="auroc", curve="ROC", num_thresholds=1000),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        ],
        run_eagerly=False,
    )
    print("\nModel:", model.name)
    print("Input:", model.input_shape, "Output:", model.output_shape)
    print("Parameters:", f"{model.count_params():,}")
    print("run_eagerly:", model.run_eagerly)

    # Create output only after all manifest/runtime preconditions pass.
    checkpoints_dir = output_dir / "checkpoints"
    validation_dir = output_dir / "evaluation" / "validation"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    validation_dir.mkdir(parents=True, exist_ok=False)

    started_at = utc_now()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        # train_ds is already shuffled inside build_dataset(training=True).
        # Keras ignores its own shuffle argument for tf.data datasets.
        shuffle=False,
        verbose=1,
    )

    checkpoint_path = checkpoints_dir / "pilot_epoch_001.keras"
    model.save(checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)

    # Reload the saved checkpoint and generate Validation scores through the
    # same controlled graph inference path. No threshold is optimized here.
    reloaded = tf.keras.models.load_model(checkpoint_path, compile=False)
    y_true, y_score = infer_validation(reloaded, val_ds, val_records)

    # Verify that predictions remain aligned with the original Validation CSV.
    expected_val_labels = np.asarray(val_labels, dtype=np.int32)
    if not np.array_equal(y_true, expected_val_labels):
        raise RuntimeError("Validation prediction labels/order differ from metadata")

    y_pred_05 = (y_score >= 0.5).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred_05, labels=[0, 1])

    validation_metrics = {
        "status": "pilot_diagnostic_not_final_model_result",
        "split": "validation",
        "num_images": int(len(y_true)),
        "inference_device": "TensorFlow default device placement; per-op placement was not explicitly logged",
        "inference_mode": "tf.function graph; checkpoint reloaded before inference",
        "input_preprocessing": "decode_resize [0,255] then normalize_for_scratch exactly once to [0,1]",
        "validation_dataset_pre_normalized_before_infer_validation": True,
        "meta_optimizer_disabled": True,
        "threshold": 0.5,
        "threshold_selected_on_test": False,
        "threshold_policy": "diagnostic_fixed_0.5_only; no threshold optimization",
        "auroc_sklearn": float(roc_auc_score(y_true, y_score)),
        "accuracy_at_0.5": float(accuracy_score(y_true, y_pred_05)),
        "precision_at_0.5": float(precision_score(y_true, y_pred_05, zero_division=0)),
        "recall_at_0.5": float(recall_score(y_true, y_pred_05, zero_division=0)),
        "macro_f1_at_0.5": float(f1_score(y_true, y_pred_05, average="macro", zero_division=0)),
        "confusion_matrix_at_0.5": cm.tolist(),
        "threshold_locked": False,
    }

    predictions_path = validation_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["relative_path", "y_true", "y_score", "y_pred_at_0.5"])
        for record, label, score, prediction in zip(
            val_records, y_true, y_score, y_pred_05
        ):
            writer.writerow([
                record["relative_path"], int(label), float(score), int(prediction)
            ])

    (validation_dir / "metrics.json").write_text(
        json.dumps(validation_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    history_rows = [
        {key: float(values[index]) for key, values in history.history.items()}
        for index in range(len(history.history.get("loss", [])))
    ]
    history_path = output_dir / "history.csv"
    if history_rows:
        with history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history_rows[0]))
            writer.writeheader()
            writer.writerows(history_rows)

    summary = {
        "experiment_name": args.run_name,
        "experiment_type": "one_epoch_pipeline_pilot_not_final_10pct_experiment",
        "created_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "status": "pilot_completed",
        "train_manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "train_manifest_sha256": manifest_hash,
        "train_images": len(train_paths),
        "train_patients": len({r["patient_id"] for r in train_rows}),
        "validation_images": len(val_paths),
        "validation_used_for_monitoring": True,
        "test_split_loaded": False,
        "test_split_evaluated": False,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "epochs": 1,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "BinaryCrossentropy",
        "run_eagerly": False,
        "disable_meta_optimizer": True,
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "gpu_devices": [str(device) for device in tf.config.list_physical_devices("GPU")],
        "model_name": model.name,
        "total_params": model.count_params(),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "keras_history": history.history,
        "validation_metrics": validation_metrics,
        "notes": [
            "One-epoch operational pilot only; not the final 10% label-efficiency result.",
            "Validation metrics are diagnostic and threshold 0.5 is not locked.",
            "Validation dataset is normalized exactly once before model.fit and reused for post-checkpoint inference.",
            "infer_validation validates the [0,1] input range and does not call prepare_batch again.",
            "Test was not loaded or evaluated.",
        ],
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 75)
    print("CNN V2 10% PILOT COMPLETED")
    print("=" * 75)
    print("Status:", summary["status"])
    print("Checkpoint:", checkpoint_path)
    print("Checkpoint SHA256:", checkpoint_hash)
    print("Validation AUROC (sklearn):", f"{validation_metrics['auroc_sklearn']:.6f}")
    print("Validation metrics are diagnostic only; threshold is not locked.")
    print("Output directory:", output_dir)
    print("Test split loaded/evaluated: NO")
    print("This run is a pilot, not a final label-efficiency result.")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
