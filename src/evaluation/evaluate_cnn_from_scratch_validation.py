from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve

from src.data.input_pipeline import (
    IMAGE_SIZE,
    build_dataset,
    load_split_metadata,
    normalize_for_scratch,
)

from src.models.baseline_models import (
    TARGET_SENSITIVITY,
    calculate_metrics,
    select_threshold,
)


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "cnn_from_scratch_v1_batch8_20261117"
)

CHECKPOINT_PATH = (
    EXPERIMENT_DIR
    / "checkpoints"
    / "best.keras"
)

SUMMARY_PATH = EXPERIMENT_DIR / "training_summary.json"

OUTPUT_DIR = (
    EXPERIMENT_DIR
    / "evaluation"
    / "validation"
)

PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
THRESHOLD_PATH = OUTPUT_DIR / "locked_threshold.json"
ROC_PATH = OUTPUT_DIR / "roc_curve.csv"

EXPECTED_VALIDATION_IMAGES = 1042


# ============================================================
# 2. UTILITIES
# ============================================================

def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_json(path: Path, payload: dict) -> None:
    temporary_path = path.with_name(path.name + ".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)
        file.write("\n")

    temporary_path.replace(path)


# ============================================================
# 3. VALIDATE TRAINING ARTIFACTS
# ============================================================

def validate_artifacts():

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(SUMMARY_PATH)

    with open(SUMMARY_PATH, "r", encoding="utf-8") as file:
        summary = json.load(file)

    expected_relative_path = str(
        CHECKPOINT_PATH.relative_to(PROJECT_ROOT)
    )

    if summary.get("training_status") != "early_stopped":
        raise RuntimeError("Training status is not early_stopped.")

    if summary.get("best_model_path") != expected_relative_path:
        raise RuntimeError("Best checkpoint path does not match summary.")

    if summary.get("best_epoch") != 32:
        raise RuntimeError("Unexpected best epoch. Inspect training summary.")

    if summary.get("test_split_evaluated") is not False:
        raise RuntimeError("Training summary indicates Test was evaluated.")

    return summary


# ============================================================
# 4. VALIDATION-ONLY PREDICTION
# ============================================================

def generate_validation_predictions():

    print("=" * 70)
    print("CNN FROM SCRATCH V1 - VALIDATION EVALUATION")
    print("=" * 70)

    print("\n[1] Loading Validation metadata only...")

    paths, labels, records = load_split_metadata("validation")

    y_expected = np.asarray(labels, dtype=np.int32)

    if len(paths) != EXPECTED_VALIDATION_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_IMAGES} images, "
            f"found {len(paths)}."
        )

    print("Validation images:", len(paths))
    print("NORMAL:", int(np.sum(y_expected == 0)))
    print("PNEUMONIA:", int(np.sum(y_expected == 1)))

    print("\n[2] Loading selected checkpoint...")

    model = tf.keras.models.load_model(
        CHECKPOINT_PATH,
        compile=False,
    )

    if tuple(model.input_shape[1:]) != (
        IMAGE_SIZE, IMAGE_SIZE, 3
    ):
        raise ValueError(f"Unexpected model input: {model.input_shape}")

    if tuple(model.output_shape[1:]) != (1,):
        raise ValueError(f"Unexpected model output: {model.output_shape}")

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    print("\n[3] Building Validation dataset...")

    validation_dataset = build_dataset(
        paths,
        labels,
        training=False,
    )

    score_batches = []
    label_batches = []

    print("\n[4] Generating predictions...")

    for images, batch_labels in validation_dataset:

        images, batch_labels = normalize_for_scratch(
            images,
            batch_labels,
        )

        probabilities = model(
            images,
            training=False,
        )

        scores = np.asarray(
            probabilities.numpy()
        ).reshape(-1)

        batch_y = np.asarray(
            batch_labels.numpy()
        ).reshape(-1)

        if not np.all(np.isfinite(scores)):
            raise ValueError("Non-finite prediction detected.")

        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("Predictions are outside [0, 1].")

        score_batches.append(scores)
        label_batches.append(batch_y)

    y_score = np.concatenate(score_batches).astype(np.float64)
    y_true = np.concatenate(label_batches).astype(np.int32)

    if len(y_score) != EXPECTED_VALIDATION_IMAGES:
        raise ValueError("Unexpected number of predictions.")

    if not np.array_equal(y_true, y_expected):
        raise ValueError("Prediction labels do not match metadata order.")

    print("Prediction count:", len(y_score))
    print("Prediction range:", float(y_score.min()), "to", float(y_score.max()))

    return model, records, y_true, y_score


# ============================================================
# 5. VALIDATION THRESHOLD SELECTION
# ============================================================

def evaluate_validation(records, y_true, y_score, summary):

    print("\n[5] Selecting threshold using Validation only...")

    threshold_result = select_threshold(
        y_true,
        y_score,
        target_sensitivity=TARGET_SENSITIVITY,
    )

    locked_threshold = threshold_result["threshold"]

    y_pred = (
        y_score >= locked_threshold
    ).astype(np.int32)

    metrics = calculate_metrics(
        y_true,
        y_score,
        y_pred,
        locked_threshold,
    )

    if metrics["sensitivity"] < TARGET_SENSITIVITY:
        raise RuntimeError("Sensitivity target was not achieved.")

    print("\n[VALIDATION THRESHOLD]")
    print(f"Target sensitivity: {TARGET_SENSITIVITY:.4f}")
    print(f"Selected threshold: {locked_threshold:.8f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")

    print("\n[VALIDATION METRICS]")
    print(f"AUROC:       {metrics['auroc']:.6f}")
    print(f"Accuracy:    {metrics['accuracy']:.6f}")
    print(f"Precision:   {metrics['precision']:.6f}")
    print(f"Sensitivity: {metrics['sensitivity']:.6f}")
    print(f"Specificity: {metrics['specificity']:.6f}")
    print(f"Macro-F1:    {metrics['macro_f1']:.6f}")

    print("\nConfusion matrix:")
    print(np.asarray(metrics["confusion_matrix"]))

    checkpoint_hash = calculate_sha256(CHECKPOINT_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    expected_outputs = [
        PREDICTIONS_PATH,
        METRICS_PATH,
        THRESHOLD_PATH,
        ROC_PATH,
    ]

    existing = [str(path) for path in expected_outputs if path.exists()]

    if existing:
        raise FileExistsError(
            "Evaluation output already exists. Refusing to overwrite:\n"
            + "\n".join(existing)
        )

    # --------------------------------------------------------
    # Save per-image predictions
    # --------------------------------------------------------

    with open(
        PREDICTIONS_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "row_index",
            "relative_path",
            "patient_id",
            "y_true",
            "y_score",
            "threshold",
            "y_pred",
            "correct",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index, (record, true_label, score, prediction) in enumerate(
            zip(records, y_true, y_score, y_pred)
        ):

            writer.writerow({
                "row_index": index,
                "relative_path": record["relative_path"],
                "patient_id": record["patient_id"],
                "y_true": int(true_label),
                "y_score": float(score),
                "threshold": float(locked_threshold),
                "y_pred": int(prediction),
                "correct": int(true_label == prediction),
            })

    # --------------------------------------------------------
    # Save ROC curve data
    # --------------------------------------------------------

    fpr, tpr, roc_thresholds = roc_curve(
        y_true,
        y_score,
        drop_intermediate=False,
    )

    with open(
        ROC_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)
        writer.writerow(["false_positive_rate", "true_positive_rate", "threshold"])

        writer.writerows(zip(fpr, tpr, roc_thresholds))

    # --------------------------------------------------------
    # Save locked threshold
    # --------------------------------------------------------

    locked_decision = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "cnn_from_scratch_v1_batch8_20261117",
        "model_name": "CNNFromScratch_V1",
        "best_epoch": summary["best_epoch"],
        "checkpoint": str(
            CHECKPOINT_PATH.relative_to(PROJECT_ROOT)
        ),
        "checkpoint_sha256": checkpoint_hash,
        "positive_class": "PNEUMONIA",
        "negative_class": "NORMAL",
        "threshold": float(locked_threshold),
        "threshold_selection_policy": (
            "Maximize specificity subject to "
            "Validation sensitivity >= target sensitivity."
        ),
        "target_sensitivity": float(TARGET_SENSITIVITY),
        "validation_sensitivity": float(metrics["sensitivity"]),
        "validation_specificity": float(metrics["specificity"]),
        "test_evaluated": False,
    }

    save_json(THRESHOLD_PATH, locked_decision)

    # --------------------------------------------------------
    # Save complete Validation metrics
    # --------------------------------------------------------

    validation_results = {
        "experiment": "cnn_from_scratch_v1_batch8_20261117",
        "model_name": "CNNFromScratch_V1",
        "best_epoch": summary["best_epoch"],
        "checkpoint": str(
            CHECKPOINT_PATH.relative_to(PROJECT_ROOT)
        ),
        "checkpoint_sha256": checkpoint_hash,
        "split": "validation",
        "image_count": len(y_true),
        "class_counts": {
            "NORMAL": int(np.sum(y_true == 0)),
            "PNEUMONIA": int(np.sum(y_true == 1)),
        },
        "threshold_selection": threshold_result,
        "metrics_at_selected_threshold": metrics,
        "locked_threshold_file": str(
            THRESHOLD_PATH.relative_to(PROJECT_ROOT)
        ),
        "test_evaluated": False,
    }

    save_json(METRICS_PATH, validation_results)

    print("\n[OUTPUT FILES]")
    print("Predictions:", PREDICTIONS_PATH)
    print("ROC data:", ROC_PATH)
    print("Metrics:", METRICS_PATH)
    print("Locked threshold:", THRESHOLD_PATH)

    print("\nPASS: CNN Validation evaluation completed.")
    print("PASS: Threshold selected using Validation only.")
    print("PASS: Test split was not loaded or evaluated.")


# ============================================================
# 6. ENTRY POINT
# ============================================================

def main():

    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass

    summary = validate_artifacts()

    model, records, y_true, y_score = (
        generate_validation_predictions()
    )

    evaluate_validation(
        records,
        y_true,
        y_score,
        summary,
    )


if __name__ == "__main__":
    main()
