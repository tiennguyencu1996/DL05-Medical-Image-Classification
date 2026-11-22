"""Train CNN V2 for one frozen patient-level Label Efficiency manifest.

Run one epoch per process invocation to suit the fanless 8-GB MacBook Air.
Repeat the same command while status is ``session_complete``. Training state,
optimizer state, epoch history, and best checkpoint are stored per run.

This script reads Train and Validation only. It never loads Test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
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
    roc_curve,
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
MANIFEST_METADATA_PATH = MANIFEST_DIR / "metadata.json"
SOURCE_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "train_v3.csv"
VALIDATION_SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "validation_v3.csv"

LEVELS = ("010pct", "025pct", "050pct", "100pct")
EXPECTED_SOURCE_TRAIN_SHA256 = "a016befaa42bf8dcdfd95628859ca6edf0a81c553aec96edac2574e755200f93"
EXPECTED_MANIFEST_SHA256 = {
    "010pct": "4cb39546eb800e0156f5ab5c819dcd71ce14b94512898e4759264e9c5bb09259",
    "025pct": "744a1689fb3f3fe967c30204cb9aeef696e109debf0aac0ba69020f0175ce95c",
    "050pct": "87829c18a4d26b46b20b6b58ef7d11ee5e50420b15e5f8c62599ca241773dc9c",
    "100pct": "e6635a99a819421c95ad79396a28848d7defe970d64fdb8445192537be1148f0",
}
EXPECTED_LEVEL_COUNTS = {
    "010pct": {"images": 416, "patients": 204, "NORMAL": 108, "PNEUMONIA": 308},
    "025pct": {"images": 1042, "patients": 529, "NORMAL": 269, "PNEUMONIA": 773},
    "050pct": {"images": 2082, "patients": 1037, "NORMAL": 539, "PNEUMONIA": 1543},
    "100pct": {"images": 4164, "patients": 2120, "NORMAL": 1078, "PNEUMONIA": 3086},
}

SEED = RANDOM_SEED
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
SESSION_EPOCHS = 1
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-4
REDUCE_LR_PATIENCE = 3
REDUCE_LR_MIN_DELTA = 1e-4
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6
DIAGNOSTIC_THRESHOLD = 0.5
STATE_VERSION = 1

HISTORY_FIELDS = [
    "epoch",
    "loss",
    "accuracy",
    "keras_auroc_approx",
    "precision",
    "recall",
    "val_loss",
    "val_accuracy",
    "val_keras_auroc_approx",
    "val_precision",
    "val_recall",
    "val_sklearn_auroc",
    "learning_rate_used",
    "learning_rate_after_epoch",
]


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(temp, path)


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    with temp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def directory_has_files(path: Path) -> bool:
    return path.exists() and any(item.is_file() for item in path.rglob("*"))


def to_float(value: Any) -> float:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return float(value)


def get_learning_rate(optimizer: tf.keras.optimizers.Optimizer) -> float:
    return to_float(optimizer.learning_rate)


def set_learning_rate(optimizer: tf.keras.optimizers.Optimizer, value: float) -> None:
    learning_rate = optimizer.learning_rate
    if not hasattr(learning_rate, "assign"):
        raise TypeError("Expected an assignable scalar optimizer learning rate")
    learning_rate.assign(value)


def configure_runtime() -> dict[str, Any]:
    # Must be set before model construction, tracing, or model.fit().
    options = tf.config.optimizer.get_experimental_options()
    options["disable_meta_optimizer"] = True
    tf.config.optimizer.set_experimental_options(options)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        threading = "configured intra_op=1, inter_op=2"
    except RuntimeError as exc:
        threading = f"unchanged because TensorFlow runtime was already initialized: {exc}"
    actual = tf.config.optimizer.get_experimental_options()
    if actual.get("disable_meta_optimizer") is not True:
        raise RuntimeError("Could not disable TensorFlow Meta Optimizer")
    return {"optimizer_options": actual, "threading": threading}


# ---------------------------------------------------------------------------
# Frozen manifest validation
# ---------------------------------------------------------------------------

def validate_and_load_manifest(level: str):
    manifest_path = MANIFEST_DIR / f"train_{level}.csv"
    for path in (manifest_path, MANIFEST_METADATA_PATH, SOURCE_TRAIN_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest_hash = sha256_file(manifest_path)
    source_hash = sha256_file(SOURCE_TRAIN_PATH)
    if source_hash != EXPECTED_SOURCE_TRAIN_SHA256:
        raise RuntimeError("Original Train CSV SHA256 differs from the frozen audit")
    if manifest_hash != EXPECTED_MANIFEST_SHA256[level]:
        raise RuntimeError(f"Frozen {level} manifest SHA256 mismatch")

    metadata = json.loads(MANIFEST_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("source_csv_sha256") != source_hash:
        raise RuntimeError("metadata.json source CSV SHA256 mismatch")

    rows = read_csv(manifest_path)
    source_rows = read_csv(SOURCE_TRAIN_PATH)
    required = {"relative_path", "label", "patient_id", "split", "source_split", "sha256"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("Manifest is empty or missing required columns")

    if len(source_rows) != 4164:
        raise RuntimeError(f"Expected 4,164 source Train records, found {len(source_rows)}")

    source_by_path = {row["relative_path"]: row for row in source_rows}
    source_groups: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        source_groups[row["patient_id"]].add(row["relative_path"])

    data_root = DATASET_ROOT.resolve()
    seen_paths: set[str] = set()
    selected_groups: dict[str, set[str]] = defaultdict(set)
    paths: list[str] = []
    labels: list[int] = []
    observed_class_counts: Counter[str] = Counter()

    for row in rows:
        rel = row["relative_path"]
        patient_id = row["patient_id"]
        label_name = row["label"]
        if rel in seen_paths:
            raise RuntimeError(f"Duplicate relative_path in {level}: {rel}")
        seen_paths.add(rel)
        if row["split"] != "train" or row["source_split"] != "train":
            raise RuntimeError(f"Non-Train row found in {level}: {rel}")
        if label_name not in LABEL_TO_ID:
            raise RuntimeError(f"Unknown label {label_name!r}")
        original = source_by_path.get(rel)
        if original is None:
            raise RuntimeError(f"Manifest path not in original Train: {rel}")
        for key in ("patient_id", "label", "sha256"):
            if row[key] != original[key]:
                raise RuntimeError(f"Manifest {key} mismatch for {rel}")

        selected_groups[patient_id].add(rel)
        image_path = (PROJECT_ROOT / rel).resolve()
        try:
            image_path.relative_to(data_root)
        except ValueError as exc:
            raise RuntimeError(f"Image path escapes DATASET_ROOT: {rel}") from exc
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        paths.append(str(image_path))
        labels.append(LABEL_TO_ID[label_name])
        observed_class_counts[label_name] += 1

    for patient_id, selected_paths in selected_groups.items():
        if selected_paths != source_groups.get(patient_id, set()):
            raise RuntimeError(f"Partial patient group found: {patient_id}")

    expected = EXPECTED_LEVEL_COUNTS[level]
    actual_patients = len(selected_groups)
    actual_counts = {
        "images": len(rows),
        "patients": actual_patients,
        "NORMAL": observed_class_counts["NORMAL"],
        "PNEUMONIA": observed_class_counts["PNEUMONIA"],
    }
    if actual_counts != expected:
        raise RuntimeError(f"{level} counts differ from frozen audit: {actual_counts}")

    level_metadata = metadata.get("levels", {}).get(level, {})
    if int(level_metadata.get("images", -1)) != len(rows):
        raise RuntimeError(f"{level} image count differs from metadata.json")

    if level == "100pct" and seen_paths != set(source_by_path):
        raise RuntimeError("100% manifest is not identical to original Train paths")

    return paths, labels, rows, manifest_hash, source_hash, selected_groups


def prepare_batch(images: tf.Tensor, labels: tf.Tensor):
    """Normalize raw [0,255] images once to [0,1]."""
    images, labels = normalize_for_scratch(images, labels)
    images = tf.cast(images, tf.float32)
    labels = tf.cast(tf.reshape(labels, (-1, 1)), tf.float32)
    tf.debugging.assert_all_finite(images, "Non-finite input image")
    tf.debugging.assert_all_finite(labels, "Non-finite label")
    tf.debugging.assert_greater_equal(tf.reduce_min(images), 0.0)
    tf.debugging.assert_less_equal(tf.reduce_max(images), 1.0 + 1e-6)
    return images, labels


def verify_train_validation_separation(train_rows, validation_records) -> None:
    train_patients = {row["patient_id"] for row in train_rows}
    validation_patients = {row["patient_id"] for row in validation_records}
    patient_overlap = train_patients & validation_patients
    if patient_overlap:
        raise RuntimeError(f"Train/Validation patient overlap: {len(patient_overlap)} patients")

    train_hashes = {row["sha256"] for row in train_rows}
    validation_hashes = {row["sha256"] for row in validation_records}
    hash_overlap = train_hashes & validation_hashes
    if hash_overlap:
        raise RuntimeError(f"Train/Validation SHA256 overlap: {len(hash_overlap)} hashes")


# ---------------------------------------------------------------------------
# Experiment state and outputs
# ---------------------------------------------------------------------------

def make_initial_state(config: dict[str, Any], run_name: str) -> dict[str, Any]:
    return {
        "state_version": STATE_VERSION,
        "experiment_name": run_name,
        "experiment_config": config,
        "training_status": "ready",
        "completed_epochs": 0,
        "session_count": 0,
        "session_started_at": None,
        "session_finished_at": None,
        "stop_reason": None,
        "best_validation_sklearn_auroc": None,
        "best_epoch": None,
        "early_stopping_best_auroc": None,
        "early_stopping_wait": 0,
        "lr_best_validation_loss": None,
        "lr_wait": 0,
        "lr_reductions": 0,
        "current_learning_rate": LEARNING_RATE,
        "last_epoch_metrics": None,
        "history": [],
        "test_split_loaded": False,
        "test_split_evaluated": False,
    }


def load_or_initialize_state(run_dir: Path, config: dict[str, Any], run_name: str):
    summary_path = run_dir / "training_summary.json"
    checkpoints_dir = run_dir / "checkpoints"
    training_state_dir = run_dir / "training_state"

    if summary_path.exists():
        state = json.loads(summary_path.read_text(encoding="utf-8"))
        if state.get("state_version") != STATE_VERSION:
            raise RuntimeError("Unsupported state version; inspect before resuming")
        if state.get("experiment_config") != config:
            raise RuntimeError("Saved experiment configuration differs; refusing to resume")
        if len(state.get("history", [])) != int(state.get("completed_epochs", 0)):
            raise RuntimeError("History and completed_epochs disagree")
        if state.get("completed_epochs", 0) > 0 and not directory_has_files(training_state_dir):
            raise RuntimeError("BackupAndRestore files are missing; refusing to restart from random weights")
        if state.get("best_epoch") is not None and not (checkpoints_dir / "best.keras").is_file():
            raise RuntimeError("Best checkpoint is missing")
        return state

    if run_dir.exists() and directory_has_files(run_dir):
        raise RuntimeError(
            f"Run directory contains files but no training_summary.json: {run_dir}. "
            "Archive/inspect it; do not overwrite."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    training_state_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation" / "validation_best").mkdir(parents=True, exist_ok=True)
    state = make_initial_state(config, run_name)
    atomic_write_json(summary_path, state)
    atomic_write_csv(run_dir / "history.csv", HISTORY_FIELDS, [])
    return state


def save_best_validation_outputs(
    run_dir: Path,
    epoch_number: int,
    records: list[dict[str, str]],
    y_true: np.ndarray,
    y_score: np.ndarray,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    y_pred = (y_score >= DIAGNOSTIC_THRESHOLD).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    metrics: dict[str, Any] = {
        "status": "best_checkpoint_validation_diagnostic",
        "split": "validation",
        "epoch": int(epoch_number),
        "num_images": int(len(y_true)),
        "num_normal": int(np.sum(y_true == 0)),
        "num_pneumonia": int(np.sum(y_true == 1)),
        "inference_method": "model.predict on dataset normalized exactly once",
        "meta_optimizer_disabled": True,
        "auroc_sklearn": float(roc_auc_score(y_true, y_score)),
        "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
        "threshold_locked": False,
        "threshold_optimization_performed": False,
        "accuracy_at_0.5": float(accuracy_score(y_true, y_pred)),
        "precision_at_0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity_at_0.5": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_at_0.5": float(tn / (tn + fp)) if (tn + fp) else None,
        "macro_f1_at_0.5": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix_at_0.5": cm.tolist(),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "score_min": float(np.min(y_score)),
        "score_max": float(np.max(y_score)),
        "checkpoint_sha256": checkpoint_sha256,
    }

    eval_dir = run_dir / "evaluation" / "validation_best"
    prediction_rows = [
        {
            "relative_path": record["relative_path"],
            "y_true": int(label),
            "y_score": float(score),
            "y_pred_at_0.5": int(prediction),
        }
        for record, label, score, prediction in zip(records, y_true, y_score, y_pred)
    ]
    atomic_write_csv(
        eval_dir / "predictions.csv",
        ["relative_path", "y_true", "y_score", "y_pred_at_0.5"],
        prediction_rows,
    )

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_rows = [
        {"fpr": float(x), "tpr": float(y), "threshold": float(th)}
        for x, y, th in zip(fpr, tpr, thresholds)
    ]
    atomic_write_csv(eval_dir / "roc_curve.csv", ["fpr", "tpr", "threshold"], roc_rows)
    atomic_write_json(eval_dir / "metrics.json", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class PersistentValidationAUROCController(tf.keras.callbacks.Callback):
    """Compute exact sklearn Validation AUROC and persist best model/state."""

    def __init__(
        self,
        state: dict[str, Any],
        run_dir: Path,
        validation_dataset,
        validation_labels: list[int],
        validation_records: list[dict[str, str]],
    ):
        super().__init__()
        self.state = state
        self.run_dir = run_dir
        self.validation_dataset = validation_dataset
        self.validation_labels = np.asarray(validation_labels, dtype=np.int32)
        self.validation_records = validation_records
        self.session_start_epoch = int(state["completed_epochs"])

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs if logs is not None else {}
        completed_epoch = int(epoch) + 1
        expected_epoch = int(self.state["completed_epochs"]) + 1
        if completed_epoch != expected_epoch:
            raise RuntimeError(
                f"Epoch mismatch: Keras={completed_epoch}, state={expected_epoch}"
            )

        # validation_dataset is already normalized by prepare_batch exactly once.
        # Do not call prepare_batch here.
        scores = self.model.predict(self.validation_dataset, verbose=0)
        y_score = np.asarray(scores, dtype=np.float32).reshape(-1)
        y_true = self.validation_labels
        if len(y_score) != len(y_true) or len(y_score) != len(self.validation_records):
            raise RuntimeError("Validation prediction count/order contract failed")
        if not np.all(np.isfinite(y_score)):
            raise FloatingPointError("Non-finite Validation predictions")
        if np.any(y_score < 0.0) or np.any(y_score > 1.0):
            raise RuntimeError("Sigmoid predictions are outside [0, 1]")

        val_sklearn_auroc = float(roc_auc_score(y_true, y_score))
        logs["val_sklearn_auroc"] = val_sklearn_auroc
        val_loss = to_float(logs["val_loss"])
        learning_rate_used = get_learning_rate(self.model.optimizer)

        # Save the best model by exact sklearn AUROC, not approximate Keras AUC.
        previous_best = self.state["best_validation_sklearn_auroc"]
        if previous_best is None or val_sklearn_auroc > float(previous_best):
            checkpoint_dir = self.run_dir / "checkpoints"
            best_path = checkpoint_dir / "best.keras"
            temp_path = checkpoint_dir / "best.tmp.keras"
            if temp_path.exists():
                temp_path.unlink()
            self.model.save(temp_path, overwrite=True)
            os.replace(temp_path, best_path)
            checkpoint_hash = sha256_file(best_path)
            metrics = save_best_validation_outputs(
                self.run_dir,
                completed_epoch,
                self.validation_records,
                y_true,
                y_score,
                checkpoint_hash,
            )
            self.state["best_validation_sklearn_auroc"] = val_sklearn_auroc
            self.state["best_epoch"] = completed_epoch
            self.state["best_checkpoint_path"] = str(best_path.relative_to(PROJECT_ROOT))
            self.state["best_checkpoint_sha256"] = checkpoint_hash
            self.state["best_validation_metrics"] = metrics
            print(
                f"\nBest Validation sklearn AUROC: {val_sklearn_auroc:.6f} "
                f"(epoch {completed_epoch}); checkpoint saved."
            )

        # Persistent early stopping based on exact Validation sklearn AUROC.
        early_best = self.state["early_stopping_best_auroc"]
        if early_best is None or val_sklearn_auroc > float(early_best) + EARLY_STOPPING_MIN_DELTA:
            self.state["early_stopping_best_auroc"] = val_sklearn_auroc
            self.state["early_stopping_wait"] = 0
        else:
            self.state["early_stopping_wait"] = int(self.state["early_stopping_wait"]) + 1

        # Persistent ReduceLROnPlateau based on validation loss.
        lr_best_loss = self.state["lr_best_validation_loss"]
        if lr_best_loss is None or val_loss < float(lr_best_loss) - REDUCE_LR_MIN_DELTA:
            self.state["lr_best_validation_loss"] = val_loss
            self.state["lr_wait"] = 0
        else:
            self.state["lr_wait"] = int(self.state["lr_wait"]) + 1

        if int(self.state["lr_wait"]) >= REDUCE_LR_PATIENCE:
            old_lr = get_learning_rate(self.model.optimizer)
            new_lr = max(old_lr * REDUCE_LR_FACTOR, MIN_LEARNING_RATE)
            if new_lr < old_lr - 1e-12:
                set_learning_rate(self.model.optimizer, new_lr)
                self.state["lr_reductions"] = int(self.state["lr_reductions"]) + 1
                print(f"\nReduce LR: {old_lr:.8g} -> {new_lr:.8g}")
            self.state["lr_wait"] = 0

        lr_after = get_learning_rate(self.model.optimizer)
        required = (
            "loss", "accuracy", "keras_auroc_approx", "precision", "recall",
            "val_loss", "val_accuracy", "val_keras_auroc_approx",
            "val_precision", "val_recall",
        )
        missing = [name for name in required if name not in logs]
        if missing:
            raise RuntimeError(f"Missing expected Keras metrics: {missing}")

        row = {name: to_float(logs[name]) for name in required}
        row.update({
            "epoch": completed_epoch,
            "val_sklearn_auroc": val_sklearn_auroc,
            "learning_rate_used": learning_rate_used,
            "learning_rate_after_epoch": lr_after,
        })
        self.state["history"].append(row)
        self.state["completed_epochs"] = completed_epoch
        self.state["current_learning_rate"] = lr_after
        self.state["last_epoch_metrics"] = row
        self.state["updated_at"] = utc_now()

        early_stop = int(self.state["early_stopping_wait"]) >= EARLY_STOPPING_PATIENCE
        if early_stop:
            self.state["training_status"] = "early_stopped"
            self.state["stop_reason"] = (
                f"Validation sklearn AUROC did not improve by {EARLY_STOPPING_MIN_DELTA} "
                f"for {EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
        elif completed_epoch >= MAX_EPOCHS:
            self.state["training_status"] = "completed"
            self.state["stop_reason"] = "Maximum epoch count reached."
        elif completed_epoch - self.session_start_epoch >= SESSION_EPOCHS:
            self.state["training_status"] = "session_complete"
            self.state["stop_reason"] = f"Session limit reached: {SESSION_EPOCHS} epoch(s)."
        else:
            self.state["training_status"] = "running"
            self.state["stop_reason"] = None

        atomic_write_json(self.run_dir / "training_summary.json", self.state)
        atomic_write_csv(self.run_dir / "history.csv", HISTORY_FIELDS, self.state["history"])

        if self.state["training_status"] in {"early_stopped", "completed", "session_complete"}:
            self.model.stop_training = True

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        self.state["session_finished_at"] = utc_now()
        if getattr(self.model, "optimizer", None) is not None:
            self.state["current_learning_rate"] = get_learning_rate(self.model.optimizer)
        atomic_write_json(self.run_dir / "training_summary.json", self.state)


class FiniteLossGuard(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        if "loss" in logs and not math.isfinite(to_float(logs["loss"])):
            raise FloatingPointError(f"Non-finite training loss at batch {batch}")


class ResumeStateValidator(tf.keras.callbacks.Callback):
    """Validate BackupAndRestore's resumed epoch and learning rate."""
    def __init__(self, state: dict[str, Any]):
        super().__init__()
        self.state = state

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        restored_epoch = getattr(self.model, "_initial_epoch", 0)
        restored_epoch = int(restored_epoch or 0)
        expected_epoch = int(self.state["completed_epochs"])
        if restored_epoch != expected_epoch:
            raise RuntimeError(
                f"BackupAndRestore epoch mismatch: restored={restored_epoch}, expected={expected_epoch}"
            )
        restored_lr = get_learning_rate(self.model.optimizer)
        expected_lr = float(self.state["current_learning_rate"])
        if not math.isclose(restored_lr, expected_lr, rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(
                f"Restored LR mismatch: restored={restored_lr}, expected={expected_lr}"
            )
        print(f"Resume verification: PASS (starting epoch {restored_epoch + 1})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=LEVELS, required=True,
                        help="Frozen Train manifest level to train")
    parser.add_argument("--run-name", default=None,
                        help="Optional custom experiment name; same name resumes the run")
    args = parser.parse_args()

    default_run_name = f"cnn_from_scratch_v2_{args.level}_full_20261119"
    run_name = args.run_name or default_run_name
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        parser.error("--run-name may contain only letters, numbers, underscore, and hyphen")

    run_dir = PROJECT_ROOT / "outputs" / "experiments" / run_name
    print("=" * 78)
    print("DL-05 CNN V2 — LABEL EFFICIENCY TRAINER")
    print("=" * 78)

    runtime = configure_runtime()
    print("TensorFlow:", tf.__version__)
    print("Keras:", tf.keras.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
    print("Threading:", runtime["threading"])
    print("Optimizer options:", json.dumps(runtime["optimizer_options"], default=str))

    train_paths, train_labels, train_rows, manifest_hash, source_hash, train_groups = (
        validate_and_load_manifest(args.level)
    )
    val_paths, val_labels, val_records = load_split_metadata("validation")
    verify_train_validation_separation(train_rows, val_records)

    val_split_hash = sha256_file(VALIDATION_SPLIT_PATH)
    train_counts = Counter(row["label"] for row in train_rows)
    train_patient_count = len(train_groups)
    print("\nTrain manifest:", (MANIFEST_DIR / f"train_{args.level}.csv").relative_to(PROJECT_ROOT))
    print("Train images:", len(train_paths))
    print("Train patients:", train_patient_count)
    print("Train classes:", dict(train_counts))
    print("Train manifest SHA256:", manifest_hash)
    print("Source Train SHA256:", source_hash)
    print("Validation images:", len(val_paths))
    print("Validation split SHA256:", val_split_hash)
    print("Test split loaded/evaluated: NO")

    config = {
        "experiment_name": run_name,
        "model_name": "CNNFromScratch_V2_Controlled",
        "label_efficiency_level": args.level,
        "train_manifest": str((MANIFEST_DIR / f"train_{args.level}.csv").relative_to(PROJECT_ROOT)),
        "train_manifest_sha256": manifest_hash,
        "source_train_csv_sha256": source_hash,
        "validation_csv_sha256": val_split_hash,
        "train_images": len(train_paths),
        "train_patients": train_patient_count,
        "train_class_counts": dict(train_counts),
        "validation_images": len(val_paths),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "random_seed": SEED,
        "shuffle_seed_policy": "base_seed_plus_completed_epochs",
        "max_epochs": MAX_EPOCHS,
        "epochs_per_process_invocation": SESSION_EPOCHS,
        "optimizer": "Adam",
        "initial_learning_rate": LEARNING_RATE,
        "loss": "BinaryCrossentropy",
        "class_weighting": "none",
        "normalization": "divide_pixel_values_by_255_exactly_once",
        "runtime_disable_meta_optimizer": True,
        "run_eagerly": False,
        "jit_compile": False,
        "checkpoint_monitor": "validation_sklearn_roc_auc_from_model_predict",
        "early_stopping_monitor": "validation_sklearn_roc_auc_from_model_predict",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "reduce_lr_monitor": "val_loss",
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "reduce_lr_factor": REDUCE_LR_FACTOR,
        "minimum_learning_rate": MIN_LEARNING_RATE,
        "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
        "diagnostic_threshold_locked": False,
        "test_split_loaded": False,
        "test_split_evaluated": False,
    }

    state = load_or_initialize_state(run_dir, config, run_name)
    if state["training_status"] in {"completed", "early_stopped"}:
        print("\nThis run is already terminal:", state["training_status"])
        print("Completed epochs:", state["completed_epochs"])
        print("Best epoch:", state["best_epoch"])
        print("Best Validation sklearn AUROC:", state["best_validation_sklearn_auroc"])
        print("No additional training was started.")
        return 0

    completed_epochs = int(state["completed_epochs"])
    tf.keras.utils.set_random_seed(SEED + completed_epochs)
    shuffle_seed = SEED + completed_epochs

    train_dataset = build_dataset(
        train_paths, train_labels, training=True, shuffle_seed=shuffle_seed
    ).map(prepare_batch, num_parallel_calls=1, deterministic=True)
    validation_dataset = build_dataset(
        val_paths, val_labels, training=False
    ).map(prepare_batch, num_parallel_calls=1, deterministic=True)

    model = build_cnn_from_scratch_v2()
    if tuple(model.input_shape) != (None, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise RuntimeError(f"Unexpected model input shape: {model.input_shape}")
    if tuple(model.output_shape) != (None, 1):
        raise RuntimeError(f"Unexpected model output shape: {model.output_shape}")
    if model.count_params() != 1_208_161:
        raise RuntimeError(f"Unexpected parameter count: {model.count_params():,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.AUC(name="keras_auroc_approx", curve="ROC", num_thresholds=1000),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        ],
        run_eagerly=False,
        jit_compile=False,
    )

    checkpoints_dir = run_dir / "checkpoints"
    training_state_dir = run_dir / "training_state"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    training_state_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation" / "validation_best").mkdir(parents=True, exist_ok=True)

    controller = PersistentValidationAUROCController(
        state, run_dir, validation_dataset, val_labels, val_records
    )
    finite_guard = FiniteLossGuard()
    backup = tf.keras.callbacks.BackupAndRestore(
        backup_dir=str(training_state_dir),
        save_freq="epoch",
        double_checkpoint=True,
        delete_checkpoint=False,
    )
    resume_validator = ResumeStateValidator(state)

    state["session_count"] = int(state["session_count"]) + 1
    state["session_started_at"] = utc_now()
    state["session_finished_at"] = None
    state["training_status"] = "running"
    state["stop_reason"] = None
    atomic_write_json(run_dir / "training_summary.json", state)

    print("\nModel:", model.name)
    print("Input:", model.input_shape, "Output:", model.output_shape)
    print("Parameters:", f"{model.count_params():,}")
    print("run_eagerly:", model.run_eagerly)
    print("Training shuffle seed:", shuffle_seed)
    print("Completed epochs before this session:", completed_epochs)
    print("One epoch will run in this process invocation.")
    print("Validation inference for checkpoint selection uses model.predict().")
    print("Training class weighting: none")

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=MAX_EPOCHS,
        callbacks=[controller, finite_guard, backup, resume_validator],
        verbose=1,
        shuffle=False,  # tf.data already shuffles the Train split.
    )

    final_state = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    print("\n" + "=" * 78)
    print("CNN V2 LABEL EFFICIENCY SESSION RESULT")
    print("=" * 78)
    print("Run:", run_name)
    print("Level:", args.level)
    print("Status:", final_state["training_status"])
    print("Completed epochs:", f"{final_state['completed_epochs']} / {MAX_EPOCHS}")
    print("Sessions:", final_state["session_count"])
    print("Best epoch:", final_state["best_epoch"])
    print("Best Validation sklearn AUROC:", final_state["best_validation_sklearn_auroc"])
    print("Best checkpoint:", final_state.get("best_checkpoint_path"))
    print("Best validation predictions:", run_dir / "evaluation" / "validation_best" / "predictions.csv")
    print("Test split loaded/evaluated: NO")

    if final_state["training_status"] == "session_complete":
        print("\nSESSION COMPLETE. Let the Mac cool, then repeat the SAME command to resume.")
    elif final_state["training_status"] == "early_stopped":
        print("\nTRAINING EARLY-STOPPED. This run is complete.")
    elif final_state["training_status"] == "completed":
        print("\nMAXIMUM EPOCH COUNT REACHED. This run is complete.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
