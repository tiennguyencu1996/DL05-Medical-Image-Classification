#!/usr/bin/env python3
"""Train MobileNetV2 transfer learning for the DL-05 frozen label-efficiency manifests.

Protocol
--------
* Uses the existing patient-level Train manifests at 10%, 25%, 50%, and 100%.
* Uses the fixed Validation split only for checkpoint selection and threshold selection.
* Never reads, opens, or evaluates the Test split.
* Stage 1: ImageNet MobileNetV2 backbone frozen; train classification head.
* Stage 2: unfreeze the last 30 backbone layers, keeping BatchNormalization frozen.
* One epoch per process invocation to suit the fanless 8-GB MacBook Air.
* Re-run the exact same command to resume; stage transition is automatic.

Run from the project root with the project's .venv active:
    PYTHONPATH="$PWD" python -m src.training.train_mobilenetv2_label_efficiency --level 010pct

When that run prints SESSION COMPLETE, repeat the same command. When Stage 1
is complete, the next invocation automatically starts Stage 2. Each level has
an independent run directory and must be trained separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
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


# ---------------------------------------------------------------------------
# Fixed experiment protocol
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = (
    PROJECT_ROOT / "data" / "interim" / "version3" / "CellData" / "chest_xray"
)
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MANIFEST_DIR = PROJECT_ROOT / "data" / "processed" / "label_efficiency" / "manifests_v1"
MANIFEST_METADATA_PATH = MANIFEST_DIR / "metadata.json"
SOURCE_TRAIN_PATH = SPLIT_DIR / "train_v3.csv"
VALIDATION_PATH = SPLIT_DIR / "validation_v3.csv"

LEVELS = ("010pct", "025pct", "050pct", "100pct")
EXPECTED_SOURCE_TRAIN_SHA256 = "a016befaa42bf8dcdfd95628859ca6edf0a81c553aec96edac2574e755200f93"
EXPECTED_VALIDATION_SHA256 = "cacb82bb6f50d14a5f09cadec8a602db8f61dabbd45761e184e15d504ffe1b7c"
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
EXPECTED_VALIDATION_COUNTS = {"images": 1042, "NORMAL": 270, "PNEUMONIA": 772}
LABEL_TO_ID = {"NORMAL": 0, "PNEUMONIA": 1}

IMAGE_SIZE = 224
BATCH_SIZE = 8
SEED = 42
NUM_PARALLEL_CALLS = 2
PREFETCH_BUFFER = 1
UNFREEZE_LAST_N_LAYERS = 30
FROZEN_MAX_EPOCHS = 15
FINE_TUNE_MAX_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MIN_DELTA = 1e-4
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6
FROZEN_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
TARGET_VALIDATION_SENSITIVITY = 0.99
DIAGNOSTIC_THRESHOLD = 0.5
STATE_VERSION = 1

STAGE_NAMES = ("frozen", "fine_tune")


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(temp, path)


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def to_float(value: Any) -> float:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return float(value)


def configure_runtime() -> dict[str, Any]:
    # Keep runtime behavior aligned with the existing CNN V2 pipeline.
    options = tf.config.optimizer.get_experimental_options()
    options["disable_meta_optimizer"] = True
    tf.config.optimizer.set_experimental_options(options)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        threading = "configured intra_op=1, inter_op=2"
    except RuntimeError as exc:
        threading = f"unchanged (TensorFlow already initialized): {exc}"
    actual = tf.config.optimizer.get_experimental_options()
    if actual.get("disable_meta_optimizer") is not True:
        raise RuntimeError("Could not disable TensorFlow Meta Optimizer")
    return {"threading": threading, "optimizer_options": actual}


def directory_has_files(path: Path) -> bool:
    return path.exists() and any(p.is_file() for p in path.rglob("*"))


# ---------------------------------------------------------------------------
# Frozen manifest and split validation — no Test file is opened here.
# ---------------------------------------------------------------------------

def validate_manifest_and_validation(level: str):
    manifest_path = MANIFEST_DIR / f"train_{level}.csv"
    for path in (manifest_path, MANIFEST_METADATA_PATH, SOURCE_TRAIN_PATH, VALIDATION_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest_hash = sha256_file(manifest_path)
    source_hash = sha256_file(SOURCE_TRAIN_PATH)
    validation_hash = sha256_file(VALIDATION_PATH)
    if source_hash != EXPECTED_SOURCE_TRAIN_SHA256:
        raise RuntimeError("Source Train CSV SHA256 differs from the frozen split audit")
    if validation_hash != EXPECTED_VALIDATION_SHA256:
        raise RuntimeError("Validation CSV SHA256 differs from the frozen split audit")
    if manifest_hash != EXPECTED_MANIFEST_SHA256[level]:
        raise RuntimeError(f"Frozen {level} manifest SHA256 mismatch")

    metadata = json.loads(MANIFEST_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("source_csv_sha256") != source_hash:
        raise RuntimeError("Manifest metadata source SHA256 mismatch")

    manifest_rows = read_csv(manifest_path)
    source_rows = read_csv(SOURCE_TRAIN_PATH)
    validation_rows = read_csv(VALIDATION_PATH)
    required = {"relative_path", "label", "patient_id", "split", "source_split", "sha256"}
    if not manifest_rows or not required.issubset(manifest_rows[0]):
        raise RuntimeError(f"Manifest is empty or missing columns: {required}")
    if not source_rows or not required.issubset(source_rows[0]):
        raise RuntimeError("Source Train CSV is missing required columns")
    if not validation_rows or not required.issubset(validation_rows[0]):
        raise RuntimeError("Validation CSV is missing required columns")
    if len(source_rows) != 4164:
        raise RuntimeError(f"Expected 4,164 source Train rows, found {len(source_rows)}")
    if len(validation_rows) != EXPECTED_VALIDATION_COUNTS["images"]:
        raise RuntimeError(f"Expected 1,042 Validation rows, found {len(validation_rows)}")

    source_by_path = {r["relative_path"]: r for r in source_rows}
    source_groups: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        source_groups[row["patient_id"]].add(row["relative_path"])

    counts: Counter[str] = Counter()
    selected_groups: dict[str, set[str]] = defaultdict(set)
    train_paths: list[str] = []
    train_labels: list[int] = []
    train_hashes: set[str] = set()
    seen_paths: set[str] = set()
    dataset_root = DATASET_ROOT.resolve()

    for row in manifest_rows:
        rel = row["relative_path"]
        patient_id = row["patient_id"]
        label_name = row["label"]
        if rel in seen_paths:
            raise RuntimeError(f"Duplicate relative path in {level} manifest: {rel}")
        seen_paths.add(rel)
        if row["split"] != "train" or row["source_split"] != "train":
            raise RuntimeError(f"Manifest includes a non-Train row: {rel}")
        if label_name not in LABEL_TO_ID:
            raise RuntimeError(f"Unknown label {label_name!r} in {rel}")
        original = source_by_path.get(rel)
        if original is None:
            raise RuntimeError(f"Manifest record not present in original Train CSV: {rel}")
        for key in ("patient_id", "label", "sha256"):
            if row[key] != original[key]:
                raise RuntimeError(f"Manifest {key} mismatch for {rel}")
        if not row["sha256"]:
            raise RuntimeError(f"Missing SHA256 in manifest row: {rel}")
        if row["sha256"] in train_hashes:
            raise RuntimeError(f"Duplicate SHA256 inside manifest: {rel}")
        train_hashes.add(row["sha256"])
        selected_groups[patient_id].add(rel)

        image_path = (PROJECT_ROOT / rel).resolve()
        try:
            image_path.relative_to(dataset_root)
        except ValueError as exc:
            raise RuntimeError(f"Image path escapes dataset root: {rel}") from exc
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        train_paths.append(str(image_path))
        train_labels.append(LABEL_TO_ID[label_name])
        counts[label_name] += 1

    # A selected patient must contribute every image belonging to that patient.
    for patient_id, selected_paths in selected_groups.items():
        if selected_paths != source_groups.get(patient_id, set()):
            raise RuntimeError(f"Partial patient group in {level} manifest: {patient_id}")

    expected = EXPECTED_LEVEL_COUNTS[level]
    observed = {
        "images": len(manifest_rows),
        "patients": len(selected_groups),
        "NORMAL": counts["NORMAL"],
        "PNEUMONIA": counts["PNEUMONIA"],
    }
    if observed != expected:
        raise RuntimeError(f"{level} manifest counts differ from frozen audit: {observed}")
    level_meta = metadata.get("levels", {}).get(level, {})
    if int(level_meta.get("images", -1)) != observed["images"]:
        raise RuntimeError(f"{level} manifest metadata image count mismatch")
    if int(level_meta.get("patients", -1)) != observed["patients"]:
        raise RuntimeError(f"{level} manifest metadata patient count mismatch")

    # Validate only the fixed Validation CSV. Do not inspect test_v3.csv here.
    val_counts: Counter[str] = Counter()
    val_paths: list[str] = []
    val_labels: list[int] = []
    val_patients: set[str] = set()
    val_hashes: set[str] = set()
    for row in validation_rows:
        if row["split"] != "validation" or row["source_split"] != "train":
            raise RuntimeError(f"Invalid Validation assignment: {row['relative_path']}")
        if row["label"] not in LABEL_TO_ID:
            raise RuntimeError(f"Unknown Validation label: {row['label']}")
        if row["patient_id"] in selected_groups:
            raise RuntimeError(f"Train/Validation patient overlap: {row['patient_id']}")
        if row["sha256"] in train_hashes:
            raise RuntimeError(f"Train/Validation SHA256 overlap: {row['relative_path']}")
        image_path = (PROJECT_ROOT / row["relative_path"]).resolve()
        try:
            image_path.relative_to(dataset_root)
        except ValueError as exc:
            raise RuntimeError(f"Validation path escapes dataset root: {row['relative_path']}") from exc
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        val_paths.append(str(image_path))
        val_labels.append(LABEL_TO_ID[row["label"]])
        val_patients.add(row["patient_id"])
        val_hashes.add(row["sha256"])
        val_counts[row["label"]] += 1

    if len(val_patients) != 530:
        raise RuntimeError(f"Expected 530 Validation patients, found {len(val_patients)}")
    if (
        len(validation_rows) != EXPECTED_VALIDATION_COUNTS["images"]
        or val_counts["NORMAL"] != EXPECTED_VALIDATION_COUNTS["NORMAL"]
        or val_counts["PNEUMONIA"] != EXPECTED_VALIDATION_COUNTS["PNEUMONIA"]
        or len(val_hashes) != len(validation_rows)
    ):
        raise RuntimeError(f"Validation class/hash counts mismatch: {dict(val_counts)}")

    actual_fraction = len(manifest_rows) / len(source_rows)
    report = {
        "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": manifest_hash,
        "source_train_sha256": source_hash,
        "validation_sha256": validation_hash,
        "train_images": len(train_paths),
        "train_patients": len(selected_groups),
        "train_class_counts": dict(counts),
        "actual_train_image_fraction": actual_fraction,
        "validation_images": len(val_paths),
        "validation_patients": len(val_patients),
        "validation_class_counts": dict(val_counts),
        "train_validation_patient_overlap": 0,
        "train_validation_sha256_overlap": 0,
        "test_split_loaded": False,
        "test_split_evaluated": False,
    }
    return train_paths, train_labels, manifest_rows, val_paths, val_labels, validation_rows, report


# ---------------------------------------------------------------------------
# tf.data preprocessing — mirror existing resize/padding, then MobileNetV2 input
# ---------------------------------------------------------------------------

def build_dataset(
    paths: list[str],
    labels: list[int],
    *,
    training: bool,
    shuffle_seed: int | None = None,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, np.asarray(labels, dtype=np.int32)))
    if training:
        ds = ds.shuffle(
            buffer_size=max(1, len(paths)),
            seed=shuffle_seed if shuffle_seed is not None else SEED,
            reshuffle_each_iteration=False,
        )

    def decode_resize_preprocess(path: tf.Tensor, label: tf.Tensor):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize_with_pad(
            image,
            target_height=IMAGE_SIZE,
            target_width=IMAGE_SIZE,
            method="bilinear",
            antialias=True,
        )
        image = tf.cast(image, tf.float32)
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = tf.ensure_shape(image, (IMAGE_SIZE, IMAGE_SIZE, 3))
        label = tf.expand_dims(tf.cast(label, tf.float32), axis=-1)
        return image, label

    ds = ds.map(
        decode_resize_preprocess,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(PREFETCH_BUFFER)
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = NUM_PARALLEL_CALLS
    options.threading.max_intra_op_parallelism = 1
    return ds.with_options(options)


# ---------------------------------------------------------------------------
# Model architecture and trainability protocol
# ---------------------------------------------------------------------------

def build_frozen_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        name="mobilenetv2_backbone",
    )
    backbone.trainable = False
    # `training=False` keeps BatchNormalization in inference mode in both stages.
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.30, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_probability")(x)
    return tf.keras.Model(inputs, outputs, name="DL05_MobileNetV2_Transfer")


def configure_fine_tuning(model: tf.keras.Model) -> tuple[int, int]:
    backbone = model.get_layer("mobilenetv2_backbone")
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - UNFREEZE_LAST_N_LAYERS)
    for index, layer in enumerate(backbone.layers):
        if index < cutoff or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    # Ensure the classification head remains trainable.
    for name in ("global_average_pooling", "head_dropout", "pneumonia_probability"):
        model.get_layer(name).trainable = True
    return len(backbone.layers), cutoff


def compile_model(model: tf.keras.Model, stage: str) -> None:
    learning_rate = FROZEN_LEARNING_RATE if stage == "frozen" else FINE_TUNE_LEARNING_RATE
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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


def count_trainable_params(model: tf.keras.Model) -> int:
    return int(sum(np.prod(tuple(int(d) for d in w.shape)) for w in model.trainable_weights))


def save_model_summary(model: tf.keras.Model, path: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append, show_trainable=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# State and Validation artifact helpers
# ---------------------------------------------------------------------------

def initial_stage_state(stage: str, max_epochs: int, learning_rate: float) -> dict[str, Any]:
    return {
        "state_version": STATE_VERSION,
        "stage": stage,
        "training_status": "not_started",
        "completed_epochs": 0,
        "session_count": 0,
        "session_started_at": None,
        "session_finished_at": None,
        "best_validation_sklearn_auroc": None,
        "best_epoch": None,
        "early_stopping_best_auroc": None,
        "early_stopping_wait": 0,
        "lr_best_validation_loss": None,
        "lr_wait": 0,
        "lr_reductions": 0,
        "current_learning_rate": learning_rate,
        "last_epoch_metrics": None,
        "history": [],
        "max_epochs": max_epochs,
        "test_split_loaded": False,
        "test_split_evaluated": False,
    }


def load_stage_state(path: Path, stage: str, max_epochs: int, learning_rate: float):
    if not path.exists():
        return initial_stage_state(stage, max_epochs, learning_rate)
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("state_version") != STATE_VERSION or state.get("stage") != stage:
        raise RuntimeError(f"Invalid {stage} stage state; inspect before continuing: {path}")
    if len(state.get("history", [])) != int(state.get("completed_epochs", 0)):
        raise RuntimeError(f"{stage} history and completed_epochs disagree")
    if int(state.get("max_epochs", -1)) != max_epochs:
        raise RuntimeError(f"{stage} max_epochs changed; refusing to resume")
    if state.get("best_epoch") is not None:
        ckpt = path.parent.parent / "checkpoints" / f"best_{stage}.keras"
        if not ckpt.is_file():
            raise RuntimeError(f"Best {stage} checkpoint is missing: {ckpt}")
    return state


def save_validation_outputs(
    eval_dir: Path,
    records: list[dict[str, str]],
    y_true: np.ndarray,
    y_score: np.ndarray,
    epoch_number: int,
    stage: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    y_pred = (y_score >= DIAGNOSTIC_THRESHOLD).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    metrics = {
        "status": "best_checkpoint_validation_diagnostic",
        "split": "validation",
        "stage": stage,
        "epoch_within_stage": int(epoch_number),
        "num_images": int(len(y_true)),
        "num_normal": int(np.sum(y_true == 0)),
        "num_pneumonia": int(np.sum(y_true == 1)),
        "auroc_sklearn": float(roc_auc_score(y_true, y_score)),
        "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
        "threshold_locked": False,
        "threshold_optimization_performed": False,
        "accuracy_at_0.5": float(accuracy_score(y_true, y_pred)),
        "precision_at_0.5": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity_at_0.5": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_at_0.5": float(tn / (tn + fp)) if tn + fp else None,
        "macro_f1_at_0.5": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix_labels": ["NORMAL (0)", "PNEUMONIA (1)"],
        "confusion_matrix_at_0.5": cm.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "checkpoint_sha256": checkpoint_sha256,
    }
    pred_rows = [
        {
            "relative_path": row["relative_path"],
            "y_true": int(label),
            "y_score": float(score),
            "y_pred_at_0.5": int(pred),
        }
        for row, label, score, pred in zip(records, y_true, y_score, y_pred)
    ]
    atomic_write_csv(
        eval_dir / "predictions.csv",
        ["relative_path", "y_true", "y_score", "y_pred_at_0.5"],
        pred_rows,
    )
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    atomic_write_csv(
        eval_dir / "roc_curve.csv",
        ["fpr", "tpr", "threshold"],
        [
            {"fpr": float(x), "tpr": float(y), "threshold": float(t)}
            for x, y, t in zip(fpr, tpr, thresholds)
        ],
    )
    atomic_write_json(eval_dir / "metrics.json", metrics)
    return metrics


def update_overall_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    summary["updated_at"] = utc_now()
    frozen_status = summary["stages"]["frozen"].get("training_status", "not_started")
    fine_status = summary["stages"]["fine_tune"].get("training_status", "not_started")
    terminal = {"completed", "early_stopped"}
    if frozen_status in terminal and fine_status in terminal:
        summary["training_status"] = "completed"
        summary["active_stage"] = None
    elif frozen_status in terminal:
        summary["training_status"] = "stage1_complete_stage2_pending"
        summary["active_stage"] = "fine_tune"
    else:
        summary["training_status"] = "running"
        summary["active_stage"] = "frozen"
    atomic_write_json(run_dir / "training_summary.json", summary)


def select_validation_threshold(run_dir: Path) -> dict[str, Any]:
    pred_path = run_dir / "evaluation" / "validation_best" / "predictions.csv"
    rows = read_csv(pred_path)
    y_true = np.asarray([int(r["y_true"]) for r in rows], dtype=np.int32)
    y_score = np.asarray([float(r["y_score"]) for r in rows], dtype=np.float64)
    candidates = np.unique(y_score)[::-1]
    selected = None
    for threshold in candidates:
        y_pred = (y_score >= threshold).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
        if sensitivity >= TARGET_VALIDATION_SENSITIVITY:
            specificity = float(tn / (tn + fp)) if tn + fp else 0.0
            selected = {
                "threshold": float(threshold),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            }
            break  # Descending candidates: this is the highest threshold meeting target.
    if selected is None:
        raise RuntimeError(
            f"No observed Validation threshold satisfies Sensitivity >= {TARGET_VALIDATION_SENSITIVITY:.3f}"
        )
    payload = {
        "status": "locked_on_validation_before_test",
        "split_used_for_selection": "validation",
        "target_sensitivity": TARGET_VALIDATION_SENSITIVITY,
        "threshold_selection_method": "highest observed validation score whose sensitivity meets target",
        "threshold_tuned_on_test": False,
        "test_split_loaded": False,
        "test_split_evaluated": False,
        **selected,
    }
    out_dir = run_dir / "evaluation" / "validation_threshold_locked"
    atomic_write_json(out_dir / "threshold_selection.json", payload)
    locked_rows = []
    for row in rows:
        pred = int(float(row["y_score"]) >= selected["threshold"])
        locked_rows.append({**row, "y_pred_at_locked_threshold": pred})
    atomic_write_csv(
        out_dir / "predictions_at_locked_threshold.csv",
        ["relative_path", "y_true", "y_score", "y_pred_at_0.5", "y_pred_at_locked_threshold"],
        locked_rows,
    )
    return payload


# ---------------------------------------------------------------------------
# Callbacks — one epoch per process, persisted state and BackupAndRestore.
# ---------------------------------------------------------------------------

class PersistentValidationAUROCController(tf.keras.callbacks.Callback):
    def __init__(
        self,
        stage: str,
        state: dict[str, Any],
        summary: dict[str, Any],
        run_dir: Path,
        validation_dataset: tf.data.Dataset,
        validation_labels: list[int],
        validation_records: list[dict[str, str]],
        session_start_epoch: int,
    ):
        super().__init__()
        self.stage = stage
        self.state = state
        self.summary = summary
        self.run_dir = run_dir
        self.validation_dataset = validation_dataset
        self.validation_labels = np.asarray(validation_labels, dtype=np.int32)
        self.validation_records = validation_records
        self.session_start_epoch = session_start_epoch

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs if logs is not None else {}
        completed_epoch = int(epoch) + 1
        expected_epoch = int(self.state["completed_epochs"]) + 1
        if completed_epoch != expected_epoch:
            raise RuntimeError(f"Epoch mismatch for {self.stage}: Keras={completed_epoch}, state={expected_epoch}")

        scores = self.model.predict(self.validation_dataset, verbose=0)
        y_score = np.asarray(scores, dtype=np.float32).reshape(-1)
        y_true = self.validation_labels
        if len(y_score) != len(y_true) or len(y_score) != len(self.validation_records):
            raise RuntimeError("Validation prediction count/order contract failed")
        if not np.all(np.isfinite(y_score)) or np.any(y_score < 0.0) or np.any(y_score > 1.0):
            raise FloatingPointError("Validation predictions contain invalid probabilities")

        val_auc = float(roc_auc_score(y_true, y_score))
        logs["val_sklearn_auroc"] = val_auc
        val_loss = to_float(logs["val_loss"])
        lr_used = to_float(self.model.optimizer.learning_rate)

        stage_best = self.state["best_validation_sklearn_auroc"]
        if stage_best is None or val_auc > float(stage_best):
            ckpt_dir = self.run_dir / "checkpoints"
            stage_best_path = ckpt_dir / f"best_{self.stage}.keras"
            temp_path = ckpt_dir / f"best_{self.stage}.tmp.keras"
            if temp_path.exists():
                temp_path.unlink()
            self.model.save(temp_path, overwrite=True)
            os.replace(temp_path, stage_best_path)
            stage_hash = sha256_file(stage_best_path)
            stage_eval_dir = self.run_dir / "evaluation" / f"validation_{self.stage}_best"
            metrics = save_validation_outputs(
                stage_eval_dir, self.validation_records, y_true, y_score,
                completed_epoch, self.stage, stage_hash,
            )
            self.state["best_validation_sklearn_auroc"] = val_auc
            self.state["best_epoch"] = completed_epoch
            self.state["best_checkpoint_path"] = str(stage_best_path.relative_to(PROJECT_ROOT))
            self.state["best_checkpoint_sha256"] = stage_hash
            self.state["best_validation_metrics"] = metrics

            overall_best = self.summary.get("best_validation_sklearn_auroc")
            if overall_best is None or val_auc > float(overall_best):
                overall_path = ckpt_dir / "best.keras"
                shutil.copy2(stage_best_path, overall_path)
                overall_hash = sha256_file(overall_path)
                overall_eval_dir = self.run_dir / "evaluation" / "validation_best"
                overall_metrics = save_validation_outputs(
                    overall_eval_dir, self.validation_records, y_true, y_score,
                    completed_epoch, self.stage, overall_hash,
                )
                self.summary["best_validation_sklearn_auroc"] = val_auc
                self.summary["best_stage"] = self.stage
                self.summary["best_epoch_within_stage"] = completed_epoch
                self.summary["best_checkpoint_path"] = str(overall_path.relative_to(PROJECT_ROOT))
                self.summary["best_checkpoint_sha256"] = overall_hash
                self.summary["best_validation_metrics"] = overall_metrics
            print(f"\nBest {self.stage} Validation sklearn AUROC: {val_auc:.6f} (epoch {completed_epoch})")

        early_best = self.state["early_stopping_best_auroc"]
        if early_best is None or val_auc > float(early_best) + EARLY_STOPPING_MIN_DELTA:
            self.state["early_stopping_best_auroc"] = val_auc
            self.state["early_stopping_wait"] = 0
        else:
            self.state["early_stopping_wait"] = int(self.state["early_stopping_wait"]) + 1

        lr_best_loss = self.state["lr_best_validation_loss"]
        if lr_best_loss is None or val_loss < float(lr_best_loss) - 1e-4:
            self.state["lr_best_validation_loss"] = val_loss
            self.state["lr_wait"] = 0
        else:
            self.state["lr_wait"] = int(self.state["lr_wait"]) + 1
        if int(self.state["lr_wait"]) >= REDUCE_LR_PATIENCE:
            old_lr = to_float(self.model.optimizer.learning_rate)
            new_lr = max(old_lr * REDUCE_LR_FACTOR, MIN_LEARNING_RATE)
            if new_lr < old_lr - 1e-12:
                self.model.optimizer.learning_rate.assign(new_lr)
                self.state["lr_reductions"] = int(self.state["lr_reductions"]) + 1
                print(f"Reduce LR: {old_lr:.8g} -> {new_lr:.8g}")
            self.state["lr_wait"] = 0

        required = (
            "loss", "accuracy", "keras_auroc_approx", "precision", "recall",
            "val_loss", "val_accuracy", "val_keras_auroc_approx", "val_precision", "val_recall",
        )
        missing = [key for key in required if key not in logs]
        if missing:
            raise RuntimeError(f"Missing Keras training metrics: {missing}")
        row = {key: to_float(logs[key]) for key in required}
        row.update({
            "epoch": completed_epoch,
            "val_sklearn_auroc": val_auc,
            "learning_rate_used": lr_used,
            "learning_rate_after_epoch": to_float(self.model.optimizer.learning_rate),
        })
        self.state["history"].append(row)
        self.state["completed_epochs"] = completed_epoch
        self.state["current_learning_rate"] = to_float(self.model.optimizer.learning_rate)
        self.state["last_epoch_metrics"] = row
        self.state["updated_at"] = utc_now()

        max_epochs = int(self.state["max_epochs"])
        if int(self.state["early_stopping_wait"]) >= EARLY_STOPPING_PATIENCE:
            self.state["training_status"] = "early_stopped"
            self.state["stop_reason"] = (
                f"Validation sklearn AUROC did not improve by {EARLY_STOPPING_MIN_DELTA} "
                f"for {EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
        elif completed_epoch >= max_epochs:
            self.state["training_status"] = "completed"
            self.state["stop_reason"] = "Maximum stage epoch count reached."
        elif completed_epoch - self.session_start_epoch >= 1:
            self.state["training_status"] = "session_complete"
            self.state["stop_reason"] = "One-epoch process session limit reached."
        else:
            self.state["training_status"] = "running"
            self.state["stop_reason"] = None

        atomic_write_json(self.run_dir / "training_state" / f"{self.stage}_state.json", self.state)
        history_fields = list(row.keys())
        atomic_write_csv(self.run_dir / f"history_{self.stage}.csv", history_fields, self.state["history"])
        self.summary["stages"][self.stage] = {
            key: self.state.get(key)
            for key in (
                "training_status", "completed_epochs", "max_epochs", "session_count",
                "best_validation_sklearn_auroc", "best_epoch", "best_checkpoint_path",
                "best_checkpoint_sha256", "early_stopping_wait", "lr_reductions", "last_epoch_metrics",
            )
        }
        update_overall_summary(self.run_dir, self.summary)
        if self.state["training_status"] in {"early_stopped", "completed", "session_complete"}:
            self.model.stop_training = True

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        self.state["session_finished_at"] = utc_now()
        if getattr(self.model, "optimizer", None) is not None:
            self.state["current_learning_rate"] = to_float(self.model.optimizer.learning_rate)
        atomic_write_json(self.run_dir / "training_state" / f"{self.stage}_state.json", self.state)
        self.summary["stages"][self.stage] = {
            key: self.state.get(key)
            for key in (
                "training_status", "completed_epochs", "max_epochs", "session_count",
                "best_validation_sklearn_auroc", "best_epoch", "best_checkpoint_path",
                "best_checkpoint_sha256", "early_stopping_wait", "lr_reductions", "last_epoch_metrics",
            )
        }
        update_overall_summary(self.run_dir, self.summary)


class FiniteLossGuard(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        if "loss" in logs and not math.isfinite(to_float(logs["loss"])):
            raise FloatingPointError(f"Non-finite loss at training batch {batch}")


class ResumeStateValidator(tf.keras.callbacks.Callback):
    def __init__(self, state: dict[str, Any]):
        super().__init__()
        self.state = state

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        restored_epoch = int(getattr(self.model, "_initial_epoch", 0) or 0)
        expected_epoch = int(self.state["completed_epochs"])
        if restored_epoch != expected_epoch:
            raise RuntimeError(
                f"BackupAndRestore epoch mismatch: restored={restored_epoch}, expected={expected_epoch}"
            )
        restored_lr = to_float(self.model.optimizer.learning_rate)
        expected_lr = float(self.state["current_learning_rate"])
        if not math.isclose(restored_lr, expected_lr, rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(f"Restored LR mismatch: restored={restored_lr}, expected={expected_lr}")
        print(f"Resume verification: PASS — {self.state['stage']} stage, epoch {restored_epoch + 1}")


# ---------------------------------------------------------------------------
# Experiment state
# ---------------------------------------------------------------------------

def make_experiment_config(level: str, run_name: str, data_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_name": run_name,
        "model_name": "MobileNetV2_ImageNet_Transfer_Learning",
        "backbone": "tf.keras.applications.MobileNetV2",
        "backbone_weights": "imagenet",
        "backbone_include_top": False,
        "label_efficiency_level": level,
        "train_manifest": data_report["manifest_path"],
        "train_manifest_sha256": data_report["manifest_sha256"],
        "source_train_sha256": data_report["source_train_sha256"],
        "validation_sha256": data_report["validation_sha256"],
        "train_images": data_report["train_images"],
        "train_patients": data_report["train_patients"],
        "train_class_counts": data_report["train_class_counts"],
        "validation_images": data_report["validation_images"],
        "validation_class_counts": data_report["validation_class_counts"],
        "actual_train_image_fraction": data_report["actual_train_image_fraction"],
        "image_size": IMAGE_SIZE,
        "resize_method": "tf.image.resize_with_pad_bilinear_antialias",
        "input_preprocessing": "tf.keras.applications.mobilenet_v2.preprocess_input; raw pixels [0,255] mapped to [-1,1]",
        "batch_size": BATCH_SIZE,
        "random_seed": SEED,
        "shuffle_seed_policy": "seed_plus_completed_stage_epochs",
        "augmentation": "none; same deterministic resize/pad protocol as CNN V2",
        "class_weighting": "none; same unweighted binary cross entropy protocol as CNN V2",
        "stage1": {
            "name": "frozen_backbone_feature_extraction",
            "max_epochs": FROZEN_MAX_EPOCHS,
            "learning_rate": FROZEN_LEARNING_RATE,
            "backbone_trainable": False,
        },
        "stage2": {
            "name": "fine_tuning_last_backbone_layers",
            "max_epochs": FINE_TUNE_MAX_EPOCHS,
            "learning_rate": FINE_TUNE_LEARNING_RATE,
            "unfreeze_last_n_backbone_layers": UNFREEZE_LAST_N_LAYERS,
            "batch_normalization_trainable": False,
            "backbone_called_with_training_false": True,
        },
        "loss": "BinaryCrossentropy",
        "optimizer": "Adam",
        "early_stopping_monitor": "exact_sklearn_validation_auroc",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "epochs_per_process_invocation": 1,
        "checkpoint_selection": "highest exact sklearn Validation AUROC across both stages",
        "operating_threshold_policy": {
            "target_validation_sensitivity": TARGET_VALIDATION_SENSITIVITY,
            "method": "highest observed Validation score meeting target sensitivity",
        },
        "test_split_loaded": False,
        "test_split_evaluated": False,
    }


def load_or_initialize_experiment(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    summary_path = run_dir / "training_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("state_version") != STATE_VERSION:
            raise RuntimeError("Unsupported experiment state version; inspect before resuming")
        if summary.get("experiment_config") != config:
            raise RuntimeError("Saved experiment config differs; refusing to overwrite or resume")
        return summary
    if run_dir.exists() and directory_has_files(run_dir):
        raise RuntimeError(
            f"Run directory contains files but has no training_summary.json: {run_dir}. "
            "Inspect/archive it; this script will not overwrite it."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("checkpoints", "training_state", "evaluation"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    summary = {
        "state_version": STATE_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "experiment_config": config,
        "training_status": "not_started",
        "active_stage": "frozen",
        "stages": {
            "frozen": {"training_status": "not_started", "completed_epochs": 0},
            "fine_tune": {"training_status": "not_started", "completed_epochs": 0},
        },
        "best_validation_sklearn_auroc": None,
        "best_stage": None,
        "best_epoch_within_stage": None,
        "best_checkpoint_path": None,
        "best_checkpoint_sha256": None,
        "best_validation_metrics": None,
        "test_split_loaded": False,
        "test_split_evaluated": False,
    }
    atomic_write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=LEVELS, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--preflight-only", action="store_true", help="Validate frozen data/manifests; do not load pretrained weights or train")
    args = parser.parse_args()

    default_name = f"mobilenetv2_{args.level}_imagenet_20261122"
    run_name = args.run_name or default_name
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        parser.error("--run-name may contain only letters, numbers, underscore, and hyphen")
    run_dir = PROJECT_ROOT / "outputs" / "experiments" / run_name

    print("=" * 78)
    print("DL-05 — MobileNetV2 Transfer Learning / Label Efficiency")
    print("=" * 78)
    print("Project root:", PROJECT_ROOT)
    print("Level:", args.level)

    data = validate_manifest_and_validation(args.level)
    train_paths, train_labels, train_rows, val_paths, val_labels, val_rows, data_report = data
    print("\n[FROZEN DATA AUDIT: PASS]")
    print(json.dumps(data_report, indent=2, ensure_ascii=False))
    print("Test split loaded/evaluated: NO")
    if args.preflight_only:
        print("\nPreflight only: no pretrained model loaded and no training started.")
        return 0

    runtime = configure_runtime()
    print("\nTensorFlow:", tf.__version__)
    print("Keras:", tf.keras.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print("Threading:", runtime["threading"])

    config = make_experiment_config(args.level, run_name, data_report)
    summary = load_or_initialize_experiment(run_dir, config)

    frozen_state_path = run_dir / "training_state" / "frozen_state.json"
    fine_state_path = run_dir / "training_state" / "fine_tune_state.json"
    frozen_state = load_stage_state(frozen_state_path, "frozen", FROZEN_MAX_EPOCHS, FROZEN_LEARNING_RATE)
    fine_state = load_stage_state(fine_state_path, "fine_tune", FINE_TUNE_MAX_EPOCHS, FINE_TUNE_LEARNING_RATE)
    terminal = {"completed", "early_stopped"}

    if frozen_state["training_status"] not in terminal:
        stage = "frozen"
        stage_state = frozen_state
        stage_state_path = frozen_state_path
        max_epochs = FROZEN_MAX_EPOCHS
    elif fine_state["training_status"] not in terminal:
        stage = "fine_tune"
        stage_state = fine_state
        stage_state_path = fine_state_path
        max_epochs = FINE_TUNE_MAX_EPOCHS
    else:
        if not (run_dir / "checkpoints" / "best.keras").is_file():
            raise RuntimeError("Both stages are complete but overall best checkpoint is missing")
        if not (run_dir / "evaluation" / "validation_threshold_locked" / "threshold_selection.json").is_file():
            threshold_result = select_validation_threshold(run_dir)
            summary["validation_operating_threshold"] = threshold_result
            summary["training_status"] = "completed"
            summary["active_stage"] = None
            atomic_write_json(run_dir / "training_summary.json", summary)
        print("\nThis Transfer Learning run is already complete.")
        print("Best validation AUROC:", summary.get("best_validation_sklearn_auroc"))
        print("Best stage:", summary.get("best_stage"))
        print("Operating threshold:", summary.get("validation_operating_threshold", {}).get("threshold"))
        print("Test split loaded/evaluated: NO")
        return 0

    if stage_state["training_status"] in terminal:
        raise RuntimeError(f"Internal stage selection error: {stage} is already terminal")
    if stage_state["completed_epochs"] > 0 and not directory_has_files(run_dir / "training_state" / "keras_backup" / stage):
        raise RuntimeError(
            f"BackupAndRestore files missing for {stage}; refusing to restart stage from a different state."
        )
    if stage == "fine_tune" and not (run_dir / "checkpoints" / "best_frozen.keras").is_file():
        raise RuntimeError("Stage 1 best checkpoint is required before fine-tuning")

    # Recreate the same stage architecture and compile settings on every invocation.
    if stage == "frozen":
        model = build_frozen_model()
    else:
        model = tf.keras.models.load_model(
            run_dir / "checkpoints" / "best_frozen.keras",
            compile=False,
        )
        num_backbone_layers, cutoff = configure_fine_tuning(model)
        print(f"Fine-tune backbone: {num_backbone_layers} layers; unfreeze indices [{cutoff}, {num_backbone_layers}) except BatchNormalization")
    compile_model(model, stage)

    if tuple(model.input_shape) != (None, IMAGE_SIZE, IMAGE_SIZE, 3) or tuple(model.output_shape) != (None, 1):
        raise RuntimeError(f"Unexpected model tensor shapes: input={model.input_shape}, output={model.output_shape}")

    save_model_summary(model, run_dir / f"model_summary_{stage}.txt")
    total_params = int(model.count_params())
    trainable_params = count_trainable_params(model)
    non_trainable_params = total_params - trainable_params

    completed_before = int(stage_state["completed_epochs"])
    tf.keras.utils.set_random_seed(SEED + completed_before)
    train_dataset = build_dataset(
        train_paths,
        train_labels,
        training=True,
        shuffle_seed=SEED + completed_before,
    )
    validation_dataset = build_dataset(val_paths, val_labels, training=False)

    # Persist/verify architecture counts as experiment metadata.
    stage_state["total_params"] = total_params
    stage_state["trainable_params"] = trainable_params
    stage_state["non_trainable_params"] = non_trainable_params
    stage_state["input_shape"] = list(model.input_shape)
    stage_state["output_shape"] = list(model.output_shape)
    atomic_write_json(stage_state_path, stage_state)

    controller = PersistentValidationAUROCController(
        stage=stage,
        state=stage_state,
        summary=summary,
        run_dir=run_dir,
        validation_dataset=validation_dataset,
        validation_labels=val_labels,
        validation_records=val_rows,
        session_start_epoch=completed_before,
    )
    backup_dir = run_dir / "training_state" / "keras_backup" / stage
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = tf.keras.callbacks.BackupAndRestore(
        backup_dir=str(backup_dir),
        save_freq="epoch",
        double_checkpoint=True,
        delete_checkpoint=False,
    )
    resume_validator = ResumeStateValidator(stage_state)

    stage_state["session_count"] = int(stage_state["session_count"]) + 1
    stage_state["session_started_at"] = utc_now()
    stage_state["session_finished_at"] = None
    stage_state["training_status"] = "running"
    stage_state["stop_reason"] = None
    atomic_write_json(stage_state_path, stage_state)

    print("\nStage:", stage)
    print("Train images/patients:", len(train_paths), len({r['patient_id'] for r in train_rows}))
    print("Train classes:", dict(Counter(r["label"] for r in train_rows)))
    print("Validation images:", len(val_paths))
    print("Model tensor shapes:", model.input_shape, "->", model.output_shape)
    print("Total/trainable/non-trainable parameters:", f"{total_params:,}", f"{trainable_params:,}", f"{non_trainable_params:,}")
    print("Stage max epochs:", max_epochs)
    print("Completed epochs before this invocation:", completed_before)
    print("One epoch will run in this process invocation.")
    print("Class weighting: none")
    print("Operating threshold: selected later on Validation only; target sensitivity >= 0.99")
    print("Test split loaded/evaluated: NO")

    # BackupAndRestore restores model/optimizer state from the previous one-epoch session.
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=max_epochs,
        callbacks=[controller, backup, resume_validator, FiniteLossGuard()],
        verbose=1,
        shuffle=False,
    )

    latest_summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
    latest_stage_state = json.loads(stage_state_path.read_text(encoding="utf-8"))
    print("\n" + "=" * 78)
    print("MOBILENETV2 TRAINING SESSION RESULT")
    print("=" * 78)
    print("Run:", run_name)
    print("Level:", args.level)
    print("Stage:", stage)
    print("Status:", latest_stage_state["training_status"])
    print("Completed epochs:", f"{latest_stage_state['completed_epochs']} / {max_epochs}")
    print("Stage best Validation sklearn AUROC:", latest_stage_state["best_validation_sklearn_auroc"])
    print("Overall best Validation sklearn AUROC:", latest_summary.get("best_validation_sklearn_auroc"))
    print("Overall best checkpoint:", latest_summary.get("best_checkpoint_path"))
    print("Test split loaded/evaluated: NO")

    if latest_stage_state["training_status"] == "session_complete":
        print("\nSESSION COMPLETE. Repeat the exact same command to resume this stage.")
    elif latest_stage_state["training_status"] in terminal and stage == "frozen":
        print("\nSTAGE 1 COMPLETE. Repeat the exact same command; the next invocation will start fine-tuning.")
    elif latest_stage_state["training_status"] in terminal and stage == "fine_tune":
        threshold_result = select_validation_threshold(run_dir)
        latest_summary["validation_operating_threshold"] = threshold_result
        latest_summary["training_status"] = "completed"
        latest_summary["active_stage"] = None
        atomic_write_json(run_dir / "training_summary.json", latest_summary)
        print("\nTRANSFER LEARNING RUN COMPLETE.")
        print("Validation threshold:", threshold_result["threshold"])
        print("Validation sensitivity:", threshold_result["sensitivity"])
        print("Validation specificity:", threshold_result["specificity"])
        print("No Test data was accessed.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
