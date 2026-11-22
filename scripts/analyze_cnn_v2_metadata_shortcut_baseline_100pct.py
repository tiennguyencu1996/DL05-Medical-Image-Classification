#!/usr/bin/env python3
"""Patient-level metadata-only diagnostic baseline for CNN V2 100%.

Purpose:
- Test whether simple acquisition/geometry/storage descriptors alone carry
  information about NORMAL vs PNEUMONIA in Train and Validation.
- Aggregate image descriptors to one median row per patient first.
- Use 5-fold stratified CV on Train patients and one diagnostic Validation run.

Strict split policy:
- Reads only train_validation_image_quality_features.csv.
- Does not open Test manifests, Test images, Test predictions, or Test features.
- Does not load the CNN, run inference, tune thresholds, or retrain the CNN.
- This is an auxiliary, metadata-only diagnostic baseline, not a replacement
  model and not a clinical evaluation.

Run from the project root:
    PYTHONPATH="$PWD" python scripts/analyze_cnn_v2_metadata_shortcut_baseline_100pct.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "cnn_from_scratch_v2_100pct_full_20261121"
RUN_DIR = PROJECT_ROOT / "outputs" / "experiments" / RUN_NAME
INPUT_CSV = (
    RUN_DIR / "evaluation" / "source_split_distribution_analysis_100pct"
    / "train_validation_image_quality_features.csv"
)
OUTPUT_DIR = RUN_DIR / "evaluation" / "metadata_shortcut_audit_100pct"

EXPECTED_PATIENT_COUNTS = {
    "Train": {"NORMAL": 806, "PNEUMONIA": 1314},
    "Validation": {"NORMAL": 202, "PNEUMONIA": 328},
}
EXPECTED_IMAGE_COUNTS = {"Train": 4164, "Validation": 1042}

FEATURE_SETS = {
    # Geometry can affect the visible image framing/padding after resize_with_pad.
    "geometry_only": [
        "width_px", "height_px", "aspect_ratio_w_over_h", "pixel_count",
    ],
    # File size is NOT fed to the CNN; this set is only a source/storage proxy.
    "jpeg_storage_proxy_only": ["file_size_kb"],
    # Grayscale/texture proxies; not clinical features.
    "appearance_quality_proxies": [
        "gray_mean_0_255", "gray_std_0_255", "gray_p01", "gray_p05",
        "gray_median", "gray_p95", "gray_p99",
        "robust_contrast_p95_minus_p05", "low_intensity_fraction_le_5",
        "high_intensity_fraction_ge_250", "gray_entropy_bits",
        "laplacian_variance_after_224_longest_side_resize",
    ],
    "all_descriptors": [
        "width_px", "height_px", "aspect_ratio_w_over_h", "pixel_count",
        "file_size_kb", "gray_mean_0_255", "gray_std_0_255", "gray_p01",
        "gray_p05", "gray_median", "gray_p95", "gray_p99",
        "robust_contrast_p95_minus_p05", "low_intensity_fraction_le_5",
        "high_intensity_fraction_ge_250", "gray_entropy_bits",
        "laplacian_variance_after_224_longest_side_resize",
    ],
}

REQUIRED_COLUMNS = {
    "split", "relative_path", "patient_id", "sha256", "y_true",
    "y_true_name", *{f for fields in FEATURE_SETS.values() for f in fields},
}


def read_patient_rows(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required Train/Validation feature file not found: {path}\n"
            "Run analyze_cnn_v2_source_split_distribution_100pct.py first."
        )

    split_patient_images: dict[str, dict[str, list[dict[str, Any]]]] = {
        "Train": defaultdict(list), "Validation": defaultdict(list)
    }
    seen_paths: dict[str, set[str]] = {"Train": set(), "Validation": set()}
    seen_hashes: dict[str, set[str]] = {"Train": set(), "Validation": set()}
    patient_labels: dict[str, dict[str, int]] = {"Train": {}, "Validation": {}}
    image_counts: Counter[str] = Counter()

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            split = row["split"]
            if split not in ("Train", "Validation"):
                raise ValueError(
                    f"Unexpected split {split!r} at row {row_number}; "
                    "this script must contain Train/Validation only."
                )
            label_name = row["y_true_name"]
            if label_name not in ("NORMAL", "PNEUMONIA"):
                raise ValueError(f"Unexpected class {label_name!r} at row {row_number}")
            label = int(row["y_true"])
            if label != (0 if label_name == "NORMAL" else 1):
                raise ValueError(f"Label mapping mismatch at row {row_number}")

            patient_id = row["patient_id"]
            rel_path = row["relative_path"]
            digest = row["sha256"].lower()
            if not patient_id or not rel_path or not digest:
                raise ValueError(f"Missing patient/path/hash at row {row_number}")
            if rel_path in seen_paths[split]:
                raise ValueError(f"Duplicate image path in {split}: {rel_path}")
            if digest in seen_hashes[split]:
                raise ValueError(f"Duplicate SHA-256 in {split}: {digest}")
            seen_paths[split].add(rel_path)
            seen_hashes[split].add(digest)

            previous_label = patient_labels[split].get(patient_id)
            if previous_label is not None and previous_label != label:
                raise ValueError(f"Patient {patient_id} has inconsistent labels in {split}")
            patient_labels[split][patient_id] = label

            features: dict[str, float] = {}
            for feature in REQUIRED_COLUMNS:
                if feature in ("split", "relative_path", "patient_id", "sha256", "y_true", "y_true_name"):
                    continue
                value = float(row[feature])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {feature} at row {row_number}")
                features[feature] = value

            split_patient_images[split][patient_id].append({
                "relative_path": rel_path,
                "y_true": label,
                "y_true_name": label_name,
                **features,
            })
            image_counts[split] += 1

    # Ensure Train and Validation do not share a patient or source hash.
    train_patients = set(split_patient_images["Train"])
    val_patients = set(split_patient_images["Validation"])
    if train_patients & val_patients:
        raise RuntimeError("Train/Validation patient overlap found in feature file")
    if seen_hashes["Train"] & seen_hashes["Validation"]:
        raise RuntimeError("Train/Validation SHA-256 overlap found in feature file")

    for split in ("Train", "Validation"):
        if image_counts[split] != EXPECTED_IMAGE_COUNTS[split]:
            raise ValueError(
                f"{split}: expected {EXPECTED_IMAGE_COUNTS[split]} image rows, "
                f"found {image_counts[split]}"
            )
        observed = Counter(
            "NORMAL" if label == 0 else "PNEUMONIA"
            for label in patient_labels[split].values()
        )
        if dict(observed) != EXPECTED_PATIENT_COUNTS[split]:
            raise ValueError(
                f"{split}: patient class counts mismatch: {dict(observed)}"
            )

    patient_data: dict[str, list[dict[str, Any]]] = {"Train": [], "Validation": []}
    for split in ("Train", "Validation"):
        for patient_id, image_rows in sorted(split_patient_images[split].items()):
            label = image_rows[0]["y_true"]
            record: dict[str, Any] = {
                "split": split,
                "patient_id": patient_id,
                "y_true": label,
                "class": "NORMAL" if label == 0 else "PNEUMONIA",
                "num_images": len(image_rows),
            }
            for feature in sorted({f for fs in FEATURE_SETS.values() for f in fs}):
                record[feature] = float(np.median([r[feature] for r in image_rows]))
            patient_data[split].append(record)

    audit = {
        "input_csv": str(path.relative_to(PROJECT_ROOT)),
        "image_rows": dict(image_counts),
        "patient_counts": {
            split: len(patient_data[split]) for split in ("Train", "Validation")
        },
        "patient_class_counts": {
            split: dict(Counter(r["class"] for r in patient_data[split]))
            for split in ("Train", "Validation")
        },
        "train_validation_patient_overlap": 0,
        "train_validation_sha256_overlap": 0,
        "test_data_read": False,
    }
    return patient_data, audit


def make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])


def metrics_at_default_05(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    y_pred = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "threshold_for_descriptive_class_metrics": 0.5,
        "threshold_was_tuned": False,
        "accuracy_at_0_5": float(accuracy_score(y_true, y_pred)),
        "sensitivity_at_0_5": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity_at_0_5": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision_at_0_5": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1_at_0_5": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "confusion_matrix_labels": ["NORMAL (0)", "PNEUMONIA (1)"],
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def main() -> int:
    print("=" * 78)
    print("CNN V2 100% — PATIENT-LEVEL METADATA-ONLY DIAGNOSTIC BASELINE")
    print("=" * 78)
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {OUTPUT_DIR}")

    patients, audit = read_patient_rows(INPUT_CSV)
    train = patients["Train"]
    validation = patients["Validation"]
    y_train = np.asarray([r["y_true"] for r in train], dtype=np.int32)
    y_val = np.asarray([r["y_true"] for r in validation], dtype=np.int32)

    print("Train patients:", len(train), "| Validation patients:", len(validation))
    print("Test data read: NO")
    print("Patient aggregation: median of each patient's image features")

    split_results: dict[str, Any] = {}
    cv_prediction_rows: list[dict[str, Any]] = []
    val_prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for set_name, feature_names in FEATURE_SETS.items():
        X_train = np.asarray([[r[f] for f in feature_names] for r in train], dtype=np.float64)
        X_val = np.asarray([[r[f] for f in feature_names] for r in validation], dtype=np.float64)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_scores = cross_val_predict(
            make_model(), X_train, y_train, cv=cv,
            method="predict_proba", n_jobs=1,
        )[:, 1]
        cv_result = metrics_at_default_05(y_train, oof_scores)

        # Fit once on all Train patients; Validation is diagnostic only.
        fitted = make_model()
        fitted.fit(X_train, y_train)
        val_scores = fitted.predict_proba(X_val)[:, 1]
        val_result = metrics_at_default_05(y_val, val_scores)

        split_results[set_name] = {
            "features": feature_names,
            "train_5fold_patient_level_oof": cv_result,
            "validation_diagnostic_only": val_result,
            "cv_method": "5-fold StratifiedKFold over one aggregated row per Train patient",
            "model": "StandardScaler + LogisticRegression(class_weight='balanced', C=1.0 default)",
        }

        for rec, score in zip(train, oof_scores):
            cv_prediction_rows.append({
                "feature_set": set_name, "patient_id": rec["patient_id"],
                "class": rec["class"], "y_true": rec["y_true"],
                "oof_score": format(float(score), ".17g"),
                "pred_at_0.5": int(score >= 0.5), "num_images": rec["num_images"],
            })
        for rec, score in zip(validation, val_scores):
            val_prediction_rows.append({
                "feature_set": set_name, "patient_id": rec["patient_id"],
                "class": rec["class"], "y_true": rec["y_true"],
                "score": format(float(score), ".17g"),
                "pred_at_0.5": int(score >= 0.5), "num_images": rec["num_images"],
            })

        coefs = fitted.named_steps["logistic_regression"].coef_[0]
        for feature, coef in zip(feature_names, coefs):
            coefficient_rows.append({
                "feature_set": set_name,
                "feature": feature,
                "standardized_logistic_coefficient": float(coef),
            })

        print(f"\n[{set_name}]")
        print("Train 5-fold OOF AUROC:", f"{cv_result['auroc']:.6f}")
        print("Validation diagnostic AUROC:", f"{val_result['auroc']:.6f}")

    result = {
        "status": "completed_patient_level_metadata_only_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": RUN_NAME,
        "purpose": "Determine whether non-CNN acquisition/geometry/storage and image-quality descriptors are associated with class labels.",
        "data_audit": audit,
        "patient_aggregation": "Median feature value across each patient's images; one row per patient.",
        "model_protocol": {
            "model": "StandardScaler + LogisticRegression",
            "class_weight": "balanced",
            "regularization_C": 1.0,
            "hyperparameter_search": False,
            "threshold_tuning": False,
            "descriptive_classification_threshold": 0.5,
            "train_evaluation": "5-fold stratified out-of-fold predictions at patient level",
            "validation_evaluation": "single diagnostic holdout; not used for tuning or model selection",
        },
        "feature_set_results": split_results,
        "interpretation_limits": [
            "This is a simple metadata/descriptor-only diagnostic, not a CNN explanation.",
            "File size is not directly input to the CNN; it is only a storage/source proxy.",
            "Geometry/aspect ratio may become visible to the CNN through resizing and padding, but this experiment does not prove the CNN used those cues.",
            "Grayscale and sharpness descriptors are crude proxies and are not clinical features.",
            "The patient-level aggregation reduces image-level dependence but does not establish causality.",
            "Validation metrics are diagnostic only and must not be used to tune the CNN or its threshold.",
            "Test manifests, Test images, Test predictions, and Test feature files were not read.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    (OUTPUT_DIR / "metadata_baseline_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def write_rows(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
        with (OUTPUT_DIR / name).open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write_rows("train_patient_oof_predictions.csv", cv_prediction_rows,
               ["feature_set", "patient_id", "class", "y_true", "oof_score", "pred_at_0.5", "num_images"])
    write_rows("validation_patient_diagnostic_predictions.csv", val_prediction_rows,
               ["feature_set", "patient_id", "class", "y_true", "score", "pred_at_0.5", "num_images"])
    write_rows("logistic_coefficients.csv", coefficient_rows,
               ["feature_set", "feature", "standardized_logistic_coefficient"])

    readme = """CNN V2 100% — Patient-Level Metadata-Only Diagnostic Baseline

Purpose:
Assess whether image geometry, storage metadata, and simple grayscale/quality
proxies alone are associated with NORMAL/PNEUMONIA labels.

Method:
- Reads only train_validation_image_quality_features.csv.
- Collapses multiple images per patient using per-feature medians.
- Uses 5-fold stratified out-of-fold evaluation on Train patients.
- Fits on all Train patients and reports Validation as diagnostic only.
- Uses StandardScaler + LogisticRegression (class_weight=balanced, C=1.0).
- No hyperparameter search and no threshold tuning; descriptive predictions use 0.5.

Strict exclusion:
- Does not read Test manifest, Test source images, Test feature CSV, Test predictions,
  or Test metrics.
- Does not load or modify the CNN checkpoint.
- Does not tune the CNN threshold, select the CNN model, or retrain the CNN.

Caution:
A strong metadata-only score would show that dataset descriptors are associated
with the label. It would not by itself prove that the CNN relied on those cues.
File size is not directly provided to the CNN; it is a source/storage proxy.
"""
    (OUTPUT_DIR / "README.txt").write_text(readme, encoding="utf-8")

    print("\n[COMPLETE]")
    print("Output directory:", OUTPUT_DIR)
    print("Metrics:", OUTPUT_DIR / "metadata_baseline_metrics.json")
    print("Train OOF predictions:", OUTPUT_DIR / "train_patient_oof_predictions.csv")
    print("Validation diagnostic predictions:", OUTPUT_DIR / "validation_patient_diagnostic_predictions.csv")
    print("Test data read: NO")
    print("CNN, checkpoint, and locked threshold modified: NO")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
