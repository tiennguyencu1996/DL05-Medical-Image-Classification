#!/usr/bin/env python3
"""Post-hoc descriptive image-quality analysis for CNN V2 100% Test errors.

This script MUST NOT be used to tune the model, threshold, or training procedure.
It reads the existing one-time Test predictions and source images, verifies each
image SHA-256 against the prediction manifest, computes simple non-diagnostic
image descriptors, and writes descriptive group summaries.

Run from the project root:
    PYTHONPATH="$PWD" python scripts/analyze_cnn_v2_image_quality_100pct.py

The script does not run model inference, modify source images, or modify the
existing Test evaluation artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "cnn_from_scratch_v2_100pct_full_20261121"
RUN_DIR = PROJECT_ROOT / "outputs" / "experiments" / RUN_NAME
TEST_DIR = RUN_DIR / "evaluation" / "test_locked_threshold_100pct"
PREDICTIONS_CSV = TEST_DIR / "predictions.csv"
METRICS_JSON = TEST_DIR / "metrics.json"
THRESHOLD_JSON = RUN_DIR / "evaluation" / "threshold_selection.json"
TEST_SPLIT_CSV = PROJECT_ROOT / "data" / "processed" / "splits" / "test_v3.csv"
DATASET_ROOT = PROJECT_ROOT / "data" / "interim" / "version3" / "CellData" / "chest_xray"
OUTPUT_DIR = RUN_DIR / "evaluation" / "image_quality_analysis_100pct"

EXPECTED_ROWS = 621
EXPECTED_LABEL_COUNTS = {0: 234, 1: 387}
EXPECTED_GROUP_COUNTS = {"TN": 137, "FP": 97, "FN": 4, "TP": 383}
EXPECTED_THRESHOLD = 0.5910010933876038
EXPECTED_CHECKPOINT_SHA256 = (
    "7b0b066fcde0b053cc5580c5de2e9ccb"
    "83e7a453a64a39a138705de6115bb292"
)
IMAGE_SIZE_FOR_SHARPNESS = 224


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def classify_group(y_true: int, y_pred: int) -> str:
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 1 and y_pred == 0:
        return "FN"
    if y_true == 1 and y_pred == 1:
        return "TP"
    raise ValueError(f"Invalid binary pair: y_true={y_true}, y_pred={y_pred}")


def resized_for_sharpness(gray: Image.Image, longest_side: int = IMAGE_SIZE_FOR_SHARPNESS) -> np.ndarray:
    width, height = gray.size
    scale = longest_side / max(width, height)
    new_width = max(3, int(round(width * scale)))
    new_height = max(3, int(round(height * scale)))
    resampling = getattr(Image, "Resampling", Image)
    resized = gray.resize((new_width, new_height), resample=resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32)


def image_descriptors(image_path: Path) -> dict[str, Any]:
    file_size = image_path.stat().st_size
    with Image.open(image_path) as source:
        source_format = source.format or "unknown"
        original_mode = source.mode
        width, height = source.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")
        gray = source.convert("L")
        gray.load()
        arr = np.asarray(gray, dtype=np.uint8)

    flat = arr.reshape(-1)
    q01, q05, q25, q50, q75, q95, q99 = np.percentile(
        flat, [1, 5, 25, 50, 75, 95, 99]
    )
    hist = np.bincount(flat, minlength=256).astype(np.float64)
    probabilities = hist[hist > 0] / flat.size
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))

    standardized = resized_for_sharpness(gray)
    center = standardized[1:-1, 1:-1]
    laplacian = (
        standardized[:-2, 1:-1]
        + standardized[2:, 1:-1]
        + standardized[1:-1, :-2]
        + standardized[1:-1, 2:]
        - 4.0 * center
    )

    return {
        "image_format": source_format,
        "original_mode": original_mode,
        "width_px": int(width),
        "height_px": int(height),
        "aspect_ratio_w_over_h": float(width / height),
        "pixel_count": int(width * height),
        "file_size_kb": float(file_size / 1024.0),
        "gray_mean_0_255": float(np.mean(flat)),
        "gray_std_0_255": float(np.std(flat)),
        "gray_p01": float(q01),
        "gray_p05": float(q05),
        "gray_median": float(q50),
        "gray_p95": float(q95),
        "gray_p99": float(q99),
        "robust_contrast_p95_minus_p05": float(q95 - q05),
        "low_intensity_fraction_le_5": float(np.mean(flat <= 5)),
        "high_intensity_fraction_ge_250": float(np.mean(flat >= 250)),
        "gray_entropy_bits": entropy,
        "laplacian_variance_after_224_longest_side_resize": float(np.var(laplacian)),
    }


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "q25": None,
                "median": None, "q75": None, "max": None}
    data = np.asarray(values, dtype=np.float64)
    q25, median, q75 = np.percentile(data, [25, 50, 75])
    return {
        "n": int(data.size),
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=0)),
        "min": float(np.min(data)),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "max": float(np.max(data)),
    }


def main() -> int:
    print("=" * 78)
    print("CNN V2 100% — POST-HOC TEST IMAGE QUALITY ANALYSIS")
    print("=" * 78)

    for required in (PREDICTIONS_CSV, METRICS_JSON, THRESHOLD_JSON, TEST_SPLIT_CSV):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_DIR}. Refusing to overwrite evidence."
        )

    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    if metrics.get("run_name") != RUN_NAME:
        raise RuntimeError("Unexpected run_name in metrics.json")
    if metrics.get("best_epoch") != 33:
        raise RuntimeError("Unexpected best epoch in metrics.json")
    if metrics.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint SHA-256 differs from the verified checkpoint")
    if not math.isclose(float(metrics.get("locked_threshold", float("nan"))),
                        EXPECTED_THRESHOLD, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("Locked threshold differs from the previously recorded value")
    if metrics.get("threshold_tuned_on_test") is not False:
        raise RuntimeError("metrics.json does not confirm Validation-only threshold selection")
    threshold_hash = sha256_file(THRESHOLD_JSON)
    if metrics.get("threshold_artifact_sha256") != threshold_hash:
        raise RuntimeError("threshold_selection.json SHA-256 differs from Test metrics")
    threshold_record = json.loads(THRESHOLD_JSON.read_text(encoding="utf-8"))
    threshold_config = threshold_record.get("threshold_selection", {})
    if threshold_record.get("experiment") != RUN_NAME:
        raise RuntimeError("Unexpected experiment in threshold_selection.json")
    if threshold_config.get("split") != "validation":
        raise RuntimeError("Threshold artifact was not selected on Validation")
    if threshold_config.get("positive_class") != "PNEUMONIA":
        raise RuntimeError("Unexpected positive class in threshold artifact")
    if not math.isclose(float(threshold_config.get("threshold", float("nan"))),
                        EXPECTED_THRESHOLD, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("Threshold artifact does not contain the expected threshold")

    test_manifest_hash = sha256_file(TEST_SPLIT_CSV)
    if metrics.get("test_split_csv_sha256") != test_manifest_hash:
        raise RuntimeError("Test split manifest SHA-256 differs from Test metrics")
    with TEST_SPLIT_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        manifest_rows = list(csv.DictReader(f))
    if len(manifest_rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows in test_v3.csv, got {len(manifest_rows)}")

    with PREDICTIONS_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required_columns = {
            "relative_path", "patient_id", "sha256", "y_true", "y_score",
            "y_pred_at_locked_threshold", "locked_threshold",
        }
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing prediction columns: {required_columns - set(reader.fieldnames or [])}")
        rows = list(reader)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} Test predictions, got {len(rows)}")

    # The saved predictions must retain the exact fixed Test manifest row order
    # and identity metadata used by the one-time evaluator.
    for index, (pred_row, manifest_row) in enumerate(zip(rows, manifest_rows), start=1):
        expected_label = 1 if manifest_row.get("label") == "PNEUMONIA" else 0 if manifest_row.get("label") == "NORMAL" else None
        if pred_row.get("relative_path") != manifest_row.get("relative_path"):
            raise RuntimeError(f"Prediction/Test manifest path mismatch at row {index}")
        if pred_row.get("patient_id") != manifest_row.get("patient_id"):
            raise RuntimeError(f"Prediction/Test manifest patient mismatch at row {index}")
        if pred_row.get("sha256") != manifest_row.get("sha256"):
            raise RuntimeError(f"Prediction/Test manifest SHA mismatch at row {index}")
        if expected_label is None or int(pred_row["y_true"]) != expected_label:
            raise RuntimeError(f"Prediction/Test manifest label mismatch at row {index}")

    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    label_counts: Counter[int] = Counter()
    group_counts: Counter[str] = Counter()
    feature_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        rel = row["relative_path"]
        if rel in seen_paths:
            raise ValueError(f"Duplicate relative_path at row {index}: {rel}")
        seen_paths.add(rel)

        y_true = int(row["y_true"])
        y_pred = int(row["y_pred_at_locked_threshold"])
        y_score = float(row["y_score"])
        row_threshold = float(row["locked_threshold"])
        expected_pred = int(y_score >= EXPECTED_THRESHOLD)
        if y_true not in (0, 1) or y_pred not in (0, 1):
            raise ValueError(f"Invalid binary label/prediction at row {index}")
        if not math.isfinite(y_score) or not 0.0 <= y_score <= 1.0:
            raise ValueError(f"Invalid y_score at row {index}: {y_score}")
        if not math.isclose(row_threshold, EXPECTED_THRESHOLD, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Threshold mismatch in predictions.csv at row {index}")
        if y_pred != expected_pred:
            raise ValueError(f"Saved prediction does not match locked threshold at row {index}")

        label_counts[y_true] += 1
        group = classify_group(y_true, y_pred)
        group_counts[group] += 1
        patient_id = row["patient_id"]

        image_path = (PROJECT_ROOT / rel).resolve()
        try:
            image_path.relative_to(DATASET_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"Image path escapes dataset root: {rel}") from exc
        if not image_path.is_file():
            errors.append({"relative_path": rel, "error": "file_not_found"})
            continue

        actual_hash = sha256_file(image_path)
        if actual_hash != row["sha256"]:
            errors.append({"relative_path": rel, "error": "sha256_mismatch"})
            continue

        try:
            descriptors = image_descriptors(image_path)
        except Exception as exc:  # preserve case-level diagnostics and continue audit
            errors.append({"relative_path": rel, "error": f"image_read_error: {type(exc).__name__}: {exc}"})
            continue

        feature_rows.append({
            "relative_path": rel,
            "patient_id": patient_id,
            "sha256": actual_hash,
            "y_true": y_true,
            "y_true_name": "PNEUMONIA" if y_true == 1 else "NORMAL",
            "y_score": y_score,
            "y_pred": y_pred,
            "y_pred_name": "PNEUMONIA" if y_pred == 1 else "NORMAL",
            "error_group": group,
            "is_error": int(group in {"FP", "FN"}),
            **descriptors,
        })

    if dict(label_counts) != EXPECTED_LABEL_COUNTS:
        raise RuntimeError(f"Unexpected Test class counts: {dict(label_counts)}")
    if dict(group_counts) != EXPECTED_GROUP_COUNTS:
        raise RuntimeError(f"Unexpected confusion groups: {dict(group_counts)}")

    if errors or len(feature_rows) != EXPECTED_ROWS:
        error_report = OUTPUT_DIR.with_name(
            OUTPUT_DIR.name + "_errors_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".csv"
        )
        with error_report.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["relative_path", "error"])
            writer.writeheader()
            writer.writerows(errors)
        print(f"Image/hash/read problems: {len(errors)}")
        print(f"Successfully analyzed: {len(feature_rows)} / {EXPECTED_ROWS}")
        print("Diagnostic error list:", error_report)
        raise RuntimeError(
            "Not all Test images passed image/hash verification. "
            "No partial feature summary was written; inspect the error report."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    errors_path = OUTPUT_DIR / "image_read_or_hash_errors.csv"
    with errors_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "error"])
        writer.writeheader()

    feature_fields = list(feature_rows[0].keys())
    features_path = OUTPUT_DIR / "image_quality_features.csv"
    with features_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=feature_fields)
        writer.writeheader()
        writer.writerows(feature_rows)

    numeric_features = [
        "width_px", "height_px", "aspect_ratio_w_over_h", "pixel_count", "file_size_kb",
        "gray_mean_0_255", "gray_std_0_255", "gray_p01", "gray_p05", "gray_median",
        "gray_p95", "gray_p99", "robust_contrast_p95_minus_p05",
        "low_intensity_fraction_le_5", "high_intensity_fraction_ge_250",
        "gray_entropy_bits", "laplacian_variance_after_224_longest_side_resize",
    ]
    summary: dict[str, Any] = {
        "status": "completed_descriptive_post_hoc_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": RUN_NAME,
        "best_epoch": 33,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "locked_threshold": EXPECTED_THRESHOLD,
        "source_predictions_csv": str(PREDICTIONS_CSV.relative_to(PROJECT_ROOT)),
        "source_metrics_json": str(METRICS_JSON.relative_to(PROJECT_ROOT)),
        "test_images": EXPECTED_ROWS,
        "test_patients": len({row["patient_id"] for row in feature_rows}),
        "class_counts": {"NORMAL": label_counts[0], "PNEUMONIA": label_counts[1]},
        "confusion_group_counts": dict(group_counts),
        "image_hashes_verified": True,
        "image_read_errors": 0,
        "groups": {},
        "feature_definitions": {
            "gray_*": "Descriptive statistics of the original JPEG converted to 8-bit grayscale.",
            "robust_contrast_p95_minus_p05": "95th percentile minus 5th percentile grayscale intensity.",
            "low_intensity_fraction_le_5": "Fraction of original grayscale pixels <= 5; not a padding detector.",
            "high_intensity_fraction_ge_250": "Fraction of original grayscale pixels >= 250.",
            "gray_entropy_bits": "Shannon entropy of the 256-bin grayscale histogram.",
            "laplacian_variance_after_224_longest_side_resize": (
                "Variance of a simple 4-neighbor Laplacian after resizing the grayscale image "
                "so its longest side is 224 pixels, without adding padding; a rough sharpness proxy only."
            ),
        },
        "interpretation_limits": [
            "These are descriptive image-quality proxies, not clinical image interpretations.",
            "Low-intensity fraction is not equivalent to black-padding fraction.",
            "Laplacian variance is affected by anatomy, acquisition, noise, and resizing; it is not a definitive blur score.",
            "Images may be clustered within patients; image rows are not necessarily independent observations.",
            "This post-hoc Test analysis must not be used to tune thresholds, select models, or retrain and then report the same Test set as an untouched evaluation.",
            "No p-values or causal conclusions are calculated by this script.",
        ],
    }

    for group in ("TN", "FP", "FN", "TP"):
        selected = [row for row in feature_rows if row["error_group"] == group]
        group_summary: dict[str, Any] = {
            "images": len(selected),
            "patients": len({row["patient_id"] for row in selected}),
            "features": {},
        }
        for feature in numeric_features:
            group_summary["features"][feature] = summarize(
                [float(row[feature]) for row in selected]
            )
        summary["groups"][group] = group_summary

    summary_path = OUTPUT_DIR / "group_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    notes_path = OUTPUT_DIR / "README.txt"
    notes_path.write_text(
        "CNN V2 100% — Post-hoc Test Image Quality Analysis\n"
        "===================================================\n\n"
        "This analysis is descriptive and retrospective. It reads the already evaluated Test predictions and source JPEGs.\n"
        "It does not perform model inference, threshold selection, model selection, or training.\n"
        "Every source JPEG is SHA-256 checked against the corresponding Test predictions row.\n\n"
        "Files:\n"
        "- image_quality_features.csv: one row per Test image, with metadata and descriptive image features.\n"
        "- group_summary.json: descriptive summaries for TN, FP, FN, and TP.\n"
        "- image_read_or_hash_errors.csv: should contain only the header when all checks pass.\n\n"
        "Interpretation cautions:\n"
        "- These features are not clinical findings or a diagnosis.\n"
        "- Low pixel intensity is not the same as padding.\n"
        "- Laplacian variance is only a rough sharpness proxy.\n"
        "- Images can be clustered within patients.\n"
        "- Do not use this Test analysis to tune threshold/model and then claim this same Test set remains untouched.\n",
        encoding="utf-8",
    )

    print("\n[RESULT]")
    print("Images analyzed:", len(feature_rows))
    print("Image hashes verified:", len(feature_rows) == EXPECTED_ROWS)
    print("Image/read errors:", len(errors))
    print("Group counts:", dict(group_counts))
    print("Distinct patients:", len({row['patient_id'] for row in feature_rows}))
    print("Output directory:", OUTPUT_DIR)
    print("Feature table:", features_path)
    print("Group summary:", summary_path)
    print("Error list:", errors_path)
    print("No source images or prior Test evaluation artifacts were modified.")
    print("No inference, threshold tuning, model selection, or training was performed.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
