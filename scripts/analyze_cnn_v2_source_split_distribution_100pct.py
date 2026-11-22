#!/usr/bin/env python3
"""Compare image-level descriptors across Train, Validation, and saved Test features.

This is a descriptive, post-hoc dataset audit for the DL-05 CNN V2 100% run.

IMPORTANT:
- This script computes descriptors from Train and Validation source images only.
- It does NOT open the Test split manifest or any original Test image.
- Test descriptors are reused from the previously generated, hash-verified
  evaluation/image_quality_analysis_100pct/image_quality_features.csv artifact.
- It does not load a model, run inference, tune thresholds, select a model,
  modify images, or modify prior experiment/evaluation artifacts.

Run from the project root:
    PYTHONPATH="$PWD" python scripts/analyze_cnn_v2_source_split_distribution_100pct.py
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
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
DATASET_ROOT = PROJECT_ROOT / "data" / "interim" / "version3" / "CellData" / "chest_xray"

TRAIN_CSV = SPLIT_DIR / "train_v3.csv"
VALIDATION_CSV = SPLIT_DIR / "validation_v3.csv"
TEST_FEATURES_CSV = RUN_DIR / "evaluation" / "image_quality_analysis_100pct" / "image_quality_features.csv"
TEST_GROUP_SUMMARY_JSON = RUN_DIR / "evaluation" / "image_quality_analysis_100pct" / "group_summary.json"
TEST_METRICS_JSON = RUN_DIR / "evaluation" / "test_locked_threshold_100pct" / "metrics.json"
THRESHOLD_JSON = RUN_DIR / "evaluation" / "threshold_selection.json"
OUTPUT_DIR = RUN_DIR / "evaluation" / "source_split_distribution_analysis_100pct"

EXPECTED = {
    "Train": {"images": 4164, "patients": 2120, "class_counts": {"NORMAL": 1078, "PNEUMONIA": 3086}, "source_split": "train"},
    "Validation": {"images": 1042, "patients": 530, "class_counts": {"NORMAL": 270, "PNEUMONIA": 772}, "source_split": "train"},
    "Test": {"images": 621, "patients": 431, "class_counts": {"NORMAL": 234, "PNEUMONIA": 387}},
}
EXPECTED_BEST_EPOCH = 33
EXPECTED_THRESHOLD = 0.5910010933876038
EXPECTED_CHECKPOINT_SHA256 = (
    "7b0b066fcde0b053cc5580c5de2e9ccb"
    "83e7a453a64a39a138705de6115bb292"
)
IMAGE_SIZE_FOR_SHARPNESS = 224

FEATURE_COLUMNS = [
    "width_px",
    "height_px",
    "aspect_ratio_w_over_h",
    "pixel_count",
    "file_size_kb",
    "gray_mean_0_255",
    "gray_std_0_255",
    "gray_p01",
    "gray_p05",
    "gray_median",
    "gray_p95",
    "gray_p99",
    "robust_contrast_p95_minus_p05",
    "low_intensity_fraction_le_5",
    "high_intensity_fraction_ge_250",
    "gray_entropy_bits",
    "laplacian_variance_after_224_longest_side_resize",
]

REQUIRED_MANIFEST_COLUMNS = {
    "relative_path", "label", "patient_id", "split", "source_split", "sha256"
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


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
    q01, q05, q50, q95, q99 = np.percentile(flat, [1, 5, 50, 95, 99])
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


def safe_project_image_path(relative_path: str) -> Path:
    path = (PROJECT_ROOT / relative_path).resolve()
    root = DATASET_ROOT.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Image path escapes dataset root: {relative_path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_source_split(split_name: str, csv_path: Path) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    expected = EXPECTED[split_name]
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    fieldnames, rows = read_csv(csv_path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(fieldnames)
    if missing:
        raise ValueError(f"{split_name} manifest missing columns: {sorted(missing)}")
    if len(rows) != expected["images"]:
        raise ValueError(f"{split_name}: expected {expected['images']} rows, got {len(rows)}")

    label_counts: Counter[str] = Counter()
    patient_labels: dict[str, str] = {}
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    output: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        label = row["label"]
        patient_id = row["patient_id"]
        relative_path = row["relative_path"]
        expected_split = split_name.lower()

        if row["split"] != expected_split:
            raise ValueError(f"{split_name}: wrong split value at row {index}: {row['split']}")
        if row["source_split"] != expected["source_split"]:
            raise ValueError(f"{split_name}: wrong source_split at row {index}: {row['source_split']}")
        if label not in ("NORMAL", "PNEUMONIA"):
            raise ValueError(f"{split_name}: unexpected label {label!r} at row {index}")
        if not patient_id or not relative_path or not row["sha256"]:
            raise ValueError(f"{split_name}: missing patient/path/hash at row {index}")
        if relative_path in seen_paths:
            raise ValueError(f"{split_name}: duplicate relative_path {relative_path}")
        if row["sha256"] in seen_hashes:
            raise ValueError(f"{split_name}: duplicate image SHA-256 {row['sha256']}")
        if patient_id in patient_labels and patient_labels[patient_id] != label:
            raise ValueError(f"{split_name}: patient has multiple labels: {patient_id}")

        seen_paths.add(relative_path)
        seen_hashes.add(row["sha256"])
        patient_labels[patient_id] = label
        label_counts[label] += 1

        image_path = safe_project_image_path(relative_path)
        actual_hash = sha256_file(image_path)
        if actual_hash.lower() != row["sha256"].lower():
            raise RuntimeError(f"{split_name}: source image SHA-256 mismatch: {relative_path}")

        descriptors = image_descriptors(image_path)
        output.append({
            "split": split_name,
            "relative_path": relative_path,
            "patient_id": patient_id,
            "sha256": row["sha256"].lower(),
            "y_true": 0 if label == "NORMAL" else 1,
            "y_true_name": label,
            "source_method": "computed_from_source_image_same_descriptor_pipeline",
            **descriptors,
        })

    if dict(label_counts) != expected["class_counts"]:
        raise ValueError(f"{split_name}: class counts mismatch: {dict(label_counts)}")
    if len(patient_labels) != expected["patients"]:
        raise ValueError(
            f"{split_name}: expected {expected['patients']} patients, got {len(patient_labels)}"
        )
    print(f"[{split_name}] verified {len(output)} images; SHA-256 OK; patients={len(patient_labels)}")
    return output, set(patient_labels), seen_hashes


def load_saved_test_features() -> tuple[list[dict[str, Any]], set[str], set[str]]:
    for path in (TEST_FEATURES_CSV, TEST_GROUP_SUMMARY_JSON, TEST_METRICS_JSON, THRESHOLD_JSON):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(TEST_GROUP_SUMMARY_JSON.read_text(encoding="utf-8"))
    metrics = json.loads(TEST_METRICS_JSON.read_text(encoding="utf-8"))
    threshold_record = json.loads(THRESHOLD_JSON.read_text(encoding="utf-8"))

    if summary.get("run_name") != RUN_NAME or metrics.get("run_name") != RUN_NAME:
        raise RuntimeError("Saved Test artifacts refer to a different experiment")
    if summary.get("best_epoch") != EXPECTED_BEST_EPOCH or metrics.get("best_epoch") != EXPECTED_BEST_EPOCH:
        raise RuntimeError("Saved Test artifacts do not identify the expected best epoch")
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Saved Test feature summary checkpoint SHA-256 mismatch")
    if metrics.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Saved Test metrics checkpoint SHA-256 mismatch")
    if summary.get("image_hashes_verified") is not True or summary.get("image_read_errors") != 0:
        raise RuntimeError("Prior Test image quality artifact is not marked hash-verified/error-free")
    if metrics.get("threshold_tuned_on_test") is not False:
        raise RuntimeError("Test metrics do not confirm Validation-only threshold selection")
    if not math.isclose(float(metrics.get("locked_threshold", float("nan"))), EXPECTED_THRESHOLD, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("Saved Test metrics threshold mismatch")
    threshold = threshold_record.get("threshold_selection", {})
    if threshold_record.get("experiment") != RUN_NAME or threshold.get("split") != "validation":
        raise RuntimeError("Threshold artifact is not the expected Validation artifact")
    if not math.isclose(float(threshold.get("threshold", float("nan"))), EXPECTED_THRESHOLD, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("Validation threshold artifact value mismatch")

    fieldnames, rows = read_csv(TEST_FEATURES_CSV)
    required = {"relative_path", "patient_id", "sha256", "y_true", "y_true_name", *FEATURE_COLUMNS}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"Saved Test feature CSV missing columns: {sorted(missing)}")
    if len(rows) != EXPECTED["Test"]["images"]:
        raise ValueError(f"Expected 621 saved Test feature rows, got {len(rows)}")

    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    patients: set[str] = set()
    counts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if row["relative_path"] in seen_paths or row["sha256"].lower() in seen_hashes:
            raise ValueError(f"Saved Test feature artifact has duplicate path/hash at row {index}")
        label = row["y_true_name"]
        if label not in ("NORMAL", "PNEUMONIA") or int(row["y_true"]) != (0 if label == "NORMAL" else 1):
            raise ValueError(f"Saved Test feature label mismatch at row {index}")
        seen_paths.add(row["relative_path"])
        seen_hashes.add(row["sha256"].lower())
        patients.add(row["patient_id"])
        counts[label] += 1
        record: dict[str, Any] = {
            "split": "Test",
            "relative_path": row["relative_path"],
            "patient_id": row["patient_id"],
            "sha256": row["sha256"].lower(),
            "y_true": int(row["y_true"]),
            "y_true_name": label,
            "source_method": "reused_previously_hash_verified_test_feature_artifact",
        }
        for col in FEATURE_COLUMNS:
            value = float(row[col])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite Test feature {col} at row {index}")
            record[col] = value
        for col in ("image_format", "original_mode"):
            if col in row:
                record[col] = row[col]
        output.append(record)

    if counts != Counter(EXPECTED["Test"]["class_counts"]):
        raise ValueError(f"Saved Test feature class counts mismatch: {dict(counts)}")
    if len(patients) != EXPECTED["Test"]["patients"]:
        raise ValueError(f"Saved Test patient count mismatch: {len(patients)}")

    print(f"[Test] reused {len(output)} saved feature rows; no Test manifest or source image opened")
    return output, patients, seen_hashes


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


def make_summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["y_true_name"])].append(row)

    image_summary: list[dict[str, Any]] = []
    patient_summary: list[dict[str, Any]] = []
    patient_feature_rows: list[dict[str, Any]] = []
    json_groups: dict[str, Any] = {}

    split_order = ("Train", "Validation", "Test")
    class_order = ("NORMAL", "PNEUMONIA")
    for split in split_order:
        json_groups[split] = {}
        for label in class_order:
            group_rows = grouped[(split, label)]
            if not group_rows:
                raise RuntimeError(f"Missing group: {split}/{label}")
            patient_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in group_rows:
                patient_map[row["patient_id"]].append(row)

            image_group_json: dict[str, Any] = {
                "images": len(group_rows),
                "patients": len(patient_map),
                "features": {},
            }
            patient_group_json: dict[str, Any] = {
                "patients": len(patient_map),
                "features": {},
            }

            for feature in FEATURE_COLUMNS:
                image_values = [float(row[feature]) for row in group_rows]
                image_stats = summarize(image_values)
                image_summary.append({
                    "split": split, "class": label, "unit": "image",
                    "n_images": len(group_rows), "n_patients": len(patient_map),
                    "feature": feature, **image_stats,
                })
                image_group_json["features"][feature] = image_stats

                per_patient = [
                    float(np.median([float(row[feature]) for row in patient_rows]))
                    for patient_rows in patient_map.values()
                ]
                patient_stats = summarize(per_patient)
                patient_summary.append({
                    "split": split, "class": label, "unit": "patient_median",
                    "n_images": len(group_rows), "n_patients": len(patient_map),
                    "feature": feature, **patient_stats,
                })
                patient_group_json["features"][feature] = patient_stats

            for patient_id, patient_rows in sorted(patient_map.items()):
                patient_record: dict[str, Any] = {
                    "split": split,
                    "class": label,
                    "patient_id": patient_id,
                    "num_images": len(patient_rows),
                }
                for feature in FEATURE_COLUMNS:
                    patient_record[feature] = float(np.median([float(row[feature]) for row in patient_rows]))
                patient_feature_rows.append(patient_record)

            json_groups[split][label] = {
                "image_level": image_group_json,
                "patient_level_median_per_patient": patient_group_json,
            }

    # Difference in medians (PNEUMONIA minus NORMAL) within each split.
    gap_rows: list[dict[str, Any]] = []
    for split in split_order:
        for feature in FEATURE_COLUMNS:
            normal = grouped[(split, "NORMAL")]
            pneumonia = grouped[(split, "PNEUMONIA")]
            normal_median = float(np.median([float(r[feature]) for r in normal]))
            pneumonia_median = float(np.median([float(r[feature]) for r in pneumonia]))
            normal_patient_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
            pneumonia_patient_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in normal:
                normal_patient_map[row["patient_id"]].append(row)
            for row in pneumonia:
                pneumonia_patient_map[row["patient_id"]].append(row)
            normal_patient_medians = [float(np.median([float(r[feature]) for r in rs])) for rs in normal_patient_map.values()]
            pneumonia_patient_medians = [float(np.median([float(r[feature]) for r in rs])) for rs in pneumonia_patient_map.values()]
            normal_patient_median = float(np.median(normal_patient_medians))
            pneumonia_patient_median = float(np.median(pneumonia_patient_medians))
            gap_rows.append({
                "split": split,
                "feature": feature,
                "normal_image_median": normal_median,
                "pneumonia_image_median": pneumonia_median,
                "image_median_gap_pneumonia_minus_normal": pneumonia_median - normal_median,
                "normal_patient_median_of_patient_medians": normal_patient_median,
                "pneumonia_patient_median_of_patient_medians": pneumonia_patient_median,
                "patient_median_gap_pneumonia_minus_normal": pneumonia_patient_median - normal_patient_median,
            })

    json_result = {
        "status": "completed_descriptive_split_distribution_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": RUN_NAME,
        "best_epoch": EXPECTED_BEST_EPOCH,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "locked_threshold_reference_only": EXPECTED_THRESHOLD,
        "split_sources": {
            "Train": str(TRAIN_CSV.relative_to(PROJECT_ROOT)),
            "Validation": str(VALIDATION_CSV.relative_to(PROJECT_ROOT)),
            "Test": str(TEST_FEATURES_CSV.relative_to(PROJECT_ROOT)),
        },
        "test_handling": "Reused previously generated and hash-verified Test feature CSV; did not open test_v3.csv or any original Test image.",
        "split_counts": {
            split: {
                "images": sum(1 for row in rows if row["split"] == split),
                "patients": len({row["patient_id"] for row in rows if row["split"] == split}),
                "NORMAL": sum(1 for row in rows if row["split"] == split and row["y_true_name"] == "NORMAL"),
                "PNEUMONIA": sum(1 for row in rows if row["split"] == split and row["y_true_name"] == "PNEUMONIA"),
            } for split in split_order
        },
        "patient_and_sha256_separation": "passed across Train, Validation, and saved Test feature records",
        "image_hash_verification": {
            "Train": "verified against source files",
            "Validation": "verified against source files",
            "Test": "reused prior artifact marked image_hashes_verified=true; source Test images were not reopened",
        },
        "groups": json_groups,
        "feature_definitions": {
            "gray_*": "Descriptive statistics of the original JPEG converted to 8-bit grayscale.",
            "robust_contrast_p95_minus_p05": "95th percentile minus 5th percentile grayscale intensity.",
            "low_intensity_fraction_le_5": "Fraction of original grayscale pixels <= 5; not a padding detector.",
            "high_intensity_fraction_ge_250": "Fraction of original grayscale pixels >= 250.",
            "gray_entropy_bits": "Shannon entropy of the 256-bin grayscale histogram.",
            "laplacian_variance_after_224_longest_side_resize": "Variance of a 4-neighbor Laplacian after resizing the grayscale image so its longest side is 224 pixels, without padding; rough sharpness proxy only.",
            "patient_level_summary": "Each patient's per-feature median across their images is calculated first; group summaries are then calculated across patients.",
        },
        "interpretation_limits": [
            "This is descriptive data-distribution analysis, not a model explanation or causal analysis.",
            "Image-level rows may be correlated within patients; patient-level summaries are provided to reduce this dependence.",
            "Low-intensity fraction is not equivalent to black-padding fraction.",
            "Laplacian variance is affected by anatomy, acquisition, noise, and resizing; it is not a definitive blur score.",
            "No p-values, statistical significance claims, or causal conclusions are calculated.",
            "This post-hoc analysis must not be used to tune thresholds, select models, retrain, or present the same Test set as untouched after such changes.",
        ],
    }
    return image_summary, patient_summary, gap_rows, {"json": json_result, "patient_feature_rows": patient_feature_rows}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    print("=" * 78)
    print("CNN V2 100% — TRAIN / VALIDATION / SAVED TEST FEATURE COMPARISON")
    print("=" * 78)

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(DATASET_ROOT)
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {OUTPUT_DIR}")

    train_rows, train_patients, train_hashes = load_source_split("Train", TRAIN_CSV)
    validation_rows, validation_patients, validation_hashes = load_source_split("Validation", VALIDATION_CSV)

    patient_overlap = train_patients & validation_patients
    hash_overlap = train_hashes & validation_hashes
    if patient_overlap:
        raise RuntimeError(f"Train/Validation patient overlap: {len(patient_overlap)}")
    if hash_overlap:
        raise RuntimeError(f"Train/Validation SHA-256 overlap: {len(hash_overlap)}")

    test_rows, test_patients, test_hashes = load_saved_test_features()
    for name, first, second in (
        ("Train/Validation", train_patients, validation_patients),
        ("Train/Test", train_patients, test_patients),
        ("Validation/Test", validation_patients, test_patients),
    ):
        overlap = first & second
        if overlap:
            raise RuntimeError(f"Patient overlap detected for {name}: {len(overlap)}")
    for name, first, second in (
        ("Train/Validation", train_hashes, validation_hashes),
        ("Train/Test", train_hashes, test_hashes),
        ("Validation/Test", validation_hashes, test_hashes),
    ):
        overlap = first & second
        if overlap:
            raise RuntimeError(f"SHA-256 overlap detected for {name}: {len(overlap)}")

    all_rows = train_rows + validation_rows + test_rows
    if len(all_rows) != 5827:
        raise RuntimeError(f"Expected 5827 combined feature rows, got {len(all_rows)}")

    image_summary, patient_summary, gap_rows, extras = make_summaries(all_rows)
    patient_feature_rows = extras["patient_feature_rows"]
    json_result = extras["json"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    feature_fields = [
        "split", "relative_path", "patient_id", "sha256", "y_true", "y_true_name", "source_method",
        "image_format", "original_mode", *FEATURE_COLUMNS,
    ]
    write_csv(OUTPUT_DIR / "train_validation_image_quality_features.csv",
              train_rows + validation_rows, feature_fields)
    write_csv(OUTPUT_DIR / "all_split_image_quality_features.csv", all_rows, feature_fields)

    summary_fields = ["split", "class", "unit", "n_images", "n_patients", "feature", "n", "mean", "std", "min", "q25", "median", "q75", "max"]
    write_csv(OUTPUT_DIR / "image_level_split_class_summary.csv", image_summary, summary_fields)
    write_csv(OUTPUT_DIR / "patient_level_split_class_summary.csv", patient_summary, summary_fields)
    patient_feature_fields = ["split", "class", "patient_id", "num_images", *FEATURE_COLUMNS]
    write_csv(OUTPUT_DIR / "patient_level_feature_table.csv", patient_feature_rows, patient_feature_fields)
    write_csv(OUTPUT_DIR / "class_median_gaps_by_split.csv", gap_rows, [
        "split", "feature", "normal_image_median", "pneumonia_image_median",
        "image_median_gap_pneumonia_minus_normal",
        "normal_patient_median_of_patient_medians", "pneumonia_patient_median_of_patient_medians",
        "patient_median_gap_pneumonia_minus_normal",
    ])

    (OUTPUT_DIR / "split_distribution_summary.json").write_text(
        json.dumps(json_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    readme = """CNN V2 100% — Split Distribution Analysis

This output compares descriptive image features for NORMAL and PNEUMONIA
across Train, Validation, and Test.

Method:
- Train and Validation features were computed from source images using the
  same Pillow/NumPy descriptor implementation as the prior Test audit.
- Train and Validation image SHA-256 values were checked against their split
  manifests. Patient IDs and image hashes were checked for cross-split overlap.
- Test features were reused from the existing image_quality_features.csv,
  which was previously generated with image hash verification. This script
  did not open the Test split manifest or original Test images again.
- Both image-level summaries and patient-level summaries are provided.
  Patient-level summaries first take the median feature value per patient.

Limitations:
- The features are descriptive proxies, not clinical image interpretations.
- Low-intensity pixel fraction is not a direct padding measurement.
- Laplacian variance is only a rough sharpness proxy.
- This analysis does not establish causality or prove that the model used any
  particular image feature. No p-values or causal claims are produced.
- Do not use this post-hoc Test analysis to tune a threshold, select a model,
  retrain, or then report the same Test set as an untouched final evaluation.

No inference, threshold tuning, model selection, retraining, or source-image
modification is performed by this script.
"""
    (OUTPUT_DIR / "README.txt").write_text(readme, encoding="utf-8")

    print("\n[COMPLETE]")
    print("Combined feature rows:", len(all_rows))
    print("Train images/patients:", len(train_rows), "/", len(train_patients))
    print("Validation images/patients:", len(validation_rows), "/", len(validation_patients))
    print("Test feature rows reused:", len(test_rows), "(no Test source images reopened)")
    print("Patient/SHA-256 separation: PASS")
    print("Output directory:", OUTPUT_DIR)
    print("Image-level summary:", OUTPUT_DIR / "image_level_split_class_summary.csv")
    print("Patient-level summary:", OUTPUT_DIR / "patient_level_split_class_summary.csv")
    print("Class median gaps:", OUTPUT_DIR / "class_median_gaps_by_split.csv")
    print("Combined JSON:", OUTPUT_DIR / "split_distribution_summary.json")
    print("No inference, threshold tuning, model selection, or training was performed.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
