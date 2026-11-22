#!/usr/bin/env python3
"""Create nested, patient-level, class-stratified Train manifests for DL-05.

Only data/processed/splits/train_v3.csv is read. Validation and Test are never
opened. Patient groups are kept intact; each class is sampled independently so
the class-wise image fractions stay close to the requested 10/25/50/100% levels.
Because patient groups can contain different numbers of images, realized image
fractions may differ slightly from nominal fractions. The exact realized
fractions are reported in metadata.json and stdout.

The script is intentionally fail-safe: it refuses to overwrite an existing
output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "processed" / "splits" / "train_v3.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "label_efficiency" / "manifests_v1"
)
DEFAULT_SEED = 42
FRACTIONS = (0.10, 0.25, 0.50, 1.00)
EXPECTED_CLASS_COUNTS = {"NORMAL": 1078, "PNEUMONIA": 3086}
REQUIRED_COLUMNS = {
    "relative_path", "label", "patient_id", "split", "source_split", "sha256"
}
LABEL_ORDER = ("NORMAL", "PNEUMONIA")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_train_rows(source_csv: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not source_csv.is_file():
        raise FileNotFoundError(f"Train split CSV not found: {source_csv}")

    with source_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("Train split CSV has no header.")
        fieldnames = list(reader.fieldnames)
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(f"Train CSV missing required columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Train split CSV is empty.")

    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    patient_label: dict[str, str] = {}
    class_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, start=2):
        label = row["label"]
        patient_id = row["patient_id"]
        relative_path = row["relative_path"]

        if label not in LABEL_ORDER:
            raise ValueError(f"Unexpected label at CSV row {index}: {label!r}")
        if not patient_id:
            raise ValueError(f"Empty patient_id at CSV row {index}.")
        if not relative_path:
            raise ValueError(f"Empty relative_path at CSV row {index}.")
        if row["split"] != "train" or row["source_split"] != "train":
            raise ValueError(
                f"Row {index} is not an original/final Train record: "
                f"split={row['split']!r}, source_split={row['source_split']!r}"
            )
        if not row["sha256"]:
            raise ValueError(f"Missing SHA256 at CSV row {index}.")
        if row["sha256"] in seen_hashes:
            raise ValueError(f"Duplicate SHA256 in Train at CSV row {index}.")
        seen_hashes.add(row["sha256"])
        if relative_path in seen_paths:
            raise ValueError(f"Duplicate relative_path in Train: {relative_path}")
        seen_paths.add(relative_path)

        previous_label = patient_label.setdefault(patient_id, label)
        if previous_label != label:
            raise ValueError(
                f"Patient {patient_id!r} has multiple labels: "
                f"{previous_label!r} and {label!r}"
            )
        class_counts[label] += 1

    if dict(class_counts) != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            "Train class counts differ from the fixed DL-05 split. "
            f"Expected {EXPECTED_CLASS_COUNTS}, observed {dict(class_counts)}. "
            "Inspect the split manifest before changing this guard."
        )

    return fieldnames, rows


def choose_nested_prefixes(
    patient_groups: dict[str, list[dict[str, str]]],
    seed: int,
) -> dict[float, set[str]]:
    """Choose nested prefixes approximating image fractions within one class."""
    patient_ids = sorted(patient_groups)
    if len(patient_ids) < 4:
        raise ValueError("Not enough patients in a class to build four nested levels.")

    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    cumulative_images = [0]
    for patient_id in patient_ids:
        cumulative_images.append(
            cumulative_images[-1] + len(patient_groups[patient_id])
        )

    total_images = cumulative_images[-1]
    chosen_prefixes: dict[float, set[str]] = {}
    previous_count = 0

    for fraction in FRACTIONS[:-1]:
        target_images = fraction * total_images
        candidate_counts = range(previous_count + 1, len(patient_ids) + 1)
        # Prefer the closest image count; on an exact tie, prefer not to exceed
        # the target, then prefer fewer selected patients.
        best_count = min(
            candidate_counts,
            key=lambda count: (
                abs(cumulative_images[count] - target_images),
                cumulative_images[count] > target_images,
                count,
            ),
        )
        chosen_prefixes[fraction] = set(patient_ids[:best_count])
        previous_count = best_count

    chosen_prefixes[1.0] = set(patient_ids)
    return chosen_prefixes


def build_selections(
    rows: list[dict[str, str]], seed: int
) -> tuple[dict[float, set[str]], dict[str, dict[str, list[dict[str, str]]]]]:
    groups: dict[str, dict[str, list[dict[str, str]]]] = {
        label: defaultdict(list) for label in LABEL_ORDER
    }
    for row in rows:
        groups[row["label"]][row["patient_id"]].append(row)

    per_class: dict[str, dict[float, set[str]]] = {}
    for class_index, label in enumerate(LABEL_ORDER):
        per_class[label] = choose_nested_prefixes(
            groups[label], seed=seed + class_index * 10_007
        )

    selected_by_fraction: dict[float, set[str]] = {}
    for fraction in FRACTIONS:
        selected = set().union(
            *(per_class[label][fraction] for label in LABEL_ORDER)
        )
        selected_by_fraction[fraction] = selected

    # Explicitly prove nesting and full-data identity.
    for smaller, larger in zip(FRACTIONS, FRACTIONS[1:]):
        if not selected_by_fraction[smaller].issubset(selected_by_fraction[larger]):
            raise AssertionError(f"Patient selections are not nested: {smaller} -> {larger}")

    all_patients = {row["patient_id"] for row in rows}
    if selected_by_fraction[1.0] != all_patients:
        raise AssertionError("The 100% manifest does not contain every Train patient.")

    return selected_by_fraction, groups


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fraction_label(fraction: float) -> str:
    return f"{int(round(fraction * 100)):03d}pct"


def create_manifests(source_csv: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    source_csv = source_csv.resolve()
    output_dir = output_dir.resolve()

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}\n"
            "Refusing to overwrite. Choose a new --output-dir or move the old output."
        )

    fieldnames, rows = read_train_rows(source_csv)
    selected_by_fraction, groups = build_selections(rows, seed)
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata: dict[str, Any] = {
        "experiment": "DL-05 label efficiency nested patient-stratified manifests",
        "source_csv": str(source_csv.relative_to(PROJECT_ROOT.resolve()))
        if source_csv.is_relative_to(PROJECT_ROOT.resolve())
        else str(source_csv),
        "source_csv_sha256": sha256_file(source_csv),
        "seed": seed,
        "sampling_unit": "patient_id",
        "stratification": "independent within NORMAL and PNEUMONIA patient groups",
        "selection_method": (
            "For each class, shuffle sorted patient IDs with a fixed class-specific seed; "
            "select nested prefixes whose cumulative image count is closest to the target "
            "class-wise image fraction. Keep every image belonging to a selected patient."
        ),
        "nominal_fractions": [fraction for fraction in FRACTIONS],
        "validation_or_test_opened": False,
        "levels": {},
    }

    patient_selection_rows: list[dict[str, Any]] = []
    all_patients = sorted({row["patient_id"] for row in rows})

    for fraction in FRACTIONS:
        selected_patients = selected_by_fraction[fraction]
        selected_rows = [row for row in rows if row["patient_id"] in selected_patients]
        filename = f"train_{fraction_label(fraction)}.csv"
        write_csv(output_dir / filename, fieldnames, selected_rows)

        per_class_stats: dict[str, Any] = {}
        for label in LABEL_ORDER:
            selected_class_rows = [row for row in selected_rows if row["label"] == label]
            total_class_rows = [row for row in rows if row["label"] == label]
            selected_class_patients = {
                row["patient_id"] for row in selected_class_rows
            }
            per_class_stats[label] = {
                "images": len(selected_class_rows),
                "total_train_images": len(total_class_rows),
                "actual_image_fraction": len(selected_class_rows) / len(total_class_rows),
                "patients": len(selected_class_patients),
                "total_train_patients": len(groups[label]),
                "actual_patient_fraction": len(selected_class_patients) / len(groups[label]),
            }

        selected_paths = {row["relative_path"] for row in selected_rows}
        expected_paths = {
            row["relative_path"] for row in rows if row["patient_id"] in selected_patients
        }
        if selected_paths != expected_paths:
            raise AssertionError(f"Image set mismatch for {fraction_label(fraction)}")

        level = {
            "manifest": filename,
            "images": len(selected_rows),
            "total_train_images": len(rows),
            "actual_overall_image_fraction": len(selected_rows) / len(rows),
            "patients": len(selected_patients),
            "total_train_patients": len(all_patients),
            "actual_patient_fraction": len(selected_patients) / len(all_patients),
            "class_distribution": per_class_stats,
        }
        metadata["levels"][fraction_label(fraction)] = level

        for patient_id in all_patients:
            patient_rows = next(
                (groups[label][patient_id] for label in LABEL_ORDER if patient_id in groups[label]),
                [],
            )
            if not patient_rows:
                raise AssertionError(f"Patient group unexpectedly empty: {patient_id}")
            patient_selection_rows.append({
                "fraction": fraction_label(fraction),
                "patient_id": patient_id,
                "label": patient_rows[0]["label"],
                "num_images": len(patient_rows),
                "selected": patient_id in selected_patients,
            })

    # Nested image-path assertions.
    for smaller, larger in zip(FRACTIONS, FRACTIONS[1:]):
        smaller_paths = {
            row["relative_path"] for row in rows
            if row["patient_id"] in selected_by_fraction[smaller]
        }
        larger_paths = {
            row["relative_path"] for row in rows
            if row["patient_id"] in selected_by_fraction[larger]
        }
        if not smaller_paths.issubset(larger_paths):
            raise AssertionError(f"Image manifests are not nested: {smaller} -> {larger}")

    write_csv(
        output_dir / "patient_selection_audit.csv",
        ["fraction", "patient_id", "label", "num_images", "selected"],
        patient_selection_rows,
    )

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")

    readme = """# DL-05 Label Efficiency Manifests\n\n"""
    readme += (
        "These CSV files contain Train-only patient-level subsets. Each selected patient "
        "contributes all of their Train images. The four subsets are nested.\n\n"
        "- `train_010pct.csv`\n- `train_025pct.csv`\n- `train_050pct.csv`\n"
        "- `train_100pct.csv`\n- `patient_selection_audit.csv`\n- `metadata.json`\n\n"
        "Nominal percentages are image-fraction targets applied separately within each "
        "class. Since patients may contribute different numbers of images, actual image "
        "fractions may differ slightly; use the reported actual fractions in the report. "
        "The script reads only the Train split and does not open Validation or Test.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    try:
        metadata = create_manifests(args.source, args.output_dir, args.seed)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("=" * 76)
    print("DL-05 LABEL EFFICIENCY MANIFESTS CREATED")
    print("=" * 76)
    print(f"Seed: {args.seed}")
    print(f"Source: {args.source}")
    print(f"Output: {args.output_dir}")
    print("Validation and Test were not opened.")
    print("\nActual subset sizes:")
    for level_name, level in metadata["levels"].items():
        print(
            f"{level_name}: {level['images']}/{level['total_train_images']} images "
            f"({level['actual_overall_image_fraction']:.2%}); "
            f"{level['patients']}/{level['total_train_patients']} patients"
        )
        for label in LABEL_ORDER:
            stats = level["class_distribution"][label]
            print(
                f"  {label}: {stats['images']}/{stats['total_train_images']} images "
                f"({stats['actual_image_fraction']:.2%}), "
                f"{stats['patients']}/{stats['total_train_patients']} patients"
            )
    print("\nPASS: Patient groups are intact and all four manifests are nested.")
    print("PASS: 100% manifest contains all Train images.")
    print("PASS: Source CSV SHA256 recorded in metadata.json.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
