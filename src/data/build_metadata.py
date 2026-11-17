
"""
Build metadata for Chest X-Ray Pneumonia Dataset.

Dataset: ZhangLabData Version 3
DOI: 10.17632/rscbjbr9sj.3

This script:
1. Scans extracted Chest X-Ray images.
2. Parses patient IDs and image numbers.
3. Maps image labels to binary classes.
4. Validates image counts and patient-level separation.
5. Exports metadata and dataset summary CSV files.

Original train/test assignments are preserved.
"""

from pathlib import Path
from collections import defaultdict, Counter
import csv
import re


# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "version3"
    / "CellData"
    / "chest_xray"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = ["train", "test"]

CLASSES = ["NORMAL", "PNEUMONIA"]

VALID_EXTENSIONS = {".jpeg", ".jpg", ".png"}

PREFIX_MAPPING = {
    "NORMAL": "NORMAL",
    "BACTERIA": "PNEUMONIA",
    "VIRUS": "PNEUMONIA",
}

EXPECTED_COUNTS = {
    ("train", "NORMAL"): 1349,
    ("train", "PNEUMONIA"): 3883,
    ("test", "NORMAL"): 234,
    ("test", "PNEUMONIA"): 390,
}

FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>[A-Z]+)-"
    r"(?P<patient_id>\d+)-"
    r"(?P<image_number>\d+)$",
    re.IGNORECASE,
)


# ==================================================
# 2. VALIDATE INPUT
# ==================================================

if not DATASET_ROOT.is_dir():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_ROOT}"
    )


# ==================================================
# 3. BUILD METADATA RECORDS
# ==================================================

records = []

patient_splits = defaultdict(set)
patient_labels = defaultdict(set)

counts = Counter()
patient_counts = defaultdict(set)

print("=" * 70)
print("BUILDING CHEST X-RAY METADATA")
print("=" * 70)

for split in SPLITS:

    for label in CLASSES:

        class_dir = DATASET_ROOT / split / label

        if not class_dir.is_dir():
            raise FileNotFoundError(class_dir)

        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        )

        for image_path in image_paths:

            match = FILENAME_PATTERN.match(image_path.stem)

            if match is None:
                raise ValueError(
                    f"Unrecognized filename: {image_path.name}"
                )

            prefix = match.group("prefix").upper()
            patient_id = match.group("patient_id")
            image_number = match.group("image_number")

            if PREFIX_MAPPING.get(prefix) != label:
                raise ValueError(
                    f"Label mismatch: {image_path}"
                )

            binary_label = 0 if label == "NORMAL" else 1

            relative_path = image_path.relative_to(
                PROJECT_ROOT
            ).as_posix()

            record = {
                "relative_path": relative_path,
                "filename": image_path.name,
                "split": split,
                "label": label,
                "binary_label": binary_label,
                "filename_prefix": prefix,
                "patient_id": patient_id,
                "image_number": image_number,
                "file_extension": image_path.suffix.lower(),
                "file_size_bytes": image_path.stat().st_size,
            }

            records.append(record)

            counts[(split, label)] += 1

            patient_splits[patient_id].add(split)

            patient_labels[(split, patient_id)].add(label)

            patient_counts[(split, label)].add(patient_id)


# ==================================================
# 4. VALIDATE EXPECTED IMAGE COUNTS
# ==================================================

print("\n[1] IMAGE COUNT VALIDATION")

for key, expected in EXPECTED_COUNTS.items():

    actual = counts[key]

    print(
        f"{key[0]:8s} | {key[1]:12s} | "
        f"Expected: {expected:5d} | "
        f"Actual: {actual:5d}"
    )

    if actual != expected:
        raise ValueError(
            f"Image count mismatch for {key}: "
            f"expected {expected}, got {actual}"
        )


# ==================================================
# 5. VALIDATE PATIENT SEPARATION
# ==================================================

train_patients = {
    patient_id
    for patient_id, splits in patient_splits.items()
    if "train" in splits
}

test_patients = {
    patient_id
    for patient_id, splits in patient_splits.items()
    if "test" in splits
}

overlap = train_patients & test_patients

if overlap:
    raise ValueError(
        f"Patient overlap detected: {len(overlap)}"
    )

multi_label = [
    (split, patient_id, labels)
    for (split, patient_id), labels in patient_labels.items()
    if len(labels) > 1
]

if multi_label:
    raise ValueError(
        f"Patients with multiple labels detected: "
        f"{len(multi_label)}"
    )


# ==================================================
# 6. EXPORT METADATA CSV
# ==================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metadata_path = OUTPUT_DIR / "metadata_v3.csv"

fieldnames = [
    "relative_path",
    "filename",
    "split",
    "label",
    "binary_label",
    "filename_prefix",
    "patient_id",
    "image_number",
    "file_extension",
    "file_size_bytes",
]

records.sort(
    key=lambda row: (
        row["split"],
        row["label"],
        row["patient_id"],
        int(row["image_number"]),
        row["filename"],
    )
)

with metadata_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(records)


# ==================================================
# 7. EXPORT DATASET SUMMARY
# ==================================================

summary_path = OUTPUT_DIR / "dataset_summary_v3.csv"

summary_rows = []

for split in SPLITS:

    for label in CLASSES:

        summary_rows.append({
            "split": split,
            "label": label,
            "num_images": counts[(split, label)],
            "num_unique_patients": len(
                patient_counts[(split, label)]
            ),
        })

with summary_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "split",
            "label",
            "num_images",
            "num_unique_patients",
        ],
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# ==================================================
# 8. FINAL REPORT
# ==================================================

print("\n" + "=" * 70)
print("METADATA GENERATION COMPLETED")
print("=" * 70)

print("Total images:", len(records))
print("Unique train patients:", len(train_patients))
print("Unique test patients:", len(test_patients))
print("Train/test patient overlap:", len(overlap))

print("\nMetadata:", metadata_path)
print("Summary:", summary_path)

print("\nAll validation checks passed.")
print("=" * 70)
