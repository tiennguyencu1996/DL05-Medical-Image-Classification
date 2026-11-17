
"""
Patient ID audit for Chest X-Ray Pneumonia Dataset.

Dataset:
ZhangLabData Version 3
DOI: 10.17632/rscbjbr9sj.3

Purpose:
1. Parse patient IDs from image filenames.
2. Count unique patients by split and class.
3. Detect patient ID overlap between train and test.
4. Detect patient IDs associated with multiple labels.
5. Summarize the number of images per patient.
"""

from pathlib import Path
from collections import Counter, defaultdict
from statistics import median
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

SPLITS = ["train", "test"]

CLASSES = ["NORMAL", "PNEUMONIA"]

VALID_EXTENSIONS = {".jpeg", ".jpg", ".png"}

EXPECTED_PREFIXES = {
    "NORMAL": {"NORMAL"},
    "PNEUMONIA": {"BACTERIA", "VIRUS"},
}

# Filename pattern:
# NORMAL-2552119-0002.jpeg
# BACTERIA-4038442-0001.jpeg
# VIRUS-3443773-0006.jpeg

FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>[A-Z]+)-"
    r"(?P<patient_id>\d+)-"
    r"(?P<image_number>\d+)$",
    re.IGNORECASE,
)


# ==================================================
# 2. VALIDATE DATASET DIRECTORY
# ==================================================

if not DATASET_ROOT.is_dir():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_ROOT}"
    )


# ==================================================
# 3. COLLECT IMAGE METADATA
# ==================================================

records = []

unparsed_files = []
prefix_mismatches = []

patient_to_splits = defaultdict(set)

patient_to_labels = defaultdict(set)

patient_to_prefixes = defaultdict(set)

images_per_patient = Counter()

patients_by_split_class = defaultdict(set)

prefix_counts = Counter()

for split in SPLITS:

    for label in CLASSES:

        class_dir = DATASET_ROOT / split / label

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing directory: {class_dir}"
            )

        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        )

        for image_path in image_paths:

            match = FILENAME_PATTERN.match(
                image_path.stem
            )

            if match is None:

                unparsed_files.append(
                    str(image_path.relative_to(PROJECT_ROOT))
                )

                continue

            prefix = match.group("prefix").upper()

            patient_id = match.group("patient_id")

            image_number = match.group("image_number")

            # Check consistency between folder class
            # and filename prefix.

            allowed_prefixes = EXPECTED_PREFIXES[label]

            if prefix not in allowed_prefixes:

                prefix_mismatches.append(
                    {
                        "path": str(image_path),
                        "label": label,
                        "prefix": prefix,
                    }
                )

            # Store metadata in memory.

            record = {
                "path": str(
                    image_path.relative_to(PROJECT_ROOT)
                ),
                "split": split,
                "label": label,
                "prefix": prefix,
                "patient_id": patient_id,
                "image_number": image_number,
            }

            records.append(record)

            patient_to_splits[patient_id].add(split)

            patient_to_labels[(split, patient_id)].add(label)

            patient_to_prefixes[patient_id].add(prefix)

            images_per_patient[
                (split, label, patient_id)
            ] += 1

            patients_by_split_class[
                (split, label)
            ].add(patient_id)

            prefix_counts[(split, label, prefix)] += 1


# ==================================================
# 4. SUMMARY BY SPLIT AND CLASS
# ==================================================

print("=" * 70)
print("PATIENT ID AUDIT - CHEST X-RAY VERSION 3")
print("=" * 70)

print("\n[1] IMAGE AND PATIENT COUNTS")

for split in SPLITS:

    for label in CLASSES:

        image_count = sum(
            1
            for record in records
            if record["split"] == split
            and record["label"] == label
        )

        patient_count = len(
            patients_by_split_class[(split, label)]
        )

        print(
            f"{split:8s} | {label:12s} | "
            f"Images: {image_count:5d} | "
            f"Unique patients: {patient_count:5d}"
        )


# ==================================================
# 5. PATIENT ID OVERLAP BETWEEN TRAIN AND TEST
# ==================================================

print("\n[2] PATIENT ID OVERLAP")

train_patients = {
    patient_id
    for patient_id, splits in patient_to_splits.items()
    if "train" in splits
}

test_patients = {
    patient_id
    for patient_id, splits in patient_to_splits.items()
    if "test" in splits
}

overlapping_patients = train_patients & test_patients

print("Unique train patients:", len(train_patients))

print("Unique test patients:", len(test_patients))

print("Overlapping patient IDs:", len(overlapping_patients))

if overlapping_patients:

    print("\nWARNING: Patient ID overlap detected!")

    for patient_id in sorted(overlapping_patients)[:20]:

        print(
            patient_id,
            "| prefixes:",
            sorted(patient_to_prefixes[patient_id]),
        )

else:

    print(
        "No matching Patient IDs found between "
        "train and test."
    )


# ==================================================
# 6. PATIENT IDS ASSOCIATED WITH MULTIPLE LABELS
# ==================================================

print("\n[3] PATIENT IDS WITH MULTIPLE LABELS")

multi_label_patients = []

for (split, patient_id), labels in patient_to_labels.items():

    if len(labels) > 1:

        multi_label_patients.append(
            (split, patient_id, sorted(labels))
        )

print(
    "Patient IDs with multiple labels in the same split:",
    len(multi_label_patients),
)

for item in multi_label_patients[:20]:
    print(item)


# ==================================================
# 7. IMAGE COUNTS PER PATIENT
# ==================================================

print("\n[4] IMAGES PER PATIENT")

for split in SPLITS:

    for label in CLASSES:

        counts = [
            count
            for (s, l, patient_id), count
            in images_per_patient.items()
            if s == split and l == label
        ]

        if not counts:
            continue

        print(
            f"{split:8s} | {label:12s} | "
            f"Min: {min(counts):3d} | "
            f"Median: {median(counts):5.1f} | "
            f"Max: {max(counts):3d}"
        )


# ==================================================
# 8. FILENAME PREFIX DISTRIBUTION
# ==================================================

print("\n[5] FILENAME PREFIX DISTRIBUTION")

for key in sorted(prefix_counts):

    split, label, prefix = key

    print(
        f"{split:8s} | {label:12s} | "
        f"{prefix:10s} | {prefix_counts[key]:5d}"
    )


# ==================================================
# 9. FILENAME VALIDATION
# ==================================================

print("\n[6] FILENAME VALIDATION")

print("Images successfully parsed:", len(records))

print("Unparsed filenames:", len(unparsed_files))

print("Folder/prefix mismatches:", len(prefix_mismatches))

if unparsed_files:

    print("\nUnparsed examples:")

    for path in unparsed_files[:20]:
        print(path)

if prefix_mismatches:

    print("\nPrefix mismatch examples:")

    for item in prefix_mismatches[:20]:
        print(item)


# ==================================================
# 10. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 70)
print("FINAL AUDIT SUMMARY")
print("=" * 70)

print("Total parsed images:", len(records))

print("Unique Patient IDs:", len(patient_to_splits))

print("Train/TEST patient overlap:", len(overlapping_patients))

print("Multi-label patient cases:", len(multi_label_patients))

print("Unparsed filenames:", len(unparsed_files))

print("Prefix mismatches:", len(prefix_mismatches))

print("=" * 70)
