
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import re

from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "splits"

INPUT_METADATA = (
    PROCESSED_DIR / "metadata_deduplicated_v3.csv"
)

OUTPUT_TRAIN = OUTPUT_DIR / "train_v3.csv"
OUTPUT_VALIDATION = OUTPUT_DIR / "validation_v3.csv"
OUTPUT_TEST = OUTPUT_DIR / "test_v3.csv"

OUTPUT_ALL = OUTPUT_DIR / "split_assignments_v3.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "split_summary_v3.json"

RANDOM_STATE = 42
N_SPLITS = 5
VALIDATION_FOLD = 0

EXPECTED_COUNTS = {
    ("train", "NORMAL"): 1348,
    ("train", "PNEUMONIA"): 3858,
    ("test", "NORMAL"): 234,
    ("test", "PNEUMONIA"): 387,
}

EXPECTED_PATIENTS = {
    "train": 2650,
    "test": 431,
}

EXPECTED_TOTAL_IMAGES = 5827

LABELS = ("NORMAL", "PNEUMONIA")

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def read_metadata(file_path):
    """Read metadata CSV."""
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("Metadata CSV has no header.")

        records = list(reader)
        fields = reader.fieldnames

    if not records:
        raise ValueError("Metadata CSV is empty.")

    return records, fields


def write_csv(file_path, records, fields):
    """Write CSV records."""
    with open(
        file_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="raise"
        )

        writer.writeheader()
        writer.writerows(records)


def count_images(records):
    """Count images by split and class."""
    return Counter(
        (row["split"], row["label"])
        for row in records
    )


def get_patient_labels(records):
    """
    Map each patient ID to one class label.
    Raise an error if a patient has multiple labels.
    """
    patient_labels = {}

    for row in records:

        patient_id = row["patient_id"]
        label = row["label"]

        if patient_id in patient_labels:

            if patient_labels[patient_id] != label:
                raise ValueError(
                    f"Patient {patient_id} has multiple labels."
                )

        patient_labels[patient_id] = label

    return patient_labels


def get_patient_ids(records):
    """Return unique patient IDs."""
    return {
        row["patient_id"]
        for row in records
    }


def calculate_split_statistics(records):
    """Calculate image-level and patient-level statistics."""

    image_counts = Counter(
        row["label"] for row in records
    )

    patient_labels = get_patient_labels(records)

    patient_counts = Counter(
        patient_labels.values()
    )

    total_images = len(records)
    total_patients = len(patient_labels)

    image_distribution = {}
    patient_distribution = {}

    for label in LABELS:

        image_count = image_counts[label]
        patient_count = patient_counts[label]

        image_distribution[label] = {
            "count": image_count,
            "prevalence_percent": (
                round(image_count / total_images * 100, 2)
                if total_images else 0
            ),
        }

        patient_distribution[label] = {
            "count": patient_count,
            "prevalence_percent": (
                round(patient_count / total_patients * 100, 2)
                if total_patients else 0
            ),
        }

    return {
        "number_of_images": total_images,
        "number_of_unique_patients": total_patients,
        "image_distribution": image_distribution,
        "patient_distribution": patient_distribution,
    }


def assign_final_split(records, final_split):
    """
    Preserve original dataset provenance in source_split.
    Set split to the final experimental assignment.
    """
    output = []

    for row in records:

        new_row = dict(row)

        new_row["source_split"] = new_row["split"]
        new_row["split"] = final_split

        output.append(new_row)

    return output


# ============================================================
# 3. LOAD AND VALIDATE INPUT METADATA
# ============================================================

def main():

    print("=" * 70)
    print("PATIENT-LEVEL DATASET SPLITTING")
    print("=" * 70)

    records, input_fields = read_metadata(INPUT_METADATA)

    print(f"Input metadata records: {len(records):,}")

    required_fields = {
        "relative_path",
        "split",
        "label",
        "patient_id",
        "sha256",
    }

    missing_fields = required_fields - set(input_fields)

    if missing_fields:
        raise ValueError(
            f"Missing required columns: {sorted(missing_fields)}"
        )

    if len(records) != EXPECTED_TOTAL_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_IMAGES} images, "
            f"found {len(records)}."
        )

    # --------------------------------------------------------
    # Validate source split and class names
    # --------------------------------------------------------

    for row in records:

        if row["split"] not in ("train", "test"):
            raise ValueError(
                f"Unexpected source split: {row['split']}"
            )

        if row["label"] not in LABELS:
            raise ValueError(
                f"Unexpected label: {row['label']}"
            )

        if not row["patient_id"]:
            raise ValueError("Empty patient ID detected.")

        if not SHA256_PATTERN.fullmatch(row["sha256"]):
            raise ValueError(
                f"Invalid SHA-256: {row['relative_path']}"
            )

    # --------------------------------------------------------
    # Validate unique paths and hashes
    # --------------------------------------------------------

    paths = [row["relative_path"] for row in records]
    hashes = [row["sha256"] for row in records]

    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate relative paths detected.")

    if len(hashes) != len(set(hashes)):
        raise ValueError(
            "Duplicate SHA-256 hashes remain in input metadata."
        )

    # --------------------------------------------------------
    # Validate class counts
    # --------------------------------------------------------

    actual_counts = count_images(records)

    if actual_counts != Counter(EXPECTED_COUNTS):
        raise ValueError(
            f"Unexpected class counts: {dict(actual_counts)}"
        )

    print("\n[1] INPUT CLASS COUNTS")

    for (split, label), expected in EXPECTED_COUNTS.items():

        actual = actual_counts[(split, label)]

        print(
            f"{split:8s} | {label:12s} | "
            f"{actual:,}/{expected:,}"
        )

    # --------------------------------------------------------
    # Validate original patient separation
    # --------------------------------------------------------

    source_train = [
        row for row in records
        if row["split"] == "train"
    ]

    source_test = [
        row for row in records
        if row["split"] == "test"
    ]

    original_train_patients = get_patient_ids(source_train)
    original_test_patients = get_patient_ids(source_test)

    if len(original_train_patients) != EXPECTED_PATIENTS["train"]:
        raise ValueError("Unexpected original train patient count.")

    if len(original_test_patients) != EXPECTED_PATIENTS["test"]:
        raise ValueError("Unexpected original test patient count.")

    original_overlap = (
        original_train_patients & original_test_patients
    )

    if original_overlap:
        raise ValueError(
            "Patient ID overlap between original train and test."
        )

    get_patient_labels(source_train)
    get_patient_labels(source_test)

    print("\n[2] ORIGINAL PATIENT SEPARATION")
    print(f"Train patients: {len(original_train_patients):,}")
    print(f"Test patients: {len(original_test_patients):,}")
    print(f"Overlapping patients: {len(original_overlap)}")

    # ========================================================
    # 4. STRATIFIED GROUP SPLITTING
    # ========================================================

    print("\n[3] STRATIFIED GROUP SPLITTING")

    # Sort records to make input ordering deterministic.
    source_train = sorted(
        source_train,
        key=lambda row: row["relative_path"]
    )

    X = list(range(len(source_train)))

    y = [
        row["label"]
        for row in source_train
    ]

    groups = [
        row["patient_id"]
        for row in source_train
    ]

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    folds = list(
        splitter.split(X, y, groups)
    )

    if len(folds) != N_SPLITS:
        raise ValueError(
            f"Expected {N_SPLITS} folds, got {len(folds)}."
        )

    train_indices, validation_indices = folds[VALIDATION_FOLD]

    train_records = [
        source_train[int(index)]
        for index in train_indices
    ]

    validation_records = [
        source_train[int(index)]
        for index in validation_indices
    ]

    # Keep original test set completely unchanged.
    test_records = sorted(
        source_test,
        key=lambda row: row["relative_path"]
    )

    print(f"StratifiedGroupKFold folds: {N_SPLITS}")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Selected validation fold: {VALIDATION_FOLD}")

    # ========================================================
    # 5. PATIENT-LEVEL SEPARATION VALIDATION
    # ========================================================

    print("\n[4] PATIENT-LEVEL SEPARATION")

    train_patients = get_patient_ids(train_records)
    validation_patients = get_patient_ids(validation_records)
    test_patients = get_patient_ids(test_records)

    train_val_overlap = train_patients & validation_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = validation_patients & test_patients

    if train_val_overlap:
        raise ValueError(
            "Patient overlap between final train and validation."
        )

    if train_test_overlap:
        raise ValueError(
            "Patient overlap between final train and test."
        )

    if val_test_overlap:
        raise ValueError(
            "Patient overlap between final validation and test."
        )

    if (
        train_patients | validation_patients
    ) != original_train_patients:
        raise ValueError(
            "Train and validation do not cover all original train patients."
        )

    if test_patients != original_test_patients:
        raise ValueError(
            "Original test patient set has changed."
        )

    print(f"Train patients: {len(train_patients):,}")
    print(f"Validation patients: {len(validation_patients):,}")
    print(f"Test patients: {len(test_patients):,}")

    print(f"Train/Validation overlap: {len(train_val_overlap)}")
    print(f"Train/Test overlap: {len(train_test_overlap)}")
    print(f"Validation/Test overlap: {len(val_test_overlap)}")

    # ========================================================
    # 6. ASSIGN FINAL SPLIT NAMES
    # ========================================================

    final_train = assign_final_split(
        train_records,
        "train"
    )

    final_validation = assign_final_split(
        validation_records,
        "validation"
    )

    final_test = assign_final_split(
        test_records,
        "test"
    )

    all_final_records = (
        final_train
        + final_validation
        + final_test
    )

    # ========================================================
    # 7. VERIFY IMAGE AND PATIENT COUNTS
    # ========================================================

    print("\n[5] FINAL IMAGE COUNTS")

    final_splits = {
        "train": final_train,
        "validation": final_validation,
        "test": final_test,
    }

    for split_name, split_records in final_splits.items():

        class_counts = Counter(
            row["label"]
            for row in split_records
        )

        patient_count = len(
            get_patient_ids(split_records)
        )

        print(
            f"{split_name:12s} | "
            f"Images: {len(split_records):,} | "
            f"NORMAL: {class_counts['NORMAL']:,} | "
            f"PNEUMONIA: {class_counts['PNEUMONIA']:,} | "
            f"Patients: {patient_count:,}"
        )

        if not split_records:
            raise ValueError(
                f"Empty final split: {split_name}"
            )

        if set(class_counts) != set(LABELS):
            raise ValueError(
                f"Missing a class in {split_name}."
            )

    if len(all_final_records) != EXPECTED_TOTAL_IMAGES:
        raise ValueError(
            "Final split total does not equal the original total."
        )

    final_paths = [
        row["relative_path"]
        for row in all_final_records
    ]

    if len(final_paths) != len(set(final_paths)):
        raise ValueError(
            "Duplicate relative paths across final splits."
        )

    # --------------------------------------------------------
    # Verify exact original test images are preserved
    # --------------------------------------------------------

    original_test_paths = {
        row["relative_path"] for row in source_test
    }

    final_test_paths = {
        row["relative_path"] for row in final_test
    }

    if original_test_paths != final_test_paths:
        raise ValueError(
            "Original test images were changed."
        )

    # --------------------------------------------------------
    # Verify no exact SHA-256 duplication across final splits
    # --------------------------------------------------------

    final_hashes = [
        row["sha256"] for row in all_final_records
    ]

    if len(final_hashes) != len(set(final_hashes)):
        raise ValueError(
            "Duplicate SHA-256 hashes found across final splits."
        )

    # ========================================================
    # 8. CALCULATE SPLIT STATISTICS
    # ========================================================

    print("\n[6] CLASS DISTRIBUTION")

    split_statistics = {}

    for split_name, split_records in final_splits.items():

        stats = calculate_split_statistics(split_records)

        split_statistics[split_name] = stats

        print(f"\n{split_name.upper()}")

        print("Image-level prevalence:")

        for label, values in stats["image_distribution"].items():

            print(
                f"  {label:12s}: "
                f"{values['count']:,} "
                f"({values['prevalence_percent']:.2f}%)"
            )

        print("Patient-level prevalence:")

        for label, values in stats["patient_distribution"].items():

            print(
                f"  {label:12s}: "
                f"{values['count']:,} "
                f"({values['prevalence_percent']:.2f}%)"
            )

    # ========================================================
    # 9. PREPARE OUTPUTS
    # ========================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve source split in a separate column.
    output_fields = []

    for field in input_fields:

        if field == "split":
            output_fields.append("split")
            output_fields.append("source_split")
        else:
            output_fields.append(field)

    # --------------------------------------------------------
    # Write individual split files
    # --------------------------------------------------------

    write_csv(
        OUTPUT_TRAIN,
        final_train,
        output_fields
    )

    write_csv(
        OUTPUT_VALIDATION,
        final_validation,
        output_fields
    )

    write_csv(
        OUTPUT_TEST,
        final_test,
        output_fields
    )

    # --------------------------------------------------------
    # Write combined split manifest
    # --------------------------------------------------------

    all_final_records.sort(
        key=lambda row: (
            row["split"],
            row["label"],
            row["patient_id"],
            row["relative_path"]
        )
    )

    write_csv(
        OUTPUT_ALL,
        all_final_records,
        output_fields
    )

    # ========================================================
    # 10. SAVE SUMMARY
    # ========================================================

    summary = {
        "dataset": "Chest X-Ray Pneumonia - Mendeley Data Version 3",

        "input_metadata": str(
            INPUT_METADATA.relative_to(PROJECT_ROOT)
        ),

        "method": "StratifiedGroupKFold",

        "n_splits": N_SPLITS,

        "validation_fold": VALIDATION_FOLD,

        "random_state": RANDOM_STATE,

        "total_images": len(all_final_records),

        "splits": split_statistics,

        "patient_overlap_checks": {
            "train_validation": len(train_val_overlap),
            "train_test": len(train_test_overlap),
            "validation_test": len(val_test_overlap),
        },

        "original_test_preserved": (
            original_test_paths == final_test_paths
        ),

        "unique_sha256_hashes": len(set(final_hashes)),

        "outputs": {
            "train": str(OUTPUT_TRAIN.relative_to(PROJECT_ROOT)),
            "validation": str(OUTPUT_VALIDATION.relative_to(PROJECT_ROOT)),
            "test": str(OUTPUT_TEST.relative_to(PROJECT_ROOT)),
            "all_assignments": str(OUTPUT_ALL.relative_to(PROJECT_ROOT)),
        },

        "all_checks_passed": True,
    }

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # 11. FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL DATA SPLITTING RESULT")
    print("=" * 70)

    print(f"Total images: {len(all_final_records):,}")
    print(f"Train images: {len(final_train):,}")
    print(f"Validation images: {len(final_validation):,}")
    print(f"Test images: {len(final_test):,}")

    print("\nPatient overlap:")
    print(f"Train / Validation: {len(train_val_overlap)}")
    print(f"Train / Test: {len(train_test_overlap)}")
    print(f"Validation / Test: {len(val_test_overlap)}")

    print("\nOriginal test preserved: YES")
    print("Duplicate SHA-256 across final splits: 0")

    print("\n[OUTPUT FILES]")
    print(f"Train: {OUTPUT_TRAIN}")
    print(f"Validation: {OUTPUT_VALIDATION}")
    print(f"Test: {OUTPUT_TEST}")
    print(f"All assignments: {OUTPUT_ALL}")
    print(f"Summary: {OUTPUT_SUMMARY}")

    print("\n[FINAL RESULT]")
    print("PASS: Patient-level split completed.")
    print("PASS: No patient overlap across final splits.")
    print("PASS: Original test set preserved.")
    print("PASS: No exact duplicate hashes across final splits.")
    print("PASS: All images accounted for.")

    print("=" * 70)


# ============================================================
# 12. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
