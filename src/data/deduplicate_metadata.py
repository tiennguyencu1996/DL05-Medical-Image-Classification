
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import re


# ============================================================
# 1. PROJECT PATHS AND EXPECTED COUNTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

METADATA_PATH = PROCESSED_DIR / "metadata_v3.csv"

VALIDATION_PATH = PROCESSED_DIR / "image_validation_v3.csv"

OUTPUT_METADATA = (
    PROCESSED_DIR / "metadata_deduplicated_v3.csv"
)

OUTPUT_EXCLUSIONS = (
    PROCESSED_DIR / "duplicate_exclusions_v3.csv"
)

OUTPUT_SUMMARY = (
    PROCESSED_DIR / "deduplication_summary_v3.json"
)

EXPECTED_COUNTS_BEFORE = {
    ("train", "NORMAL"): 1349,
    ("train", "PNEUMONIA"): 3883,
    ("test", "NORMAL"): 234,
    ("test", "PNEUMONIA"): 390,
}

EXPECTED_COUNTS_AFTER = {
    ("train", "NORMAL"): 1348,
    ("train", "PNEUMONIA"): 3858,
    ("test", "NORMAL"): 234,
    ("test", "PNEUMONIA"): 387,
}

EXPECTED_TOTAL_BEFORE = 5856
EXPECTED_TOTAL_AFTER = 5827

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def read_csv(file_path):
    """Read a CSV file and return its records."""
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {file_path}")

        records = list(reader)

    return records, reader.fieldnames


def write_csv(file_path, records, fieldnames):
    """Write records to CSV, including the header."""
    with open(file_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="raise"
        )

        writer.writeheader()
        writer.writerows(records)


def count_classes(records):
    """Count images by split and label."""
    return Counter(
        (row["split"], row["label"])
        for row in records
    )


def format_class_counts(counter):
    """Create a readable class-count dictionary."""
    return {
        f"{split}/{label}": counter[(split, label)]
        for split, label in EXPECTED_COUNTS_BEFORE
    }


def get_patient_sets(records):
    """Get unique patient IDs for each split."""
    return {
        split: {
            row["patient_id"]
            for row in records
            if row["split"] == split
        }
        for split in ("train", "test")
    }


def representative_sort_key(row):
    """
    Select a deterministic representative.

    Prefer the smallest image_number.
    Use relative_path as a tie-breaker.
    """
    try:
        image_number = int(row["image_number"])
    except (KeyError, ValueError, TypeError):
        image_number = float("inf")

    return (
        image_number,
        row["relative_path"]
    )


# ============================================================
# 3. MAIN DEDUPLICATION
# ============================================================

def main():

    print("=" * 70)
    print("CHEST X-RAY METADATA DEDUPLICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load metadata and image validation results
    # --------------------------------------------------------

    metadata, metadata_fields = read_csv(METADATA_PATH)

    validation, validation_fields = read_csv(VALIDATION_PATH)

    required_metadata_fields = {
        "relative_path",
        "split",
        "label",
        "patient_id",
        "image_number",
    }

    required_validation_fields = {
        "relative_path",
        "split",
        "label",
        "patient_id",
        "status",
        "sha256",
    }

    missing_metadata_fields = (
        required_metadata_fields - set(metadata_fields)
    )

    missing_validation_fields = (
        required_validation_fields - set(validation_fields)
    )

    if missing_metadata_fields:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing_metadata_fields)}"
        )

    if missing_validation_fields:
        raise ValueError(
            f"Missing validation columns: {sorted(missing_validation_fields)}"
        )

    print(f"Original metadata records: {len(metadata):,}")
    print(f"Validation records: {len(validation):,}")

    # --------------------------------------------------------
    # Validate original dataset counts
    # --------------------------------------------------------

    if len(metadata) != EXPECTED_TOTAL_BEFORE:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_BEFORE} metadata records, "
            f"found {len(metadata)}."
        )

    original_counts = count_classes(metadata)

    if original_counts != Counter(EXPECTED_COUNTS_BEFORE):
        raise ValueError(
            "Original class counts do not match expected counts.\n"
            f"Actual: {format_class_counts(original_counts)}"
        )

    # --------------------------------------------------------
    # Build path lookup tables
    # --------------------------------------------------------

    metadata_by_path = {}

    for row in metadata:

        path = row["relative_path"]

        if path in metadata_by_path:
            raise ValueError(
                f"Duplicate relative_path in metadata: {path}"
            )

        metadata_by_path[path] = row

    validation_by_path = {}

    for row in validation:

        path = row["relative_path"]

        if path in validation_by_path:
            raise ValueError(
                f"Duplicate relative_path in validation CSV: {path}"
            )

        validation_by_path[path] = row

    metadata_paths = set(metadata_by_path)
    validation_paths = set(validation_by_path)

    if metadata_paths != validation_paths:

        missing_validation = metadata_paths - validation_paths
        missing_metadata = validation_paths - metadata_paths

        raise ValueError(
            "Metadata and validation file lists do not match.\n"
            f"Missing validation records: {len(missing_validation)}\n"
            f"Extra validation records: {len(missing_metadata)}"
        )

    if len(validation) != EXPECTED_TOTAL_BEFORE:
        raise ValueError(
            "Image validation record count is incorrect."
        )

    # --------------------------------------------------------
    # Verify every image passed validation
    # --------------------------------------------------------

    for path, validation_row in validation_by_path.items():

        if validation_row["status"] != "VALID":
            raise ValueError(
                f"Image has not passed validation: {path} "
                f"({validation_row['status']})"
            )

        image_hash = validation_row["sha256"]

        if not SHA256_PATTERN.fullmatch(image_hash):
            raise ValueError(
                f"Invalid SHA-256 hash for image: {path}"
            )

        metadata_row = metadata_by_path[path]

        for field in ("split", "label", "patient_id"):

            if metadata_row[field] != validation_row[field]:
                raise ValueError(
                    f"Metadata mismatch for {field}: {path}"
                )

    print("\n[1] IMAGE VALIDATION CROSS-CHECK")
    print("All metadata records match validation records.")
    print("All 5,856 images have VALID status.")

    # --------------------------------------------------------
    # Verify patient-level consistency
    # --------------------------------------------------------

    patient_labels = defaultdict(set)
    patient_splits = defaultdict(set)

    for row in metadata:

        patient_id = row["patient_id"]

        patient_labels[(row["split"], patient_id)].add(
            row["label"]
        )

        patient_splits[patient_id].add(row["split"])

    multi_label_patients = {
        key: labels
        for key, labels in patient_labels.items()
        if len(labels) > 1
    }

    if multi_label_patients:
        raise ValueError(
            "Some patients have multiple labels within a split."
        )

    overlapping_patients = {
        patient_id: splits
        for patient_id, splits in patient_splits.items()
        if len(splits) > 1
    }

    if overlapping_patients:
        raise ValueError(
            "Patient IDs overlap between train and test."
        )

    patients_before = get_patient_sets(metadata)

    # --------------------------------------------------------
    # Group images by exact SHA-256
    # --------------------------------------------------------

    hash_to_images = defaultdict(list)

    for row in metadata:

        path = row["relative_path"]

        image_hash = validation_by_path[path]["sha256"]

        hash_to_images[image_hash].append(row)

    # --------------------------------------------------------
    # Identify duplicate groups
    # --------------------------------------------------------

    deduplicated_rows = []
    exclusion_records = []

    duplicate_groups = 0
    cross_split_duplicate_groups = 0

    for image_hash, images in hash_to_images.items():

        # ----------------------------------------------------
        # Each hash must belong to only one patient, label,
        # and split. Otherwise, stop for manual investigation.
        # ----------------------------------------------------

        identities = {
            (
                row["split"],
                row["label"],
                row["patient_id"]
            )
            for row in images
        }

        if len(identities) > 1:

            raise ValueError(
                "A duplicate SHA-256 group contains different "
                "patients, labels, or splits.\n"
                f"SHA-256: {image_hash}\n"
                f"Files: {[row['relative_path'] for row in images]}"
            )

        # ----------------------------------------------------
        # Choose one deterministic representative
        # ----------------------------------------------------

        sorted_images = sorted(
            images,
            key=representative_sort_key
        )

        representative = sorted_images[0]

        deduplicated_record = dict(representative)
        deduplicated_record["sha256"] = image_hash

        deduplicated_rows.append(deduplicated_record)

        # ----------------------------------------------------
        # Record other copies as exclusions
        # ----------------------------------------------------

        if len(sorted_images) > 1:

            duplicate_groups += 1

            splits = {
                row["split"]
                for row in sorted_images
            }

            if len(splits) > 1:
                cross_split_duplicate_groups += 1

            for excluded in sorted_images[1:]:

                exclusion_records.append({
                    "sha256": image_hash,
                    "split": excluded["split"],
                    "label": excluded["label"],
                    "patient_id": excluded["patient_id"],
                    "kept_relative_path": representative["relative_path"],
                    "excluded_relative_path": excluded["relative_path"],
                    "reason": (
                        "Exact SHA-256 duplicate; "
                        "one representative retained."
                    ),
                })

    # --------------------------------------------------------
    # Sort outputs deterministically
    # --------------------------------------------------------

    deduplicated_rows.sort(
        key=lambda row: (
            row["split"],
            row["label"],
            row["patient_id"],
            representative_sort_key(row)
        )
    )

    exclusion_records.sort(
        key=lambda row: (
            row["split"],
            row["label"],
            row["patient_id"],
            row["excluded_relative_path"]
        )
    )

    # --------------------------------------------------------
    # Validate deduplicated dataset
    # --------------------------------------------------------

    deduplicated_counts = count_classes(deduplicated_rows)

    if len(deduplicated_rows) != EXPECTED_TOTAL_AFTER:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_AFTER} deduplicated records, "
            f"found {len(deduplicated_rows)}."
        )

    if deduplicated_counts != Counter(EXPECTED_COUNTS_AFTER):
        raise ValueError(
            "Deduplicated class counts do not match expected counts.\n"
            f"Actual: {format_class_counts(deduplicated_counts)}"
        )

    if len(exclusion_records) != (
        EXPECTED_TOTAL_BEFORE - EXPECTED_TOTAL_AFTER
    ):
        raise ValueError(
            "Unexpected number of excluded duplicate files."
        )

    if len({row["sha256"] for row in deduplicated_rows}) != (
        len(deduplicated_rows)
    ):
        raise ValueError(
            "Duplicate SHA-256 hashes remain after deduplication."
        )

    if cross_split_duplicate_groups != 0:
        raise ValueError(
            "Cross-split duplicate groups were detected."
        )

    # --------------------------------------------------------
    # Verify that deduplication did not remove any patient
    # --------------------------------------------------------

    patients_after = get_patient_sets(deduplicated_rows)

    if patients_before != patients_after:
        raise ValueError(
            "Patient IDs changed after deduplication."
        )

    # --------------------------------------------------------
    # Save deduplicated metadata
    # --------------------------------------------------------

    output_metadata_fields = list(metadata_fields)

    if "sha256" not in output_metadata_fields:
        output_metadata_fields.append("sha256")

    write_csv(
        OUTPUT_METADATA,
        deduplicated_rows,
        output_metadata_fields
    )

    # --------------------------------------------------------
    # Save duplicate exclusion log
    # --------------------------------------------------------

    exclusion_fields = [
        "sha256",
        "split",
        "label",
        "patient_id",
        "kept_relative_path",
        "excluded_relative_path",
        "reason",
    ]

    write_csv(
        OUTPUT_EXCLUSIONS,
        exclusion_records,
        exclusion_fields
    )

    # --------------------------------------------------------
    # Build summary report
    # --------------------------------------------------------

    summary = {
        "original_total_images": len(metadata),
        "deduplicated_total_images": len(deduplicated_rows),
        "excluded_duplicate_files": len(exclusion_records),
        "duplicate_groups": duplicate_groups,
        "cross_split_duplicate_groups": cross_split_duplicate_groups,

        "class_counts_before": format_class_counts(
            original_counts
        ),

        "class_counts_after": format_class_counts(
            deduplicated_counts
        ),

        "unique_patients_before": {
            split: len(ids)
            for split, ids in patients_before.items()
        },

        "unique_patients_after": {
            split: len(ids)
            for split, ids in patients_after.items()
        },

        "patient_ids_preserved": patients_before == patients_after,

        "unique_sha256_after": len({
            row["sha256"]
            for row in deduplicated_rows
        }),

        "deduplication_policy": (
            "For each exact SHA-256 group, retain one image "
            "with the smallest image_number. Use relative_path "
            "as deterministic tie-breaker. Do not delete source images."
        ),

        "all_checks_passed": True,
    }

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # 4. PRINT FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("DEDUPLICATION SUMMARY")
    print("=" * 70)

    print(f"Original images: {len(metadata):,}")
    print(f"Duplicate groups: {duplicate_groups:,}")
    print(f"Excluded duplicate files: {len(exclusion_records):,}")
    print(f"Cross-split duplicate groups: {cross_split_duplicate_groups:,}")
    print(f"Deduplicated images: {len(deduplicated_rows):,}")

    print("\n[CLASS DISTRIBUTION BEFORE]")

    for key, value in format_class_counts(original_counts).items():
        print(f"{key:25s}: {value:,}")

    print("\n[CLASS DISTRIBUTION AFTER]")

    for key, value in format_class_counts(deduplicated_counts).items():
        print(f"{key:25s}: {value:,}")

    print("\n[UNIQUE PATIENTS]")

    for split in ("train", "test"):
        print(
            f"{split:10s}: "
            f"{len(patients_before[split]):,} "
            f"before -> {len(patients_after[split]):,} after"
        )

    print("\n[OUTPUT FILES]")
    print(f"Deduplicated metadata: {OUTPUT_METADATA}")
    print(f"Duplicate exclusions: {OUTPUT_EXCLUSIONS}")
    print(f"Summary: {OUTPUT_SUMMARY}")

    print("\n[FINAL RESULT]")
    print("PASS: Deduplication completed successfully.")
    print("PASS: No patient IDs were removed.")
    print("PASS: No cross-split duplicate groups.")
    print("PASS: No exact duplicate hashes remain.")

    print("=" * 70)


# ============================================================
# 5. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
