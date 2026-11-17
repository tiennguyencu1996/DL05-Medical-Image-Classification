
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import statistics

from PIL import Image


# ============================================================
# 1. PROJECT PATHS AND EXPECTED COUNTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "version3"
    / "CellData"
    / "chest_xray"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "metadata_v3.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

VALIDATION_CSV = OUTPUT_DIR / "image_validation_v3.csv"
SUMMARY_JSON = OUTPUT_DIR / "image_validation_summary_v3.json"

EXPECTED_COUNTS = {
    ("train", "NORMAL"): 1349,
    ("train", "PNEUMONIA"): 3883,
    ("test", "NORMAL"): 234,
    ("test", "PNEUMONIA"): 390,
}

EXPECTED_TOTAL_IMAGES = sum(EXPECTED_COUNTS.values())

CHUNK_SIZE = 1024 * 1024  # 1 MB


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def calculate_sha256(file_path):
    """
    Calculate SHA-256 using chunks.
    This avoids loading the entire file into RAM.
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(CHUNK_SIZE)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def write_csv(file_path, records):
    """Write validation records to CSV."""
    if not records:
        return

    fieldnames = list(records[0].keys())

    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


# ============================================================
# 3. MAIN VALIDATION
# ============================================================

def main():

    print("=" * 70)
    print("CHEST X-RAY IMAGE VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check required paths
    # --------------------------------------------------------

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    with open(METADATA_PATH, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        metadata = list(reader)

    if not metadata:
        raise ValueError("Metadata CSV is empty.")

    required_columns = {
        "relative_path",
        "split",
        "label",
        "patient_id",
    }

    missing_columns = required_columns - set(metadata[0].keys())

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing_columns)}"
        )

    print(f"Metadata records: {len(metadata):,}")
    print(f"Dataset root: {DATASET_ROOT}")
    print("\nStarting image validation...\n")

    # --------------------------------------------------------
    # Initialize containers
    # --------------------------------------------------------

    validation_records = []

    # SHA-256 -> list of images sharing the same exact file hash
    hash_to_images = defaultdict(list)

    formats = Counter()
    modes = Counter()

    widths = []
    heights = []

    dataset_root_resolved = DATASET_ROOT.resolve()

    # --------------------------------------------------------
    # Validate each image
    # --------------------------------------------------------

    for index, row in enumerate(metadata, start=1):

        relative_path = row["relative_path"]

        result = {
            "relative_path": relative_path,
            "split": row["split"],
            "label": row["label"],
            "patient_id": row["patient_id"],
            "status": "",
            "width": "",
            "height": "",
            "mode": "",
            "format": "",
            "file_size_bytes": "",
            "sha256": "",
            "error": "",
        }

        # ----------------------------------------------------
        # Resolve image path
        #
        # IMPORTANT:
        # relative_path in metadata is relative to PROJECT_ROOT,
        # not relative to DATASET_ROOT.
        # ----------------------------------------------------

        image_path = (PROJECT_ROOT / relative_path).resolve()

        # ----------------------------------------------------
        # Prevent paths from escaping the dataset directory
        # ----------------------------------------------------

        try:
            image_path.relative_to(dataset_root_resolved)

        except ValueError:
            result["status"] = "INVALID_PATH"
            result["error"] = "Path is outside dataset directory"

            validation_records.append(result)
            continue

        # ----------------------------------------------------
        # Check file existence
        # ----------------------------------------------------

        if not image_path.is_file():
            result["status"] = "MISSING"
            result["error"] = "File does not exist"

            validation_records.append(result)
            continue

        result["file_size_bytes"] = image_path.stat().st_size

        # ----------------------------------------------------
        # Verify and decode image
        # ----------------------------------------------------

        try:
            # First pass: verify file structure.
            with Image.open(image_path) as image:
                image.verify()

            # Second pass: fully decode image data.
            with Image.open(image_path) as image:
                image.load()

                width, height = image.size
                image_mode = image.mode
                image_format = image.format or "UNKNOWN"

            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions")

            # ------------------------------------------------
            # Calculate exact file hash
            # ------------------------------------------------

            file_hash = calculate_sha256(image_path)

            # ------------------------------------------------
            # Record successful validation
            # ------------------------------------------------

            result.update({
                "status": "VALID",
                "width": width,
                "height": height,
                "mode": image_mode,
                "format": image_format,
                "sha256": file_hash,
            })

            widths.append(width)
            heights.append(height)

            formats[image_format] += 1
            modes[image_mode] += 1

            # Record image information for duplicate detection.
            hash_to_images[file_hash].append({
                "relative_path": relative_path,
                "split": row["split"],
                "label": row["label"],
                "patient_id": row["patient_id"],
            })

        except Exception as error:
            result["status"] = "INVALID"
            result["error"] = (
                f"{type(error).__name__}: {str(error)[:250]}"
            )

        validation_records.append(result)

        # ----------------------------------------------------
        # Progress report
        # ----------------------------------------------------

        if index % 500 == 0 or index == len(metadata):
            print(
                f"Processed {index:,}/{len(metadata):,} images"
            )

    # ========================================================
    # 4. SAVE DETAILED VALIDATION REPORT
    # ========================================================

    write_csv(VALIDATION_CSV, validation_records)

    # ========================================================
    # 5. COUNT VALIDATION RESULTS
    # ========================================================

    status_counts = Counter(
        record["status"]
        for record in validation_records
    )

    actual_counts = Counter(
        (record["split"], record["label"])
        for record in metadata
    )

    expected_counts_match = (
        actual_counts == Counter(EXPECTED_COUNTS)
    )

    total_records_match = (
        len(metadata) == EXPECTED_TOTAL_IMAGES
    )

    # ========================================================
    # 6. EXACT DUPLICATE DETECTION
    # ========================================================

    duplicate_groups = []

    for file_hash, images in hash_to_images.items():

        if len(images) > 1:

            splits = sorted({
                image["split"]
                for image in images
            })

            duplicate_groups.append({
                "sha256": file_hash,
                "number_of_files": len(images),
                "cross_split": len(splits) > 1,
                "files": images,
            })

    cross_split_duplicates = [
        group
        for group in duplicate_groups
        if group["cross_split"]
    ]

    duplicate_extra_files = sum(
        len(group["files"]) - 1
        for group in duplicate_groups
    )

    # ========================================================
    # 7. BUILD SUMMARY
    # ========================================================

    valid_count = status_counts["VALID"]

    invalid_count = (
        len(validation_records) - valid_count
    )

    summary = {
        "total_metadata_records": len(metadata),

        "valid_images": valid_count,

        "invalid_or_missing_images": invalid_count,

        "status_counts": dict(status_counts),

        "expected_class_counts_match": expected_counts_match,

        "expected_total_records_match": total_records_match,

        "class_counts": {
            f"{split}/{label}": actual_counts[(split, label)]
            for split, label in EXPECTED_COUNTS
        },

        "image_formats": dict(formats),

        "image_modes": dict(modes),

        "dimensions": {
            "min_width": min(widths) if widths else None,
            "max_width": max(widths) if widths else None,
            "median_width": (
                statistics.median(widths)
                if widths else None
            ),

            "min_height": min(heights) if heights else None,
            "max_height": max(heights) if heights else None,
            "median_height": (
                statistics.median(heights)
                if heights else None
            ),
        },

        "unique_sha256_hashes": len(hash_to_images),

        "exact_duplicate_groups": len(duplicate_groups),

        "exact_duplicate_extra_files": duplicate_extra_files,

        "cross_split_duplicate_groups": len(
            cross_split_duplicates
        ),

        "duplicate_groups": duplicate_groups,

        "overall_integrity_pass": (
            total_records_match
            and expected_counts_match
            and valid_count == EXPECTED_TOTAL_IMAGES
            and invalid_count == 0
        ),
    }

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------

    with open(SUMMARY_JSON, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # 8. PRINT FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("IMAGE VALIDATION SUMMARY")
    print("=" * 70)

    print(f"Total metadata records: {len(metadata):,}")
    print(f"Valid images: {valid_count:,}")
    print(f"Invalid or missing images: {invalid_count:,}")

    print("\n[VALIDATION STATUS COUNTS]")

    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count:,}")

    print("\n[IMAGE FORMATS]")

    for image_format, count in sorted(formats.items()):
        print(f"{image_format}: {count:,}")

    print("\n[IMAGE MODES]")

    for image_mode, count in sorted(modes.items()):
        print(f"{image_mode}: {count:,}")

    print("\n[IMAGE DIMENSIONS]")

    print(json.dumps(
        summary["dimensions"],
        indent=4
    ))

    print("\n[EXACT DUPLICATE DETECTION]")

    print(f"Unique SHA-256 hashes: {len(hash_to_images):,}")

    print(f"Duplicate groups: {len(duplicate_groups):,}")

    print(f"Duplicate extra files: {duplicate_extra_files:,}")

    print(
        "Cross-split duplicate groups: "
        f"{len(cross_split_duplicates):,}"
    )

    print("\n[FINAL RESULT]")

    if summary["overall_integrity_pass"]:
        print("PASS: All images passed integrity validation.")

    else:
        print("ATTENTION: Some validation checks did not pass.")

    # --------------------------------------------------------
    # Show first errors, if any
    # --------------------------------------------------------

    error_records = [
        record
        for record in validation_records
        if record["status"] != "VALID"
    ]

    if error_records:

        print("\n[FIRST 10 INVALID OR MISSING IMAGES]")

        for record in error_records[:10]:
            print(
                f"{record['status']} | "
                f"{record['relative_path']} | "
                f"{record['error']}"
            )

    # --------------------------------------------------------
    # Show cross-split duplicate groups, if any
    # --------------------------------------------------------

    if cross_split_duplicates:

        print("\n[CROSS-SPLIT DUPLICATE DETAILS]")

        for group in cross_split_duplicates[:10]:

            print(
                f"\nSHA-256: {group['sha256']}"
            )

            for image in group["files"]:
                print(
                    f"  {image['split']} | "
                    f"{image['label']} | "
                    f"{image['relative_path']}"
                )

    print(f"\nDetailed report: {VALIDATION_CSV}")
    print(f"Summary report: {SUMMARY_JSON}")

    print("=" * 70)


# ============================================================
# 9. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
