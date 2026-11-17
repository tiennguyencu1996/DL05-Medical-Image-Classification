
"""
Extract only the Chest X-Ray portion of ZhangLabData Version 3.

Source:
Kermany et al. (2018)
DOI: 10.17632/rscbjbr9sj.3

The original ZIP archive is preserved.
OCT images and macOS .DS_Store files are excluded.
"""

from pathlib import Path, PurePosixPath
from collections import Counter
from zipfile import ZipFile
import shutil


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARCHIVE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "ZhangLabData.zip"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "version3"
    / "CellData"
    / "chest_xray"
)

ZIP_PREFIX = PurePosixPath("CellData/chest_xray")

VALID_SPLITS = {"train", "test"}
VALID_CLASSES = {"NORMAL", "PNEUMONIA"}
VALID_EXTENSIONS = {".jpeg", ".jpg", ".png"}


# --------------------------------------------------
# 2. VALIDATE INPUT
# --------------------------------------------------

if not ARCHIVE_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset archive not found: {ARCHIVE_PATH}"
    )

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. EXTRACT SELECTED FILES
# --------------------------------------------------

def main():

    counts = Counter()
    extracted_size = 0
    skipped_existing = 0

    print("=" * 65)
    print("CHEST X-RAY DATASET EXTRACTION")
    print("=" * 65)

    print(f"Archive: {ARCHIVE_PATH}")
    print(f"Output:  {OUTPUT_ROOT}")

    with ZipFile(ARCHIVE_PATH, "r") as archive:

        selected_entries = []

        # First pass: identify image files only.
        for info in archive.infolist():

            if info.is_dir():
                continue

            member = PurePosixPath(info.filename)

            if member.parts[:len(ZIP_PREFIX.parts)] != ZIP_PREFIX.parts:
                continue

            relative_parts = member.parts[len(ZIP_PREFIX.parts):]

            if len(relative_parts) != 3:
                continue

            split, label, filename = relative_parts

            if split not in VALID_SPLITS:
                continue

            if label not in VALID_CLASSES:
                continue

            if PurePosixPath(filename).suffix.lower() not in VALID_EXTENSIONS:
                continue

            # Basic archive path safety check.
            if ".." in relative_parts or member.is_absolute():
                raise ValueError(
                    f"Unsafe archive path: {info.filename}"
                )

            selected_entries.append(
                (info, split, label, filename)
            )

        required_bytes = sum(
            info.file_size
            for info, _, _, _ in selected_entries
        )

        available_bytes = shutil.disk_usage(
            PROJECT_ROOT
        ).free

        print(f"\nSelected image files: {len(selected_entries)}")
        print(f"Uncompressed size: {required_bytes / 1024**3:.2f} GiB")
        print(f"Available disk: {available_bytes / 1024**3:.2f} GiB")

        if available_bytes < required_bytes:
            raise OSError(
                "Insufficient disk space for extraction."
            )

        # Second pass: extract selected files.
        for index, (info, split, label, filename) in enumerate(
            selected_entries, start=1
        ):

            relative_path = Path(split) / label / filename
            output_path = OUTPUT_ROOT / relative_path

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # Avoid extracting an already complete file again.
            if (
                output_path.is_file()
                and output_path.stat().st_size == info.file_size
            ):
                skipped_existing += 1
                counts[(split, label)] += 1
                continue

            with archive.open(info, "r") as source:
                with output_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

            extracted_size += info.file_size
            counts[(split, label)] += 1

            if index % 500 == 0:
                print(
                    f"Processed {index}/{len(selected_entries)} images"
                )

    # --------------------------------------------------
    # 4. REPORT
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("EXTRACTION SUMMARY")
    print("=" * 65)

    for split in sorted(VALID_SPLITS):
        for label in sorted(VALID_CLASSES):
            print(
                f"{split:8s} | {label:10s} | "
                f"{counts[(split, label)]:5d} images"
            )

    total_images = sum(counts.values())

    print("-" * 65)
    print(f"Total images: {total_images}")
    print(f"Newly extracted: {total_images - skipped_existing}")
    print(f"Already existing and skipped: {skipped_existing}")
    print(f"Newly extracted size: {extracted_size / 1024**3:.2f} GiB")
    print(f"Output directory: {OUTPUT_ROOT}")

    print("\nExtraction completed.")


if __name__ == "__main__":
    main()
