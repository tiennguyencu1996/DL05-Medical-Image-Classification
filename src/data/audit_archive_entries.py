
from zipfile import ZipFile
from pathlib import PurePosixPath
from collections import Counter

# Original Version 3 archive
ARCHIVE_PATH = "data/raw/ZhangLabData.zip"

PREFIX = PurePosixPath("CellData/chest_xray")

VALID_SPLITS = {"train", "test"}
VALID_CLASSES = {"NORMAL", "PNEUMONIA"}
VALID_EXTENSIONS = {".jpeg", ".jpg", ".png"}

raw_counts = Counter()
image_counts = Counter()

non_image_files = []
unexpected_files = []

with ZipFile(ARCHIVE_PATH, "r") as archive:

    for info in archive.infolist():

        if info.is_dir():
            continue

        path = PurePosixPath(info.filename)

        # Check whether this file belongs to Chest X-Ray.
        if path.parts[:len(PREFIX.parts)] != PREFIX.parts:
            continue

        relative_parts = path.parts[len(PREFIX.parts):]

        # Expected structure: split/class/filename
        if len(relative_parts) != 3:
            unexpected_files.append(
                (info.filename, info.file_size)
            )
            continue

        split, label, filename = relative_parts

        if split not in VALID_SPLITS:
            continue

        if label not in VALID_CLASSES:
            continue

        key = (split, label)

        # Count all files inside the class directories.
        raw_counts[key] += 1

        # Identify supported image extensions.
        if path.suffix.lower() in VALID_EXTENSIONS:
            image_counts[key] += 1
        else:
            non_image_files.append(
                (info.filename, info.file_size)
            )


# Print audit results.
print("=" * 65)
print("CHEST X-RAY ARCHIVE AUDIT")
print("=" * 65)

for key in sorted(raw_counts):

    print(
        f"{key[0]:8s} {key[1]:12s} "
        f"All files: {raw_counts[key]:5d} | "
        f"Images: {image_counts[key]:5d}"
    )

print("\n========== NON-IMAGE FILES ==========")

for filename, size in non_image_files:
    print(filename, "|", size, "bytes")

print("\n========== UNEXPECTED FILES ==========")

for filename, size in unexpected_files:
    print(filename, "|", size, "bytes")

print("\n========== TOTALS ==========")

print("Total class-directory files:", sum(raw_counts.values()))
print("Total recognized images:", sum(image_counts.values()))
print("Non-image files:", len(non_image_files))
print("Unexpected files:", len(unexpected_files))
