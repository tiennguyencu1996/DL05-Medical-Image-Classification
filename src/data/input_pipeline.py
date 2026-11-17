
from pathlib import Path
from collections import Counter
import csv

import tensorflow as tf


# ============================================================
# 1. CONFIGURATION
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

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

SPLIT_FILES = {
    "train": SPLIT_DIR / "train_v3.csv",
    "validation": SPLIT_DIR / "validation_v3.csv",
    "test": SPLIT_DIR / "test_v3.csv",
}

IMAGE_SIZE = 224
BATCH_SIZE = 8
RANDOM_SEED = 42

SHUFFLE_BUFFER = 2048
NUM_PARALLEL_CALLS = 2
PREFETCH_BUFFER = 1

LABEL_TO_ID = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}

EXPECTED_CLASS_COUNTS = {
    "train": {
        "NORMAL": 1078,
        "PNEUMONIA": 3086,
    },
    "validation": {
        "NORMAL": 270,
        "PNEUMONIA": 772,
    },
    "test": {
        "NORMAL": 234,
        "PNEUMONIA": 387,
    },
}

EXPECTED_IMAGE_COUNTS = {
    "train": 4164,
    "validation": 1042,
    "test": 621,
}


# ============================================================
# 2. LOAD AND VALIDATE METADATA
# ============================================================

def load_split_metadata(split_name):
    """
    Load one split CSV and validate its records.

    Returns:
        paths: absolute image paths
        labels: integer labels
        records: original CSV records
    """

    if split_name not in SPLIT_FILES:
        raise ValueError(f"Unknown split: {split_name}")

    csv_path = SPLIT_FILES[split_name]

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Split CSV not found: {csv_path}"
        )

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)
        records = list(reader)

    if not records:
        raise ValueError(
            f"{split_name} metadata is empty."
        )

    required_columns = {
        "relative_path",
        "label",
        "patient_id",
        "split",
        "source_split",
        "sha256",
    }

    missing_columns = (
        required_columns - set(records[0].keys())
    )

    if missing_columns:
        raise ValueError(
            f"{split_name}: missing columns "
            f"{sorted(missing_columns)}"
        )

    if len(records) != EXPECTED_IMAGE_COUNTS[split_name]:
        raise ValueError(
            f"{split_name}: expected "
            f"{EXPECTED_IMAGE_COUNTS[split_name]} images, "
            f"found {len(records)}."
        )

    paths = []
    labels = []

    observed_class_counts = Counter()
    seen_paths = set()
    patient_labels = {}

    expected_source_split = (
        "test" if split_name == "test" else "train"
    )

    dataset_root_resolved = DATASET_ROOT.resolve()

    for row in records:

        # Check final split assignment.
        if row["split"] != split_name:
            raise ValueError(
                f"Incorrect split assignment: "
                f"{row['relative_path']}"
            )

        # Check original source split.
        if row["source_split"] != expected_source_split:
            raise ValueError(
                f"Incorrect source split: "
                f"{row['relative_path']}"
            )

        # Check label.
        label_name = row["label"]

        if label_name not in LABEL_TO_ID:
            raise ValueError(
                f"Unknown label: {label_name}"
            )

        # Check Patient ID.
        patient_id = row["patient_id"]

        if not patient_id:
            raise ValueError("Empty patient ID detected.")

        # Ensure one patient has only one label.
        if patient_id in patient_labels:

            if patient_labels[patient_id] != label_name:
                raise ValueError(
                    f"Patient {patient_id} has multiple labels."
                )

        else:
            patient_labels[patient_id] = label_name

        # Check duplicate relative paths.
        relative_path = row["relative_path"]

        if relative_path in seen_paths:
            raise ValueError(
                f"Duplicate image path: {relative_path}"
            )

        seen_paths.add(relative_path)

        # Resolve image path relative to project root.
        image_path = (
            PROJECT_ROOT / relative_path
        ).resolve()

        # Prevent paths from escaping dataset directory.
        try:
            image_path.relative_to(dataset_root_resolved)

        except ValueError:
            raise ValueError(
                f"Image path is outside dataset directory: "
                f"{relative_path}"
            )

        # Check image existence.
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        paths.append(str(image_path))
        labels.append(LABEL_TO_ID[label_name])

        observed_class_counts[label_name] += 1

    # Check class distribution against fixed split.
    expected_counts = EXPECTED_CLASS_COUNTS[split_name]

    if observed_class_counts != Counter(expected_counts):
        raise ValueError(
            f"{split_name}: class counts do not match.\n"
            f"Expected: {expected_counts}\n"
            f"Observed: {dict(observed_class_counts)}"
        )

    return paths, labels, records


# ============================================================
# 3. VERIFY PATIENT AND HASH SEPARATION
# ============================================================

def validate_all_splits(split_records):

    patient_sets = {}
    hash_sets = {}

    for split_name, records in split_records.items():

        # Collect Patient IDs.
        patient_sets[split_name] = {
            row["patient_id"] for row in records
        }

        # Collect SHA-256 hashes.
        hashes = [
            row["sha256"] for row in records
        ]

        # Check for duplicate hashes within a split.
        if len(hashes) != len(set(hashes)):
            raise ValueError(
                f"Duplicate SHA-256 found within {split_name}."
            )

        # Check for missing hashes.
        if any(not value for value in hashes):
            raise ValueError(
                f"Empty SHA-256 found within {split_name}."
            )

        hash_sets[split_name] = set(hashes)

    split_names = ("train", "validation", "test")

    for index, first in enumerate(split_names):

        for second in split_names[index + 1:]:

            patient_overlap = (
                patient_sets[first] & patient_sets[second]
            )

            hash_overlap = (
                hash_sets[first] & hash_sets[second]
            )

            if patient_overlap:
                raise ValueError(
                    f"Patient overlap: {first}/{second}"
                )

            if hash_overlap:
                raise ValueError(
                    f"SHA-256 overlap: {first}/{second}"
                )

    print("\n[DATA SEPARATION]")
    print("All patient and SHA-256 overlap checks passed.")


# ============================================================
# 4. IMAGE DECODING AND RESIZING
# ============================================================

def decode_and_resize(image_path, label):
    """
    Decode JPEG as RGB and resize with padding.

    Output:
        image: float32 [224, 224, 3], pixel range [0, 255]
        label: int32
    """

    # --------------------------------------------------------
    # Read image file.
    # --------------------------------------------------------

    image_content = tf.io.read_file(image_path)

    # --------------------------------------------------------
    # Decode JPEG into 3 channels.
    #
    # Grayscale images are converted to 3 channels.
    # Existing RGB images retain their RGB information.
    # --------------------------------------------------------

    image = tf.io.decode_jpeg(
        image_content,
        channels=3
    )

    # --------------------------------------------------------
    # Preserve original aspect ratio.
    #
    # resize_with_pad automatically adds padding.
    # Do not use the unsupported pad_to_aspect_ratio argument.
    # --------------------------------------------------------

    image = tf.image.resize_with_pad(
        image,
        target_height=IMAGE_SIZE,
        target_width=IMAGE_SIZE,
        method="bilinear",
        antialias=True,
    )

    # --------------------------------------------------------
    # Convert to float32.
    # --------------------------------------------------------

    image = tf.cast(image, tf.float32)

    # --------------------------------------------------------
    # Clip minor floating-point interpolation overshoots.
    #
    # Bilinear interpolation may produce values slightly
    # outside [0, 255], for example 255.00002.
    # Clipping guarantees the required pixel range.
    # --------------------------------------------------------

    image = tf.clip_by_value(
        image,
        clip_value_min=0.0,
        clip_value_max=255.0
    )

    # --------------------------------------------------------
    # Ensure a fixed output shape.
    # --------------------------------------------------------

    image = tf.ensure_shape(
        image,
        (IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    # Convert label to int32.
    label = tf.cast(label, tf.int32)

    return image, label


# ============================================================
# 5. BUILD TF.DATA DATASET
# ============================================================

def build_dataset(
    paths,
    labels,
    training=False,
    shuffle_seed=None,
):
    """
    Build a tf.data pipeline.

    Training:
        Shuffle -> Decode/Resize -> Batch -> Prefetch

    Validation/Test:
        Decode/Resize -> Batch -> Prefetch
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    # Default to the project seed for backward compatibility.
    if shuffle_seed is None:
        shuffle_seed = RANDOM_SEED

    # Shuffle only training data.
    if training:

        dataset = dataset.shuffle(
            buffer_size=min(
                len(paths),
                SHUFFLE_BUFFER
            ),
            seed=shuffle_seed,
            reshuffle_each_iteration=True,
        )

    # Decode and resize images.
    dataset = dataset.map(
        decode_and_resize,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        deterministic=True,
    )

    # Create batches.
    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    # Configure TensorFlow data pipeline.
    options = tf.data.Options()

    options.deterministic = True

    options.threading.private_threadpool_size = (
        NUM_PARALLEL_CALLS
    )

    options.threading.max_intra_op_parallelism = 1

    dataset = dataset.with_options(options)

    # Limit prefetching to reduce memory usage.
    dataset = dataset.prefetch(PREFETCH_BUFFER)

    return dataset


# ============================================================
# 6. BUILD ALL DATASETS
# ============================================================

def build_datasets(shuffle_seed=None):
    """Build train, validation, and test datasets.

    Args:
        shuffle_seed: Optional seed for the training split shuffle.
            Validation and test datasets are never shuffled.
            If omitted, RANDOM_SEED is used as before.
    """

    if shuffle_seed is None:
        shuffle_seed = RANDOM_SEED

    print("=" * 70)
    print("BUILDING CHEST X-RAY INPUT PIPELINE")
    print("=" * 70)

    split_paths = {}
    split_labels = {}
    split_records = {}

    # --------------------------------------------------------
    # Load all split metadata.
    # --------------------------------------------------------

    for split_name in ("train", "validation", "test"):

        paths, labels, records = load_split_metadata(
            split_name
        )

        split_paths[split_name] = paths
        split_labels[split_name] = labels
        split_records[split_name] = records

        counts = Counter(
            row["label"] for row in records
        )

        print(f"\n{split_name.upper()}")
        print(f"Images: {len(paths):,}")
        print(f"NORMAL: {counts['NORMAL']:,}")
        print(f"PNEUMONIA: {counts['PNEUMONIA']:,}")

        print(
            f"Patients: "
            f"{len({row['patient_id'] for row in records}):,}"
        )

    # --------------------------------------------------------
    # Verify patient and duplicate separation.
    # --------------------------------------------------------

    validate_all_splits(split_records)

    # --------------------------------------------------------
    # Build TensorFlow datasets.
    # --------------------------------------------------------

    datasets = {
        "train": build_dataset(
            split_paths["train"],
            split_labels["train"],
            training=True,
            shuffle_seed=shuffle_seed,
        ),

        "validation": build_dataset(
            split_paths["validation"],
            split_labels["validation"],
            training=False,
        ),

        "test": build_dataset(
            split_paths["test"],
            split_labels["test"],
            training=False,
        ),
    }

    return datasets


# ============================================================
# 7. NORMALIZATION HELPER FOR CNN FROM SCRATCH
# ============================================================

def normalize_for_scratch(images, labels):
    """
    Convert pixel range from [0, 255] to [0, 1].

    This function can be applied to batches used by
    the CNN-from-scratch model.

    Transfer-learning preprocessing will be selected
    separately according to the pretrained architecture.
    """

    images = images / 255.0

    return images, labels


# ============================================================
# 8. PIPELINE SMOKE TEST
# ============================================================

def main():

    print("=" * 70)
    print("INPUT PIPELINE SMOKE TEST")
    print("=" * 70)

    print("\nTensorFlow:", tf.__version__)

    print(
        "Detected GPU devices:",
        tf.config.list_physical_devices("GPU")
    )

    # Build all datasets.
    datasets = build_datasets()

    print("\n" + "=" * 70)
    print("CHECKING FIRST BATCH OF EACH SPLIT")
    print("=" * 70)

    for split_name, dataset in datasets.items():

        print(f"\n[{split_name.upper()}]")

        # Check dataset output specification.
        print("Element spec:", dataset.element_spec)

        # Inspect only the first batch.
        for images, labels in dataset.take(1):

            # Check finite image values.
            tf.debugging.assert_all_finite(
                images,
                "Non-finite image values detected."
            )

            # Check pixel range.
            tf.debugging.assert_greater_equal(
                tf.reduce_min(images),
                tf.constant(0.0, dtype=tf.float32)
            )

            tf.debugging.assert_less_equal(
                tf.reduce_max(images),
                tf.constant(255.0, dtype=tf.float32)
            )

            # Check label range.
            tf.debugging.assert_greater_equal(
                tf.reduce_min(labels),
                tf.constant(0, dtype=tf.int32)
            )

            tf.debugging.assert_less_equal(
                tf.reduce_max(labels),
                tf.constant(1, dtype=tf.int32)
            )

            print("Image batch shape:", images.shape)
            print("Label batch shape:", labels.shape)

            print("Image dtype:", images.dtype.name)
            print("Label dtype:", labels.dtype.name)

            print(
                "Pixel range:",
                float(tf.reduce_min(images).numpy()),
                "to",
                float(tf.reduce_max(images).numpy())
            )

            label_counts = Counter(
                labels.numpy().tolist()
            )

            print("Batch label counts:", dict(label_counts))

            # Expected batch shape.
            expected_shape = (
                BATCH_SIZE,
                IMAGE_SIZE,
                IMAGE_SIZE,
                3
            )

            if tuple(images.shape) != expected_shape:
                raise ValueError(
                    f"Unexpected image batch shape: {images.shape}"
                )

            if images.dtype != tf.float32:
                raise TypeError(
                    "Image dtype must be float32."
                )

            if labels.dtype != tf.int32:
                raise TypeError(
                    "Label dtype must be int32."
                )

    print("\n" + "=" * 70)
    print("PASS: INPUT PIPELINE SMOKE TEST COMPLETED")
    print("=" * 70)


# ============================================================
# 9. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
