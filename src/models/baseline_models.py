
from pathlib import Path
from collections import Counter
import csv
import json
import warnings

import joblib
import numpy as np

from PIL import Image, ImageOps

from skimage.feature import hog

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)


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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

METRICS_CSV = OUTPUT_DIR / "baseline_metrics_v3.csv"
METRICS_JSON = OUTPUT_DIR / "baseline_results_v3.json"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"

HOG_MODEL_PATH = (
    CHECKPOINT_DIR / "hog_logistic_regression_v3.joblib"
)

LABEL_TO_ID = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}

ID_TO_LABEL = {
    0: "NORMAL",
    1: "PNEUMONIA",
}

EXPECTED_COUNTS = {
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

EXPECTED_PATIENTS = {
    "train": 2120,
    "validation": 530,
    "test": 431,
}

# HOG image preprocessing
HOG_IMAGE_SIZE = 128

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"
HOG_TRANSFORM_SQRT = True

# Number of HOG features:
# 15 x 15 blocks x 2 x 2 cells x 9 orientations = 8100
EXPECTED_HOG_FEATURES = 8100

# Logistic Regression configuration
LOGISTIC_C = 1.0
LOGISTIC_SOLVER = "liblinear"
LOGISTIC_MAX_ITER = 2000
RANDOM_STATE = 42

# Select threshold using validation only.
TARGET_SENSITIVITY = 0.85

# Feature extraction progress interval.
PROGRESS_INTERVAL = 500


# ============================================================
# 2. LOAD AND VALIDATE DATA SPLITS
# ============================================================

def load_split(split_name):
    """
    Load one CSV split and validate:
        - Image paths
        - Labels
        - Patient IDs
        - Original split provenance
        - SHA-256 uniqueness
        - Expected image counts
    """

    if split_name not in SPLIT_FILES:
        raise ValueError(f"Unknown split: {split_name}")

    csv_path = SPLIT_FILES[split_name]

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Split metadata not found: {csv_path}"
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
        raise ValueError(f"{split_name} CSV is empty.")

    required_columns = {
        "relative_path",
        "split",
        "source_split",
        "label",
        "patient_id",
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

    expected_total = sum(EXPECTED_COUNTS[split_name].values())

    if len(records) != expected_total:
        raise ValueError(
            f"{split_name}: expected {expected_total} records, "
            f"found {len(records)}."
        )

    paths = []
    labels = []

    class_counts = Counter()
    patient_labels = {}
    seen_paths = set()
    seen_hashes = set()

    expected_source = (
        "test" if split_name == "test" else "train"
    )

    dataset_root_resolved = DATASET_ROOT.resolve()

    for row in records:

        if row["split"] != split_name:
            raise ValueError(
                f"Incorrect split assignment: {row['relative_path']}"
            )

        if row["source_split"] != expected_source:
            raise ValueError(
                f"Incorrect source split: {row['relative_path']}"
            )

        label_name = row["label"]

        if label_name not in LABEL_TO_ID:
            raise ValueError(f"Unknown label: {label_name}")

        patient_id = row["patient_id"]

        if not patient_id:
            raise ValueError("Empty Patient ID detected.")

        # Each patient must have one class label.
        if patient_id in patient_labels:

            if patient_labels[patient_id] != label_name:
                raise ValueError(
                    f"Patient {patient_id} has multiple labels."
                )

        else:
            patient_labels[patient_id] = label_name

        # Verify image path uniqueness.
        relative_path = row["relative_path"]

        if relative_path in seen_paths:
            raise ValueError(
                f"Duplicate image path: {relative_path}"
            )

        seen_paths.add(relative_path)

        # Resolve image path.
        image_path = (
            PROJECT_ROOT / relative_path
        ).resolve()

        # Prevent paths from escaping dataset directory.
        try:
            image_path.relative_to(dataset_root_resolved)

        except ValueError:
            raise ValueError(
                f"Image outside dataset directory: {relative_path}"
            )

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        # Verify SHA-256 metadata.
        image_hash = row["sha256"]

        if not image_hash:
            raise ValueError(
                f"Missing SHA-256: {relative_path}"
            )

        if image_hash in seen_hashes:
            raise ValueError(
                f"Duplicate SHA-256 in {split_name}."
            )

        seen_hashes.add(image_hash)

        paths.append(str(image_path))
        labels.append(LABEL_TO_ID[label_name])

        class_counts[label_name] += 1

    if class_counts != Counter(EXPECTED_COUNTS[split_name]):
        raise ValueError(
            f"{split_name}: class distribution mismatch.\n"
            f"Expected: {EXPECTED_COUNTS[split_name]}\n"
            f"Actual: {dict(class_counts)}"
        )

    if len(patient_labels) != EXPECTED_PATIENTS[split_name]:
        raise ValueError(
            f"{split_name}: unexpected patient count."
        )

    return paths, np.asarray(labels, dtype=np.int32), records


def validate_split_separation(split_records):

    patient_sets = {}
    hash_sets = {}

    for split_name, records in split_records.items():

        patient_sets[split_name] = {
            row["patient_id"] for row in records
        }

        hash_sets[split_name] = {
            row["sha256"] for row in records
        }

    split_names = ("train", "validation", "test")

    for index, first in enumerate(split_names):

        for second in split_names[index + 1:]:

            if patient_sets[first] & patient_sets[second]:
                raise ValueError(
                    f"Patient overlap: {first}/{second}"
                )

            if hash_sets[first] & hash_sets[second]:
                raise ValueError(
                    f"SHA-256 overlap: {first}/{second}"
                )

    print("\n[DATA SEPARATION]")
    print("Patient overlap: 0")
    print("SHA-256 overlap: 0")


# ============================================================
# 3. HOG IMAGE PREPROCESSING
# ============================================================

def prepare_hog_image(image_path):
    """
    Convert image to grayscale and resize with padding.

    Preserve aspect ratio.
    Do not crop image regions.

    Output:
        Grayscale float32 image with shape (128, 128).
        Pixel range [0, 1].
    """

    with Image.open(image_path) as image:

        # Convert both grayscale and RGB images to grayscale.
        image = image.convert("L")

        # Resize while preserving aspect ratio.
        resized = ImageOps.contain(
            image,
            (HOG_IMAGE_SIZE, HOG_IMAGE_SIZE),
            method=Image.Resampling.BILINEAR,
        )

        # Create a black square canvas.
        canvas = Image.new(
            "L",
            (HOG_IMAGE_SIZE, HOG_IMAGE_SIZE),
            color=0,
        )

        # Center the resized image.
        left = (HOG_IMAGE_SIZE - resized.width) // 2
        top = (HOG_IMAGE_SIZE - resized.height) // 2

        canvas.paste(resized, (left, top))

        # Convert to float32 in [0, 1].
        image_array = (
            np.asarray(canvas, dtype=np.float32) / 255.0
        )

    return image_array


# ============================================================
# 4. HOG FEATURE EXTRACTION
# ============================================================

def extract_hog_features(paths, split_name):
    """
    Extract HOG features for one split.

    Feature extraction is deterministic and does not fit
    any parameters using validation or test data.
    """

    print("\n" + "-" * 65)
    print(f"EXTRACTING HOG FEATURES: {split_name.upper()}")
    print("-" * 65)

    features = np.empty(
        (len(paths), EXPECTED_HOG_FEATURES),
        dtype=np.float32
    )

    for index, image_path in enumerate(paths):

        try:
            image_array = prepare_hog_image(image_path)

            feature_vector = hog(
                image_array,
                orientations=HOG_ORIENTATIONS,
                pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK,
                block_norm=HOG_BLOCK_NORM,
                transform_sqrt=HOG_TRANSFORM_SQRT,
                feature_vector=True,
                channel_axis=None,
            )

            if feature_vector.shape != (
                EXPECTED_HOG_FEATURES,
            ):
                raise ValueError(
                    f"Unexpected HOG shape: {feature_vector.shape}"
                )

            if not np.isfinite(feature_vector).all():
                raise ValueError(
                    "Non-finite HOG features detected."
                )

            features[index] = feature_vector.astype(
                np.float32,
                copy=False
            )

        except Exception as error:
            raise RuntimeError(
                f"HOG extraction failed for image: {image_path}"
            ) from error

        if (
            (index + 1) % PROGRESS_INTERVAL == 0
            or index + 1 == len(paths)
        ):
            print(
                f"Processed {index + 1:,}/{len(paths):,} images"
            )

    print("HOG feature matrix shape:", features.shape)
    print("Feature dtype:", features.dtype)

    return features


# ============================================================
# 5. MAJORITY CLASS BASELINE
# ============================================================

def run_majority_baseline(y_train, y_validation, y_test):

    print("\n" + "=" * 70)
    print("BASELINE 1: MAJORITY CLASS CLASSIFIER")
    print("=" * 70)

    class_counts = Counter(y_train.tolist())

    majority_class = class_counts.most_common(1)[0][0]

    majority_label = ID_TO_LABEL[majority_class]

    print("Training class counts:", dict(class_counts))
    print("Predicted class:", majority_label)

    results = {}

    for split_name, y_true in (
        ("validation", y_validation),
        ("test", y_test),
    ):

        # A constant prediction for every sample.
        y_pred = np.full(
            len(y_true),
            majority_class,
            dtype=np.int32
        )

        # Constant scores provide no ranking information.
        # ROC-AUC is 0.5 for tied constant scores.
        y_score = np.full(
            len(y_true),
            float(majority_class),
            dtype=np.float64
        )

        metrics = calculate_metrics(
            y_true,
            y_score,
            y_pred,
            threshold=None,
        )

        results[split_name] = {
            "metrics": metrics,
            "y_true": y_true,
            "y_score": y_score,
            "y_pred": y_pred,
            "threshold": None,
        }

        print(f"\n{split_name.upper()}")
        print_metrics(metrics)

    return majority_class, results


# ============================================================
# 6. METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true,
    y_score,
    y_pred,
    threshold=None
):
    """
    Calculate classification metrics.

    Positive class:
        PNEUMONIA = 1
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0 else 0.0
    )

    auc = roc_auc_score(
        y_true,
        y_score
    )

    return {
        "threshold": (
            float(threshold)
            if threshold is not None else None
        ),

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "sensitivity": float(sensitivity),

        "specificity": float(specificity),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "auroc": float(auc),

        "confusion_matrix": cm.tolist(),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def print_metrics(metrics):

    print(
        f"AUROC:       {metrics['auroc']:.4f}"
    )

    print(
        f"Sensitivity: {metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity: {metrics['specificity']:.4f}"
    )

    print(
        f"Macro-F1:    {metrics['macro_f1']:.4f}"
    )

    print(
        f"Accuracy:    {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:   {metrics['precision']:.4f}"
    )

    print("Confusion matrix:")
    print(
        np.asarray(
            metrics["confusion_matrix"]
        )
    )


# ============================================================
# 7. VALIDATION-BASED THRESHOLD SELECTION
# ============================================================

def select_threshold(
    y_true,
    y_score,
    target_sensitivity=TARGET_SENSITIVITY
):
    """
    Select the threshold on validation data.

    Objective:
        Maximize specificity subject to:
        Sensitivity >= target_sensitivity.

    Test data must never be used here.
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score,
        drop_intermediate=False,
    )

    specificity = 1.0 - fpr

    valid_indices = np.flatnonzero(
        tpr >= target_sensitivity
    )

    if len(valid_indices) == 0:
        raise RuntimeError(
            "No threshold achieves the required sensitivity."
        )

    best_specificity = np.max(
        specificity[valid_indices]
    )

    # If multiple thresholds have the same specificity,
    # select the highest threshold deterministically.
    candidates = [
        index
        for index in valid_indices
        if np.isclose(
            specificity[index],
            best_specificity
        )
    ]

    best_index = max(
        candidates,
        key=lambda index: thresholds[index]
    )

    best_threshold = float(
        thresholds[best_index]
    )

    selected_sensitivity = float(
        tpr[best_index]
    )

    selected_specificity = float(
        specificity[best_index]
    )

    if selected_sensitivity < target_sensitivity:
        raise RuntimeError(
            "Selected threshold does not meet sensitivity target."
        )

    return {
        "threshold": best_threshold,
        "sensitivity": selected_sensitivity,
        "specificity": selected_specificity,
        "target_sensitivity": target_sensitivity,
    }


# ============================================================
# 8. LOGISTIC REGRESSION WITH HOG
# ============================================================

def run_hog_logistic_regression(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
):
    """
    Fit Logistic Regression on training HOG features.

    Select decision threshold on validation data.

    Evaluate test only after threshold has been locked.
    """

    print("\n" + "=" * 70)
    print("BASELINE 2: LOGISTIC REGRESSION WITH HOG")
    print("=" * 70)

    model = LogisticRegression(
        C=LOGISTIC_C,
        solver=LOGISTIC_SOLVER,
        max_iter=LOGISTIC_MAX_ITER,
        class_weight=None,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Fit on training data only.
    # --------------------------------------------------------

    print("\nFitting Logistic Regression on training data...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")
    print("Solver iterations:", model.n_iter_.tolist())

    if np.any(model.n_iter_ >= LOGISTIC_MAX_ITER):

        warnings.warn(
            "Logistic Regression may not have converged. "
            "Review n_iter_ and convergence warnings.",
            RuntimeWarning
        )

    # --------------------------------------------------------
    # Validation probabilities.
    # --------------------------------------------------------

    positive_index = list(model.classes_).index(1)

    validation_scores = model.predict_proba(
        X_validation
    )[:, positive_index]

    # --------------------------------------------------------
    # Choose threshold using validation only.
    # --------------------------------------------------------

    threshold_result = select_threshold(
        y_validation,
        validation_scores,
        TARGET_SENSITIVITY,
    )

    locked_threshold = threshold_result["threshold"]

    validation_predictions = (
        validation_scores >= locked_threshold
    ).astype(np.int32)

    validation_metrics = calculate_metrics(
        y_validation,
        validation_scores,
        validation_predictions,
        locked_threshold,
    )

    print("\n[VALIDATION THRESHOLD SELECTION]")

    print(
        "Target sensitivity:",
        threshold_result["target_sensitivity"]
    )

    print(
        "Selected threshold:",
        f"{locked_threshold:.6f}"
    )

    print(
        "Sensitivity:",
        f"{threshold_result['sensitivity']:.4f}"
    )

    print(
        "Specificity:",
        f"{threshold_result['specificity']:.4f}"
    )

    print("\n[VALIDATION METRICS]")
    print_metrics(validation_metrics)

    # --------------------------------------------------------
    # Test evaluation.
    #
    # Important:
    # Use the locked validation threshold.
    # Do not optimize it using test labels.
    # --------------------------------------------------------

    print("\n[TEST EVALUATION]")
    print("Using previously locked validation threshold.")

    test_scores = model.predict_proba(
        X_test
    )[:, positive_index]

    test_predictions = (
        test_scores >= locked_threshold
    ).astype(np.int32)

    test_metrics = calculate_metrics(
        y_test,
        test_scores,
        test_predictions,
        locked_threshold,
    )

    print_metrics(test_metrics)

    results = {
        "threshold_selection": threshold_result,

        "validation": {
            "metrics": validation_metrics,
            "y_true": y_validation,
            "y_score": validation_scores,
            "y_pred": validation_predictions,
            "threshold": locked_threshold,
        },

        "test": {
            "metrics": test_metrics,
            "y_true": y_test,
            "y_score": test_scores,
            "y_pred": test_predictions,
            "threshold": locked_threshold,
        },
    }

    return model, results


# ============================================================
# 9. SAVE ROC CURVE DATA
# ============================================================

def save_roc_curve(
    y_true,
    y_score,
    output_path
):
    """
    Save ROC coordinates to CSV for plotting and reporting.
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score,
        drop_intermediate=False,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "false_positive_rate",
            "true_positive_rate",
            "threshold",
        ])

        writer.writerows(
            zip(fpr, tpr, thresholds)
        )


# ============================================================
# 10. SAVE PREDICTIONS
# ============================================================

def save_predictions(
    prediction_records,
    output_path
):
    """
    Save predictions without Patient IDs or image paths.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "model",
        "split",
        "row_index",
        "y_true",
        "y_score",
        "threshold",
        "y_pred",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(prediction_records)


# ============================================================
# 11. SAVE METRICS
# ============================================================

def build_metric_row(model_name, split_name, metrics):

    return {
        "model": model_name,
        "split": split_name,
        "threshold": metrics["threshold"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "macro_f1": metrics["macro_f1"],
        "auroc": metrics["auroc"],
        "true_negative": metrics["true_negative"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
        "true_positive": metrics["true_positive"],
    }


def save_metrics_csv(rows, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(rows[0].keys())

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 12. MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("DL-05 BASELINE EXPERIMENTS")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directories.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load fixed data splits.
    # --------------------------------------------------------

    split_paths = {}
    split_labels = {}
    split_records = {}

    for split_name in ("train", "validation", "test"):

        paths, labels, records = load_split(split_name)

        split_paths[split_name] = paths
        split_labels[split_name] = labels
        split_records[split_name] = records

        counts = Counter(labels.tolist())

        print(f"\n{split_name.upper()}")
        print("Images:", len(paths))
        print("NORMAL:", counts[0])
        print("PNEUMONIA:", counts[1])

    # --------------------------------------------------------
    # Verify patient and hash separation.
    # --------------------------------------------------------

    validate_split_separation(split_records)

    # ========================================================
    # BASELINE 1: MAJORITY CLASS
    # ========================================================

    majority_class, majority_results = run_majority_baseline(
        split_labels["train"],
        split_labels["validation"],
        split_labels["test"],
    )

    # ========================================================
    # BASELINE 2: HOG FEATURE EXTRACTION
    # ========================================================

    features = {}

    for split_name in ("train", "validation", "test"):

        features[split_name] = extract_hog_features(
            split_paths[split_name],
            split_name,
        )

    # --------------------------------------------------------
    # Validate feature matrices.
    # --------------------------------------------------------

    for split_name, matrix in features.items():

        expected_shape = (
            EXPECTED_COUNTS[split_name]["NORMAL"]
            + EXPECTED_COUNTS[split_name]["PNEUMONIA"],
            EXPECTED_HOG_FEATURES,
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                f"{split_name}: unexpected HOG matrix shape."
            )

    # ========================================================
    # FIT AND EVALUATE LOGISTIC REGRESSION
    # ========================================================

    hog_model, hog_results = run_hog_logistic_regression(
        features["train"],
        split_labels["train"],
        features["validation"],
        split_labels["validation"],
        features["test"],
        split_labels["test"],
    )

    # Save trained model artifact.
    joblib.dump(
        hog_model,
        HOG_MODEL_PATH
    )

    # ========================================================
    # SAVE ROC CURVES
    # ========================================================

    print("\n[SAVING ROC CURVE DATA]")

    for split_name in ("validation", "test"):

        save_roc_curve(
            hog_results[split_name]["y_true"],
            hog_results[split_name]["y_score"],
            OUTPUT_DIR / f"hog_roc_{split_name}_v3.csv",
        )

        save_roc_curve(
            majority_results[split_name]["y_true"],
            majority_results[split_name]["y_score"],
            OUTPUT_DIR / f"majority_roc_{split_name}_v3.csv",
        )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_records = []

    for model_name, model_results in (
        ("Majority Class", majority_results),
        ("HOG Logistic Regression", hog_results),
    ):

        for split_name in ("validation", "test"):

            result = model_results[split_name]

            for index, (true_label, score, prediction) in enumerate(
                zip(
                    result["y_true"],
                    result["y_score"],
                    result["y_pred"],
                )
            ):

                prediction_records.append({
                    "model": model_name,
                    "split": split_name,
                    "row_index": index,
                    "y_true": int(true_label),
                    "y_score": float(score),
                    "threshold": (
                        result["threshold"]
                        if result["threshold"] is not None
                        else ""
                    ),
                    "y_pred": int(prediction),
                })

    save_predictions(
        prediction_records,
        PREDICTIONS_CSV
    )

    # ========================================================
    # SAVE METRICS CSV
    # ========================================================

    metric_rows = []

    for model_name, model_results in (
        ("Majority Class", majority_results),
        ("HOG Logistic Regression", hog_results),
    ):

        for split_name in ("validation", "test"):

            metric_rows.append(
                build_metric_row(
                    model_name,
                    split_name,
                    model_results[split_name]["metrics"],
                )
            )

    save_metrics_csv(
        metric_rows,
        METRICS_CSV
    )

    # ========================================================
    # SAVE JSON SUMMARY
    # ========================================================

    summary = {
        "project": "DL-05 Medical Image Classification",

        "dataset": "Chest X-Ray Pneumonia - Mendeley Data Version 3",

        "positive_class": "PNEUMONIA",

        "label_mapping": {
            "NORMAL": 0,
            "PNEUMONIA": 1,
        },

        "data_splits": {
            split_name: {
                "images": len(split_paths[split_name]),
                "patients": EXPECTED_PATIENTS[split_name],
                "class_counts": EXPECTED_COUNTS[split_name],
            }
            for split_name in ("train", "validation", "test")
        },

        "majority_class_baseline": {
            "predicted_class": ID_TO_LABEL[majority_class],
            "evaluation": {
                split_name: majority_results[split_name]["metrics"]
                for split_name in ("validation", "test")
            },
            "auroc_interpretation": (
                "0.5 is the theoretical ROC-AUC of a constant "
                "score with no ranking discrimination."
            ),
        },

        "hog_logistic_regression": {
            "feature_extraction": {
                "image_size": [128, 128],
                "color": "Grayscale",
                "resize_method": "Aspect-ratio preserving with black padding",
                "orientations": HOG_ORIENTATIONS,
                "pixels_per_cell": list(HOG_PIXELS_PER_CELL),
                "cells_per_block": list(HOG_CELLS_PER_BLOCK),
                "block_norm": HOG_BLOCK_NORM,
                "transform_sqrt": HOG_TRANSFORM_SQRT,
                "feature_dimension": EXPECTED_HOG_FEATURES,
            },

            "logistic_regression": {
                "C": LOGISTIC_C,
                "solver": LOGISTIC_SOLVER,
                "max_iter": LOGISTIC_MAX_ITER,
                "class_weight": None,
                "random_state": RANDOM_STATE,
            },

            "threshold_selection": hog_results["threshold_selection"],

            "evaluation": {
                split_name: hog_results[split_name]["metrics"]
                for split_name in ("validation", "test")
            },

            "test_threshold_policy": (
                "The threshold was selected using validation data "
                "only and then fixed for test evaluation."
            ),
        },

        "outputs": {
            "metrics_csv": str(
                METRICS_CSV.relative_to(PROJECT_ROOT)
            ),
            "metrics_json": str(
                METRICS_JSON.relative_to(PROJECT_ROOT)
            ),
            "predictions_csv": str(
                PREDICTIONS_CSV.relative_to(PROJECT_ROOT)
            ),
            "hog_model": str(
                HOG_MODEL_PATH.relative_to(PROJECT_ROOT)
            ),
        },
    }

    with open(
        METRICS_JSON,
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
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("BASELINE EXPERIMENTS COMPLETED")
    print("=" * 70)

    print("\n[MODEL 1]")
    print("Majority Class Classifier")
    print("Predicted class:", ID_TO_LABEL[majority_class])

    print("\n[MODEL 2]")
    print("Logistic Regression with HOG")
    print("HOG feature dimension:", EXPECTED_HOG_FEATURES)
    print(
        "Validation-selected threshold:",
        f"{hog_results['threshold_selection']['threshold']:.6f}"
    )

    print("\n[OUTPUT FILES]")

    print("Metrics CSV:", METRICS_CSV)
    print("Metrics JSON:", METRICS_JSON)
    print("Predictions CSV:", PREDICTIONS_CSV)
    print("HOG model:", HOG_MODEL_PATH)

    print("\n[FINAL RESULT]")
    print("PASS: Baseline experiments completed.")
    print("PASS: Logistic Regression fitted on training data only.")
    print("PASS: Threshold selected using validation only.")
    print("PASS: Test evaluated using locked threshold.")
    print("PASS: Original patient-level splits preserved.")

    print("=" * 70)


# ============================================================
# 13. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
