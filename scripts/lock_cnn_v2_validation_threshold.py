from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_NAME = "cnn_from_scratch_v2_025pct_full_20261120"
RUN_DIR = PROJECT_ROOT / "outputs" / "experiments" / RUN_NAME

INPUT_CSV = (
    RUN_DIR
    / "evaluation"
    / "validation_best"
    / "predictions.csv"
)

OUTPUT_DIR = (
    RUN_DIR
    / "evaluation"
    / "validation_threshold_locked"
)

MIN_SENSITIVITY = 0.85

EXPECTED_ROWS = 1042
EXPECTED_LABEL_COUNTS = {0: 270, 1: 772}

EXPECTED_BEST_EPOCH = 21
EXPECTED_CHECKPOINT_SHA256 = (
    "fbf1b49bcf55c5cb40ca3a80adbad04baec05465e1ae6ffd7d26adbe7d79321d"
)
EXPECTED_VALIDATION_AUROC = 0.9909398388025331

REQUIRED_COLUMNS = [
    "relative_path",
    "y_true",
    "y_score",
    "y_pred_at_0.5",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_metrics(
    labels: list[int],
    scores: list[float],
    threshold: float,
) -> dict:
    predictions = [
        int(score >= threshold)
        for score in scores
    ]

    tp = sum(
        y == 1 and pred == 1
        for y, pred in zip(labels, predictions)
    )
    tn = sum(
        y == 0 and pred == 0
        for y, pred in zip(labels, predictions)
    )
    fp = sum(
        y == 0 and pred == 1
        for y, pred in zip(labels, predictions)
    )
    fn = sum(
        y == 1 and pred == 0
        for y, pred in zip(labels, predictions)
    )

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)

    f1_positive = safe_divide(
        2 * precision * sensitivity,
        precision + sensitivity,
    )

    # F1 for class 0, treating NORMAL as the positive class.
    precision_negative = safe_divide(tn, tn + fn)
    recall_negative = specificity

    f1_negative = safe_divide(
        2 * precision_negative * recall_negative,
        precision_negative + recall_negative,
    )

    accuracy = safe_divide(tp + tn, len(labels))
    macro_f1 = (f1_positive + f1_negative) / 2

    return {
        "threshold": float(threshold),
        "comparison_operator": ">=",
        "num_images": len(labels),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_pneumonia": f1_positive,
        "f1_normal": f1_negative,
        "macro_f1": macro_f1,
    }


def main() -> None:
    print("=" * 75)
    print("CNN V2 — VALIDATION THRESHOLD LOCK")
    print("=" * 75)

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Validation predictions not found: {INPUT_CSV}"
        )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_DIR}. "
            "Refusing to overwrite existing evidence."
        )

    input_sha256 = sha256_file(INPUT_CSV)

    with INPUT_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError(
                f"Unexpected CSV columns: {reader.fieldnames}"
            )

        rows = list(reader)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, found {len(rows)}"
        )

    paths = [row["relative_path"] for row in rows]

    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate relative_path entries detected")

    labels = []
    scores = []

    for index, row in enumerate(rows, start=1):
        label = int(row["y_true"])
        score = float(row["y_score"])
        prediction_at_05 = int(row["y_pred_at_0.5"])

        if label not in (0, 1):
            raise ValueError(
                f"Invalid label at row {index}: {label}"
            )

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Invalid y_score at row {index}: {score}"
            )

        if prediction_at_05 != int(score >= 0.5):
            raise ValueError(
                f"y_pred_at_0.5 mismatch at row {index}"
            )

        labels.append(label)
        scores.append(score)

    label_counts = dict(Counter(labels))

    if label_counts != EXPECTED_LABEL_COUNTS:
        raise ValueError(
            f"Unexpected Validation class counts: {label_counts}"
        )

    # Each unique observed score represents a distinct prediction state
    # under the rule y_score >= threshold.
    candidate_thresholds = sorted(set(scores))

    feasible_results = []

    for threshold in candidate_thresholds:
        result = calculate_metrics(
            labels,
            scores,
            threshold,
        )

        if result["sensitivity"] >= MIN_SENSITIVITY:
            feasible_results.append(result)

    if not feasible_results:
        raise RuntimeError(
            "No threshold satisfies the minimum Sensitivity requirement."
        )

    # Primary objective: maximize Specificity.
    # Tie-break 1: maximize Sensitivity.
    # Tie-break 2: choose the higher threshold.
    best = max(
        feasible_results,
        key=lambda item: (
            item["specificity"],
            item["sensitivity"],
            item["threshold"],
        ),
    )

    locked_threshold = best["threshold"]

    locked_predictions = [
        int(score >= locked_threshold)
        for score in scores
    ]

    metrics = {
        "status": "validation_threshold_locked",
        "experiment_name": RUN_NAME,
        "split": "validation",
        "selection_policy": {
            "minimum_sensitivity": MIN_SENSITIVITY,
            "primary_objective": "maximize_specificity",
            "tie_breakers": [
                "maximize_sensitivity",
                "choose_higher_threshold",
            ],
            "prediction_rule": "y_score >= threshold",
            "candidate_thresholds": (
                "unique observed Validation prediction scores"
            ),
            "candidate_threshold_count": len(candidate_thresholds),
            "feasible_threshold_count": len(feasible_results),
        },
        "locked_validation_metrics": best,
        "validation_class_counts": {
            "NORMAL_0": label_counts[0],
            "PNEUMONIA_1": label_counts[1],
        },
        "source_predictions_csv": str(
            INPUT_CSV.relative_to(PROJECT_ROOT)
        ),
        "source_predictions_csv_sha256": input_sha256,
        "checkpoint_reference": {
            "best_epoch": EXPECTED_BEST_EPOCH,
            "sha256_from_read_only_verification": (
                EXPECTED_CHECKPOINT_SHA256
            ),
            "saved_validation_auroc_from_verification": (
                EXPECTED_VALIDATION_AUROC
            ),
        },
        "test_split_loaded": False,
        "test_split_evaluated": False,
        "test_used_for_threshold_selection": False,
        "notes": [
            "Threshold selected using Validation predictions only.",
            "Original predictions.csv was not modified.",
            "No model training or checkpoint modification was performed.",
            "Test split was not accessed by this script.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    output_predictions = (
        OUTPUT_DIR / "predictions_at_locked_threshold.csv"
    )

    with output_predictions.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = REQUIRED_COLUMNS + [
            "y_pred_at_locked_threshold"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()

        for row, prediction in zip(rows, locked_predictions):
            output_row = dict(row)
            output_row["y_pred_at_locked_threshold"] = prediction
            writer.writerow(output_row)

    metrics_path = OUTPUT_DIR / "threshold_selection.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 75)
    print("THRESHOLD SELECTION RESULT")
    print("=" * 75)
    print("Run:", RUN_NAME)
    print("Validation images:", len(labels))
    print("Validation class counts:", label_counts)
    print("Input CSV SHA256:", input_sha256)
    print("Candidate thresholds:", len(candidate_thresholds))
    print("Feasible thresholds:", len(feasible_results))
    print()
    print(f"LOCKED THRESHOLD: {locked_threshold:.17g}")
    print(f"Sensitivity:      {best['sensitivity']:.6f}")
    print(f"Specificity:      {best['specificity']:.6f}")
    print(f"Precision:        {best['precision']:.6f}")
    print(f"Accuracy:         {best['accuracy']:.6f}")
    print(f"Macro-F1:         {best['macro_f1']:.6f}")
    print("Confusion matrix:")
    print(best["confusion_matrix"])
    print()
    print("Threshold selection JSON:", metrics_path)
    print("Locked predictions CSV:", output_predictions)
    print()
    print("Test loaded/evaluated: NO")
    print("Original predictions.csv modified: NO")
    print("Checkpoint modified: NO")
    print("=" * 75)


if __name__ == "__main__":
    main()
