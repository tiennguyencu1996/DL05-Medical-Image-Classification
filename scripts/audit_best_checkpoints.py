from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
from pathlib import Path

ROOT = Path.cwd()
OUTPUT_DIR = ROOT / "outputs/reports/verified_results"

EXPECTED = {
    "cnn": {
        "total": 1_208_161,
        "trainable": 1_206_241,
        "non_trainable": 1_920,
    },
    "mobilenet": {
        "total": 2_259_265,
        "trainable": 1_512_001,
        "non_trainable": 747_264,
    },
}

PATTERNS = [
    "outputs/experiments/cnn_from_scratch_v2_*_full_*/training_summary.json",
    "outputs/experiments/mobilenetv2_*_imagenet_*/training_summary.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def count_parameters(weights) -> int:
    total = 0

    for weight in weights:
        shape = tuple(weight.shape)

        if any(dim is None for dim in shape):
            raise ValueError(
                f"Dynamic weight shape encountered: {shape}"
            )

        total += math.prod(int(dim) for dim in shape)

    return total


def save_results(records, tensorflow_version=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "checkpoint_audit.csv"
    json_path = OUTPUT_DIR / "checkpoint_audit_summary.json"

    fields = [
        "experiment",
        "model_type",
        "training_status",
        "best_epoch",
        "best_validation_auroc",
        "checkpoint",
        "file_size_mb",
        "expected_sha256",
        "actual_sha256",
        "hash_match",
        "load_status",
        "input_shape",
        "output_shape",
        "total_params",
        "expected_total_params",
        "trainable_params",
        "expected_trainable_params",
        "non_trainable_params",
        "expected_non_trainable_params",
        "parameter_check",
        "error",
    ]

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    verified = sum(
        r["hash_match"] is True
        and r["load_status"] == "PASS"
        and r["parameter_check"] is True
        for r in records
    )

    summary = {
        "project": "DL-05 Medical Image Classification",
        "checkpoint_count": len(records),
        "verified_count": verified,
        "expected_checkpoint_count": 8,
        "all_checkpoints_verified": (
            len(records) == 8 and verified == 8
        ),
        "tensorflow_version": tensorflow_version,
        "training_performed": False,
        "inference_performed": False,
        "notes": [
            "SHA-256 is compared against training_summary.json.",
            "Only hash-matched checkpoints are loaded.",
            "Keras models are loaded with compile=False.",
            "No training or prediction is performed.",
        ],
        "results": records,
    }

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n========== CHECKPOINT AUDIT ==========")
    print(f"Discovered: {len(records)}")
    print(f"Fully verified: {verified}/8")
    print(
        f"All verified: "
        f"{summary['all_checkpoints_verified']}"
    )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print("======================================\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    summaries = []

    # Discover the 4 CNN V2 and 4 MobileNetV2 experiments.
    for pattern in PATTERNS:
        summaries.extend(sorted(ROOT.glob(pattern)))

    summaries = sorted(set(summaries))

    if not summaries:
        raise SystemExit(
            "ERROR: No matching training_summary.json files found. "
            "Run this script from the project root."
        )

    # Phase 1: Verify checkpoint existence and SHA-256.
    for summary_path in summaries:
        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )

        experiment = summary.get(
            "experiment_name",
            summary_path.parent.name,
        )

        model_type = (
            "cnn"
            if experiment.startswith("cnn_from_scratch_v2_")
            else "mobilenet"
        )

        expected = EXPECTED[model_type]

        checkpoint_value = summary.get(
            "best_checkpoint_path"
        )

        expected_hash = summary.get(
            "best_checkpoint_sha256"
        )

        record = {
            "experiment": experiment,
            "model_type": model_type,
            "training_status": summary.get(
                "training_status"
            ),
            "best_epoch": summary.get(
                "best_epoch",
                summary.get("best_epoch_within_stage"),
            ),
            "best_validation_auroc": summary.get(
                "best_validation_sklearn_auroc"
            ),
            "checkpoint": checkpoint_value,
            "file_size_mb": None,
            "expected_sha256": expected_hash,
            "actual_sha256": None,
            "hash_match": False,
            "load_status": "NOT_ATTEMPTED",
            "input_shape": None,
            "output_shape": None,
            "total_params": None,
            "expected_total_params": expected["total"],
            "trainable_params": None,
            "expected_trainable_params": expected["trainable"],
            "non_trainable_params": None,
            "expected_non_trainable_params": expected[
                "non_trainable"
            ],
            "parameter_check": False,
            "error": None,
        }

        if not checkpoint_value or not expected_hash:
            record["load_status"] = "MISSING_HASH_OR_PATH"
            records.append(record)
            continue

        checkpoint = Path(checkpoint_value)

        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint

        if not checkpoint.is_file():
            record["load_status"] = "FILE_NOT_FOUND"
            records.append(record)
            continue

        record["checkpoint"] = str(
            checkpoint.relative_to(ROOT)
        )

        record["file_size_mb"] = round(
            checkpoint.stat().st_size / (1024 * 1024),
            3,
        )

        actual_hash = sha256_file(checkpoint)

        record["actual_sha256"] = actual_hash
        record["hash_match"] = (
            actual_hash.lower() == expected_hash.lower()
        )

        if not record["hash_match"]:
            record["load_status"] = "SKIPPED_HASH_MISMATCH"

        records.append(record)

    print("\nSHA-256 verification completed.")
    for r in records:
        print(
            f"{r['experiment']}: "
            f"{'MATCH' if r['hash_match'] else 'NOT MATCH'}"
        )

    # Phase 2: Load only checkpoints whose SHA-256 matches.
    try:
        import tensorflow as tf
    except ImportError as exc:
        for r in records:
            if r["hash_match"]:
                r["load_status"] = "TENSORFLOW_IMPORT_FAILED"
                r["error"] = str(exc)

        save_results(records)
        raise SystemExit(
            "ERROR: TensorFlow is unavailable. "
            "Activate the project's virtual environment."
        )

    # Hide GPU for this read-only checkpoint audit.
    try:
        tf.config.set_visible_devices([], "GPU")
    except (RuntimeError, ValueError) as exc:
        print(
            "WARNING: Could not change TensorFlow GPU visibility:",
            exc,
        )

    print(f"\nTensorFlow version: {tf.__version__}")

    for record in records:
        if not record["hash_match"]:
            continue

        checkpoint = ROOT / record["checkpoint"]

        print(f"\nLoading: {record['experiment']}")

        model = None

        try:
            model = tf.keras.models.load_model(
                str(checkpoint),
                compile=False,
            )

            record["input_shape"] = str(model.input_shape)
            record["output_shape"] = str(model.output_shape)

            record["total_params"] = int(
                model.count_params()
            )

            record["trainable_params"] = count_parameters(
                model.trainable_weights
            )

            record["non_trainable_params"] = count_parameters(
                model.non_trainable_weights
            )

            expected = EXPECTED[record["model_type"]]

            total_ok = (
                record["total_params"] == expected["total"]
            )

            trainable_ok = (
                record["trainable_params"]
                == expected["trainable"]
            )

            non_trainable_ok = (
                record["non_trainable_params"]
                == expected["non_trainable"]
            )

            record["parameter_check"] = (
                total_ok and trainable_ok and non_trainable_ok
            )

            record["load_status"] = "PASS"

            print(
                f"  Total params: "
                f"{record['total_params']:,}"
            )
            print(
                f"  Trainable: "
                f"{record['trainable_params']:,}"
            )
            print(
                f"  Non-trainable: "
                f"{record['non_trainable_params']:,}"
            )
            print(
                f"  Parameter check: "
                f"{record['parameter_check']}"
            )

        except Exception as exc:
            record["load_status"] = "LOAD_FAILED"
            record["error"] = repr(exc)
            print(f"  LOAD FAILED: {exc}")

        finally:
            if model is not None:
                del model

            tf.keras.backend.clear_session()
            gc.collect()

    save_results(
        records,
        tensorflow_version=tf.__version__,
    )


if __name__ == "__main__":
    main()
