"""Read-only verification of the best CNN V2 checkpoint vs saved Validation CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from src.data.input_pipeline import (
    IMAGE_SIZE,
    build_dataset,
    load_split_metadata,
    normalize_for_scratch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "outputs" / "experiments" / args.run_name
    summary_path = run_dir / "training_summary.json"
    checkpoint_path = run_dir / "checkpoints" / "best.keras"
    prediction_path = run_dir / "evaluation" / "validation_best" / "predictions.csv"

    for path in (summary_path, checkpoint_path, prediction_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test_split_loaded") or summary.get("test_split_evaluated"):
        raise RuntimeError("Run summary unexpectedly indicates Test usage")
    if summary["experiment_config"].get("runtime_disable_meta_optimizer") is not True:
        raise RuntimeError("Run was not configured with Meta Optimizer disabled")

    options = tf.config.optimizer.get_experimental_options()
    options["disable_meta_optimizer"] = True
    tf.config.optimizer.set_experimental_options(options)

    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != summary.get("best_checkpoint_sha256"):
        raise RuntimeError("Best checkpoint SHA256 differs from training summary")

    rows: list[dict[str, str]] = []
    with prediction_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    val_paths, val_labels, val_records = load_split_metadata("validation")
    expected_paths = [row["relative_path"] for row in val_records]
    saved_paths = [row["relative_path"] for row in rows]
    if saved_paths != expected_paths:
        raise RuntimeError("Saved Validation prediction path order mismatch")

    saved_labels = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int32)
    saved_scores = np.asarray([float(row["y_score"]) for row in rows], dtype=np.float32)
    y_true = np.asarray(val_labels, dtype=np.int32)
    if not np.array_equal(saved_labels, y_true):
        raise RuntimeError("Saved Validation labels differ from metadata")

    def prepare_batch(images, labels):
        images, labels = normalize_for_scratch(images, labels)
        labels = tf.cast(tf.reshape(labels, (-1, 1)), tf.float32)
        tf.debugging.assert_all_finite(images, "Non-finite image")
        tf.debugging.assert_greater_equal(tf.reduce_min(images), 0.0)
        tf.debugging.assert_less_equal(tf.reduce_max(images), 1.0 + 1e-6)
        return images, labels

    dataset = build_dataset(val_paths, val_labels, training=False).map(
        prepare_batch, num_parallel_calls=1, deterministic=True
    )
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    reloaded_scores = model.predict(dataset, verbose=0).reshape(-1).astype(np.float32)

    if len(reloaded_scores) != len(saved_scores):
        raise RuntimeError("Reloaded prediction count mismatch")
    max_diff = float(np.max(np.abs(reloaded_scores - saved_scores)))
    mean_diff = float(np.mean(np.abs(reloaded_scores - saved_scores)))
    allclose = bool(np.allclose(reloaded_scores, saved_scores, atol=1e-6, rtol=0))

    print("=" * 72)
    print("CNN V2 BEST CHECKPOINT — READ-ONLY VALIDATION VERIFICATION")
    print("=" * 72)
    print("Run:", args.run_name)
    print("Level:", summary["experiment_config"]["label_efficiency_level"])
    print("Best epoch:", summary["best_epoch"])
    print("Checkpoint SHA256:", checkpoint_hash)
    print("Validation images:", len(y_true))
    print("Saved Validation AUROC:", f"{roc_auc_score(y_true, saved_scores):.10f}")
    print("Reloaded Validation AUROC:", f"{roc_auc_score(y_true, reloaded_scores):.10f}")
    print("Maximum score difference:", max_diff)
    print("Mean score difference:", mean_diff)
    print("Allclose (atol=1e-6):", allclose)
    print("Test loaded/evaluated: NO")
    print("No files were modified.")
    print("=" * 72)
    if not allclose:
        raise RuntimeError("Checkpoint predictions do not reproduce saved Validation scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
