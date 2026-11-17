"""
Train CNN From Scratch V1 for Chest X-Ray Classification.

Experiment:
    CNNFromScratch_V1, batch size 8, 224 x 224 RGB input.

Dataset:
    Chest X-Ray Pneumonia (Kermany)

Labels:
    NORMAL = 0
    PNEUMONIA = 1

This script runs one training epoch per invocation by default. The
BackupAndRestore callback preserves the latest completed training state,
including model/optimizer state and epoch progress, so the same command can
be run again after the Mac has cooled down.

The test split is never passed to model.fit, used for checkpoint selection,
or evaluated by this script.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorflow as tf

from src.data.input_pipeline import (
    BATCH_SIZE as PIPELINE_BATCH_SIZE,
    IMAGE_SIZE,
    RANDOM_SEED as PIPELINE_RANDOM_SEED,
    build_datasets,
    normalize_for_scratch,
)
from src.models.cnn_from_scratch import build_cnn_from_scratch


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Keep this name unchanged when resuming the same Batch-8 experiment.
# The Batch-16 pilot is stored in a different directory and is untouched.
EXPERIMENT_NAME = "cnn_from_scratch_v1_batch8_20261117"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "experiments" / EXPERIMENT_NAME

CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
TRAINING_STATE_DIR = EXPERIMENT_DIR / "training_state"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best.keras"
HISTORY_PATH = EXPERIMENT_DIR / "history.csv"
SUMMARY_PATH = EXPERIMENT_DIR / "training_summary.json"

RANDOM_SEED = PIPELINE_RANDOM_SEED
BATCH_SIZE = PIPELINE_BATCH_SIZE

MAX_EPOCHS = 50
SESSION_EPOCHS = 1  # One complete epoch per process; then let the Mac cool.

INITIAL_LEARNING_RATE = 1e-3

EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-4

REDUCE_LR_PATIENCE = 3
REDUCE_LR_MIN_DELTA = 1e-4
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

EXPECTED_INPUT_SHAPE = (None, IMAGE_SIZE, IMAGE_SIZE, 3)
EXPECTED_OUTPUT_SHAPE = (None, 1)
EXPECTED_TOTAL_PARAMS = 1_208_161
STATE_VERSION = 1

HISTORY_FIELDS = [
    "epoch",
    "accuracy",
    "auroc",
    "learning_rate",
    "loss",
    "precision",
    "recall",
    "val_accuracy",
    "val_auroc",
    "val_loss",
    "val_precision",
    "val_recall",
]

EXPERIMENT_CONFIG = {
    "experiment_name": EXPERIMENT_NAME,
    "model_name": "CNNFromScratch_V1",
    "input_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "epochs_per_session": SESSION_EPOCHS,
    "random_seed": RANDOM_SEED,
    "training_shuffle_seed_policy": "random_seed_plus_completed_epochs",
    "optimizer": "Adam",
    "initial_learning_rate": INITIAL_LEARNING_RATE,
    "loss": "BinaryCrossentropy",
    "class_weighting": "none",
    "normalization": "pixel_values_divided_by_255",
    "checkpoint_monitor": "val_auroc",
    "checkpoint_mode": "max",
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
    "reduce_lr_patience": REDUCE_LR_PATIENCE,
    "reduce_lr_factor": REDUCE_LR_FACTOR,
    "minimum_learning_rate": MIN_LEARNING_RATE,
    "test_split_used_for_training": False,
    "test_split_used_for_checkpoint_selection": False,
    "test_split_evaluated": False,
}


# ============================================================
# 2. GENERAL UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_float(value: Any) -> float:
    """Convert a scalar, including a Tensor, into a Python float."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    return float(value)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically to reduce the risk of partial files."""
    temporary_path = path.with_name(path.name + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)
        file.write("\n")
    os.replace(temporary_path, path)


def write_history_csv(path: Path, history_rows: list[dict[str, Any]]) -> None:
    """Regenerate CSV from the canonical JSON state after each epoch."""
    temporary_path = path.with_name(path.name + ".tmp")
    with open(temporary_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=HISTORY_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(history_rows)
    os.replace(temporary_path, path)


def directory_has_files(path: Path) -> bool:
    return path.exists() and any(item.is_file() for item in path.rglob("*"))


def make_initial_state() -> dict[str, Any]:
    return {
        "state_version": STATE_VERSION,
        "experiment_config": EXPERIMENT_CONFIG,
        "training_status": "ready",
        "completed_epochs": 0,
        "session_count": 0,
        "session_started_at": None,
        "session_finished_at": None,
        "stop_reason": None,
        "best_validation_auroc": None,
        "best_epoch": None,
        "validation_loss_at_best_auroc": None,
        "early_stopping_best_auroc": None,
        "early_stopping_wait": 0,
        "lr_best_validation_loss": None,
        "lr_wait": 0,
        "lr_reductions": 0,
        "current_learning_rate": INITIAL_LEARNING_RATE,
        "last_epoch_metrics": None,
        "history": [],
        "test_split_used_for_training": False,
        "test_split_used_for_checkpoint_selection": False,
        "test_split_evaluated": False,
        "best_model_path": str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "training_state_path": str(TRAINING_STATE_DIR.relative_to(PROJECT_ROOT)),
        "history_path": str(HISTORY_PATH.relative_to(PROJECT_ROOT)),
    }


def load_or_initialize_state() -> dict[str, Any]:
    """Load this experiment's state, or create a genuinely fresh run."""
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_STATE_DIR.mkdir(parents=True, exist_ok=True)

    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r", encoding="utf-8") as file:
            state = json.load(file)

        if state.get("state_version") != STATE_VERSION:
            raise RuntimeError(
                "Unsupported training-state version. Do not delete the experiment; "
                "inspect the saved state before proceeding."
            )

        if state.get("experiment_config") != EXPERIMENT_CONFIG:
            raise RuntimeError(
                "The saved experiment configuration does not match this script. "
                "Do not resume with changed batch size, model, optimizer, or fit settings. "
                "Use a new experiment directory for a different configuration."
            )

        if len(state.get("history", [])) != int(state.get("completed_epochs", 0)):
            raise RuntimeError(
                "Training summary and epoch history are inconsistent. "
                "Stop here and inspect the experiment files; do not overwrite them."
            )

        completed_epochs = int(state.get("completed_epochs", 0))
        if completed_epochs > 0 and not directory_has_files(TRAINING_STATE_DIR):
            raise RuntimeError(
                "The summary records completed epochs, but BackupAndRestore files "
                "are missing. Refusing to start from random weights."
            )

        if state.get("best_epoch") is not None and not BEST_MODEL_PATH.is_file():
            raise RuntimeError(
                "The summary records a best epoch, but best.keras is missing. "
                "Do not resume until the checkpoint is recovered."
            )

        # Rebuild CSV from the canonical JSON in case a prior shutdown occurred
        # after the JSON was replaced but before the CSV was updated.
        write_history_csv(HISTORY_PATH, state.get("history", []))
        return state

    # A folder containing any prior artifacts but no summary is ambiguous.
    if (
        HISTORY_PATH.exists()
        or BEST_MODEL_PATH.exists()
        or directory_has_files(TRAINING_STATE_DIR)
    ):
        raise RuntimeError(
            "Experiment artifacts exist but training_summary.json is missing. "
            "Refusing to overwrite or silently start a new run."
        )

    state = make_initial_state()
    atomic_write_json(SUMMARY_PATH, state)
    write_history_csv(HISTORY_PATH, state["history"])
    return state


def configure_tensorflow_runtime() -> None:
    """Use conservative CPU thread limits for this 8-GB fanless laptop."""
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        print("TensorFlow CPU thread limits: intra_op=1, inter_op=2")
    except RuntimeError as error:
        # This can happen only if a TensorFlow runtime was initialized earlier.
        # Continue without changing runtime thread settings.
        print(f"Runtime thread limits unchanged: {error}")


def set_optimizer_learning_rate(optimizer: tf.keras.optimizers.Optimizer, value: float) -> None:
    learning_rate = optimizer.learning_rate
    if hasattr(learning_rate, "assign"):
        learning_rate.assign(value)
    else:
        raise TypeError(
            "This training script expects an assignable scalar optimizer learning rate."
        )


def get_optimizer_learning_rate(optimizer: tf.keras.optimizers.Optimizer) -> float:
    return to_float(optimizer.learning_rate)


def save_best_model_atomically(model: tf.keras.Model) -> None:
    """Save the best complete Keras model without replacing it mid-write."""
    temporary_path = BEST_MODEL_PATH.with_name("best.tmp.keras")
    if temporary_path.exists():
        temporary_path.unlink()
    model.save(temporary_path, overwrite=True)
    os.replace(temporary_path, BEST_MODEL_PATH)


# ============================================================
# 3. DATA BATCH PREPARATION
# ============================================================

def prepare_batch(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Normalize images to [0, 1] and reshape labels to (batch, 1)."""
    images, labels = normalize_for_scratch(images, labels)
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, (-1, 1))
    return images, labels


# ============================================================
# 4. PERSISTENT CALLBACKS
# ============================================================

class PersistentTrainingController(tf.keras.callbacks.Callback):
    """Persist best-model selection, LR scheduling, early stopping, and history.

    The callback state is kept in training_summary.json so that patience
    counters continue across one-epoch process sessions.
    """

    def __init__(self, state: dict[str, Any]):
        super().__init__()
        self.state = state
        self.session_start_epoch = int(state["completed_epochs"])

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        completed_epoch = int(epoch) + 1
        expected_epoch = int(self.state["completed_epochs"]) + 1

        if completed_epoch != expected_epoch:
            raise RuntimeError(
                f"Epoch sequence mismatch: Keras completed {completed_epoch}, "
                f"but experiment state expected {expected_epoch}."
            )

        required_metrics = (
            "accuracy", "auroc", "loss", "precision", "recall",
            "val_accuracy", "val_auroc", "val_loss", "val_precision", "val_recall",
        )
        missing = [name for name in required_metrics if name not in logs]
        if missing:
            raise RuntimeError(f"Missing required training metrics: {missing}")

        metrics = {name: to_float(logs[name]) for name in required_metrics}
        non_finite_metrics = [
            name for name, value in metrics.items() if not math.isfinite(value)
        ]
        if non_finite_metrics:
            raise FloatingPointError(
                "Non-finite epoch metrics detected: "
                f"{non_finite_metrics}. The previous completed-epoch backup "
                "will remain the recovery point."
            )

        learning_rate_used = get_optimizer_learning_rate(self.model.optimizer)
        val_auroc = metrics["val_auroc"]
        val_loss = metrics["val_loss"]

        # Save a separate best model using validation AUROC only.
        previous_best_auroc = self.state["best_validation_auroc"]
        if previous_best_auroc is None or val_auroc > float(previous_best_auroc):
            save_best_model_atomically(self.model)
            self.state["best_validation_auroc"] = val_auroc
            self.state["best_epoch"] = completed_epoch
            self.state["validation_loss_at_best_auroc"] = val_loss
            print(
                f"\nBest validation AUROC updated: {val_auroc:.6f} "
                f"(epoch {completed_epoch}); saved {BEST_MODEL_PATH}"
            )

        # Persistent EarlyStopping logic (monitoring validation AUROC).
        early_best = self.state["early_stopping_best_auroc"]
        if early_best is None or val_auroc > float(early_best) + EARLY_STOPPING_MIN_DELTA:
            self.state["early_stopping_best_auroc"] = val_auroc
            self.state["early_stopping_wait"] = 0
        else:
            self.state["early_stopping_wait"] = int(
                self.state["early_stopping_wait"]
            ) + 1

        # Persistent ReduceLROnPlateau logic (monitoring validation loss).
        lr_best_loss = self.state["lr_best_validation_loss"]
        if lr_best_loss is None or val_loss < float(lr_best_loss) - REDUCE_LR_MIN_DELTA:
            self.state["lr_best_validation_loss"] = val_loss
            self.state["lr_wait"] = 0
        else:
            self.state["lr_wait"] = int(self.state["lr_wait"]) + 1

        if int(self.state["lr_wait"]) >= REDUCE_LR_PATIENCE:
            old_lr = get_optimizer_learning_rate(self.model.optimizer)
            new_lr = max(old_lr * REDUCE_LR_FACTOR, MIN_LEARNING_RATE)
            if new_lr < old_lr - 1e-12:
                set_optimizer_learning_rate(self.model.optimizer, new_lr)
                self.state["lr_reductions"] = int(self.state["lr_reductions"]) + 1
                print(
                    f"\nReducing learning rate: {old_lr:.8g} -> {new_lr:.8g}"
                )
            self.state["lr_wait"] = 0

        current_learning_rate = get_optimizer_learning_rate(self.model.optimizer)

        row = {
            "epoch": completed_epoch,
            "accuracy": metrics["accuracy"],
            "auroc": metrics["auroc"],
            "learning_rate": learning_rate_used,
            "loss": metrics["loss"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "val_accuracy": metrics["val_accuracy"],
            "val_auroc": metrics["val_auroc"],
            "val_loss": metrics["val_loss"],
            "val_precision": metrics["val_precision"],
            "val_recall": metrics["val_recall"],
        }

        self.state["history"].append(row)
        self.state["completed_epochs"] = completed_epoch
        self.state["current_learning_rate"] = current_learning_rate
        self.state["last_epoch_metrics"] = row
        self.state["updated_at"] = utc_now()

        early_stop_triggered = (
            int(self.state["early_stopping_wait"]) >= EARLY_STOPPING_PATIENCE
        )

        if early_stop_triggered:
            self.state["training_status"] = "early_stopped"
            self.state["stop_reason"] = (
                "Validation AUROC did not improve by the configured min_delta "
                f"for {EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
        elif completed_epoch >= MAX_EPOCHS:
            self.state["training_status"] = "completed"
            self.state["stop_reason"] = "Maximum epoch count reached."
        elif completed_epoch - self.session_start_epoch >= SESSION_EPOCHS:
            self.state["training_status"] = "session_complete"
            self.state["stop_reason"] = (
                f"Session limit reached: {SESSION_EPOCHS} completed epoch(s)."
            )
        else:
            self.state["training_status"] = "running"
            self.state["stop_reason"] = None

        # JSON is the canonical record; CSV is rebuilt from the same history.
        atomic_write_json(SUMMARY_PATH, self.state)
        write_history_csv(HISTORY_PATH, self.state["history"])

        if self.state["training_status"] in {
            "early_stopped", "completed", "session_complete"
        }:
            self.model.stop_training = True

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        self.state["session_finished_at"] = utc_now()
        self.state["current_learning_rate"] = get_optimizer_learning_rate(
            self.model.optimizer
        )
        atomic_write_json(SUMMARY_PATH, self.state)


class FiniteLossGuard(tf.keras.callbacks.Callback):
    """Abort immediately if a training batch produces a non-finite loss."""

    def on_train_batch_end(
        self,
        batch: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        logs = logs or {}
        if "loss" in logs and not math.isfinite(to_float(logs["loss"])):
            raise FloatingPointError(
                f"Non-finite training loss detected at batch {batch}. "
                "The incomplete epoch will not replace the last completed backup."
            )


class ResumeStateValidator(tf.keras.callbacks.Callback):
    """Check BackupAndRestore recovered exactly the epoch recorded in JSON."""

    def __init__(self, state: dict[str, Any]):
        super().__init__()
        self.state = state

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        restored_epoch = getattr(self.model, "_initial_epoch", 0)
        if restored_epoch is None:
            restored_epoch = 0
        restored_epoch = int(restored_epoch)
        expected_epoch = int(self.state["completed_epochs"])

        if restored_epoch != expected_epoch:
            raise RuntimeError(
                "BackupAndRestore and training_summary.json disagree: "
                f"restored epoch={restored_epoch}, summary epoch={expected_epoch}. "
                "Training has been stopped to avoid silently restarting or skipping epochs."
            )

        restored_lr = get_optimizer_learning_rate(self.model.optimizer)
        expected_lr = float(self.state["current_learning_rate"])
        if not math.isclose(restored_lr, expected_lr, rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(
                "Restored optimizer learning rate does not match the experiment state: "
                f"restored={restored_lr}, expected={expected_lr}."
            )

        print(f"Resume verification: PASS (starting at epoch {restored_epoch + 1})")


# ============================================================
# 5. MODEL AND DATASET VALIDATION
# ============================================================

def validate_model(model: tf.keras.Model) -> None:
    if model.input_shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(f"Unexpected input shape: {model.input_shape}")
    if model.output_shape != EXPECTED_OUTPUT_SHAPE:
        raise ValueError(f"Unexpected output shape: {model.output_shape}")
    if model.count_params() != EXPECTED_TOTAL_PARAMS:
        raise ValueError(
            f"Unexpected parameter count: {model.count_params():,}; "
            f"expected {EXPECTED_TOTAL_PARAMS:,}."
        )


def validate_experiment_artifacts(state: dict[str, Any]) -> None:
    completed_epochs = int(state["completed_epochs"])
    if completed_epochs > 0 and not directory_has_files(TRAINING_STATE_DIR):
        raise RuntimeError(
            "No BackupAndRestore files were found although completed_epochs > 0."
        )
    if state["best_epoch"] is not None and not BEST_MODEL_PATH.is_file():
        raise RuntimeError("Best checkpoint is missing; refusing to continue.")


# ============================================================
# 6. MAIN TRAINING FUNCTION
# ============================================================

def main() -> None:
    print("=" * 75)
    print("CNN FROM SCRATCH V1 - BATCH 8 TRAINING SESSION")
    print("=" * 75)

    configure_tensorflow_runtime()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Input resolution: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Epochs per invocation: {SESSION_EPOCHS}")
    print(f"Maximum total epochs: {MAX_EPOCHS}")
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print(f"Experiment directory: {EXPERIMENT_DIR}")

    state = load_or_initialize_state()
    validate_experiment_artifacts(state)

    if state["training_status"] in {"completed", "early_stopped"}:
        print(
            f"\nThis experiment is already {state['training_status']}. "
            "No further training was started."
        )
        print(f"Best validation AUROC: {state['best_validation_auroc']}")
        print(f"Best epoch: {state['best_epoch']}")
        return

    # Create a different deterministic shuffle order for each completed epoch.
    # If an epoch is interrupted, the same seed is reused when that epoch is retried.
    shuffle_seed = RANDOM_SEED + int(state["completed_epochs"])
    print(f"Training shuffle seed for this session: {shuffle_seed}")

    print("\n[STEP 1] Building datasets...")
    datasets = build_datasets(shuffle_seed=shuffle_seed)

    train_dataset = datasets["train"].map(
        prepare_batch,
        num_parallel_calls=1,
        deterministic=True,
    )
    validation_dataset = datasets["validation"].map(
        prepare_batch,
        num_parallel_calls=1,
        deterministic=True,
    )

    # The test dataset is deliberately not mapped, iterated, or passed to fit.
    del datasets["test"]

    print("Train and validation datasets are ready.")

    print("\n[STEP 2] Building CNN From Scratch V1...")
    model = build_cnn_from_scratch()
    validate_model(model)
    print("Architecture verification: PASS")
    print(f"Total parameters: {model.count_params():,}")
    model.summary()

    print("\n[STEP 3] Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=INITIAL_LEARNING_RATE,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.AUC(
                name="auroc",
                curve="ROC",
                num_thresholds=1000,
            ),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        ],
    )
    print("Model compilation: PASS")

    print("\n[STEP 4] Configuring resumable callbacks...")
    controller_callback = PersistentTrainingController(state)
    backup_callback = tf.keras.callbacks.BackupAndRestore(
        backup_dir=str(TRAINING_STATE_DIR),
        save_freq="epoch",
        double_checkpoint=True,
        delete_checkpoint=False,
    )
    resume_validator = ResumeStateValidator(state)
    finite_loss_guard = FiniteLossGuard()

    # Order matters:
    # - controller persists epoch metrics and the selected best model;
    # - BackupAndRestore then saves the latest model/optimizer/epoch state;
    # - validator checks the restored epoch and LR at the next invocation.
    callbacks = [
        controller_callback,
        finite_loss_guard,
        backup_callback,
        resume_validator,
    ]

    state["session_count"] = int(state["session_count"]) + 1
    state["session_started_at"] = utc_now()
    state["session_finished_at"] = None
    state["training_status"] = "running"
    state["stop_reason"] = None
    atomic_write_json(SUMMARY_PATH, state)

    print("Callbacks configured:")
    print("- Persistent best-model selection (validation AUROC)")
    print("- BackupAndRestore (model + optimizer + completed epoch)")
    print("- Persistent validation-loss learning-rate reduction")
    print("- Persistent validation-AUROC early stopping")
    print("- One-epoch session limit")
    print("- FiniteLossGuard")

    print("\n" + "=" * 75)
    print("STARTING ONE-EPOCH TRAINING SESSION")
    print("=" * 75)

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Callback state is canonical; do not infer global epoch count from the
    # session-local History object returned by model.fit().
    with open(SUMMARY_PATH, "r", encoding="utf-8") as file:
        final_state = json.load(file)

    print("\n" + "=" * 75)
    print("CNN FROM SCRATCH V1 - SESSION RESULT")
    print("=" * 75)
    print(f"Training status: {final_state['training_status']}")
    print(f"Completed epochs: {final_state['completed_epochs']} / {MAX_EPOCHS}")
    print(f"Sessions started: {final_state['session_count']}")
    print(f"Best validation AUROC: {final_state['best_validation_auroc']}")
    print(f"Best epoch: {final_state['best_epoch']}")
    print(f"Current learning rate: {final_state['current_learning_rate']}")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Latest training state: {TRAINING_STATE_DIR}")
    print(f"History: {HISTORY_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("Test split used/evaluated: NO")

    if final_state["training_status"] == "session_complete":
        print("\nSESSION COMPLETE.")
        print("Let the Mac cool down, then run the SAME command again to resume.")
    elif final_state["training_status"] == "completed":
        print("\nTRAINING COMPLETED: maximum epoch count reached.")
    elif final_state["training_status"] == "early_stopped":
        print("\nTRAINING EARLY-STOPPED: validation AUROC plateaued.")
    else:
        print("\nTraining ended without a session-complete status; inspect the log.")

    print("=" * 75)


if __name__ == "__main__":
    main()
