
"""
Train CNN From Scratch V1 for Chest X-Ray Classification.

Dataset:
    Chest X-Ray Pneumonia (Kermany)

Labels:
    NORMAL = 0
    PNEUMONIA = 1

Important:
    - Train and validation are used during training.
    - Test data is NOT used for training.
    - Test data is NOT used for checkpoint selection.
    - Test data is NOT evaluated in this script.
"""

from pathlib import Path
import json

import tensorflow as tf

from src.data.input_pipeline import (
    build_datasets,
    normalize_for_scratch,
)

from src.models.cnn_from_scratch import (
    build_cnn_from_scratch,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "cnn_from_scratch_v1_best.keras"
)

HISTORY_PATH = (
    METRICS_DIR / "cnn_from_scratch_v1_history.csv"
)

SUMMARY_PATH = (
    METRICS_DIR / "cnn_from_scratch_v1_training_summary.json"
)

RANDOM_SEED = 42

BATCH_SIZE = 16
MAX_EPOCHS = 50

INITIAL_LEARNING_RATE = 1e-3

EARLY_STOPPING_PATIENCE = 8

REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

EXPECTED_INPUT_SHAPE = (None, 224, 224, 3)
EXPECTED_OUTPUT_SHAPE = (None, 1)
EXPECTED_TOTAL_PARAMS = 1_208_161


# ============================================================
# 2. PREPARE BATCHES
# ============================================================

def prepare_batch(images, labels):
    """
    Prepare a batch for CNN From Scratch V1.

    Images:
        float32, pixel range [0, 255]
        -> float32, pixel range [0, 1]

    Labels:
        int32, shape (batch_size,)
        -> float32, shape (batch_size, 1)
    """

    images, labels = normalize_for_scratch(
        images,
        labels,
    )

    labels = tf.cast(
        labels,
        tf.float32,
    )

    labels = tf.reshape(
        labels,
        (-1, 1),
    )

    return images, labels


# ============================================================
# 3. MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 75)
    print("CNN FROM SCRATCH V1 - TRAINING")
    print("=" * 75)

    # --------------------------------------------------------
    # 3.1 Set random seed
    # --------------------------------------------------------

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    print(f"\nTensorFlow version: {tf.__version__}")

    print(
        "Available GPU devices:",
        tf.config.list_physical_devices("GPU"),
    )

    # --------------------------------------------------------
    # 3.2 Create output directories
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 3.3 Build datasets
    # --------------------------------------------------------

    print("\n[STEP 1] Building datasets...")

    datasets = build_datasets()

    # Use only train and validation in this experiment.
    # The test dataset is intentionally not passed to model.fit.

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

    print("\nTrain and validation datasets are ready.")

    # --------------------------------------------------------
    # 3.4 Build CNN model
    # --------------------------------------------------------

    print("\n[STEP 2] Building CNN From Scratch V1...")

    model = build_cnn_from_scratch()

    if model.input_shape != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"Unexpected input shape: {model.input_shape}"
        )

    if model.output_shape != EXPECTED_OUTPUT_SHAPE:
        raise ValueError(
            f"Unexpected output shape: {model.output_shape}"
        )

    if model.count_params() != EXPECTED_TOTAL_PARAMS:
        raise ValueError(
            "Unexpected parameter count: "
            f"{model.count_params():,}"
        )

    print("\nArchitecture verification: PASS")

    print(f"Total parameters: {model.count_params():,}")

    model.summary()

    # --------------------------------------------------------
    # 3.5 Compile model
    # --------------------------------------------------------

    print("\n[STEP 3] Compiling model...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=INITIAL_LEARNING_RATE,
        ),

        loss=tf.keras.losses.BinaryCrossentropy(),

        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy",
                threshold=0.5,
            ),

            tf.keras.metrics.AUC(
                name="auroc",
                curve="ROC",
                num_thresholds=1000,
            ),

            tf.keras.metrics.Precision(
                name="precision",
                thresholds=0.5,
            ),

            tf.keras.metrics.Recall(
                name="recall",
                thresholds=0.5,
            ),
        ],
    )

    print("Model compilation: PASS")

    # --------------------------------------------------------
    # 3.6 Define callbacks
    # --------------------------------------------------------

    print("\n[STEP 4] Configuring training callbacks...")

    checkpoint_callback = (
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_PATH),
            monitor="val_auroc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
    )

    reduce_lr_callback = (
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        )
    )

    early_stopping_callback = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auroc",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
    )

    csv_logger_callback = (
        tf.keras.callbacks.CSVLogger(
            filename=str(HISTORY_PATH),
            append=False,
        )
    )

    terminate_on_nan_callback = (
        tf.keras.callbacks.TerminateOnNaN()
    )

    callbacks = [
        checkpoint_callback,
        reduce_lr_callback,
        early_stopping_callback,
        csv_logger_callback,
        terminate_on_nan_callback,
    ]

    print("Callbacks configured:")
    print("- ModelCheckpoint")
    print("- ReduceLROnPlateau")
    print("- EarlyStopping")
    print("- CSVLogger")
    print("- TerminateOnNaN")

    # --------------------------------------------------------
    # 3.7 Start training
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("STARTING MODEL TRAINING")
    print("=" * 75)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # --------------------------------------------------------
    # 3.8 Identify best epoch
    # --------------------------------------------------------

    validation_auroc = history.history.get(
        "val_auroc",
        [],
    )

    validation_loss = history.history.get(
        "val_loss",
        [],
    )

    if not validation_auroc:
        raise RuntimeError(
            "Validation AUROC history is empty."
        )

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            "Best model checkpoint was not created."
        )

    best_epoch_index = max(
        range(len(validation_auroc)),
        key=lambda index: validation_auroc[index],
    )

    best_epoch = best_epoch_index + 1

    best_val_auroc = float(
        validation_auroc[best_epoch_index]
    )

    best_val_loss = float(
        validation_loss[best_epoch_index]
    )

    epochs_completed = len(history.epoch)

    # --------------------------------------------------------
    # 3.9 Save training summary
    # --------------------------------------------------------

    summary = {
        "model_name": "CNNFromScratch_V1",

        "training_status": "completed",

        "random_seed": RANDOM_SEED,

        "input_shape": [224, 224, 3],

        "total_parameters": model.count_params(),

        "batch_size": BATCH_SIZE,

        "maximum_epochs": MAX_EPOCHS,

        "epochs_completed": epochs_completed,

        "optimizer": "Adam",

        "initial_learning_rate": INITIAL_LEARNING_RATE,

        "loss": "BinaryCrossentropy",

        "class_weighting": "none",

        "normalization": "pixel_values_divided_by_255",

        "checkpoint_monitor": "val_auroc",

        "checkpoint_mode": "max",

        "best_epoch": best_epoch,

        "best_validation_auroc": best_val_auroc,

        "validation_loss_at_best_auroc": best_val_loss,

        "early_stopping_patience": EARLY_STOPPING_PATIENCE,

        "test_split_used_for_training": False,

        "test_split_used_for_checkpoint_selection": False,

        "test_split_evaluated": False,

        "checkpoint_path": str(
            CHECKPOINT_PATH.relative_to(PROJECT_ROOT)
        ),

        "history_path": str(
            HISTORY_PATH.relative_to(PROJECT_ROOT)
        ),
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # 3.10 Final report
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("CNN FROM SCRATCH V1 - TRAINING COMPLETED")
    print("=" * 75)

    print(f"Epochs completed: {epochs_completed}")

    print(f"Best epoch: {best_epoch}")

    print(f"Best validation AUROC: {best_val_auroc:.4f}")

    print(f"Validation loss at best epoch: {best_val_loss:.4f}")

    print(f"\nBest model: {CHECKPOINT_PATH}")

    print(f"Training history: {HISTORY_PATH}")

    print(f"Training summary: {SUMMARY_PATH}")

    print("\nTEST SPLIT:")
    print("Not used for training.")
    print("Not used for checkpoint selection.")
    print("Not evaluated in this script.")

    print("\nTRAINING RESULT: COMPLETED")
    print("=" * 75)


# ============================================================
# 4. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
