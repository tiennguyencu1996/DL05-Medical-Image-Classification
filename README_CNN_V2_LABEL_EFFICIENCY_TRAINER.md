# DL-05 CNN V2 — Shared Label Efficiency Trainer

This document describes the resumable CNN V2 trainer and the formal Label Efficiency runs recorded for the four **frozen, patient-level nested Train manifests** (`010pct`, `025pct`, `050pct`, `100pct`). The same trainer source and protocol were used at every level. The four formal runs have been completed and their saved training summaries and Validation predictions are retained as experiment evidence. This formal series is separate from the historical one-epoch pilot; the fixed original splits and frozen manifests remain unchanged.

## Included files

- `src/training/train_cnn_v2_label_efficiency.py`: shared trainer, one epoch per process invocation, per-level isolated output directory.
- `scripts/verify_cnn_v2_experiment.py`: read-only check that the saved best checkpoint reproduces its saved Validation predictions.

## What is fixed across all four runs

- Architecture: `CNNFromScratch_V2_Controlled`, same verified V1 topology and 1,208,161 parameters.
- Image size: 224 x 224 RGB.
- Batch size: 8.
- Input normalization: divide [0,255] pixel values by 255 exactly once.
- Optimizer: Adam, initial learning rate 1e-3.
- Loss: Binary Crossentropy.
- Class weighting: none.
- Runtime: TensorFlow Meta Optimizer disabled before model construction/tracing; `run_eagerly=False`, `jit_compile=False`.
- Maximum epochs: 50; one epoch is run per process invocation so the Mac can cool between sessions.
- Checkpoint selection: highest **scikit-learn ROC AUC** from `model.predict()` on the fixed Validation split.
- Early stopping: validation scikit-learn AUROC, patience 8, min_delta 1e-4.
- Reduce LR: validation loss, patience 3, factor 0.5, minimum LR 1e-6.
- Validation is not shuffled. Train shuffling occurs inside `tf.data` with seed `base_seed + completed_epochs`.
- Test is never loaded or evaluated.

The Keras AUC metric is retained as a diagnostic approximation and is named `keras_auroc_approx`. It is **not** used for checkpoint selection. At each epoch end, the controller computes scikit-learn AUROC from a single-normalized Validation dataset via `model.predict()`; that exact score is the checkpoint-selection metric.

## Environment and required inputs

Run commands from the repository root with the project's `.venv` activated. The trainer source and verification script are already part of this repository; do not extract the original development ZIP over the finalized project.

The public Git repository intentionally excludes row-level patient metadata and CSV files. Before running the trainer, restore the required private inputs to their original paths:

- `data/processed/splits/train_v3.csv` and `validation_v3.csv`.
- `data/processed/label_efficiency/manifests_v1/metadata.json`.
- `train_010pct.csv`, `train_025pct.csv`, `train_050pct.csv`, and `train_100pct.csv` under `data/processed/label_efficiency/manifests_v1/`.
- The extracted Chest X-Ray images under `data/interim/version3/CellData/chest_xray/`.

The trainer checks the frozen metadata/manifests and the Train/Validation protocol. It does not load the Test split. The private evidence archive is required to restore patient-level CSV inputs; the original image dataset is additionally required for training or fresh image evaluation.

## Re-run a label level (optional)

The following commands document how to run a new formal experiment. They are not required to inspect the completed results, rebuild the saved-result tables, or prepare the report. The original four formal runs are already complete. Preserve their experiment directories and evidence; do not delete or overwrite them to rerun training.

For a deliberately planned new run in an isolated working copy with the required private inputs restored, the command form is:

```bash
python -m src.training.train_cnn_v2_label_efficiency --level 010pct
```

The script runs **one epoch per invocation**. When it prints `Status: session_complete`, let the Mac cool and run the exact same command again. It resumes the model and optimizer state from that level's `training_state/` folder. Continue until it reports `early_stopped` or `completed`.

Then repeat, one level at a time:

```bash
python -m src.training.train_cnn_v2_label_efficiency --level 025pct
python -m src.training.train_cnn_v2_label_efficiency --level 050pct
python -m src.training.train_cnn_v2_label_efficiency --level 100pct
```

Do not start multiple runs concurrently on the 8-GB Mac. Each level has a separate experiment directory and independent model initialization. The 10% one-epoch pilot directory is not reused.

## Output layout

For example, the 10% full run uses:

```text
outputs/experiments/cnn_from_scratch_v2_010pct_full_20261119/
├── checkpoints/
│   └── best.keras
├── training_state/
│   └── ... BackupAndRestore state ...
├── evaluation/
│   └── validation_best/
│       ├── predictions.csv
│       ├── metrics.json
│       └── roc_curve.csv
├── history.csv
└── training_summary.json
```

The `validation_best` outputs always correspond to the best checkpoint saved in `checkpoints/best.keras`. Threshold 0.5 is included only as a diagnostic in the per-epoch best metrics; **no operating threshold is locked by this trainer**. Threshold policy/selection will be applied later using Validation, consistently across models.

## Verify a finished or paused run

After at least one best checkpoint has been saved:

```bash
python scripts/verify_cnn_v2_experiment.py \
  --run-name cnn_from_scratch_v2_010pct_full_20261119
```

This reloads the best checkpoint, rebuilds Validation with one normalization step, and checks the predictions against the saved Validation CSV. It does not modify any files and does not access Test.

## Safety and recovery rules

- The script verifies each manifest SHA256 against the frozen audit and checks complete patient groups, expected image/patient/class counts, original Train row metadata, source Train SHA256, image paths, and Train/Validation patient/hash separation.
- It refuses to resume if the saved config differs, if summary/history disagree, or if the BackupAndRestore state is missing.
- It refuses to overwrite incompatible or partial experiment directories.
- If a process errors after creating its output directory, **do not delete the directory or run with a new name immediately**. Inspect the saved summary, history and `training_state/` first; share the full log before recovery.
- Do not edit `manifests_v1` during this experiment series.
- Do not read, train on, select checkpoints from, tune thresholds on, or otherwise use Test during these runs.
- Do not use pilot metrics as formal Label Efficiency results. These are new full runs, starting from random initialization for each percentage.

## Interpretation

The four recorded CNN V2 formal runs reached the `early_stopped` terminal status. Their saved best-checkpoint Validation AUROCs are 0.984444 (10%), 0.990940 (25%), 0.995845 (50%), and 0.995956 (100%). These saved Validation results form the CNN-from-scratch Label Efficiency curve reported for this project. The separate MobileNetV2 comparison uses the same four manifest levels and fixed Validation split. Do not substitute pilot metrics for any of these formal results.

## Recorded execution and verification status

The four formal CNN V2 runs were executed in the recorded macOS Apple-Silicon environment (Python 3.11.4, TensorFlow 2.18.0, and `tensorflow-metal` 1.2.0). Their training summaries record terminal status `early_stopped`; the saved best Validation predictions and histories are retained under their respective `outputs/experiments/` directories.

The project-wide checkpoint audit separately verified the four official CNN V2 best checkpoints as part of the eight-checkpoint audit: their recorded SHA-256 hashes matched, Keras loading passed, and total/trainable/non-trainable parameter counts matched the expected CNN V2 architecture. This checkpoint-load audit did not run inference. It must not be described as a fresh prediction-reproduction test. The script `scripts/verify_cnn_v2_experiment.py` is a separate read-only check for comparing a run's reloaded-checkpoint Validation predictions with its saved Validation CSV; report that check as completed only when its own output is available.
