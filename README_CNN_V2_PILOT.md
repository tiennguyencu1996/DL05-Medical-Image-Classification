# DL-05 CNN V2: training-mode preflight and 10% pilot

This package adds two scripts for the next controlled CNN-from-scratch experiment. It does **not** modify CNN v1, the frozen manifests, the original Train/Validation/Test splits, or existing archives.

## Included files

- `src/models/cnn_from_scratch_v2.py`: V2 wrapper around the already verified V1 architecture; expected counts remain 1,208,161 total parameters, 1,206,241 trainable, and 1,920 non-trainable.
- `src/training/check_training_mode_cnn_v2.py`: checks finite/non-None gradients with `training=True`, verifies BatchNorm moving-statistic updates, and runs one ephemeral graph-mode training step. It reads eight balanced Train examples only.
- `src/training/train_cnn_v2_pilot.py`: runs one epoch on the frozen `train_010pct.csv` manifest and evaluates diagnostic metrics on Validation. It does not load Test.

## Install

From the project root:

```bash
cd ~/Projects/DL05-Medical-Image-Classification
unzip -n ~/Downloads/DL05_CNN_V2_pilot.zip -d .
source .venv/bin/activate
```

`unzip -n` will not overwrite files already present, including the previously installed V2 model definition.

## Step 1: training-mode gradient preflight

```bash
python -m src.training.check_training_mode_cnn_v2
```

Expected checks:

- TensorFlow Meta Optimizer disabled before model construction/tracing.
- Eager and graph training-mode gradient tensors are non-None and finite.
- At least one nonzero gradient exists.
- BatchNormalization moving statistics update during `training=True`.
- One graph-mode `train_on_batch` step has finite metrics and finite weights.

This script intentionally does not compare eager and graph gradient values exactly because Dropout is stochastic in training mode. The earlier deterministic inference-mode comparison remains a separate check.

It reads only eight balanced Train examples, does not load Validation/Test, and writes no checkpoint or experiment directory.

## Step 2: one-epoch 10% pilot

Only proceed if Step 1 passes:

```bash
python -m src.training.train_cnn_v2_pilot
```

Defaults are deliberately fixed to one epoch and the 10% frozen manifest. The script:

- verifies the manifest's source Train SHA256 against `metadata.json`;
- checks that all selected patient groups are complete;
- checks image paths and class counts;
- disables the TensorFlow Meta Optimizer before model construction/tracing;
- trains with batch size 8, Adam, learning rate 0.001 and Binary Crossentropy;
- validates on the fixed Validation split;
- writes a new experiment directory, checkpoint, training history, Validation predictions, metrics and summary;
- uses a fixed diagnostic threshold of 0.5 only; it does not select/lock an operating threshold;
- never loads or evaluates Test.

Default output directory:

`outputs/experiments/cnn_from_scratch_v2_010pct_pilot_20261119/`

The script refuses to overwrite an existing run directory. If a run was partially created and aborted, inspect and archive it before choosing a new explicit `--run-name`.

## Important interpretation

The one-epoch run is an **operational pilot**, not the final 10% Label Efficiency result. Do not include its diagnostic metrics in the final model comparison. Once the pipeline is accepted, create a separate resumable trainer for the formal 10%, 25%, 50%, and 100% runs, with the final training protocol fixed consistently across levels.

## Frozen inputs and data policy

- Train subsets: `data/processed/label_efficiency/manifests_v1/`
- Validation: original `validation_v3.csv`
- Test: never loaded by either script
- Never regenerate or modify `manifests_v1` during this experiment series.
