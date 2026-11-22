# DL-05 CNN V2 controlled-runtime preflight

This package adds two new files to the existing project:

- `src/models/cnn_from_scratch_v2.py`
- `src/training/smoke_test_cnn_v2.py`

It does not modify V1 source code, checkpoints, outputs, archives, split CSVs, or data.

## Install into the project

From the project root, extract the ZIP with `unzip -n` so existing files are not overwritten:

```bash
unzip -n ~/Downloads/DL05_CNN_V2_preflight.zip -d .
```

## Run preflight

Activate the existing environment and run:

```bash
source .venv/bin/activate
python -m src.training.smoke_test_cnn_v2
```

The script configures `disable_meta_optimizer=True` before dataset/model execution, uses eight balanced images from the Train split only, then checks:

1. Eager vs graph forward predictions.
2. Eager vs graph loss and gradients (inference-mode path, with Dropout disabled and BatchNorm using moving statistics).
3. One graph-mode `train_on_batch` step and finite weights/metrics.
4. Keras checkpoint save/reload consistency using a temporary directory.

The script does not load Validation or Test, does not save any experiment checkpoint in the project, and does not start full training. A PASS result is a preflight check, not proof that the full multi-epoch run is correct; the first full-training epoch must still be checked for finite loss/gradients and execution consistency.

## Important

The V2 model intentionally reuses the V1 topology and parameter counts. The purpose is to isolate runtime/execution differences, not to introduce an architectural change. Full V2 training should only be prepared after this preflight passes and the training script has an independent V2 output directory/state.
