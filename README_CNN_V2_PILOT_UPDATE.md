# DL05 CNN v2 pilot update

This package contains a corrected full pilot-training script and a separate
inference-only correction script for the already completed 10% pilot.

## Bug fixed

The old pilot built `val_ds` with `.map(prepare_batch, ...)`, then
`infer_validation()` called `prepare_batch()` again for each batch. Since
`prepare_batch()` divides pixel values by 255, validation images were
normalized twice during post-checkpoint prediction generation.

The updated training script now treats `val_ds` as already normalized,
validates the [0,1] range, and does not normalize again. It also passes
`shuffle=False` to `model.fit()` because shuffling is already performed
inside the `tf.data` Train pipeline.

## Files

- `src/training/train_cnn_v2_pilot.py`
  Full updated pilot training script. It remains a one-epoch pilot script.
- `scripts/evaluate_cnn_v2_pilot_validation_corrected.py`
  Inference-only script for the existing checkpoint
  `cnn_from_scratch_v2_010pct_pilot_20261119`. It writes a separate folder
  `evaluation/validation_corrected_single_normalization`; it never overwrites
  the original incorrect pilot predictions and does not retrain.

## Recommended sequence

1. Back up the existing source file before replacing it:

   ```bash
   cp src/training/train_cnn_v2_pilot.py \
      src/training/train_cnn_v2_pilot.py.before_double_normalization_fix
   ```

2. Install the updated files from the project root:

   ```bash
   unzip -o ~/Downloads/DL05_CNN_V2_PILOT_UPDATED.zip -d .
   ```

3. Re-evaluate the already trained checkpoint without training again:

   ```bash
   source .venv/bin/activate
   python scripts/evaluate_cnn_v2_pilot_validation_corrected.py
   ```

4. Review the new predictions and metrics. The original folder
   `evaluation/validation/` remains untouched.

5. Do not run `python -m src.training.train_cnn_v2_pilot` with its default
   run name for the existing pilot. Its output directory already exists and
   the script intentionally refuses to overwrite it. A new pilot run would
   require a new `--run-name` and would train a new model from scratch.

## Safety

- Validation is used only for diagnostic inference and metrics.
- Test is not loaded or evaluated by either script.
- The existing checkpoint is read-only in the evaluation-only script.
- No threshold is selected or locked by these scripts.
- The historical incorrect predictions file is retained for auditability.
