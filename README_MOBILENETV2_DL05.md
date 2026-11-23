# DL-05 — MobileNetV2 Transfer Learning

This document describes the MobileNetV2 transfer-learning trainer and the four formal label-efficiency runs (10%, 25%, 50%, and 100%) recorded for DL-05. All four formal runs have been completed using the same fixed patient-level Train/Validation protocol and the same four nested Train manifests used by CNN V2. The historical CNN V2 pilot and the separate CNN V1 experiments are not part of the MobileNetV2 formal results.

## Repository status and required inputs

The trainer source is already part of this repository at `src/training/train_mobilenetv2_label_efficiency.py`. Do not copy or extract an earlier development package over the finalized project.

The public Git repository intentionally excludes row-level patient metadata, split CSVs, manifest CSVs, and the raw Chest X-Ray images. Before running a new training session, restore the required private files to their original project-relative paths:

```text
data/processed/splits/train_v3.csv
data/processed/splits/validation_v3.csv
data/processed/label_efficiency/manifests_v1/metadata.json
data/processed/label_efficiency/manifests_v1/train_010pct.csv
data/processed/label_efficiency/manifests_v1/train_025pct.csv
data/processed/label_efficiency/manifests_v1/train_050pct.csv
data/processed/label_efficiency/manifests_v1/train_100pct.csv
data/interim/version3/CellData/chest_xray/...
```

These row-level metadata, split, and manifest files are included in the private reproducibility archive `archives/DL05_PRIVATE_REPRODUCIBILITY_EVIDENCE_20261123_FINAL.zip`. The archive is excluded from public Git and must be transferred separately when using a clean public clone. The original image dataset is not included and must be supplied separately for training or fresh image evaluation. The script checks the frozen SHA-256 hashes, class/image/patient counts, patient-group completeness, and Train–Validation separation. The training script does not open `test_v3.csv`; Test evaluation is handled separately by `scripts/evaluate_mobilenetv2_test.py`.

## Model and protocol

- Backbone: `tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet")`.
- Input size: 224 × 224 × 3.
- Image geometry: same `resize_with_pad` bilinear + antialias operation used by the existing project.
- Input preprocessing: `tf.keras.applications.mobilenet_v2.preprocess_input`, mapping the decoded [0, 255] pixels to [-1, 1]. Do not add `/255` before this preprocessing.
- Head: GlobalAveragePooling2D → Dropout(0.30) → Dense(1, sigmoid).
- Stage 1: frozen backbone, Adam 1e-3.
- Stage 2: initialize from best Stage-1 Validation checkpoint; unfreeze last 30 backbone layers except BatchNormalization; Adam 1e-5. The backbone is called with `training=False` so BatchNormalization statistics remain in inference mode.
- Loss: binary cross-entropy; no class weights, consistent with the recorded CNN V2 protocol.
- Batch size: 8; one epoch per process invocation to support the fanless MacBook Air.
- Checkpoint selection: exact scikit-learn AUROC computed from Validation predictions, across both stages.
- Operating threshold: after both stages finish, select the highest observed Validation score meeting Sensitivity ≥ 0.99. This is a Validation-only rule; Test remains untouched.

## Optional rerun procedure

The four recorded MobileNetV2 formal runs are already complete. Their saved Validation AUROCs are 0.993211 (10%), 0.995140 (25%), 0.997347 (50%), and 0.997990 (100%). Their training summaries, histories, Validation predictions, and best checkpoints are retained under their corresponding `outputs/experiments/` directories.

The commands below document how to deliberately rerun an experiment; they are not required to inspect the recorded results or rebuild the saved-result tables. Do not run them over the existing completed experiment directories. Use a separate working copy with the required private inputs and original image dataset restored, and verify the output-directory/run-state behavior before starting a new run.

Activate the virtual environment from that separate project working copy:

```bash
# Run these commands from the repository root.
source .venv/bin/activate
```

First run a data-only preflight for 10%:

```bash
PYTHONPATH="$PWD" python -m src.training.train_mobilenetv2_label_efficiency --level 010pct --preflight-only
```

Then start training:

```bash
PYTHONPATH="$PWD" python -m src.training.train_mobilenetv2_label_efficiency --level 010pct
```

Each invocation runs one epoch. Repeat the exact same command whenever the output says `SESSION COMPLETE`. When Stage 1 ends, the next invocation automatically begins Stage 2. A completed run will refuse to train again.

After the 10% run reaches `TRANSFER LEARNING RUN COMPLETE`, repeat the sequence with the following levels (one level at a time):

```bash
PYTHONPATH="$PWD" python -m src.training.train_mobilenetv2_label_efficiency --level 025pct --preflight-only
PYTHONPATH="$PWD" python -m src.training.train_mobilenetv2_label_efficiency --level 025pct

PYTHONPATH="$PWD" python -m src.training.train_mobilenetv2_label_efficiency --level 050pct --preflight-only
PYTHONPATH="$PWD" python -m src.training.train_mobilenetv2_label_efficiency --level 050pct

PYTHONPATH="$PWD" python -m src.training.train_mobilenetv2_label_efficiency --level 100pct --preflight-only
PYTHONPATH="$PWD" python -m src.training.train_mobilenetv2_label_efficiency --level 100pct
```

For each level, repeat its training command after every `SESSION COMPLETE` message. Do not run all four levels concurrently.

## Output structure

Each level gets a separate directory:

```text
outputs/experiments/mobilenetv2_<level>_imagenet_20261122/
├── checkpoints/
│   ├── best_frozen.keras
│   ├── best_fine_tune.keras
│   └── best.keras
├── training_state/
│   ├── frozen_state.json
│   ├── fine_tune_state.json
│   └── keras_backup/
├── evaluation/
│   ├── validation_frozen_best/
│   ├── validation_fine_tune_best/
│   ├── validation_best/
│   └── validation_threshold_locked/
├── history_frozen.csv
├── history_fine_tune.csv
├── model_summary_frozen.txt
├── model_summary_fine_tune.txt
└── training_summary.json
```

For the four recorded formal runs, each `best_fine_tune.keras` file is byte-identical by SHA-256 to its corresponding `best.keras`; `best.keras` is the final checkpoint used in the official checkpoint audit. Each `best_frozen.keras` is a distinct intermediate checkpoint and is retained as stage-level evidence. The current archive candidate manifest selects one copy of each final checkpoint plus the four distinct frozen-stage checkpoints. The verified private archive is `archives/DL05_PRIVATE_REPRODUCIBILITY_EVIDENCE_20261123_FINAL.zip`; it is excluded from public Git.

Do not delete or rename an incomplete run directory. The script refuses to overwrite runs with a mismatched configuration or missing recovery files. Preserve the completed experiment directories as evidence; do not use the optional rerun commands against those existing directories.

## Weights, citations, and caveats

The four formal MobileNetV2 runs were executed in the recorded macOS Apple-Silicon environment (Python 3.11.4, TensorFlow 2.18.0, and `tensorflow-metal` 1.2.0). Their four official final checkpoints were included in the project-wide eight-checkpoint audit: SHA-256 matched for all four, Keras loading passed, and total/trainable/non-trainable parameter counts matched the expected MobileNetV2 architecture. This checkpoint-load audit did not run inference and must not be described as a fresh prediction-reproduction test.

For a new training run, the first invocation may download the official Keras MobileNetV2 ImageNet weights if they are not already cached. Internet access is required for that first download. Preserve the actual checkpoint SHA-256 values alongside the experiment summaries.

References:

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018. https://doi.org/10.1109/CVPR.2018.00474
2. Keras API — MobileNetV2 and preprocessing: https://keras.io/api/applications/mobilenet/#mobilenetv2-function
3. TensorFlow — Transfer learning and fine-tuning (including BatchNormalization behavior): https://www.tensorflow.org/guide/keras/transfer_learning

The script uses the Keras-provided ImageNet checkpoint; the ImageNet dataset's underlying terms and the checkpoint's redistribution terms should be checked against the provider's current notices before redistributing model weights. Do not describe this model as a clinical diagnostic system.
