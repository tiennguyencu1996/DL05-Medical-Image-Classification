# DL-05 — MobileNetV2 Transfer Learning implementation

This implementation adds a separate transfer-learning experiment without modifying any existing CNN V2 experiment, checkpoint, threshold, or Test evaluation.

## Placement in the existing project

Copy the supplied Python file to:

```text
src/training/train_mobilenetv2_label_efficiency.py
```

It expects the existing project layout and these already-audited files:

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

The script checks the existing frozen SHA256 hashes, class/image/patient counts, patient-group completeness, and Train–Validation separation. It never opens `test_v3.csv`.

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

## Run sequence

Activate the existing virtual environment from the project root:

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

Do not delete or rename an incomplete run directory. The script refuses to overwrite runs with a mismatched configuration or missing recovery files.

## Weights, citations, and caveats

The first training invocation downloads the official Keras MobileNetV2 ImageNet weights if they are not already cached. Internet access is required for that first download. Keep a copy of the resulting checkpoint and record the actual checkpoint SHA256 in the report.

References:

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018. https://doi.org/10.1109/CVPR.2018.00474
2. Keras API — MobileNetV2 and preprocessing: https://keras.io/api/applications/mobilenet/#mobilenetv2-function
3. TensorFlow — Transfer learning and fine-tuning (including BatchNormalization behavior): https://www.tensorflow.org/guide/keras/transfer_learning

The script uses the Keras-provided ImageNet checkpoint; the ImageNet dataset's underlying terms and the checkpoint's redistribution terms should be checked against the provider's current notices before redistributing model weights. Do not describe this model as a clinical diagnostic system.
