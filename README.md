# DL-05 — Medical Image Classification

**A CNN Designed from Scratch versus Transfer Learning**

Phenikaa University · Deep Learning Fundamentals (EEE703035) · Topic DL-05

## 1. Project overview

This project studies binary chest X-ray image classification for **NORMAL (label 0)** and **PNEUMONIA (label 1)** using two approaches:

1. A custom CNN (CNN V2) trained from random initialization.
2. ImageNet-pretrained MobileNetV2 using a frozen-backbone stage followed by fine-tuning.

The comparison uses a fixed patient-level Train/Validation/Test protocol and four nested, patient-level Train subsets (10%, 25%, 50%, and 100%). Because the class distribution is imbalanced, results are reported with AUROC, Sensitivity, Specificity, Accuracy, Macro-F1, and confusion matrices. This is a research benchmark, **not a medical diagnostic system**.

## 2. Dataset and data policy

The project uses the Chest X-Ray Pneumonia subset associated with *Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images*, Mendeley Data, Version 3, DOI: [10.17632/rscbjbr9sj.3](https://doi.org/10.17632/rscbjbr9sj.3). The repository lists CC BY 4.0 and also includes a research-use statement. Attribute the dataset and check the provider's terms before any redistribution.

The image files are **not included** in this source repository or audit bundle. The training and image-evaluation scripts expect the extracted images at:

```text
data/interim/version3/CellData/chest_xray/
```

The code expects the fixed split metadata under `data/processed/splits/` and frozen label-efficiency manifests under `data/processed/label_efficiency/manifests_v1/`. Do not regenerate these files when reproducing the reported experiments; their SHA-256 values are checked by the training scripts.

### Audited dataset inventory

| Split | Images | Patients | NORMAL | PNEUMONIA | PNEUMONIA share |
|---|---:|---:|---:|---:|---:|
| Train | 4,164 | 2,120 | 1,078 | 3,086 | 74.11% |
| Validation | 1,042 | 530 | 270 | 772 | 74.09% |
| Test | 621 | 431 | 234 | 387 | 62.32% |
| **Total** | **5,827** | **3,081** | **1,582** | **4,245** | **72.85%** |

The audit started from 5,856 metadata records. Exact SHA-256 deduplication excluded 29 files from 27 duplicate groups, leaving 5,827 unique images. The retained-file policy is deterministic: smallest image number in each exact-hash group, with relative path as a tie-breaker. The saved split audit records zero patient overlap and zero SHA-256 overlap across the three partitions.

### Frozen Train-only label-efficiency subsets

| Nominal level | Actual image fraction | Train images | Patients | NORMAL | PNEUMONIA |
|---|---:|---:|---:|---:|---:|
| 10% | 9.990% | 416 | 204 | 108 | 308 |
| 25% | 25.024% | 1,042 | 529 | 269 | 773 |
| 50% | 50.000% | 2,082 | 1,037 | 539 | 1,543 |
| 100% | 100.000% | 4,164 | 2,120 | 1,078 | 3,086 |

The two deep-learning model families use the same manifest at each fraction. Actual image fractions differ slightly from nominal percentages because complete patient groups are retained.

## 3. Models

### CNN V2 from scratch

- Input: 224 × 224 × 3 RGB.
- Four convolutional blocks with 32, 64, 128, and 256 filters.
- Each block contains two 3 × 3 Conv2D layers, Batch Normalization, and ReLU, followed by 2 × 2 MaxPooling and Dropout.
- Classification head: Global Average Pooling → Dense(128, ReLU) → Dropout(0.40) → Dense(1, sigmoid).
- Total parameters: 1,208,161 (1,206,241 trainable; 1,920 non-trainable).

### MobileNetV2 transfer learning

- `tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet")`.
- Input: 224 × 224 × 3 RGB; use the MobileNetV2-specific preprocessing function on decoded [0, 255] pixels.
- Head: Global Average Pooling → Dropout(0.30) → Dense(1, sigmoid).
- Stage 1: frozen backbone, train the classifier head.
- Stage 2: unfreeze the last 30 backbone layers while excluding BatchNormalization layers; use a lower learning rate and keep the backbone call in inference mode for BatchNormalization behavior.
- Total parameters: 2,259,265. The saved fine-tuning checkpoint has 1,512,001 trainable and 747,264 non-trainable parameters.

## 4. Main results recorded in the report

### Validation label-efficiency AUROC

| Train fraction | CNN V2 | MobileNetV2 | MobileNetV2 − CNN V2 |
|---|---:|---:|---:|
| 10% | 0.984444 | 0.993211 | +0.008768 |
| 25% | 0.990940 | 0.995140 | +0.004200 |
| 50% | 0.995845 | 0.997347 | +0.001502 |
| 100% | 0.995956 | 0.997990 | +0.002034 |

MobileNetV2 has higher Validation AUROC at all four measured fractions. No crossover is observed at the four sampled fractions or on the piecewise-linear segments joining them. Do not infer an intersection outside the tested range.

### Test metrics at locked Validation-selected operating points

| Model | Train fraction | AUROC | Sensitivity | Specificity | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Majority Classifier | N/A | 0.5000 | 1.0000 | 0.0000 | 0.6232 | 0.3839 |
| HOG + Logistic Regression | 100% | 0.9764 | 0.9121 | 0.9103 | 0.9114 | 0.9067 |
| CNN V2 | 10% | 0.9341 | 0.9121 | 0.8162 | 0.8760 | 0.8670 |
| CNN V2 | 25% | 0.9342 | 0.9070 | 0.7821 | 0.8599 | 0.8488 |
| CNN V2 | 50% | 0.9746 | 0.9251 | 0.9103 | 0.9195 | 0.9148 |
| CNN V2 | 100% | 0.9764 | 0.9897 | 0.5855 | 0.8374 | 0.8071 |
| MobileNetV2 | 100% | 0.9882 | 0.9948 | 0.6197 | 0.8535 | 0.8277 |

Threshold-dependent metrics were computed at each model's locked Validation-selected threshold. The threshold policies are not identical across every row: HOG and CNN V2 at 10%–50% target Validation Sensitivity ≥ 0.85, while CNN V2 at 100% and MobileNetV2 target ≥ 0.99. Therefore, AUROC is the cleanest primary ranking metric; do not describe the operating-point metrics as comparisons at identical threshold-selection conditions.

The fixed Test split was evaluated in several separately defined runs. The report discloses this repeated-evaluation limitation; Test was not used for training, checkpoint selection, or threshold tuning according to the saved protocols.

## 5. Important interpretation and limitations

A patient-level metadata-only diagnostic found strong associations between class labels and non-CNN descriptors. In particular, JPEG file size alone produced very high AUROC in the saved patient-level diagnostic. Geometry and image-quality proxies were also predictive. This is a **dataset/source-confounding warning**, not proof that the CNN used file size or learned a specific shortcut. File size is not directly supplied as a CNN input; geometry may be indirectly visible after resizing and padding. These findings limit claims that the models learned generalizable disease-specific features.

Other limitations include one public dataset and one fixed split, no external hospital validation, no multi-seed confidence intervals, different threshold-selection targets, a large Validation-to-Test Specificity reduction for MobileNetV2, and no calibration, decision-curve, subgroup fairness, prospective workflow, or clinical evaluation. Do not present either model as a diagnostic tool.

## 6. Repository layout

```text
.
├── data/
│   ├── interim/version3/CellData/chest_xray/     # local dataset; not included
│   └── processed/
│       ├── splits/                               # fixed Train/Validation/Test metadata
│       └── label_efficiency/manifests_v1/        # frozen nested Train subsets
├── src/
│   ├── data/                                     # metadata, split, image pipeline utilities
│   ├── evaluation/                               # evaluation utilities
│   ├── models/                                   # CNN V1/V2 and baselines
│   └── training/                                 # CNN and MobileNetV2 trainers
├── scripts/                                      # audit, threshold, ROC, and result scripts
├── outputs/
│   ├── experiments/                              # histories, summaries, predictions, checkpoints
│   ├── analysis/test_roc_comparison/             # reconstructed Test ROC artifacts
│   └── reports/verified_results/                 # verified CSV/JSON/XLSX outputs
├── README.md
├── CHANGELOG.md
└── requirements.txt
```

The exact contents may differ as audit scripts and deliverables are added. Check the current tree rather than assuming every optional folder is present in every clone.

## 7. Environment

The recorded working environment on the MacBook Air M2 was:

- macOS 15.8.1, Apple M2, 8 GB unified memory.
- Python 3.11.4.
- TensorFlow 2.18.0, Keras 3.15.1, tensorflow-metal 1.2.0.
- NumPy 2.0.2, scikit-learn 1.9.1, Matplotlib 3.11.2.

`requirements.txt` pins the recorded 57-package environment. It includes `tensorflow-metal`, so it represents the tested macOS Apple-Silicon environment and is not a platform-neutral lock for Windows or Linux.

Create/activate the environment from the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version
python -m pip check
```

For the exact recorded run, use Python 3.11.4. Re-run `python -m pip freeze` in the final submission environment and archive the result with the submission metadata.

## 8. Read-only verification commands

Run these from the repository root with `.venv` active. They do not train models or perform new model inference:

```bash
# Recompute tables/metrics from saved prediction CSVs and run the packaged checks.
python scripts/build_verified_results.py

# Verify checkpoint file hashes and Keras loadability; no prediction is performed.
python scripts/audit_best_checkpoints.py

# Reconstruct Test ROC coordinates/figure from already-saved prediction scores.
python scripts/build_test_roc_comparison.py
```

Expected outputs include:

```text
outputs/reports/verified_results/
├── verified_test_metrics.csv
├── verified_label_efficiency.csv
├── verified_thresholds.csv
├── verified_dataset_splits.csv
├── audit_checks.csv
├── audit_summary.json
├── checkpoint_audit.csv
├── checkpoint_audit_summary.json
└── DL05_Verified_Results_v1.1.1.xlsx
```

The checkpoint audit previously reported 8/8 SHA-256 matches, 8/8 successful Keras loads, and exact CNN V2/MobileNetV2 parameter-count matches. A fresh run should be retained as evidence for the current local repository state.

## 9. Training documentation

Training scripts are provided for the formal experiments. Training is not required to regenerate the reported summary tables if the saved prediction and metrics artifacts are present.

Read the corresponding guide before running any trainer:

- `README_CNN_V2_LABEL_EFFICIENCY_TRAINER.md`
- `README_MOBILENETV2_DL05.md`
- `README_LABEL_EFFICIENCY.md`
- `README_CNN_V2_PILOT.md`
- `README_CNN_V2_PILOT_UPDATE.md`

The CNN V2 and MobileNetV2 label-efficiency trainers run one epoch per invocation and resume from their run-specific training state. Do not start multiple runs concurrently on the 8-GB MacBook Air. Do not run training commands merely to rebuild the report tables.

## 10. Reproducibility and submission notes

- Preserve the fixed split CSVs and `manifests_v1`; do not regenerate them during final reporting.
- Keep dataset images, raw archives, virtual environments, and large checkpoint files out of ordinary Git commits unless a deliberate storage method (for example, Git LFS or a separate submission archive) has been selected.
- Keep the saved predictions, metrics, thresholds, training summaries, checkpoint hashes, environment snapshot, and verified-results workbook as audit evidence.
- Ensure the final Git commit represents the source code that produced the report; the original audit snapshot showed important scripts and source files untracked at the time of capture.
- Include the required report, runnable code, slides, quantitative CSV/XLSX, and AI-use declaration in the course submission.

## 11. References

1. Kermany, D., Zhang, K., & Goldbaum, M. (2018). *Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images* (Version 3). Mendeley Data. https://doi.org/10.17632/rscbjbr9sj.3
2. Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5), 1122–1131.e9. https://doi.org/10.1016/j.cell.2018.02.010
3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*. https://doi.org/10.1109/CVPR.2018.00474
4. TensorFlow/Keras documentation: MobileNetV2 preprocessing and transfer learning/fine-tuning guides (see model-specific documentation referenced in the training README).
