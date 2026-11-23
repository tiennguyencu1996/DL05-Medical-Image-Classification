# DL-05 Verified Results Pack

This document describes the saved-artifact verification workflow for DL-05. The result builder recomputes result tables and programmed consistency checks from saved predictions, metadata, split files, manifests, thresholds, and metric JSON files. It does **not** train models or run fresh model inference.

## Contents

- `scripts/build_verified_results.py` — rebuilds the CSV result tables and automated audit checks from saved prediction files, split CSVs, manifests and metric JSON files.
- `outputs/reports/verified_results/verified_test_metrics.csv` — seven Test rows: Majority, HOG + Logistic Regression, CNN V2 at 10/25/50/100%, MobileNetV2 at 100%; includes AUROC, Sensitivity, Specificity, Precision, Accuracy, Macro-F1, TN/FP/FN/TP and artifact paths.
- `outputs/reports/verified_results/verified_label_efficiency.csv` — Validation AUROC at 10/25/50/100%, using saved best-checkpoint prediction files and Train-manifest image fractions.
- `outputs/reports/verified_results/verified_thresholds.csv` — thresholds and Validation selection policies.
- `outputs/reports/verified_results/verified_dataset_splits.csv` — image/patient/class counts for Train, Validation and Test.
- `outputs/reports/verified_results/audit_checks.csv` and `audit_summary.json` — machine-readable checks run during audit.
- `outputs/reports/verified_results/DL05_Verified_Results_FINAL.xlsx` — formatted workbook with Summary, Test_Metrics, Label_Efficiency, Thresholds, Dataset_Splits and Audit_Checks sheets.

## Required private evidence

The public Git repository intentionally excludes row-level patient metadata, split CSVs, label-efficiency manifest CSVs, experiment prediction files, and model checkpoints. A clean public clone therefore does not contain every input required by the audit scripts.

Before rebuilding the verified result tables, restore the corresponding private evidence to the original project-relative paths. At minimum, `scripts/build_verified_results.py` expects:

- Dataset audit inputs: `data/processed/metadata_v3.csv`, `metadata_deduplicated_v3.csv`, `duplicate_exclusions_v3.csv`, and `deduplication_summary_v3.json`.
- Fixed split CSVs: `data/processed/splits/train_v3.csv`, `validation_v3.csv`, and `test_v3.csv`.
- Frozen manifests: `data/processed/label_efficiency/manifests_v1/train_010pct.csv`, `train_025pct.csv`, `train_050pct.csv`, and `train_100pct.csv`.
- Baseline evidence: `outputs/metrics/predictions.csv` and `outputs/metrics/baseline_results_v3.json`.
- Formal experiment evidence: saved CNN V2 and MobileNetV2 Test/Validation predictions and metric JSON files, plus the Validation threshold-selection artifacts referenced by the builder.

The raw Chest X-Ray image files are not required to rebuild tables from saved predictions. They are required for retraining or fresh image evaluation. The eight official `.keras` best checkpoints are required for the separate checkpoint audit, not for `build_verified_results.py` itself.

## Rebuild CSV outputs

From the project root, with the project's `.venv` active:

```bash
python scripts/build_verified_results.py
```

Optional explicit output directory:

```bash
python scripts/build_verified_results.py --output-dir outputs/reports/verified_results
```

The script requires NumPy and scikit-learn. It recomputes metrics from saved predictions, checks Test threshold labels using float32 score/threshold semantics, checks prediction inventories against the split CSVs, compares recomputed Test metrics with saved metric JSONs, validates patient/SHA split separation, checks deduplication counts against the saved deduplication summary, checks manifest membership and nesting, and writes CSV/JSON outputs.

The builder's checks are scoped to the supplied artifacts and the implemented assertions. They do not constitute a fresh end-to-end reproduction of data extraction, model training, threshold optimization, or inference. In particular, the builder consumes saved threshold-selection evidence; it does not independently rerun the complete threshold-selection optimization from raw Validation scores.

## Scope and limitations

- The result builder is an **artifact audit**, not a retraining or fresh-inference run. It recomputes metrics and programmed consistency checks from stored evidence.
- Checkpoint verification is a separate operation performed by `scripts/audit_best_checkpoints.py`. The recorded local audit verified 8/8 official checkpoint SHA-256 hashes, successful Keras loads, and expected parameter counts. Loading a checkpoint for this audit does not run prediction or establish fresh-inference reproducibility.
- The earlier audit snapshot ZIPs did not include the best `.keras` checkpoint binaries. The later local checkpoint audit was run against the actual checkpoint files listed in the training summaries. Keep the checkpoint audit CSV/JSON and the corresponding private checkpoint archive together as provenance.
- The workflow uses the same fixed Test partition that the project evaluated in multiple model/fraction runs; the report must disclose this repeated-evaluation limitation.
- Threshold-dependent metrics use the declared Validation-selected thresholds. The threshold selection targets differ between methods (HOG/CNN V2 10-50% target Sensitivity >= 0.85; CNN V2 100% and MobileNetV2 target Sensitivity >= 0.99); avoid describing them as identical operating conditions.
- Metadata-only descriptors, especially JPEG file size, show strong label association. This is evidence of dataset/source confounding risk, not proof that the CNN used those proxies.
- A clean public Git clone cannot run this workflow until the required private row-level metadata, split/manifests, prediction files, metric JSONs, and threshold evidence are restored. Do not upload patient-level files or the private archive to a public repository.

## Audit run recorded for the local evidence set

- 7 Test model/result rows
- 4 label-efficiency points
- 10 threshold records
- 86 automated checks passed out of 86

The recorded local audit generated the CSV/JSON result tables from the saved evidence set. The formatted workbook is a separate deliverable and is not generated by `scripts/build_verified_results.py`; review it against the verified CSVs before submission. The reported 86/86 checks are the checks implemented by this builder for those supplied artifacts; they should not be interpreted as proof of independent end-to-end reproduction. Review the workbook, CSV tables, audit output, source commit, and private archive manifest together before submission.
