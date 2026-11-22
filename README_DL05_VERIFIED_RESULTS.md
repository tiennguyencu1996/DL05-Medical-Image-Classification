# DL-05 Verified Results Pack

This pack adds a reproducible result-table builder and the tables generated during the audit. It does **not** train models or run inference.

## Contents

- `scripts/build_verified_results.py` — rebuilds the CSV result tables and automated audit checks from saved prediction files, split CSVs, manifests and metric JSON files.
- `outputs/reports/verified_results/verified_test_metrics.csv` — seven Test rows: Majority, HOG + Logistic Regression, CNN V2 at 10/25/50/100%, MobileNetV2 at 100%; includes AUROC, Sensitivity, Specificity, Precision, Accuracy, Macro-F1, TN/FP/FN/TP and artifact paths.
- `outputs/reports/verified_results/verified_label_efficiency.csv` — Validation AUROC at 10/25/50/100%, using saved best-checkpoint prediction files and Train-manifest image fractions.
- `outputs/reports/verified_results/verified_thresholds.csv` — thresholds and Validation selection policies.
- `outputs/reports/verified_results/verified_dataset_splits.csv` — image/patient/class counts for Train, Validation and Test.
- `outputs/reports/verified_results/audit_checks.csv` and `audit_summary.json` — machine-readable checks run during audit.
- `outputs/reports/verified_results/DL05_Verified_Results_FINAL.xlsx` — formatted workbook with Summary, Test_Metrics, Label_Efficiency, Thresholds, Dataset_Splits and Audit_Checks sheets.

## Install location

Extract this pack into the **project root** (the folder containing `src/`, `data/`, `outputs/`, and `requirements.txt`). It adds a new script and a new report-output folder; it does not replace the original project source files.

## Rebuild CSV outputs

From the project root, with the project's `.venv` active:

```bash
python scripts/build_verified_results.py
```

Optional explicit output directory:

```bash
python scripts/build_verified_results.py --output-dir outputs/reports/verified_results
```

The script requires NumPy and scikit-learn. It recomputes metrics from saved predictions, checks Test threshold labels using float32 score/threshold semantics, checks prediction inventories against the split CSVs, verifies saved Test metric JSONs, validates patient/SHA split separation, checks deduplication counts and nested manifests, and writes CSV/JSON outputs.

## Scope and limitations

- This is an **artifact audit**, not a retraining or fresh-inference run.
- It does not verify model weights by restoring best checkpoints. The best `.keras` checkpoints were not present in the submitted audit ZIPs.
- It uses the same fixed Test partition that the project has evaluated in multiple model/fraction runs; the report should disclose this repeated-evaluation limitation.
- Threshold-dependent metrics use the declared Validation-selected thresholds. The threshold selection targets differ between methods (HOG/CNN V2 10-50% target Sensitivity >= 0.85; CNN V2 100% and MobileNetV2 target Sensitivity >= 0.99); avoid describing them as identical operating conditions.
- Metadata-only descriptors, especially JPEG file size, show strong label association. This is evidence of dataset/source confounding risk, not proof that the CNN used those proxies.

## Audit run recorded for the supplied bundle

- 7 Test model/result rows
- 4 label-efficiency points
- 10 threshold records
- 86 automated checks passed out of 86

The workbook and CSV tables are based on the artifacts supplied for the audit and should be reviewed by the student before submission.
