# DL-05 Label Efficiency Manifests

These CSV files contain Train-only patient-level subsets. Each selected patient contributes all of their Train images. The four subsets are nested.

- `train_010pct.csv`
- `train_025pct.csv`
- `train_050pct.csv`
- `train_100pct.csv`
- `patient_selection_audit.csv`
- `metadata.json`

Nominal percentages are image-fraction targets applied separately within each class. Since patients may contribute different numbers of images, actual image fractions may differ slightly; use the reported actual fractions in the report. The script reads only the Train split and does not open Validation or Test.
