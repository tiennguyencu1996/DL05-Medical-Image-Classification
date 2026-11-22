# DL-05 Label Efficiency: Train-only Manifest Setup

## Purpose

Create four deterministic, nested, patient-level, class-stratified subsets from the fixed `train_v3.csv` split for the mandatory 10%, 25%, 50%, and 100% label-efficiency experiment.

## Install

From the project root, extract this package so that the script is placed at:

```text
scripts/create_label_efficiency_manifests.py
```

## Run

```bash
cd ~/Projects/DL05-Medical-Image-Classification
source .venv/bin/activate
python scripts/create_label_efficiency_manifests.py
```

Defaults:

- Input: `data/processed/splits/train_v3.csv`
- Output: `data/processed/label_efficiency/manifests_v1/`
- Seed: `42`

The script reads only the Train CSV. It does not open Validation or Test and does not load image pixels. It validates the expected Train class counts, required metadata, one-label-per-patient consistency, unique relative paths, and train split tags.

## Patient-level and class-stratified sampling

Each patient is assigned to exactly one label in the source manifest. Patient IDs are sorted, independently shuffled within each class using fixed class-specific seeds, and then nested prefixes are selected to approximate the target **image fraction in each class**. All Train images for every selected patient are included. This avoids splitting images from one patient across label-efficiency levels.

Because patients can contribute different numbers of images, actual percentages will not necessarily equal 10%, 25%, or 50% exactly. The actual image and patient fractions are recorded in `metadata.json`; report those actual values rather than pretending they are exact.

## Outputs

- `train_010pct.csv`
- `train_025pct.csv`
- `train_050pct.csv`
- `train_100pct.csv`
- `patient_selection_audit.csv`
- `metadata.json` (includes source CSV SHA256, seed, counts, actual fractions)
- `README.md`

The four manifests preserve the original Train CSV columns and source row order. The script asserts patient and image nesting and verifies that the 100% manifest contains every Train record.

## Safety

The script refuses to overwrite an existing output directory. If you intentionally need a new version, choose a new `--output-dir`, for example:

```bash
python scripts/create_label_efficiency_manifests.py \
  --output-dir data/processed/label_efficiency/manifests_v2
```

Do not pass Validation or Test CSVs as input. Use the generated exact same manifest at each percentage for both CNN-from-scratch and Transfer Learning.
