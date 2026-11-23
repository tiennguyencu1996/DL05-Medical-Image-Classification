# DL-05 Label Efficiency: Train-only Manifest Setup

## Purpose

Create four deterministic, nested, patient-level, class-stratified subsets from the fixed `train_v3.csv` split for the mandatory 10%, 25%, 50%, and 100% label-efficiency experiment.

## Script location and required input

The manifest generator is already included in this repository at
`scripts/create_label_efficiency_manifests.py`. Run commands from the project
root with the project's `.venv` activated.

The generator expects the fixed, patient-level Train split at
`data/processed/splits/train_v3.csv`. That row-level CSV is intentionally
excluded from the public Git repository and must be restored from the private
project evidence before running the generator.

## Optional regeneration for a new manifest set

The four `manifests_v1` CSVs have already been generated and were used by the
formal CNN V2 and MobileNetV2 experiments. Do not rerun the default command
when inspecting or reproducing the reported experiments; preserve the frozen
`manifests_v1` files.

For a deliberately new manifest set, use a separate output directory rather
than replacing the recorded manifests:

```bash
python scripts/create_label_efficiency_manifests.py \
  --output-dir data/processed/label_efficiency/manifests_v2
```

The script refuses to overwrite an existing output directory. A newly
generated manifest set is a new experiment input and must not be substituted
silently for the recorded `manifests_v1` inputs.

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

## Safety and current project status

The recorded project uses the four frozen CSV files under
`data/processed/label_efficiency/manifests_v1/` for both CNN V2 and
MobileNetV2. These manifest CSVs contain row-level patient/image metadata and
are intentionally excluded from public Git; restore them from the private
archive `archives/DL05_PRIVATE_REPRODUCIBILITY_EVIDENCE_20261123_FINAL.zip`. The archive is excluded from public Git and must be transferred separately when using a clean public clone.

The script refuses to overwrite an existing output directory. If you intentionally need a new version, choose a new `--output-dir`, for example:

```bash
python scripts/create_label_efficiency_manifests.py \
  --output-dir data/processed/label_efficiency/manifests_v2
```

Do not pass Validation or Test CSVs as input. Use the generated exact same manifest at each percentage for both CNN-from-scratch and Transfer Learning.
