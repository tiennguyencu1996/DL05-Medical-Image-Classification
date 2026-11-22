import csv
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(".")

BASELINE = ROOT / "outputs/metrics/predictions.csv"

CNN_DIR = ROOT / "outputs/experiments/cnn_from_scratch_v2_100pct_full_20261121"
MOB_DIR = ROOT / "outputs/experiments/mobilenetv2_100pct_imagenet_20261122"

CNN_PRED = CNN_DIR / "evaluation/test_locked_threshold_100pct/predictions.csv"
MOB_PRED = MOB_DIR / "evaluation/test_locked_threshold/predictions.csv"

CNN_METRICS = CNN_PRED.parent / "metrics.json"
MOB_METRICS = MOB_PRED.parent / "metrics.json"

OUT = ROOT / "outputs/analysis/test_roc_comparison"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_AUC = {
    "Majority Class": 0.5,
    "HOG + Logistic Regression": 0.9764129066454648,
    "CNN V2 100%": 0.976358,
    "MobileNetV2 100%": 0.988168,
}

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def read_metrics(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc.get("metrics", doc)

def check_auc(name, auc, expected):
    if not np.isclose(auc, expected, rtol=0, atol=1e-6):
        raise RuntimeError(
            f"{name}: reconstructed AUROC {auc:.9f} "
            f"does not match saved AUROC {expected:.9f}"
        )

# 1. Read existing prediction artifacts.
baseline_rows = read_csv(BASELINE)
cnn_rows = read_csv(CNN_PRED)
mob_rows = read_csv(MOB_PRED)

def baseline_predictions(model_name):
    rows = [
        r for r in baseline_rows
        if r["split"] == "test" and r["model"] == model_name
    ]
    rows.sort(key=lambda r: int(r["row_index"]))
    return rows

majority_rows = baseline_predictions("Majority Class")
hog_rows = baseline_predictions("HOG Logistic Regression")

collections = {
    "Majority Class": majority_rows,
    "HOG + Logistic Regression": hog_rows,
    "CNN V2 100%": cnn_rows,
    "MobileNetV2 100%": mob_rows,
}

# 2. Verify sample counts and labels.
for name, rows in collections.items():
    if len(rows) != 621:
        raise RuntimeError(f"{name}: expected 621 rows, got {len(rows)}")

def labels(rows):
    return np.asarray([int(r["y_true"]) for r in rows], dtype=np.int32)

y_reference = labels(cnn_rows)

if Counter(y_reference) != Counter({0: 234, 1: 387}):
    raise RuntimeError("Unexpected CNN V2 Test class distribution.")

for name, rows in collections.items():
    if not np.array_equal(labels(rows), y_reference):
        raise RuntimeError(
            f"{name}: Test label sequence differs from CNN V2."
        )

# Check path-level identity between the two neural models.
for key in ("relative_path", "patient_id", "sha256"):
    cnn_values = [r[key] for r in cnn_rows]
    mob_values = [r[key] for r in mob_rows]
    if cnn_values != mob_values:
        raise RuntimeError(f"CNN/MobileNet Test {key} sequence mismatch.")

# 3. Reconstruct ROC and validate stored AUC.
series = {}
roc_records = []

for name, rows in collections.items():
    y = labels(rows)
    scores = np.asarray(
        [float(r["y_score"]) for r in rows],
        dtype=np.float64
    )

    if not np.isfinite(scores).all():
        raise RuntimeError(f"{name}: non-finite scores.")

    auc = float(roc_auc_score(y, scores))
    check_auc(name, auc, EXPECTED_AUC[name])

    fpr, tpr, thresholds = roc_curve(y, scores)
    series[name] = (fpr, tpr, auc)

    for x, z, threshold in zip(fpr, tpr, thresholds):
        roc_records.append({
            "model": name,
            "fpr": float(x),
            "tpr": float(z),
            "threshold": float(threshold),
        })

# Compare neural network metrics with their saved JSON artifacts.
for name, path in (
    ("CNN V2 100%", CNN_METRICS),
    ("MobileNetV2 100%", MOB_METRICS),
):
    saved = read_metrics(path)
    check_auc(name, series[name][2], float(saved["auroc"]))

# 4. Save ROC coordinates.
csv_path = OUT / "test_roc_curves.csv"
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["model", "fpr", "tpr", "threshold"]
    )
    writer.writeheader()
    writer.writerows(roc_records)

# 5. Draw ROC chart.
plt.figure(figsize=(10, 7))

for name, (fpr, tpr, auc) in series.items():
    plt.plot(
        fpr, tpr,
        label=f"{name} (AUROC = {auc:.4f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlim(0, 1)
plt.ylim(0, 1.02)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Test ROC Curves — DL-05 Medical Image Classification")
plt.legend(loc="lower right")
plt.grid(alpha=0.25)
plt.tight_layout()

png_path = OUT / "test_roc_comparison.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

# 6. Print audit summary.
print("=" * 72)
print("TEST ROC RECONSTRUCTION — PASS")
print("=" * 72)
print("Test samples:", len(y_reference))
print("Class distribution:", dict(Counter(y_reference)))

for name, (_, _, auc) in series.items():
    print(f"{name:30s}: AUROC = {auc:.9f}")

print("\nCSV:", csv_path)
print("PNG:", png_path)
print("Inference performed: NO")
print("Test images accessed: NO")
print("Original Test artifacts modified: NO")
