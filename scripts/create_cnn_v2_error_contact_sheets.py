#!/usr/bin/env python3
"""Create read-only visual contact sheets for CNN V2 100% Test errors.

Run from the project root:
    PYTHONPATH="$PWD" python scripts/create_cnn_v2_error_contact_sheets.py

Reads only the already-exported Test predictions and the corresponding image
files. It does not load a model, rerun inference, alter thresholds, or modify
source images / Test evaluation artifacts.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps

RUN_NAME = "cnn_from_scratch_v2_100pct_full_20261121"
DEFAULT_PREDICTIONS = (
    Path("outputs") / "experiments" / RUN_NAME / "evaluation"
    / "test_locked_threshold_100pct" / "predictions.csv"
)
DEFAULT_OUTPUT = (
    Path("outputs") / "experiments" / RUN_NAME / "evaluation"
    / "error_analysis_contact_sheets"
)
EXPECTED_ROWS = 621
EXPECTED_FP = 97
EXPECTED_FN = 4
EXPECTED_THRESHOLD = 0.5910010933876038
TEST_IMAGE_ROOT = Path("data/interim/version3/CellData/chest_xray/test")

# Layout: four columns by five rows (20 images per PNG page).
COLUMNS = 4
ROWS_PER_PAGE = 5
TILE_WIDTH = 340
TILE_HEIGHT = 310
IMAGE_AREA_WIDTH = TILE_WIDTH - 20
IMAGE_AREA_HEIGHT = 218
PAGE_MARGIN = 18
HEADER_HEIGHT = 78


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend([
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ])
    else:
        candidates.extend([
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ])
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_predictions(path: Path, project_root: Path) -> tuple[list[dict], float]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")

    required = {
        "relative_path", "patient_id", "sha256", "y_true", "y_score",
        "y_pred_at_locked_threshold", "locked_threshold",
    }
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(f"Missing prediction CSV columns: {sorted(missing)}")
        rows = list(reader)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows; found {len(rows)}")

    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    thresholds: set[float] = set()
    parsed: list[dict] = []
    test_root = (project_root / TEST_IMAGE_ROOT).resolve()

    for line_number, row in enumerate(rows, start=2):
        rel = Path(row["relative_path"])
        if rel.is_absolute():
            raise ValueError(f"Absolute image path at CSV line {line_number}")
        image_path = (project_root / rel).resolve()
        try:
            image_path.relative_to(test_root)
        except ValueError as exc:
            raise ValueError(
                f"Image path is not inside the expected Test folder: {rel}"
            ) from exc

        rel_text = rel.as_posix()
        if rel_text in seen_paths:
            raise ValueError(f"Duplicate relative_path: {rel_text}")
        seen_paths.add(rel_text)

        digest = row["sha256"].strip()
        if not digest or digest in seen_hashes:
            raise ValueError("Missing or duplicate SHA-256 in predictions CSV")
        seen_hashes.add(digest)

        y_true = int(row["y_true"])
        y_pred = int(row["y_pred_at_locked_threshold"])
        score = float(row["y_score"])
        threshold = float(row["locked_threshold"])
        if y_true not in (0, 1) or y_pred not in (0, 1):
            raise ValueError(f"Invalid class value at CSV line {line_number}")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid y_score at CSV line {line_number}")
        if y_pred != int(score >= threshold):
            raise ValueError(
                f"Prediction/threshold mismatch at CSV line {line_number}"
            )

        # Confirm the folder class agrees with the stored ground-truth label.
        folder_class = image_path.parent.name.upper()
        expected_folder = "PNEUMONIA" if y_true == 1 else "NORMAL"
        if folder_class != expected_folder:
            raise ValueError(
                f"Path class {folder_class} disagrees with y_true={y_true}: {rel}"
            )

        thresholds.add(threshold)
        row_out = dict(row)
        row_out.update({
            "image_path": image_path,
            "y_true_int": y_true,
            "y_pred_int": y_pred,
            "score_float": score,
            "threshold_float": threshold,
            "error_type": (
                "FALSE_POSITIVE" if y_true == 0 and y_pred == 1
                else "FALSE_NEGATIVE" if y_true == 1 and y_pred == 0
                else "CORRECT"
            ),
        })
        parsed.append(row_out)

    if len(thresholds) != 1:
        raise ValueError(f"Expected one locked threshold, found: {thresholds}")
    threshold = next(iter(thresholds))
    if not math.isclose(threshold, EXPECTED_THRESHOLD, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(
            f"Unexpected threshold {threshold:.17g}; expected {EXPECTED_THRESHOLD:.17g}"
        )
    return parsed, threshold


def make_tile(row: dict, rank: int, error_type: str) -> tuple[Image.Image, bool]:
    tile = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), "white")
    image_ok = True
    draw = ImageDraw.Draw(tile)
    title_font = load_font(16, bold=True)
    body_font = load_font(13)
    small_font = load_font(11)

    img_path: Path = row["image_path"]
    image_top = 34
    try:
        with Image.open(img_path) as src:
            img = ImageOps.exif_transpose(src).convert("RGB")
            resampling = getattr(Image, "Resampling", Image)
            img.thumbnail(
                (IMAGE_AREA_WIDTH, IMAGE_AREA_HEIGHT),
                resampling.LANCZOS,
            )
            x = (TILE_WIDTH - img.width) // 2
            y = image_top + (IMAGE_AREA_HEIGHT - img.height) // 2
            tile.paste(img, (x, y))
    except Exception as exc:  # Keep an index tile even if one image cannot decode.
        image_ok = False
        draw.rectangle(
            (10, image_top, TILE_WIDTH - 10, image_top + IMAGE_AREA_HEIGHT),
            outline="red", width=2,
        )
        draw.text((18, image_top + 90), f"IMAGE LOAD ERROR: {type(exc).__name__}",
                  fill="red", font=body_font)

    draw.text((10, 8), f"{error_type} #{rank:03d}", fill="black", font=title_font)
    score = row["score_float"]
    threshold = row["threshold_float"]
    filename = Path(row["relative_path"]).name
    label_top = image_top + IMAGE_AREA_HEIGHT + 4
    draw.text((10, label_top), f"score={score:.6f}  threshold={threshold:.6f}",
              fill="black", font=body_font)
    draw.text((10, label_top + 20), f"patient={row['patient_id']}",
              fill="black", font=small_font)
    # Preserve the complete relative path in the companion CSV; tile uses basename.
    max_chars = 45
    display_name = filename if len(filename) <= max_chars else filename[:max_chars - 3] + "..."
    draw.text((10, label_top + 39), display_name, fill="black", font=small_font)
    draw.rectangle((0, 0, TILE_WIDTH - 1, TILE_HEIGHT - 1), outline="#888888", width=1)
    return tile, image_ok


def build_pages(
    cases: list[dict], error_type: str, threshold: float, output_dir: Path
) -> tuple[list[Path], list[dict]]:
    page_size = COLUMNS * ROWS_PER_PAGE
    page_count = math.ceil(len(cases) / page_size) if cases else 0
    page_paths: list[Path] = []
    unreadable: list[dict] = []

    for page_index in range(page_count):
        batch = cases[page_index * page_size:(page_index + 1) * page_size]
        page_width = PAGE_MARGIN * 2 + COLUMNS * TILE_WIDTH
        page_height = HEADER_HEIGHT + PAGE_MARGIN + ROWS_PER_PAGE * TILE_HEIGHT + PAGE_MARGIN
        page = Image.new("RGB", (page_width, page_height), "#eeeeee")
        draw = ImageDraw.Draw(page)
        header_font = load_font(22, bold=True)
        sub_font = load_font(14)

        if error_type == "FALSE_POSITIVE":
            title = "FALSE POSITIVES — NORMAL predicted as PNEUMONIA"
        else:
            title = "FALSE NEGATIVES — PNEUMONIA predicted as NORMAL"
        draw.text((PAGE_MARGIN, 10), title, fill="black", font=header_font)
        start = page_index * page_size + 1
        end = start + len(batch) - 1
        draw.text(
            (PAGE_MARGIN, 43),
            f"Threshold: {threshold:.17g} | Cases {start}-{end} of {len(cases)} | "
            f"Page {page_index + 1}/{page_count}",
            fill="#333333", font=sub_font,
        )

        for local_index, row in enumerate(batch):
            global_rank = page_index * page_size + local_index + 1
            tile, image_ok = make_tile(
                row, global_rank,
                "FP" if error_type == "FALSE_POSITIVE" else "FN",
            )
            if not image_ok:
                unreadable.append(row)
            col = local_index % COLUMNS
            row_index = local_index // COLUMNS
            x = PAGE_MARGIN + col * TILE_WIDTH
            y = HEADER_HEIGHT + row_index * TILE_HEIGHT
            page.paste(tile, (x, y))

        filename = f"{ 'fp' if error_type == 'FALSE_POSITIVE' else 'fn' }_page_{page_index + 1:02d}.png"
        page_path = output_dir / filename
        page.save(page_path, format="PNG", optimize=True)
        page_paths.append(page_path)
    return page_paths, unreadable


def write_case_index(output_dir: Path, fp_cases: list[dict], fn_cases: list[dict]) -> Path:
    path = output_dir / "error_cases_index.csv"
    fields = [
        "error_type", "rank", "relative_path", "patient_id", "sha256",
        "y_true", "y_score", "y_pred_at_locked_threshold", "locked_threshold",
    ]
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for error_type, cases in (("FALSE_POSITIVE", fp_cases), ("FALSE_NEGATIVE", fn_cases)):
            for rank, row in enumerate(cases, start=1):
                writer.writerow({
                    "error_type": error_type,
                    "rank": rank,
                    "relative_path": row["relative_path"],
                    "patient_id": row["patient_id"],
                    "sha256": row["sha256"],
                    "y_true": row["y_true_int"],
                    "y_score": format(row["score_float"], ".17g"),
                    "y_pred_at_locked_threshold": row["y_pred_int"],
                    "locked_threshold": format(row["threshold_float"], ".17g"),
                })
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    predictions_path = args.predictions
    if not predictions_path.is_absolute():
        predictions_path = project_root / predictions_path
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    predictions, threshold = read_predictions(predictions_path, project_root)
    fp_cases = [r for r in predictions if r["error_type"] == "FALSE_POSITIVE"]
    fn_cases = [r for r in predictions if r["error_type"] == "FALSE_NEGATIVE"]

    # Rank high-score false alarms first; rank missed positives nearest to the cutoff first.
    fp_cases.sort(key=lambda r: (-r["score_float"], r["relative_path"]))
    fn_cases.sort(key=lambda r: (-r["score_float"], r["relative_path"]))

    if len(fp_cases) != EXPECTED_FP or len(fn_cases) != EXPECTED_FN:
        raise ValueError(
            f"Unexpected error counts: FP={len(fp_cases)} (expected {EXPECTED_FP}), "
            f"FN={len(fn_cases)} (expected {EXPECTED_FN})"
        )
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    case_index = write_case_index(output_dir, fp_cases, fn_cases)
    fp_pages, fp_unreadable = build_pages(fp_cases, "FALSE_POSITIVE", threshold, output_dir)
    fn_pages, fn_unreadable = build_pages(fn_cases, "FALSE_NEGATIVE", threshold, output_dir)

    readme = output_dir / "README.txt"
    with readme.open("x", encoding="utf-8") as f:
        f.write("CNN V2 100% — Test Error Analysis Contact Sheets\n")
        f.write("================================================\n")
        f.write(f"Threshold: {threshold:.17g}\n")
        f.write(f"False Positives: {len(fp_cases)}\n")
        f.write(f"False Negatives: {len(fn_cases)}\n")
        f.write("FP pages: NORMAL images predicted as PNEUMONIA; sorted by descending score.\n")
        f.write("FN pages: PNEUMONIA images predicted as NORMAL; sorted by descending score.\n")
        f.write("Images are read-only; thumbnails were generated for visual inspection.\n")
        f.write("No model inference, threshold tuning, or Test metric recalculation was performed.\n")
        f.write("Complete paths and identifiers are in error_cases_index.csv.\n")
        f.write(f"Image decoding errors: {len(fp_unreadable) + len(fn_unreadable)}\n")

    print("=" * 72)
    print("CNN V2 100% — TEST ERROR CONTACT SHEETS")
    print("=" * 72)
    print("Predictions:", predictions_path)
    print("Total predictions checked:", len(predictions))
    print("Locked threshold:", format(threshold, ".17g"))
    print("False Positives:", len(fp_cases), "across", len({r['patient_id'] for r in fp_cases}), "patients")
    print("False Negatives:", len(fn_cases), "across", len({r['patient_id'] for r in fn_cases}), "patients")
    print("Output directory:", output_dir)
    print("Case index:", case_index)
    print("FP contact sheets:")
    for path in fp_pages:
        print(" ", path)
    print("FN contact sheets:")
    for path in fn_pages:
        print(" ", path)
    print("Unreadable images:", len(fp_unreadable) + len(fn_unreadable))
    print("No source images or existing Test evaluation files were modified.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
