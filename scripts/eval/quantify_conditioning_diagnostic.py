#!/usr/bin/env python
"""Quantify short VG conditioning diagnostics without external model deps.

This is not a paper metric script. It summarizes image-level health signals
and metadata coverage for fixed small diagnostic runs, so we can decide whether
a conditioning change is worth scaling to CLIP/detector/FID evaluation.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def colorfulness(path):
    image = Image.open(path).convert("RGB").resize((256, 256))
    array = np.asarray(image).astype(np.float32)
    r, g, b = array[..., 0], array[..., 1], array[..., 2]
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )


def image_stats(path):
    image = Image.open(path).convert("RGB").resize((256, 256))
    rgb = np.asarray(image).astype(np.float32)
    gray = np.asarray(image.convert("L").resize((256, 256))).astype(np.float32)
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum())
    return {
        "colorfulness": colorfulness(path),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness_lapvar": float(lap.var()),
        "entropy": entropy,
        "rgb_std": float(rgb.std()),
    }


def mean_abs(path_a, path_b):
    a = np.asarray(Image.open(path_a).convert("RGB").resize((256, 256))).astype(
        np.float32
    )
    b = np.asarray(Image.open(path_b).convert("RGB").resize((256, 256))).astype(
        np.float32
    )
    return float(np.abs(a - b).mean())


def box_area(box):
    if not box or len(box) < 4:
        return 0.0
    x0, y0, x1, y1 = [float(value) for value in box[:4]]
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def load_metadata(run_dir, image_id):
    path = run_dir / "sample_metadata" / f"{image_id}.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_metadata(metadata):
    objects = metadata.get("objects", [])
    object_items = [obj for obj in objects if obj.get("role") == "object"]
    relation_token_items = [obj for obj in objects if obj.get("role") == "relation"]
    relations = metadata.get("relations", [])
    category_counts = {}
    category_areas = {}
    for obj in object_items:
        category = obj.get("category") or "unknown"
        category_counts[category] = category_counts.get(category, 0) + 1
        category_areas[category] = category_areas.get(category, 0.0) + box_area(
            obj.get("box_xyxy", [])
        )
    return {
        "caption_len": len(str(metadata.get("caption", "")).split()),
        "object_count": len(object_items),
        "relation_token_count": len(relation_token_items),
        "relation_count": len(relations),
        "foreground_count": category_counts.get("foreground", 0),
        "support_count": category_counts.get("support", 0),
        "background_count": category_counts.get("background", 0),
        "other_count": category_counts.get("other", 0),
        "unknown_count": category_counts.get("unknown", 0),
        "foreground_area": category_areas.get("foreground", 0.0),
        "support_area": category_areas.get("support", 0.0),
        "background_area": category_areas.get("background", 0.0),
        "other_area": category_areas.get("other", 0.0),
        "has_relation": 1.0 if relations else 0.0,
        "has_relation_token": 1.0 if relation_token_items else 0.0,
    }


def collect_run(name, run_dir, legacy_dir=None, baseline_dir=None):
    fake_dir = run_dir / "fake"
    real_dir = run_dir / "real"
    image_ids = sorted(path.stem for path in fake_dir.glob("*.png"))
    rows = []
    for image_id in image_ids:
        fake_path = fake_dir / f"{image_id}.png"
        real_path = real_dir / f"{image_id}.png"
        row = {"run": name, "image_id": int(image_id)}
        row.update({f"fake_{k}": v for k, v in image_stats(fake_path).items()})
        if real_path.exists():
            row.update({f"real_{k}": v for k, v in image_stats(real_path).items()})
            row["mean_abs_vs_real_255"] = mean_abs(fake_path, real_path)
        if legacy_dir is not None:
            legacy_path = legacy_dir / "fake" / f"{image_id}.png"
            if legacy_path.exists():
                row["mean_abs_vs_legacy_255"] = mean_abs(fake_path, legacy_path)
        if baseline_dir is not None:
            baseline_path = baseline_dir / "fake" / f"{image_id}.png"
            if baseline_path.exists():
                row["mean_abs_vs_baseline_255"] = mean_abs(fake_path, baseline_path)
        row.update(summarize_metadata(load_metadata(run_dir, image_id)))
        rows.append(row)
    return rows


def mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarize_rows(rows):
    numeric_keys = sorted(
        key
        for row in rows
        for key, value in row.items()
        if key not in {"run", "image_id"} and isinstance(value, (int, float))
    )
    summary = {"count": len(rows)}
    for key in numeric_keys:
        values = [row.get(key) for row in rows if row.get(key) is not None]
        summary[f"mean_{key}"] = mean(values)
    return summary


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path, summaries):
    columns = [
        ("count", "N"),
        ("mean_fake_colorfulness", "Color"),
        ("mean_fake_sharpness_lapvar", "Sharp"),
        ("mean_mean_abs_vs_real_255", "FakeRealL1"),
        ("mean_mean_abs_vs_legacy_255", "VsLegacyL1"),
        ("mean_mean_abs_vs_baseline_255", "VsInternalL1"),
        ("mean_object_count", "Obj"),
        ("mean_foreground_count", "FG"),
        ("mean_support_count", "Support"),
        ("mean_background_count", "BG"),
        ("mean_relation_count", "Rel"),
        ("mean_relation_token_count", "RelTok"),
        ("mean_has_relation_token", "RelTokRate"),
    ]
    lines = ["| Run | " + " | ".join(label for _, label in columns) + " |"]
    lines.append("|---|" + "|".join("---:" for _ in columns) + "|")
    for name, summary in summaries.items():
        values = []
        for key, _ in columns:
            value = summary.get(key)
            if value is None:
                values.append("")
            elif key == "count":
                values.append(str(int(value)))
            else:
                values.append(f"{float(value):.2f}")
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_run(value):
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/to/run")
    parser.add_argument("--legacy-run", help="NAME matching the GLIGEN-VG legacy baseline")
    parser.add_argument("--baseline-run", help="NAME matching one of --run values")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    runs = [parse_run(value) for value in args.run]
    run_map = {name: path for name, path in runs}
    legacy_dir = run_map.get(args.legacy_run) if args.legacy_run else None
    baseline_dir = run_map.get(args.baseline_run) if args.baseline_run else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = {}
    for name, run_dir in runs:
        rows = collect_run(
            name,
            run_dir,
            legacy_dir=legacy_dir,
            baseline_dir=baseline_dir,
        )
        all_rows.extend(rows)
        summaries[name] = summarize_rows(rows)

    write_csv(out_dir / "conditioning_diagnostic_per_image.csv", all_rows)
    (out_dir / "conditioning_diagnostic_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "conditioning_diagnostic_summary.md", summaries)
    print(out_dir / "conditioning_diagnostic_summary.md")


if __name__ == "__main__":
    main()
