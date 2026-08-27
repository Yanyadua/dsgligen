#!/usr/bin/env python
"""CLIP-based crop grounding diagnostics for VG generated images.

This is an exploratory proxy metric, not a detector AP replacement. It checks
whether object-conditioned boxes are locally compatible with their object text.
The script reads the per-sample metadata emitted by generate_vg_fixedsplit_eval.py.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def box_area(box):
    x0, y0, x1, y1 = [float(value) for value in box[:4]]
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def box_iou(a, b):
    ax0, ay0, ax1, ay1 = [float(value) for value in a[:4]]
    bx0, by0, bx1, by1 = [float(value) for value in b[:4]]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def clamp_box(box):
    x0, y0, x1, y1 = [float(value) for value in box[:4]]
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    return (x0, y0, x1, y1)


def crop_normalized(image, box, min_pixels=8):
    width, height = image.size
    x0, y0, x1, y1 = clamp_box(box)
    left = int(round(x0 * width))
    top = int(round(y0 * height))
    right = int(round(x1 * width))
    bottom = int(round(y1 * height))
    if right - left < min_pixels:
        pad = (min_pixels - (right - left)) // 2 + 1
        left = max(left - pad, 0)
        right = min(right + pad, width)
    if bottom - top < min_pixels:
        pad = (min_pixels - (bottom - top)) // 2 + 1
        top = max(top - pad, 0)
        bottom = min(bottom + pad, height)
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


def negative_boxes_like(box):
    x0, y0, x1, y1 = clamp_box(box)
    width = max(x1 - x0, 0.05)
    height = max(y1 - y0, 0.05)
    anchors = [
        (0.0, 0.0),
        (1.0 - width, 0.0),
        (0.0, 1.0 - height),
        (1.0 - width, 1.0 - height),
        ((1.0 - width) * 0.5, (1.0 - height) * 0.5),
    ]
    boxes = []
    for nx0, ny0 in anchors:
        candidate = clamp_box((nx0, ny0, nx0 + width, ny0 + height))
        if box_iou(candidate, (x0, y0, x1, y1)) < 0.10:
            boxes.append(candidate)
    return boxes


def parse_run(value):
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def load_metadata(run_dir, image_id):
    with open(run_dir / "sample_metadata" / f"{image_id}.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_clip(model_name, device):
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=True)
    image_size = int(getattr(model.config.vision_config, "image_size", 224))
    return torch, model, tokenizer, image_size


def preprocess_clip_image(torch, image, image_size):
    # CLIP ViT preprocessing: bicubic resize, center crop, [0,1] tensor, normalize.
    width, height = image.size
    scale = image_size / min(width, height)
    resized = (
        max(int(round(width * scale)), image_size),
        max(int(round(height * scale)), image_size),
    )
    image = image.resize(resized, Image.BICUBIC)
    left = max((image.size[0] - image_size) // 2, 0)
    top = max((image.size[1] - image_size) // 2, 0)
    image = image.crop((left, top, left + image_size, top + image_size))
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    return (tensor - mean) / std


def encode_images(torch, model, image_size, images, device, batch_size):
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            pixel_values = torch.stack(
                [preprocess_clip_image(torch, image, image_size) for image in batch],
                dim=0,
            ).to(device)
            features = model.get_image_features(pixel_values=pixel_values)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            embeddings.append(features.detach().cpu())
    return torch.cat(embeddings, dim=0)


def encode_texts(torch, model, tokenizer, texts, device, batch_size):
    embeddings = []
    prompts = [f"a photo of {text}" for text in texts]
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            features = model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            embeddings.append(features.detach().cpu())
    return torch.cat(embeddings, dim=0)


def eligible_objects(metadata, min_area):
    objects = []
    for obj in metadata.get("objects", []):
        if obj.get("role") != "object":
            continue
        text = str(obj.get("text", "")).strip()
        box = obj.get("box_xyxy", [])
        if not text or len(box) < 4:
            continue
        if box_area(clamp_box(box)) < min_area:
            continue
        objects.append(obj)
    return objects


def score_run(torch, model, tokenizer, image_size, name, run_dir, device, batch_size, min_area):
    fake_dir = run_dir / "fake"
    image_ids = sorted(
        (path.stem for path in fake_dir.glob("*.png")),
        key=lambda value: int(value),
    )
    per_object = []
    per_image = []

    for image_id in image_ids:
        image = Image.open(fake_dir / f"{image_id}.png").convert("RGB")
        metadata = load_metadata(run_dir, image_id)
        objects = eligible_objects(metadata, min_area=min_area)
        if not objects:
            per_image.append(
                {
                    "run": name,
                    "image_id": int(image_id),
                    "object_count": 0,
                    "mean_crop_text": None,
                    "mean_box_contrast": None,
                    "diag_top1": None,
                }
            )
            continue

        crops = []
        crop_objects = []
        neg_crops = []
        neg_object_indices = []
        for obj_idx, obj in enumerate(objects):
            crop = crop_normalized(image, obj["box_xyxy"])
            if crop is None:
                continue
            crop_objects.append(obj)
            crops.append(crop)
            for neg_box in negative_boxes_like(obj["box_xyxy"]):
                neg = crop_normalized(image, neg_box)
                if neg is not None:
                    neg_object_indices.append(len(crop_objects) - 1)
                    neg_crops.append(neg)

        if not crops:
            continue

        texts = [str(obj.get("text", "")).strip() for obj in crop_objects]
        image_embeddings = encode_images(
            torch,
            model,
            image_size,
            crops,
            device,
            batch_size,
        )
        text_embeddings = encode_texts(
            torch,
            model,
            tokenizer,
            texts,
            device,
            batch_size,
        )
        score_matrix = image_embeddings @ text_embeddings.T
        diag_scores = score_matrix.diag().numpy()
        top1_hits = (score_matrix.argmax(dim=0).numpy() == np.arange(len(texts))).astype(np.float32)

        neg_scores_by_obj = {idx: [] for idx in range(len(crop_objects))}
        if neg_crops:
            neg_embeddings = encode_images(
                torch,
                model,
                image_size,
                neg_crops,
                device,
                batch_size,
            )
            for neg_idx, obj_idx in enumerate(neg_object_indices):
                score = float((neg_embeddings[neg_idx] @ text_embeddings[obj_idx]).item())
                neg_scores_by_obj[obj_idx].append(score)

        image_object_rows = []
        for idx, obj in enumerate(crop_objects):
            neg_scores = neg_scores_by_obj.get(idx, [])
            max_neg = max(neg_scores) if neg_scores else None
            contrast = float(diag_scores[idx] - max_neg) if max_neg is not None else None
            row = {
                "run": name,
                "image_id": int(image_id),
                "object_index": int(obj.get("index", idx)),
                "text": str(obj.get("text", "")),
                "category": str(obj.get("category", "") or "unknown"),
                "box_area": box_area(clamp_box(obj.get("box_xyxy", []))),
                "crop_text_score": float(diag_scores[idx]),
                "diag_top1": float(top1_hits[idx]),
                "max_negative_crop_score": max_neg,
                "box_contrast_score": contrast,
            }
            image_object_rows.append(row)
            per_object.append(row)

        contrasts = [
            row["box_contrast_score"]
            for row in image_object_rows
            if row["box_contrast_score"] is not None
        ]
        per_image.append(
            {
                "run": name,
                "image_id": int(image_id),
                "object_count": len(image_object_rows),
                "mean_crop_text": float(np.mean([row["crop_text_score"] for row in image_object_rows])),
                "mean_box_contrast": float(np.mean(contrasts)) if contrasts else None,
                "diag_top1": float(np.mean([row["diag_top1"] for row in image_object_rows])),
            }
        )

    return per_image, per_object


def mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarize_per_image(rows):
    return {
        "image_count": len(rows),
        "mean_object_count": mean(row.get("object_count") for row in rows),
        "mean_crop_text": mean(row.get("mean_crop_text") for row in rows),
        "mean_box_contrast": mean(row.get("mean_box_contrast") for row in rows),
        "mean_diag_top1": mean(row.get("diag_top1") for row in rows),
    }


def summarize_per_object(rows):
    summary = {
        "object_count": len(rows),
        "mean_crop_text": mean(row.get("crop_text_score") for row in rows),
        "mean_box_contrast": mean(row.get("box_contrast_score") for row in rows),
        "mean_diag_top1": mean(row.get("diag_top1") for row in rows),
    }
    by_category = {}
    for row in rows:
        by_category.setdefault(row.get("category") or "unknown", []).append(row)
    for category, category_rows in by_category.items():
        summary[f"{category}_object_count"] = len(category_rows)
        summary[f"{category}_mean_crop_text"] = mean(
            row.get("crop_text_score") for row in category_rows
        )
        summary[f"{category}_mean_box_contrast"] = mean(
            row.get("box_contrast_score") for row in category_rows
        )
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
        ("image_count", "Nimg"),
        ("object_count", "Nobj"),
        ("mean_object_count", "ObjImg"),
        ("mean_crop_text", "CropText"),
        ("mean_box_contrast", "BoxContrast"),
        ("mean_diag_top1", "DiagTop1"),
        ("foreground_mean_crop_text", "FGCrop"),
        ("support_mean_crop_text", "SupportCrop"),
        ("background_mean_crop_text", "BGCrop"),
    ]
    lines = ["| Run | " + " | ".join(label for _, label in columns) + " |"]
    lines.append("|---|" + "|".join("---:" for _ in columns) + "|")
    for name, summary in summaries.items():
        values = []
        for key, _ in columns:
            value = summary.get(key)
            if value is None:
                values.append("")
            elif key in {"image_count", "object_count"}:
                values.append(str(int(value)))
            else:
                values.append(f"{float(value):.4f}")
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/to/run")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-area", type=float, default=0.0025)
    args = parser.parse_args()

    torch, model, tokenizer, image_size = load_clip(args.model, args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image_all = []
    per_object_all = []
    summaries = {}
    for run_value in args.run:
        name, run_dir = parse_run(run_value)
        per_image, per_object = score_run(
            torch,
            model,
            tokenizer,
            image_size,
            name,
            run_dir,
            device=args.device,
            batch_size=args.batch_size,
            min_area=args.min_area,
        )
        per_image_all.extend(per_image)
        per_object_all.extend(per_object)
        summaries[name] = {
            **summarize_per_image(per_image),
            **summarize_per_object(per_object),
        }

    write_csv(out_dir / "clip_crop_per_image.csv", per_image_all)
    write_csv(out_dir / "clip_crop_per_object.csv", per_object_all)
    (out_dir / "clip_crop_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "clip_crop_summary.md", summaries)
    print(out_dir / "clip_crop_summary.md")


if __name__ == "__main__":
    main()
