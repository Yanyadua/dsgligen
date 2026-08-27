"""Run a small, protocol-pinned InstanceDiffusion layout-only VG probe.

This script deliberately does not compute FID/IS and does not train.  It is
for the fixed ten clean test samples exported by export_vg_instance_layout_json
and keeps graph relations out of every generation prompt.
"""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import ImageDraw


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xywh_pixels_to_xyxy_normalized(box, width, height):
    x, y, w, h = (float(value) for value in box)
    return [x / width, y / height, (x + w) / width, (y + h) / height]


def save_overlay(image, annos, path):
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for index, anno in enumerate(annos):
        x, y, w, h = anno["bbox"]
        draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=3)
        draw.text((x + 3, y + 3), f"{index}: {anno['caption']}", fill=(255, 0, 0))
    overlay.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--steps", default=50, type=int)
    parser.add_argument("--guidance", default=7.5, type=float)
    parser.add_argument("--alpha", default=0.8, type=float)
    parser.add_argument("--beta", default=0.36, type=float)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    source_manifest_path = args.input_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if "relation-free object noun phrase" not in source_manifest.get("instance_caption_policy", ""):
        raise ValueError("input manifest is not the relation-free instance-layout protocol")
    args.out_dir.mkdir(parents=True)
    raw_dir = args.out_dir / "raw"
    overlay_dir = args.out_dir / "overlay"
    raw_dir.mkdir()
    overlay_dir.mkdir()

    from diffusers import StableDiffusionINSTDIFFPipeline
    import diffusers

    random.seed(args.base_seed)
    np.random.seed(args.base_seed)
    torch.manual_seed(args.base_seed)
    torch.cuda.manual_seed_all(args.base_seed)
    pipe = StableDiffusionINSTDIFFPipeline.from_pretrained(
        str(args.snapshot),
        local_files_only=True,
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    runs = []
    for sample in source_manifest["samples"]:
        name = sample["input_json"]
        record = json.loads((args.input_dir / name).read_text(encoding="utf-8"))
        width, height = int(record["width"]), int(record["height"])
        annos = record["annos"]
        boxes = [xywh_pixels_to_xyxy_normalized(anno["bbox"], width, height) for anno in annos]
        phrases = [anno["caption"] for anno in annos]
        sample_seed = int(args.base_seed) + int(sample["test_index"])
        image = pipe(
            prompt=record["caption"],
            negative_prompt=None,
            instdiff_phrases=phrases,
            instdiff_boxes=boxes,
            instdiff_scheduled_sampling_alpha=args.alpha,
            instdiff_scheduled_sampling_beta=args.beta,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            generator=torch.Generator(device="cuda").manual_seed(sample_seed),
            output_type="pil",
        ).images[0]
        stem = Path(name).stem
        raw_path = raw_dir / f"{stem}_seed{sample_seed}.png"
        overlay_path = overlay_dir / f"{stem}_seed{sample_seed}_boxes.png"
        image.save(raw_path)
        save_overlay(image, annos, overlay_path)
        runs.append(
            {
                "test_index": sample["test_index"],
                "image_id": sample["image_id"],
                "input_json": name,
                "seed": sample_seed,
                "raw_file": str(raw_path.relative_to(args.out_dir)),
                "raw_sha256": sha256(raw_path),
                "overlay_file": str(overlay_path.relative_to(args.out_dir)),
                "objects": sample["selected_object_texts"],
                "relation_target": {
                    "texts": sample["selected_relation_texts"],
                    "edges": sample["selected_relation_edges"],
                },
            }
        )
        print(f"GENERATED test_index={sample['test_index']} seed={sample_seed}", flush=True)

    manifest = {
        "status": "exploratory fixed-10 layout-only diagnostic; not FID/IS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "kyeongry/instancediffusion_sd15",
        "snapshot": str(args.snapshot.resolve()),
        "diffusers_version": diffusers.__version__,
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256(source_manifest_path),
        "sample_count": len(runs),
        "sampler": "StableDiffusionINSTDIFFPipeline default scheduler",
        "steps": args.steps,
        "guidance": args.guidance,
        "instdiff_alpha": args.alpha,
        "instdiff_beta": args.beta,
        "base_seed": args.base_seed,
        "seed_rule": "per-sample seed = base_seed + fixed test index",
        "negative_prompt": None,
        "runs": runs,
    }
    (args.out_dir / "meta.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"DONE {len(runs)} images -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
