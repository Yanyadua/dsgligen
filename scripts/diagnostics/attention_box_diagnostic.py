from pathlib import Path
import json
import os
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ldm.modules.attention import GatedSelfAttentionDense
from scripts.diagnostics.attention_box_metrics import (
    attention_inside_box_ratio,
    infer_square_grid,
)
from scripts.eval import generate_vg_fixedsplit_eval as gen
from scripts.eval.deterministic_noise import build_per_sample_noise
from scripts.eval.sample_selection import parse_sample_indices


OUT_DIR = Path(os.environ.get("OUT_DIR", "eval_outputs/attention_box_diagnostic"))
DIAGNOSTIC_TIMESTEP = int(os.environ.get("ATTENTION_DIAGNOSTIC_TIMESTEP", "500"))


def iter_gated_self_attention_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, GatedSelfAttentionDense):
            yield name, module


def set_attention_recording(model, enabled=True):
    count = 0
    for _, module in iter_gated_self_attention_modules(model):
        module.set_attention_recording(enabled)
        count += 1
    return count


def clear_attention_records(model):
    for _, module in iter_gated_self_attention_modules(model):
        module.clear_last_attention()


def summarize_layer_attention(model, batch):
    boxes = batch["boxes"].detach()
    masks = batch["masks"].detach()
    layer_summaries = []

    for name, module in iter_gated_self_attention_modules(model):
        visual_to_grounding = module.get_visual_to_grounding_attention()
        if visual_to_grounding is None:
            continue
        _, _, num_visual, num_grounding = visual_to_grounding.shape
        grid = infer_square_grid(num_visual)
        if grid is None:
            layer_summaries.append({
                "name": name,
                "num_visual_tokens": int(num_visual),
                "num_grounding_tokens": int(num_grounding),
                "skipped": "non_square_visual_tokens",
            })
            continue
        if boxes.shape[1] < num_grounding:
            layer_summaries.append({
                "name": name,
                "num_visual_tokens": int(num_visual),
                "grid": [int(grid[0]), int(grid[1])],
                "num_grounding_tokens": int(num_grounding),
                "skipped": "more_attention_grounding_tokens_than_boxes",
            })
            continue

        aligned_boxes = boxes[:, :num_grounding, :]
        aligned_masks = masks[:, :num_grounding]
        ratio, area_ratio = attention_inside_box_ratio(
            visual_to_grounding.detach(),
            aligned_boxes,
            aligned_masks,
        )
        valid = torch.isfinite(ratio)
        if valid.any():
            mean_ratio = float(ratio[valid].mean().detach().cpu())
            mean_area = float(area_ratio[valid].mean().detach().cpu())
            mean_lift = mean_ratio / max(mean_area, 1e-12)
        else:
            mean_ratio = None
            mean_area = None
            mean_lift = None

        layer_summaries.append({
            "name": name,
            "num_visual_tokens": int(num_visual),
            "grid": [int(grid[0]), int(grid[1])],
            "num_grounding_tokens": int(num_grounding),
            "valid_tokens": int(valid.sum().detach().cpu()),
            "mean_attention_inside_box_ratio": mean_ratio,
            "mean_box_area_ratio": mean_area,
            "mean_attention_lift_over_area": mean_lift,
        })
    return layer_summaries


@torch.no_grad()
def run_batch(model, text_encoder, grounding_tokenizer_input, items):
    batch = gen.items_to_batch(items)
    batch_size = len(items)
    gen.encode_text_grid(text_encoder, batch, "object_texts", "text_embeddings")
    gen.encode_text_grid(text_encoder, batch, "relation_texts", "relation_embeddings")
    context = text_encoder.encode(batch["caption"])
    grounding_input = grounding_tokenizer_input.prepare(batch)
    x = build_per_sample_noise(
        image_ids=[int(item["id"]) for item in items],
        sample_shape=(4, 64, 64),
        base_seed=gen.SEED,
        device=gen.DEVICE,
    )
    timesteps = torch.full(
        (batch_size,),
        DIAGNOSTIC_TIMESTEP,
        device=gen.DEVICE,
        dtype=torch.long,
    )
    clear_attention_records(model)
    _ = model(dict(
        x=x,
        timesteps=timesteps,
        context=context,
        inpainting_extra_input=None,
        grounding_extra_input=None,
        grounding_input=grounding_input,
    ))
    return batch, summarize_layer_attention(model, batch)


def main():
    gen.set_seed(gen.SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = gen.VGFixedSplitDataset(gen.H5_PATH, gen.VOCAB_PATH, gen.IMAGE_ROOT)
    indices = parse_sample_indices(gen.SAMPLE_INDICES, len(dataset))
    if indices is None:
        count = min(gen.NUM_SAMPLES, len(dataset) - gen.START_INDEX)
        indices = list(range(gen.START_INDEX, gen.START_INDEX + count))

    model, _, text_encoder, _, grounding_tokenizer_input = gen.load_model()
    fuser_count = set_attention_recording(model, True)
    if fuser_count == 0:
        raise RuntimeError("No GatedSelfAttentionDense modules found; cannot diagnose fuser attention")

    per_batch = []
    layer_accumulator = {}
    pending = []
    for dataset_index in indices:
        pending.append(dataset[dataset_index])
        if len(pending) < gen.EVAL_BATCH_SIZE:
            continue
        _, layers = run_batch(model, text_encoder, grounding_tokenizer_input, pending)
        per_batch.append({
            "image_ids": [int(item["id"]) for item in pending],
            "layers": layers,
        })
        for layer in layers:
            if "mean_attention_inside_box_ratio" not in layer:
                continue
            if layer["mean_attention_inside_box_ratio"] is None:
                continue
            slot = layer_accumulator.setdefault(layer["name"], {
                "ratios": [],
                "areas": [],
                "lifts": [],
                "grid": layer.get("grid"),
                "num_visual_tokens": layer.get("num_visual_tokens"),
            })
            slot["ratios"].append(layer["mean_attention_inside_box_ratio"])
            slot["areas"].append(layer["mean_box_area_ratio"])
            slot["lifts"].append(layer["mean_attention_lift_over_area"])
        print("PROGRESS", len(per_batch), "batches", flush=True)
        pending = []

    if pending:
        _, layers = run_batch(model, text_encoder, grounding_tokenizer_input, pending)
        per_batch.append({
            "image_ids": [int(item["id"]) for item in pending],
            "layers": layers,
        })
        for layer in layers:
            if "mean_attention_inside_box_ratio" not in layer:
                continue
            if layer["mean_attention_inside_box_ratio"] is None:
                continue
            slot = layer_accumulator.setdefault(layer["name"], {
                "ratios": [],
                "areas": [],
                "lifts": [],
                "grid": layer.get("grid"),
                "num_visual_tokens": layer.get("num_visual_tokens"),
            })
            slot["ratios"].append(layer["mean_attention_inside_box_ratio"])
            slot["areas"].append(layer["mean_box_area_ratio"])
            slot["lifts"].append(layer["mean_attention_lift_over_area"])

    layer_summary = []
    for name, values in sorted(layer_accumulator.items()):
        layer_summary.append({
            "name": name,
            "grid": values["grid"],
            "num_visual_tokens": values["num_visual_tokens"],
            "mean_attention_inside_box_ratio": sum(values["ratios"]) / len(values["ratios"]),
            "mean_box_area_ratio": sum(values["areas"]) / len(values["areas"]),
            "mean_attention_lift_over_area": sum(values["lifts"]) / len(values["lifts"]),
            "num_batches": len(values["ratios"]),
        })

    payload = {
        "protocol": "diagnostic_only_not_formal_eval",
        "sample_indices": indices,
        "num_samples": len(indices),
        "eval_batch_size": gen.EVAL_BATCH_SIZE,
        "checkpoint": str(gen.GROUNDING_CKPT),
        "base_checkpoint": str(gen.BASE_CKPT),
        "model_yaml": str(gen.MODEL_YAML),
        "data_yaml": str(gen.DATA_YAML),
        "timestep": DIAGNOSTIC_TIMESTEP,
        "fuser_count": fuser_count,
        "layer_summary": layer_summary,
        "per_batch": per_batch,
    }
    out_path = OUT_DIR / "attention_box_diagnostic.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print("SAVED", out_path, flush=True)


if __name__ == "__main__":
    main()
