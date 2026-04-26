from pathlib import Path
import os
import sys

import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.concat_dataset import ConCatDataset


FAKE_DIR = Path(os.environ.get("FAKE_DIR", "eval_outputs/vg_baseline_fid_5k/fake"))
DATA_YAML = os.environ.get("DATA_YAML", "configs/vg_raw_scene_graph_compatible_spatial_gat_geo_prediction_loss.yaml")
START_INDEX = int(os.environ.get("START_INDEX", "0"))
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "200"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")


def load_dataset():
    cfg = OmegaConf.load(DATA_YAML)
    cfg.DATA_ROOT = "DATA"
    cfg.train_dataset_names.VGSceneGraphRaw.random_flip = False
    cfg.train_dataset_names.VGSceneGraphRaw.random_crop = False
    needed = max(NUM_SAMPLES + START_INDEX, cfg.train_dataset_names.VGSceneGraphRaw.get("max_images", 0) or 0)
    cfg.train_dataset_names.VGSceneGraphRaw.max_images = needed
    return ConCatDataset(cfg.train_dataset_names, cfg.DATA_ROOT, train=True)


def build_caption_map():
    dataset = load_dataset()
    caption_map = {}
    end = min(len(dataset), START_INDEX + NUM_SAMPLES)
    for idx in range(START_INDEX, end):
        item = dataset[idx]
        caption_map[str(int(item["id"]))] = item["caption"]
    return caption_map


@torch.no_grad()
def main():
    if not FAKE_DIR.exists():
        raise FileNotFoundError(f"Missing FAKE_DIR: {FAKE_DIR}")

    caption_map = build_caption_map()
    image_paths = [p for p in sorted(FAKE_DIR.glob("*.png")) if p.stem in caption_map]
    if not image_paths:
        raise FileNotFoundError(f"No matching fake images with captions found in {FAKE_DIR}")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    scores = []
    for start in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[start : start + BATCH_SIZE]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        texts = [caption_map[path.stem] for path in batch_paths]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        batch_scores = torch.clamp((image_embeds * text_embeds).sum(dim=-1), min=0.0) * 100.0
        scores.extend(batch_scores.cpu().tolist())

    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    std = variance ** 0.5
    print(f"CLIPScore mean={mean:.6f} std={std:.6f} n={len(scores)}", flush=True)


if __name__ == "__main__":
    main()
