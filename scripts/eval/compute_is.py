from pathlib import Path
import math
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights


FAKE_DIR = Path(os.environ.get("FAKE_DIR", "eval_outputs/vg_baseline_fid_5k/fake"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
SPLITS = int(os.environ.get("SPLITS", "10"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_images():
    image_paths = sorted([p for p in FAKE_DIR.glob("*.png")])
    if not image_paths:
        raise FileNotFoundError(f"No png files found in {FAKE_DIR}")

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]
    )
    tensors = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        tensors.append(transform(image))
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def main():
    images = load_images()
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(DEVICE).eval()
    preds = []
    for start in range(0, images.shape[0], BATCH_SIZE):
        batch = images[start : start + BATCH_SIZE].to(DEVICE)
        logits = model(batch)
        preds.append(torch.softmax(logits, dim=1).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    split_scores = []
    splits = max(1, min(SPLITS, preds.shape[0]))
    for part in np.array_split(preds, splits):
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-12) - np.log(py + 1e-12))
        split_scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))

    mean = float(np.mean(split_scores))
    std = float(np.std(split_scores))
    print(f"IS mean={mean:.6f} std={std:.6f} n={preds.shape[0]} splits={splits}", flush=True)


if __name__ == "__main__":
    main()
