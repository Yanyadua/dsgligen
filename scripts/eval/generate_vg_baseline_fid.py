from pathlib import Path
import gc
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.concat_dataset import ConCatDataset
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from trainer import batch_to_device


DEVICE = torch.device("cuda")
BASE_CKPT = os.environ.get("BASE_CKPT", "gligen_checkpoints/diffusion_pytorch_model.bin")
DATA_YAML = os.environ.get("DATA_YAML", "configs/vg_raw_scene_graph_compatible_spatial_gat_geo.yaml")
GROUNDING_CKPT = os.environ.get("GROUNDING_CKPT")
DEFAULT_BASELINE_MODEL_YAML = "configs/vg_text_box_baseline.yaml"
MODEL_YAML = os.environ.get(
    "MODEL_YAML",
    DATA_YAML if GROUNDING_CKPT else DEFAULT_BASELINE_MODEL_YAML,
)
OUT_DIR = Path(os.environ.get("OUT_DIR", "eval_outputs/vg_baseline_fid_5k"))
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "5000"))
START_INDEX = int(os.environ.get("START_INDEX", "0"))
STEPS = int(os.environ.get("STEPS", "50"))
GUIDANCE = float(os.environ.get("GUIDANCE", "5.0"))
SEED = int(os.environ.get("SEED", "20260424"))
SAVE_SIZE = int(os.environ.get("SAVE_SIZE", "256"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_tensor_image(tensor, out_path):
    image = torch.clamp(tensor.detach().cpu(), min=-1, max=1)
    image = (image * 0.5 + 0.5).mul(255).permute(1, 2, 0).numpy().astype(np.uint8)
    pil = Image.fromarray(image)
    if SAVE_SIZE > 0 and pil.size != (SAVE_SIZE, SAVE_SIZE):
        pil = pil.resize((SAVE_SIZE, SAVE_SIZE), Image.BICUBIC)
    pil.save(out_path)


def one_item_batch(item):
    batch = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = [value]
    return batch_to_device(batch, DEVICE)


@torch.no_grad()
def encode_text_grid(text_encoder, batch, source_key, target_key):
    rows = batch.get(source_key)
    if not rows:
        return
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return
    flat = [row[i] if i < len(row) and row[i] else "" for row in rows for i in range(width)]
    _, pooled = text_encoder.encode(flat, return_pooler_output=True)
    batch[target_key] = pooled.view(len(rows), width, -1)


def load_model():
    cfg = OmegaConf.load(MODEL_YAML)
    model = instantiate_from_config(cfg.model).to(DEVICE).eval()
    autoencoder = instantiate_from_config(cfg.autoencoder).to(DEVICE).eval()
    text_encoder = instantiate_from_config(cfg.text_encoder).to(DEVICE).eval()
    diffusion = instantiate_from_config(cfg.diffusion).to(DEVICE)

    base = torch.load(BASE_CKPT, map_location="cpu")
    compatible = {
        k: v
        for k, v in base["model"].items()
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape
    }
    model.load_state_dict(compatible, strict=False)
    autoencoder.load_state_dict(base["autoencoder"])
    text_encoder.load_state_dict(base["text_encoder"], strict=False)
    diffusion.load_state_dict(base["diffusion"])

    if GROUNDING_CKPT:
        grounding_state = torch.load(GROUNDING_CKPT, map_location="cpu")
        grounding_state = grounding_state.get("model_trainable", grounding_state.get("model", {}))
        current_state = model.state_dict()
        compatible_grounding = {
            k: v
            for k, v in grounding_state.items()
            if k in current_state and current_state[k].shape == v.shape
        }
        skipped_grounding = sorted(set(grounding_state.keys()) - set(compatible_grounding.keys()))
        print(
            "GROUNDING_LOAD",
            f"ckpt={GROUNDING_CKPT}",
            f"loaded={len(compatible_grounding)}",
            f"skipped={len(skipped_grounding)}",
            flush=True,
        )
        if len(compatible_grounding) == 0:
            raise RuntimeError(
                "GROUNDING_CKPT was provided but no compatible parameters were loaded. "
                f"MODEL_YAML={MODEL_YAML} likely does not match the checkpoint architecture."
            )
        current_state.update(compatible_grounding)
        model.load_state_dict(current_state, strict=True)

    grounding_tokenizer_input = instantiate_from_config(cfg.grounding_tokenizer_input)
    model.grounding_tokenizer_input = grounding_tokenizer_input
    return model, autoencoder, text_encoder, diffusion, grounding_tokenizer_input


def load_dataset():
    cfg = OmegaConf.load(DATA_YAML)
    cfg.DATA_ROOT = "DATA"
    cfg.train_dataset_names.VGSceneGraphRaw.random_flip = False
    cfg.train_dataset_names.VGSceneGraphRaw.random_crop = False
    cfg.train_dataset_names.VGSceneGraphRaw.max_images = max(NUM_SAMPLES + START_INDEX, cfg.train_dataset_names.VGSceneGraphRaw.get("max_images", 0) or 0)
    dataset = ConCatDataset(cfg.train_dataset_names, cfg.DATA_ROOT, train=True)
    return dataset


def main():
    set_seed(SEED)
    out_real = OUT_DIR / "real"
    out_fake = OUT_DIR / "fake"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_real.mkdir(parents=True, exist_ok=True)
    out_fake.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    available = len(dataset) - START_INDEX
    count = min(NUM_SAMPLES, available)
    indices = list(range(START_INDEX, START_INDEX + count))
    print("DATASET_LEN", len(dataset), "START_INDEX", START_INDEX, "COUNT", count, flush=True)

    with open(OUT_DIR / "meta.txt", "w") as f:
        f.write(f"MODEL_YAML={MODEL_YAML}\n")
        f.write(f"DATA_YAML={DATA_YAML}\n")
        f.write(f"GROUNDING_CKPT={GROUNDING_CKPT}\n")
        f.write(f"NUM_SAMPLES={count}\n")
        f.write(f"START_INDEX={START_INDEX}\n")
        f.write(f"STEPS={STEPS}\n")
        f.write(f"GUIDANCE={GUIDANCE}\n")
        f.write(f"SEED={SEED}\n")

    model, autoencoder, text_encoder, diffusion, grounding_tokenizer_input = load_model()
    sampler = PLMSSampler(diffusion, model)
    shape = (1, 4, 64, 64)

    for local_i, dataset_idx in enumerate(indices):
        item = dataset[dataset_idx]
        image_id = f"{int(item['id'])}.png"
        out_real_path = out_real / image_id
        out_fake_path = out_fake / image_id
        if out_real_path.exists() and out_fake_path.exists():
            continue

        batch = one_item_batch(item)
        encode_text_grid(text_encoder, batch, "object_texts", "text_embeddings")
        encode_text_grid(text_encoder, batch, "relation_texts", "relation_embeddings")
        context = text_encoder.encode(batch["caption"])
        uc = text_encoder.encode([""])
        grounding_input = grounding_tokenizer_input.prepare(batch)
        input_dict = dict(
            x=torch.randn(shape, device=DEVICE),
            timesteps=None,
            context=context,
            grounding_input=grounding_input,
            inpainting_extra_input=None,
            grounding_extra_input=None,
        )
        samples = sampler.sample(S=STEPS, shape=shape, input=input_dict, uc=uc, guidance_scale=GUIDANCE)
        decoded = autoencoder.decode(samples)[0]

        save_tensor_image(batch["image"][0], out_real_path)
        save_tensor_image(decoded, out_fake_path)

        if local_i % 50 == 0:
            print("PROGRESS", local_i, "/", count, "image_id", image_id, flush=True)

        del batch, context, uc, grounding_input, input_dict, samples, decoded
        torch.cuda.empty_cache()
        gc.collect()

    print("SAVED", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
