from pathlib import Path
import gc
import json
import os
import random
import sys

import h5py
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_vg_scene_graph import (
    DEFAULT_GENERIC_OBJECTS,
    DEFAULT_PRIORITY_OBJECTS,
    compute_relation_geo_features,
)
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from trainer import batch_to_device


DEVICE = torch.device("cuda")
BASE_CKPT = os.environ.get("BASE_CKPT", "gligen_checkpoints/diffusion_pytorch_model.bin")
DATA_YAML = os.environ.get(
    "DATA_YAML",
    "configs/vg_raw_scene_graph_compatible_spatial_gat_geo_prediction_loss.yaml",
)
GROUNDING_CKPT = os.environ.get("GROUNDING_CKPT")
DEFAULT_BASELINE_MODEL_YAML = "configs/vg_text_box_baseline.yaml"
MODEL_YAML = os.environ.get(
    "MODEL_YAML",
    DATA_YAML if GROUNDING_CKPT else DEFAULT_BASELINE_MODEL_YAML,
)
H5_PATH = Path(
    os.environ.get(
        "H5_PATH",
        "/root/autodl-tmp/fixed_split_work/datasets/vg/test.h5",
    )
)
VOCAB_PATH = Path(
    os.environ.get(
        "VOCAB_PATH",
        "/root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json",
    )
)
IMAGE_ROOT = Path(
    os.environ.get(
        "IMAGE_ROOT",
        "/root/autodl-tmp/fixed_split_work/datasets/vg/images",
    )
)
OUT_DIR = Path(os.environ.get("OUT_DIR", "eval_outputs/vg_fixedsplit_fid_1000"))
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "1000"))
START_INDEX = int(os.environ.get("START_INDEX", "0"))
STEPS = int(os.environ.get("STEPS", "50"))
GUIDANCE = float(os.environ.get("GUIDANCE", "5.0"))
SEED = int(os.environ.get("SEED", "20260429"))
SAVE_SIZE = int(os.environ.get("SAVE_SIZE", "256"))
MAX_CAPTION_OBJECTS = int(os.environ.get("MAX_CAPTION_OBJECTS", "8"))
MAX_CAPTION_RELATIONS = int(os.environ.get("MAX_CAPTION_RELATIONS", "4"))
SPLIT_NAME = os.environ.get("SPLIT_NAME", "test")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_real_image(image_path, out_path):
    image = Image.open(image_path).convert("RGB")
    if SAVE_SIZE > 0 and image.size != (SAVE_SIZE, SAVE_SIZE):
        image = image.resize((SAVE_SIZE, SAVE_SIZE), Image.BICUBIC)
    image.save(out_path)


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


class VGFixedSplitDataset:
    def __init__(self, h5_path, vocab_path, image_root):
        self.h5_path = Path(h5_path)
        self.vocab_path = Path(vocab_path)
        self.image_root = Path(image_root)
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.object_idx_to_name = vocab["object_idx_to_name"]
        self.pred_idx_to_name = vocab["pred_idx_to_name"]
        self.generic_object_names = set(DEFAULT_GENERIC_OBJECTS)
        self.priority_object_names = set(DEFAULT_PRIORITY_OBJECTS)
        self.h5 = h5py.File(self.h5_path, "r")
        assert "image_ids" in self.h5 and "image_paths" in self.h5, "Invalid VG fixed-split h5 file"
        assert "object_names" in self.h5 and "relationship_predicates" in self.h5, "Missing object/relation fields"

    def __len__(self):
        return int(self.h5["image_ids"].shape[0])

    def _decode_path(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _caption_from_graph(self, object_texts, relation_edges, relation_masks, relation_texts):
        valid_object_texts = [text for text in object_texts if text]
        valid_relation_texts = [
            f"{object_texts[int(src)]} {relation_texts[i]} {object_texts[int(dst)]}"
            for i, (src, dst) in enumerate(relation_edges.long().tolist())
            if relation_masks[i] > 0 and object_texts[int(src)] and object_texts[int(dst)] and relation_texts[i]
        ]
        caption_object_texts = sorted(
            valid_object_texts,
            key=lambda text: (text in self.priority_object_names, text not in self.generic_object_names),
            reverse=True,
        )
        object_part = ", ".join(caption_object_texts[:MAX_CAPTION_OBJECTS])
        relation_part = ". ".join(valid_relation_texts[:MAX_CAPTION_RELATIONS])
        if object_part and relation_part:
            return f"A scene with {object_part}. {relation_part}."
        if object_part:
            return f"A scene with {object_part}."
        return "A scene with objects."

    def __getitem__(self, index):
        image_id = int(self.h5["image_ids"][index])
        rel_path = self._decode_path(self.h5["image_paths"][index])
        image_path = self.image_root / rel_path

        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size

        num_objects = int(self.h5["objects_per_image"][index])
        num_relations = int(self.h5["relationships_per_image"][index])
        max_boxes = int(self.h5["object_names"].shape[1])
        max_relations = int(self.h5["relationship_predicates"].shape[1])

        boxes = torch.zeros(max_boxes, 4, dtype=torch.float32)
        masks = torch.zeros(max_boxes, dtype=torch.float32)
        object_texts = [""] * max_boxes

        object_names = self.h5["object_names"][index][:num_objects].tolist()
        object_boxes_xywh = self.h5["object_boxes"][index][:num_objects]
        if object_names:
            max_object_idx = max(int(idx) for idx in object_names)
            if max_object_idx >= len(self.object_idx_to_name):
                raise IndexError(
                    f"object idx {max_object_idx} exceeds vocab size {len(self.object_idx_to_name)}"
                )

        for obj_idx, (name_idx, xywh) in enumerate(zip(object_names, object_boxes_xywh)):
            x, y, w, h = [float(v) for v in xywh.tolist()]
            x1 = x / max(width, 1)
            y1 = y / max(height, 1)
            x2 = (x + w) / max(width, 1)
            y2 = (y + h) / max(height, 1)
            boxes[obj_idx] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
            masks[obj_idx] = 1.0
            object_texts[obj_idx] = str(self.object_idx_to_name[int(name_idx)]).lower()

        relation_edges = torch.zeros(max_relations, 2, dtype=torch.float32)
        relation_masks = torch.zeros(max_relations, dtype=torch.float32)
        relation_geo_features = torch.zeros(max_relations, 12, dtype=torch.float32)
        relation_texts = [""] * max_relations

        rel_subjects = self.h5["relationship_subjects"][index][:num_relations].tolist()
        rel_predicates = self.h5["relationship_predicates"][index][:num_relations].tolist()
        rel_objects = self.h5["relationship_objects"][index][:num_relations].tolist()
        if rel_predicates:
            max_pred_idx = max(int(idx) for idx in rel_predicates)
            if max_pred_idx >= len(self.pred_idx_to_name):
                raise IndexError(
                    f"predicate idx {max_pred_idx} exceeds vocab size {len(self.pred_idx_to_name)}"
                )
        for rel_idx, (src, pred_idx, dst) in enumerate(zip(rel_subjects, rel_predicates, rel_objects)):
            src = int(src)
            dst = int(dst)
            relation_edges[rel_idx] = torch.tensor([src, dst], dtype=torch.float32)
            relation_masks[rel_idx] = 1.0
            relation_texts[rel_idx] = str(self.pred_idx_to_name[int(pred_idx)]).lower()
            relation_geo_features[rel_idx] = compute_relation_geo_features(boxes[src], boxes[dst])

        caption = self._caption_from_graph(object_texts, relation_edges, relation_masks, relation_texts)

        return {
            "id": image_id,
            "image_path": str(image_path),
            "caption": caption,
            "boxes": boxes,
            "masks": masks,
            "object_texts": object_texts,
            "relation_edges": relation_edges,
            "relation_masks": relation_masks,
            "relation_geo_features": relation_geo_features,
            "relation_texts": relation_texts,
        }


def main():
    set_seed(SEED)
    out_real = OUT_DIR / "real"
    out_fake = OUT_DIR / "fake"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_real.mkdir(parents=True, exist_ok=True)
    out_fake.mkdir(parents=True, exist_ok=True)

    dataset = VGFixedSplitDataset(H5_PATH, VOCAB_PATH, IMAGE_ROOT)
    available = len(dataset) - START_INDEX
    count = min(NUM_SAMPLES, available)
    indices = list(range(START_INDEX, START_INDEX + count))
    print("DATASET_LEN", len(dataset), "START_INDEX", START_INDEX, "COUNT", count, flush=True)

    with open(OUT_DIR / "meta.txt", "w", encoding="utf-8") as f:
        f.write("PROTOCOL=sg2i_fixed_split\n")
        f.write(f"SPLIT_NAME={SPLIT_NAME}\n")
        f.write(f"SUBSET=first_{count}_samples_from_{SPLIT_NAME}_split_starting_at_{START_INDEX}\n")
        f.write(f"MODEL_YAML={MODEL_YAML}\n")
        f.write(f"DATA_YAML={DATA_YAML}\n")
        f.write(f"GROUNDING_CKPT={GROUNDING_CKPT}\n")
        f.write(f"H5_PATH={H5_PATH}\n")
        f.write(f"VOCAB_PATH={VOCAB_PATH}\n")
        f.write(f"IMAGE_ROOT={IMAGE_ROOT}\n")
        f.write(f"NUM_SAMPLES={count}\n")
        f.write(f"START_INDEX={START_INDEX}\n")
        f.write(f"STEPS={STEPS}\n")
        f.write(f"GUIDANCE={GUIDANCE}\n")
        f.write(f"SEED={SEED}\n")
        f.write(f"SAVE_SIZE={SAVE_SIZE}\n")

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

        save_real_image(item["image_path"], out_real_path)
        save_tensor_image(decoded, out_fake_path)

        if local_i % 50 == 0:
            print("PROGRESS", local_i, "/", count, "image_id", image_id, flush=True)

        del batch, context, uc, grounding_input, input_dict, samples, decoded
        torch.cuda.empty_cache()
        gc.collect()

    print("SAVED", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
