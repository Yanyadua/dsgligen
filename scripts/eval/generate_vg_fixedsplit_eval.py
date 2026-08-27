from pathlib import Path
import gc
import hashlib
import json
import os
import random
import sys
from functools import partial

import h5py
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_vg_scene_graph import (
    compute_relation_geo_features,
)
from dataset.relation_grounding_tokens import (
    append_relation_grounding_tokens,
    parse_allowed_predicates,
)
from dataset.scene_graph_conditioning import build_clean_scene_graph_condition
from dataset.scene_graph_caption import (
    build_clean_scene_graph_caption,
    build_clean_primary_scene_graph_caption,
    build_natural_scene_graph_caption,
    build_scene_graph_caption,
)
from dataset.scene_graph_box_utils import (
    compute_center_crop_transform,
    transform_scene_graph_annotations,
)
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from scripts.eval.recovery_checks import (
    ProtocolMismatchError,
    validate_base_model_compatibility,
    validate_checkpoint_trainable_manifest,
    validate_grounding_state,
    validate_resume_metadata,
)
from scripts.eval.grounding_alpha import (
    build_grounding_alpha_schedule,
    parse_grounding_alpha_type,
    set_grounding_alpha_scale,
)
from scripts.eval.inference_ablation import (
    apply_fuser_alpha_multiplier,
    apply_fuser_alpha_profile,
    apply_graph_gate_override,
    parse_fuser_alpha_profile,
    restore_base_fuser_state,
    summarize_fuser_alpha,
)
from scripts.eval.deterministic_noise import build_per_sample_noise
from scripts.eval.sample_selection import parse_sample_indices
from scripts.eval.scene_graph_metadata import write_sample_metadata
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
SAMPLE_INDICES = os.environ.get("SAMPLE_INDICES")
SAMPLER_NAME = os.environ.get("SAMPLER", "plms").strip().lower()
STEPS = int(os.environ.get("STEPS", "50"))
GUIDANCE = float(os.environ.get("GUIDANCE", "5.0"))
GROUNDING_ALPHA_TYPE = parse_grounding_alpha_type(
    os.environ.get("GROUNDING_ALPHA_TYPE", "1,0,0")
)
SEED = int(os.environ.get("SEED", "20260429"))
SAVE_SIZE = int(os.environ.get("SAVE_SIZE", "256"))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "1"))
RESTORE_BASE_FUSER = os.environ.get(
    "RESTORE_BASE_FUSER",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}
GRAPH_GATE_OVERRIDE_VALUE = os.environ.get("GRAPH_GATE_OVERRIDE")
GRAPH_GATE_OVERRIDE = (
    None
    if GRAPH_GATE_OVERRIDE_VALUE in {None, ""}
    else float(GRAPH_GATE_OVERRIDE_VALUE)
)
TRIPLET_GATE_OVERRIDE_VALUE = os.environ.get("TRIPLET_GATE_OVERRIDE")
TRIPLET_GATE_OVERRIDE = (
    None
    if TRIPLET_GATE_OVERRIDE_VALUE in {None, ""}
    else float(TRIPLET_GATE_OVERRIDE_VALUE)
)
FUSER_ALPHA_ATTN_MULTIPLIER = float(
    os.environ.get("FUSER_ALPHA_ATTN_MULTIPLIER", "1.0")
)
FUSER_ALPHA_DENSE_MULTIPLIER = float(
    os.environ.get("FUSER_ALPHA_DENSE_MULTIPLIER", "1.0")
)
FUSER_ALPHA_ATTN_PROFILE = os.environ.get("FUSER_ALPHA_ATTN_PROFILE", "").strip()
FUSER_ALPHA_DENSE_PROFILE = os.environ.get("FUSER_ALPHA_DENSE_PROFILE", "").strip()
MAX_CAPTION_OBJECTS = int(os.environ.get("MAX_CAPTION_OBJECTS", "8"))
MAX_CAPTION_RELATIONS = int(os.environ.get("MAX_CAPTION_RELATIONS", "4"))
CAPTION_POLICY = os.environ.get("CAPTION_POLICY", "graph").strip().lower()
if CAPTION_POLICY not in {"graph", "natural", "clean", "clean_primary"}:
    raise ValueError(
        "CAPTION_POLICY must be 'graph', 'natural', 'clean', or 'clean_primary', "
        f"got {CAPTION_POLICY!r}"
    )
CAPTION_STYLE_PREFIX = os.environ.get("CAPTION_STYLE_PREFIX", "").strip()
CAPTION_STYLE_SUFFIX = os.environ.get("CAPTION_STYLE_SUFFIX", "").strip()
MAX_EVAL_OBJECTS = int(os.environ.get("MAX_EVAL_OBJECTS", "0"))
MAX_EVAL_RELATIONS = int(os.environ.get("MAX_EVAL_RELATIONS", "0"))
EVAL_SELECTION_POLICY = os.environ.get("EVAL_SELECTION_POLICY", "first").strip()
CONDITIONING_POLICY = os.environ.get("CONDITIONING_POLICY", "legacy").strip().lower()
if CONDITIONING_POLICY == "vg_conditioning_v2":
    CONDITIONING_POLICY = "clean_spatial_v2"
if CONDITIONING_POLICY in {"vg_conditioning_v2.1", "vg_conditioning_v21"}:
    CONDITIONING_POLICY = "clean_spatial_v2_1"
if CONDITIONING_POLICY not in {
    "legacy",
    "clean_spatial_v1",
    "clean_spatial_v2",
    "clean_spatial_v2_1",
}:
    raise ValueError(
        "CONDITIONING_POLICY must be 'legacy', 'clean_spatial_v1', "
        "'clean_spatial_v2', 'clean_spatial_v2_1', "
        "'vg_conditioning_v2', or 'vg_conditioning_v2.1', "
        f"got {CONDITIONING_POLICY!r}"
    )
_USES_CLEAN_CONDITIONING = CONDITIONING_POLICY in {
    "clean_spatial_v1",
    "clean_spatial_v2",
    "clean_spatial_v2_1",
}
_USES_CLEAN_CONDITIONING_V2 = CONDITIONING_POLICY in {
    "clean_spatial_v2",
    "clean_spatial_v2_1",
}
CLEAN_MAX_OBJECTS = int(
    os.environ.get("CLEAN_MAX_OBJECTS", "8" if _USES_CLEAN_CONDITIONING_V2 else "6")
)
CLEAN_MAX_RELATIONS = int(
    os.environ.get("CLEAN_MAX_RELATIONS", "2" if _USES_CLEAN_CONDITIONING_V2 else "1")
)
CLEAN_MIN_BOX_AREA = float(os.environ.get("CLEAN_MIN_BOX_AREA", "0.0025"))
CLEAN_MIN_BOX_SIDE = float(os.environ.get("CLEAN_MIN_BOX_SIDE", "0.035"))
CLEAN_RELATION_CORE_MIN_AREA = float(
    os.environ.get("CLEAN_RELATION_CORE_MIN_AREA", "0.0015")
)
CLEAN_DUPLICATE_IOU_THRESHOLD = float(
    os.environ.get("CLEAN_DUPLICATE_IOU_THRESHOLD", "0.85")
)
CLEAN_RELATION_PREDICATES = parse_allowed_predicates(
    os.environ.get("CLEAN_RELATION_PREDICATES", "")
)
CLEAN_FOREGROUND_MASK_SCALE = float(
    os.environ.get("CLEAN_FOREGROUND_MASK_SCALE", "1.0")
)
CLEAN_SUPPORT_MASK_SCALE = float(os.environ.get("CLEAN_SUPPORT_MASK_SCALE", "0.8"))
CLEAN_BACKGROUND_MASK_SCALE = float(
    os.environ.get("CLEAN_BACKGROUND_MASK_SCALE", "0.4")
)
CLEAN_OTHER_MASK_SCALE = float(os.environ.get("CLEAN_OTHER_MASK_SCALE", "0.8"))
ENABLE_RELATION_GROUNDING_TOKENS = os.environ.get(
    "ENABLE_RELATION_GROUNDING_TOKENS",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}
MAX_RELATION_GROUNDING_TOKENS = int(os.environ.get("MAX_RELATION_GROUNDING_TOKENS", "0"))
RELATION_GROUNDING_MASK_SCALE = float(
    os.environ.get("RELATION_GROUNDING_MASK_SCALE", "1.0")
)
RELATION_GROUNDING_TEMPLATE = os.environ.get(
    "RELATION_GROUNDING_TEMPLATE",
    "{subject} {predicate} {object}",
)
RELATION_GROUNDING_ALLOWED_PREDICATES = os.environ.get(
    "RELATION_GROUNDING_ALLOWED_PREDICATES",
    "",
)
DEDUP_RELATION_GROUNDING_TOKENS = os.environ.get(
    "DEDUP_RELATION_GROUNDING_TOKENS",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}
SAVE_SAMPLE_METADATA = os.environ.get(
    "SAVE_SAMPLE_METADATA",
    "1",
).strip().lower() in {"1", "true", "yes", "on"}
SPLIT_NAME = os.environ.get("SPLIT_NAME", "test")
EVAL_TRANSFORM_MODE = os.environ.get(
    "EVAL_TRANSFORM_MODE",
    "gligen_center_crop",
).strip().lower()
if EVAL_TRANSFORM_MODE not in {"gligen_center_crop", "legacy_stretch"}:
    raise ValueError(
        "EVAL_TRANSFORM_MODE must be 'gligen_center_crop' or 'legacy_stretch', "
        f"got {EVAL_TRANSFORM_MODE!r}"
    )


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(path):
    values = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            key, separator, value = line.rstrip("\n").partition("=")
            if separator:
                values[key] = value
    return values


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_real_image(image_path, out_path):
    image = Image.open(image_path).convert("RGB")
    if EVAL_TRANSFORM_MODE == "gligen_center_crop":
        transform = compute_center_crop_transform(*image.size, SAVE_SIZE)
        image = image.resize(
            (transform["resized_width"], transform["resized_height"]),
            Image.BICUBIC,
        )
        crop_x = transform["crop_x"]
        crop_y = transform["crop_y"]
        image = image.crop(
            (crop_x, crop_y, crop_x + SAVE_SIZE, crop_y + SAVE_SIZE)
        )
    elif SAVE_SIZE > 0 and image.size != (SAVE_SIZE, SAVE_SIZE):
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


def items_to_batch(items):
    batch = {}
    keys = items[0].keys()
    for key in keys:
        value0 = items[0][key]
        if torch.is_tensor(value0):
            batch[key] = torch.stack([item[key] for item in items], dim=0)
        else:
            batch[key] = [item[key] for item in items]
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
    model_state = model.state_dict()
    compatible, base_report = validate_base_model_compatibility(
        model_state,
        base["model"],
    )
    print(
        "BASE_MODEL_LOAD",
        f"loaded={base_report['loaded']}",
        f"new_scene_graph={len(base_report['new_scene_graph_tensors'])}",
        flush=True,
    )
    model.load_state_dict(compatible, strict=False)
    autoencoder.load_state_dict(base["autoencoder"])
    text_encoder.load_state_dict(base["text_encoder"], strict=False)
    diffusion.load_state_dict(base["diffusion"])

    if GROUNDING_CKPT:
        grounding_checkpoint = torch.load(GROUNDING_CKPT, map_location="cpu")
        grounding_state = grounding_checkpoint.get(
            "model_trainable",
            grounding_checkpoint.get("model", {}),
        )
        if "model_trainable" in grounding_checkpoint:
            validate_checkpoint_trainable_manifest(
                grounding_state,
                grounding_checkpoint.get("trainable_names", []),
            )
        current_state = model.state_dict()
        compatible_grounding, load_report = validate_grounding_state(
            current_state,
            grounding_state,
        )
        print(
            "GROUNDING_LOAD",
            f"ckpt={GROUNDING_CKPT}",
            f"loaded={load_report['loaded']}",
            f"skipped={len(load_report['skipped'])}",
            flush=True,
        )
        current_state.update(compatible_grounding)
        if RESTORE_BASE_FUSER:
            restored_fuser_tensors = restore_base_fuser_state(
                current_state,
                compatible,
            )
            if restored_fuser_tensors == 0:
                raise RuntimeError(
                    "RESTORE_BASE_FUSER requested but no fuser tensors were restored"
                )
            print(
                "FUSER_ABLATION",
                f"restored_base_tensors={restored_fuser_tensors}",
                flush=True,
            )
        model.load_state_dict(current_state, strict=True)

    if GRAPH_GATE_OVERRIDE is not None:
        apply_graph_gate_override(model, GRAPH_GATE_OVERRIDE)
        print(
            "GRAPH_GATE_ABLATION",
            f"override={GRAPH_GATE_OVERRIDE}",
            flush=True,
        )
    if TRIPLET_GATE_OVERRIDE is not None:
        position_net = getattr(model, "position_net", None)
        triplet_gate = getattr(position_net, "triplet_gate", None)
        if triplet_gate is None:
            raise RuntimeError(
                "TRIPLET_GATE_OVERRIDE was requested but the model has no triplet_gate"
            )
        with torch.no_grad():
            triplet_gate.fill_(TRIPLET_GATE_OVERRIDE)
        print(
            "TRIPLET_GATE_ABLATION",
            f"override_logit={TRIPLET_GATE_OVERRIDE}",
            f"effective_sigmoid={torch.sigmoid(triplet_gate).item():.6f}",
            flush=True,
        )

    fuser_alpha_summary = summarize_fuser_alpha(model)
    print(
        "FUSER_ALPHA_SUMMARY",
        f"count={fuser_alpha_summary['count']}",
        f"mean_attn={fuser_alpha_summary['mean_abs_tanh_alpha_attn']:.6f}",
        f"max_attn={fuser_alpha_summary['max_abs_tanh_alpha_attn']:.6f}",
        f"mean_dense={fuser_alpha_summary['mean_abs_tanh_alpha_dense']:.6f}",
        f"max_dense={fuser_alpha_summary['max_abs_tanh_alpha_dense']:.6f}",
        flush=True,
    )
    if (
        FUSER_ALPHA_ATTN_MULTIPLIER != 1.0
        or FUSER_ALPHA_DENSE_MULTIPLIER != 1.0
    ):
        report = apply_fuser_alpha_multiplier(
            model,
            attn_multiplier=FUSER_ALPHA_ATTN_MULTIPLIER,
            dense_multiplier=FUSER_ALPHA_DENSE_MULTIPLIER,
        )
        after = report["after"]
        print(
            "FUSER_ALPHA_ABLATION",
            f"updated={report['updated']}",
            f"attn_multiplier={report['attn_multiplier']}",
            f"dense_multiplier={report['dense_multiplier']}",
            f"after_mean_attn={after['mean_abs_tanh_alpha_attn']:.6f}",
            f"after_max_attn={after['max_abs_tanh_alpha_attn']:.6f}",
            f"after_mean_dense={after['mean_abs_tanh_alpha_dense']:.6f}",
            f"after_max_dense={after['max_abs_tanh_alpha_dense']:.6f}",
            flush=True,
        )
    if FUSER_ALPHA_ATTN_PROFILE or FUSER_ALPHA_DENSE_PROFILE:
        profile_report = apply_fuser_alpha_profile(
            model,
            attn_profile=parse_fuser_alpha_profile(FUSER_ALPHA_ATTN_PROFILE),
            dense_profile=parse_fuser_alpha_profile(FUSER_ALPHA_DENSE_PROFILE),
        )
        after = profile_report["after"]
        layer_preview = ";".join(
            f"{layer['resolution'] or 'unknown'}:{layer['attn_multiplier']}"
            for layer in profile_report["layers"][:8]
        )
        print(
            "FUSER_ALPHA_PROFILE_ABLATION",
            f"updated={profile_report['updated']}",
            f"attn_profile={FUSER_ALPHA_ATTN_PROFILE or 'None'}",
            f"dense_profile={FUSER_ALPHA_DENSE_PROFILE or 'None'}",
            f"after_mean_attn={after['mean_abs_tanh_alpha_attn']:.6f}",
            f"after_max_attn={after['max_abs_tanh_alpha_attn']:.6f}",
            f"after_mean_dense={after['mean_abs_tanh_alpha_dense']:.6f}",
            f"after_max_dense={after['max_abs_tanh_alpha_dense']:.6f}",
            f"layer_preview={layer_preview}",
            flush=True,
        )

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
        if CAPTION_POLICY == "clean_primary":
            return build_clean_primary_scene_graph_caption(
                object_texts,
                relation_edges,
                relation_masks,
                relation_texts,
                max_objects=MAX_CAPTION_OBJECTS,
                style_prefix=CAPTION_STYLE_PREFIX
                or "A full-color realistic DSLR photograph",
                style_suffix=CAPTION_STYLE_SUFFIX
                or "vivid natural colors, realistic color photography, natural lighting",
            )
        if CAPTION_POLICY == "clean" or _USES_CLEAN_CONDITIONING:
            return build_clean_scene_graph_caption(
                object_texts,
                relation_edges,
                relation_masks,
                relation_texts,
                style_prefix=CAPTION_STYLE_PREFIX
                or "A full-color realistic DSLR photograph",
                style_suffix=CAPTION_STYLE_SUFFIX
                or "vivid natural colors, realistic color photography, natural lighting",
            )
        if CAPTION_POLICY == "natural":
            return build_natural_scene_graph_caption(
                object_texts,
                relation_edges,
                relation_masks,
                relation_texts,
                max_objects=MAX_CAPTION_OBJECTS,
                max_relations=MAX_CAPTION_RELATIONS,
                style_prefix=CAPTION_STYLE_PREFIX
                or "A realistic natural color photograph",
                style_suffix=CAPTION_STYLE_SUFFIX
                or "natural lighting and realistic details",
            )
        return build_scene_graph_caption(
            object_texts,
            relation_edges,
            relation_masks,
            relation_texts,
            max_objects=MAX_CAPTION_OBJECTS,
            max_relations=MAX_CAPTION_RELATIONS,
            style_prefix=CAPTION_STYLE_PREFIX,
            style_suffix=CAPTION_STYLE_SUFFIX,
        )

    def __getitem__(self, index):
        image_id = int(self.h5["image_ids"][index])
        rel_path = self._decode_path(self.h5["image_paths"][index])
        image_path = self.image_root / rel_path

        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size

        num_objects = int(self.h5["objects_per_image"][index])
        num_relations = int(self.h5["relationships_per_image"][index])
        h5_max_boxes = int(self.h5["object_names"].shape[1])
        h5_max_relations = int(self.h5["relationship_predicates"].shape[1])
        max_boxes = h5_max_boxes if MAX_EVAL_OBJECTS <= 0 else min(h5_max_boxes, MAX_EVAL_OBJECTS)
        max_relations = (
            h5_max_relations
            if MAX_EVAL_RELATIONS <= 0
            else min(h5_max_relations, MAX_EVAL_RELATIONS)
        )

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

        rel_subjects = self.h5["relationship_subjects"][index][:num_relations].tolist()
        rel_predicates = self.h5["relationship_predicates"][index][:num_relations].tolist()
        rel_objects = self.h5["relationship_objects"][index][:num_relations].tolist()
        raw_relations = list(zip(rel_subjects, rel_predicates, rel_objects))

        conditioning_trace = None
        clean_object_categories = None
        clean_object_mask_scales = None
        if EVAL_TRANSFORM_MODE == "gligen_center_crop":
            transform_max_boxes = h5_max_boxes if _USES_CLEAN_CONDITIONING else max_boxes
            transform_max_relations = (
                h5_max_relations if _USES_CLEAN_CONDITIONING else max_relations
            )
            annotations = transform_scene_graph_annotations(
                object_names=object_names,
                object_boxes_xywh=object_boxes_xywh.tolist(),
                relations=raw_relations,
                trans_info=compute_center_crop_transform(width, height, SAVE_SIZE),
                image_size=SAVE_SIZE,
                min_box_size=0.0,
                max_boxes=transform_max_boxes,
                max_relations=transform_max_relations,
                selection_policy=(
                    "first" if _USES_CLEAN_CONDITIONING else EVAL_SELECTION_POLICY
                ),
            )
        else:
            legacy_boxes = []
            for x, y, w, h in object_boxes_xywh.tolist():
                legacy_boxes.append(
                    (
                        float(x) / max(width, 1),
                        float(y) / max(height, 1),
                        float(x + w) / max(width, 1),
                        float(y + h) / max(height, 1),
                    )
                )
            annotations = {
                "object_names": object_names,
                "boxes": legacy_boxes,
                "relations": raw_relations,
            }

        if _USES_CLEAN_CONDITIONING:
            annotation_object_texts = [
                str(self.object_idx_to_name[int(name)]).lower()
                for name in annotations["object_names"]
            ]
            annotation_relation_texts = [
                str(self.pred_idx_to_name[int(predicate)]).lower()
                for _, predicate, _ in annotations["relations"]
            ]
            clean_condition = build_clean_scene_graph_condition(
                object_names=annotations["object_names"],
                object_texts=annotation_object_texts,
                boxes=annotations["boxes"],
                relations=annotations["relations"],
                relation_texts=annotation_relation_texts,
                max_objects=min(CLEAN_MAX_OBJECTS, max_boxes),
                max_relations=min(CLEAN_MAX_RELATIONS, max_relations),
                min_box_area=CLEAN_MIN_BOX_AREA,
                min_box_side=CLEAN_MIN_BOX_SIDE,
                relation_core_min_area=CLEAN_RELATION_CORE_MIN_AREA,
                duplicate_iou_threshold=CLEAN_DUPLICATE_IOU_THRESHOLD,
                relation_predicates=CLEAN_RELATION_PREDICATES,
                policy=CONDITIONING_POLICY,
                foreground_mask_scale=CLEAN_FOREGROUND_MASK_SCALE,
                support_mask_scale=CLEAN_SUPPORT_MASK_SCALE,
                background_mask_scale=CLEAN_BACKGROUND_MASK_SCALE,
                other_mask_scale=CLEAN_OTHER_MASK_SCALE,
            )
            annotations = {
                "object_names": clean_condition.object_names,
                "boxes": clean_condition.boxes,
                "relations": clean_condition.relations,
            }
            clean_object_categories = clean_condition.object_categories
            clean_object_mask_scales = clean_condition.object_mask_scales
            conditioning_trace = clean_condition.trace

        for obj_idx, (name_idx, box) in enumerate(
            zip(annotations["object_names"], annotations["boxes"])
        ):
            boxes[obj_idx] = torch.tensor(box, dtype=torch.float32)
            if clean_object_mask_scales is not None and obj_idx < len(clean_object_mask_scales):
                masks[obj_idx] = float(clean_object_mask_scales[obj_idx])
            else:
                masks[obj_idx] = 1.0
            object_texts[obj_idx] = str(self.object_idx_to_name[int(name_idx)]).lower()

        relation_edges = torch.zeros(max_relations, 2, dtype=torch.float32)
        relation_masks = torch.zeros(max_relations, dtype=torch.float32)
        relation_geo_features = torch.zeros(max_relations, 12, dtype=torch.float32)
        relation_texts = [""] * max_relations

        if rel_predicates:
            max_pred_idx = max(int(idx) for idx in rel_predicates)
            if max_pred_idx >= len(self.pred_idx_to_name):
                raise IndexError(
                    f"predicate idx {max_pred_idx} exceeds vocab size {len(self.pred_idx_to_name)}"
                )
        for rel_idx, (src, pred_idx, dst) in enumerate(annotations["relations"]):
            src = int(src)
            dst = int(dst)
            relation_edges[rel_idx] = torch.tensor([src, dst], dtype=torch.float32)
            relation_masks[rel_idx] = 1.0
            relation_texts[rel_idx] = str(self.pred_idx_to_name[int(pred_idx)]).lower()
            relation_geo_features[rel_idx] = compute_relation_geo_features(boxes[src], boxes[dst])

        caption = self._caption_from_graph(object_texts, relation_edges, relation_masks, relation_texts)
        token_roles = [
            "object" if float(mask) > 0.0 else "padding"
            for mask in masks.detach().cpu().tolist()
        ]
        relation_token_source = torch.empty(0, dtype=torch.long)
        if ENABLE_RELATION_GROUNDING_TOKENS and MAX_RELATION_GROUNDING_TOKENS > 0:
            relation_token_result = append_relation_grounding_tokens(
                boxes=boxes,
                masks=masks,
                object_texts=object_texts,
                relation_edges=relation_edges,
                relation_masks=relation_masks,
                relation_texts=relation_texts,
                max_relation_tokens=MAX_RELATION_GROUNDING_TOKENS,
                phrase_template=RELATION_GROUNDING_TEMPLATE,
                allowed_predicates=RELATION_GROUNDING_ALLOWED_PREDICATES,
                deduplicate=DEDUP_RELATION_GROUNDING_TOKENS,
                relation_mask_scale=RELATION_GROUNDING_MASK_SCALE,
            )
            boxes = relation_token_result.boxes
            masks = relation_token_result.masks
            object_texts = relation_token_result.object_texts
            token_roles = relation_token_result.token_roles
            relation_token_source = relation_token_result.relation_token_source.cpu()

        relation_token_mask = torch.tensor(
            [1.0 if role == "relation" else 0.0 for role in token_roles],
            dtype=torch.float32,
        )

        return {
            "id": image_id,
            "image_path": str(image_path),
            "caption": caption,
            "boxes": boxes,
            "masks": masks,
            "object_texts": object_texts,
            "grounding_token_roles": token_roles,
            "object_categories": clean_object_categories,
            "relation_token_source": relation_token_source,
            "relation_token_mask": relation_token_mask,
            "conditioning_trace": conditioning_trace,
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
    indices = parse_sample_indices(SAMPLE_INDICES, len(dataset))
    if indices is None:
        available = len(dataset) - START_INDEX
        count = min(NUM_SAMPLES, available)
        indices = list(range(START_INDEX, START_INDEX + count))
        subset = f"first_{count}_samples_from_{SPLIT_NAME}_split_starting_at_{START_INDEX}"
    else:
        count = len(indices)
        subset = "fixed_indices_" + ",".join(str(index) for index in indices)
    print("DATASET_LEN", len(dataset), "START_INDEX", START_INDEX, "COUNT", count, flush=True)

    grounding_ckpt_path = Path(GROUNDING_CKPT).resolve() if GROUNDING_CKPT else None
    base_ckpt_path = Path(BASE_CKPT).resolve()
    expected_metadata = {
        "PROTOCOL": "sg2i_fixed_split",
        "SPLIT_NAME": SPLIT_NAME,
        "SUBSET": subset,
        "MODEL_YAML": MODEL_YAML,
        "DATA_YAML": DATA_YAML,
        "GROUNDING_CKPT": str(grounding_ckpt_path) if grounding_ckpt_path else "None",
        "GROUNDING_CKPT_SHA256": sha256_file(grounding_ckpt_path) if grounding_ckpt_path else "None",
        "BASE_CKPT": str(base_ckpt_path),
        "BASE_CKPT_SHA256": sha256_file(base_ckpt_path),
        "H5_PATH": str(H5_PATH.resolve()),
        "VOCAB_PATH": str(VOCAB_PATH.resolve()),
        "IMAGE_ROOT": str(IMAGE_ROOT.resolve()),
        "NUM_SAMPLES": str(count),
        "START_INDEX": str(START_INDEX),
        "SAMPLE_INDICES": ",".join(str(index) for index in indices),
        "SAMPLER": SAMPLER_NAME,
        "STEPS": str(STEPS),
        "GUIDANCE": str(GUIDANCE),
        "GROUNDING_ALPHA_TYPE": ",".join(
            f"{ratio:g}" for ratio in GROUNDING_ALPHA_TYPE
        ),
        "SEED": str(SEED),
        "SAVE_SIZE": str(SAVE_SIZE),
        "EVAL_BATCH_SIZE": str(EVAL_BATCH_SIZE),
        "NOISE_POLICY": "per_image_cpu_seed_plus_image_id_v1",
        "RESTORE_BASE_FUSER": str(RESTORE_BASE_FUSER),
        "GRAPH_GATE_OVERRIDE": (
            "None"
            if GRAPH_GATE_OVERRIDE is None
            else str(GRAPH_GATE_OVERRIDE)
        ),
        "TRIPLET_GATE_OVERRIDE": (
            "None"
            if TRIPLET_GATE_OVERRIDE is None
            else str(TRIPLET_GATE_OVERRIDE)
        ),
        "FUSER_ALPHA_ATTN_MULTIPLIER": str(FUSER_ALPHA_ATTN_MULTIPLIER),
        "FUSER_ALPHA_DENSE_MULTIPLIER": str(FUSER_ALPHA_DENSE_MULTIPLIER),
        "FUSER_ALPHA_ATTN_PROFILE": FUSER_ALPHA_ATTN_PROFILE,
        "FUSER_ALPHA_DENSE_PROFILE": FUSER_ALPHA_DENSE_PROFILE,
        "CAPTION_POLICY": CAPTION_POLICY,
        "CAPTION_STYLE_PREFIX": CAPTION_STYLE_PREFIX,
        "CAPTION_STYLE_SUFFIX": CAPTION_STYLE_SUFFIX,
        "EVAL_TRANSFORM_MODE": EVAL_TRANSFORM_MODE,
        "MAX_EVAL_OBJECTS": str(MAX_EVAL_OBJECTS),
        "MAX_EVAL_RELATIONS": str(MAX_EVAL_RELATIONS),
        "EVAL_SELECTION_POLICY": EVAL_SELECTION_POLICY,
        "CONDITIONING_POLICY": CONDITIONING_POLICY,
        "CLEAN_MAX_OBJECTS": str(CLEAN_MAX_OBJECTS),
        "CLEAN_MAX_RELATIONS": str(CLEAN_MAX_RELATIONS),
        "CLEAN_MIN_BOX_AREA": str(CLEAN_MIN_BOX_AREA),
        "CLEAN_MIN_BOX_SIDE": str(CLEAN_MIN_BOX_SIDE),
        "CLEAN_RELATION_CORE_MIN_AREA": str(CLEAN_RELATION_CORE_MIN_AREA),
        "CLEAN_DUPLICATE_IOU_THRESHOLD": str(CLEAN_DUPLICATE_IOU_THRESHOLD),
        "CLEAN_RELATION_PREDICATES": ",".join(
            sorted(CLEAN_RELATION_PREDICATES or [])
        ),
        "CLEAN_FOREGROUND_MASK_SCALE": str(CLEAN_FOREGROUND_MASK_SCALE),
        "CLEAN_SUPPORT_MASK_SCALE": str(CLEAN_SUPPORT_MASK_SCALE),
        "CLEAN_BACKGROUND_MASK_SCALE": str(CLEAN_BACKGROUND_MASK_SCALE),
        "CLEAN_OTHER_MASK_SCALE": str(CLEAN_OTHER_MASK_SCALE),
        "ENABLE_RELATION_GROUNDING_TOKENS": str(ENABLE_RELATION_GROUNDING_TOKENS),
        "MAX_RELATION_GROUNDING_TOKENS": str(MAX_RELATION_GROUNDING_TOKENS),
        "RELATION_GROUNDING_MASK_SCALE": str(RELATION_GROUNDING_MASK_SCALE),
        "RELATION_GROUNDING_TEMPLATE": RELATION_GROUNDING_TEMPLATE,
        "RELATION_GROUNDING_ALLOWED_PREDICATES": RELATION_GROUNDING_ALLOWED_PREDICATES,
        "DEDUP_RELATION_GROUNDING_TOKENS": str(DEDUP_RELATION_GROUNDING_TOKENS),
        "SAVE_SAMPLE_METADATA": str(SAVE_SAMPLE_METADATA),
    }
    existing_images = list(out_real.glob("*.png")) + list(out_fake.glob("*.png"))
    metadata_path = OUT_DIR / "meta.txt"
    if existing_images:
        if not metadata_path.exists():
            raise ProtocolMismatchError(
                f"{OUT_DIR} already contains images but has no meta.txt; refusing an unverified resume"
            )
        validate_resume_metadata(read_metadata(metadata_path), expected_metadata)

    with open(OUT_DIR / "meta.txt", "w", encoding="utf-8") as f:
        for key, value in expected_metadata.items():
            f.write(f"{key}={value}\n")
    sample_metadata_dir = OUT_DIR / "sample_metadata"
    generation_metadata = {
        "protocol": expected_metadata["PROTOCOL"],
        "split_name": SPLIT_NAME,
        "subset": subset,
        "sampler": SAMPLER_NAME,
        "steps": STEPS,
        "guidance": GUIDANCE,
        "seed": SEED,
        "grounding_alpha_type": expected_metadata["GROUNDING_ALPHA_TYPE"],
        "base_ckpt": expected_metadata["BASE_CKPT"],
        "grounding_ckpt": expected_metadata["GROUNDING_CKPT"],
        "graph_gate_override": expected_metadata["GRAPH_GATE_OVERRIDE"],
        "triplet_gate_override": expected_metadata["TRIPLET_GATE_OVERRIDE"],
        "fuser_alpha_attn_multiplier": FUSER_ALPHA_ATTN_MULTIPLIER,
        "fuser_alpha_dense_multiplier": FUSER_ALPHA_DENSE_MULTIPLIER,
        "fuser_alpha_attn_profile": FUSER_ALPHA_ATTN_PROFILE,
        "fuser_alpha_dense_profile": FUSER_ALPHA_DENSE_PROFILE,
        "caption_policy": CAPTION_POLICY,
        "restore_base_fuser": RESTORE_BASE_FUSER,
        "eval_transform_mode": EVAL_TRANSFORM_MODE,
        "max_eval_objects": MAX_EVAL_OBJECTS,
        "max_eval_relations": MAX_EVAL_RELATIONS,
        "eval_selection_policy": EVAL_SELECTION_POLICY,
        "conditioning_policy": CONDITIONING_POLICY,
        "clean_max_objects": CLEAN_MAX_OBJECTS,
        "clean_max_relations": CLEAN_MAX_RELATIONS,
        "clean_min_box_area": CLEAN_MIN_BOX_AREA,
        "clean_min_box_side": CLEAN_MIN_BOX_SIDE,
        "clean_relation_core_min_area": CLEAN_RELATION_CORE_MIN_AREA,
        "clean_duplicate_iou_threshold": CLEAN_DUPLICATE_IOU_THRESHOLD,
        "clean_relation_predicates": sorted(CLEAN_RELATION_PREDICATES or []),
        "clean_foreground_mask_scale": CLEAN_FOREGROUND_MASK_SCALE,
        "clean_support_mask_scale": CLEAN_SUPPORT_MASK_SCALE,
        "clean_background_mask_scale": CLEAN_BACKGROUND_MASK_SCALE,
        "clean_other_mask_scale": CLEAN_OTHER_MASK_SCALE,
        "enable_relation_grounding_tokens": ENABLE_RELATION_GROUNDING_TOKENS,
        "max_relation_grounding_tokens": MAX_RELATION_GROUNDING_TOKENS,
        "relation_grounding_mask_scale": RELATION_GROUNDING_MASK_SCALE,
        "relation_grounding_allowed_predicates": RELATION_GROUNDING_ALLOWED_PREDICATES,
        "dedup_relation_grounding_tokens": DEDUP_RELATION_GROUNDING_TOKENS,
    }

    model, autoencoder, text_encoder, diffusion, grounding_tokenizer_input = load_model()
    alpha_generator_func = partial(
        build_grounding_alpha_schedule,
        alpha_type=GROUNDING_ALPHA_TYPE,
    )
    if SAMPLER_NAME == "plms":
        sampler = PLMSSampler(
            diffusion,
            model,
            alpha_generator_func=alpha_generator_func,
            set_alpha_scale=set_grounding_alpha_scale,
        )
    elif SAMPLER_NAME == "ddim":
        sampler = DDIMSampler(
            diffusion,
            model,
            alpha_generator_func=alpha_generator_func,
            set_alpha_scale=set_grounding_alpha_scale,
        )
    else:
        raise ValueError(f"Unsupported SAMPLER={SAMPLER_NAME}; expected 'plms' or 'ddim'.")

    pending_items = []
    processed = 0

    def flush_batch(items, processed_count):
        if not items:
            return processed_count
        batch = items_to_batch(items)
        batch_size = len(items)
        encode_text_grid(text_encoder, batch, "object_texts", "text_embeddings")
        encode_text_grid(text_encoder, batch, "relation_texts", "relation_embeddings")
        context = text_encoder.encode(batch["caption"])
        uc = text_encoder.encode([""] * batch_size)
        grounding_input = grounding_tokenizer_input.prepare(batch)
        shape = (batch_size, 4, 64, 64)
        initial_noise = build_per_sample_noise(
            image_ids=[int(item["id"]) for item in items],
            sample_shape=shape[1:],
            base_seed=SEED,
            device=DEVICE,
        )
        input_dict = dict(
            x=initial_noise,
            timesteps=None,
            context=context,
            grounding_input=grounding_input,
            inpainting_extra_input=None,
            grounding_extra_input=None,
        )
        samples = sampler.sample(S=STEPS, shape=shape, input=input_dict, uc=uc, guidance_scale=GUIDANCE)
        decoded = autoencoder.decode(samples)

        for i, item in enumerate(items):
            image_id = f"{int(item['id'])}.png"
            out_real_path = out_real / image_id
            out_fake_path = out_fake / image_id
            save_real_image(item["image_path"], out_real_path)
            save_tensor_image(decoded[i], out_fake_path)

        processed_count += batch_size
        if processed_count % 50 == 0 or processed_count == count:
            last_image_id = f"{int(items[-1]['id'])}.png"
            print("PROGRESS", processed_count, "/", count, "image_id", last_image_id, "batch", batch_size, flush=True)

        del batch, context, uc, grounding_input, initial_noise, input_dict, samples, decoded
        torch.cuda.empty_cache()
        gc.collect()
        return processed_count

    for dataset_idx in indices:
        item = dataset[dataset_idx]
        if SAVE_SAMPLE_METADATA:
            write_sample_metadata(
                sample_metadata_dir,
                item,
                dataset_index=dataset_idx,
                generation=generation_metadata,
            )
        image_id = f"{int(item['id'])}.png"
        out_real_path = out_real / image_id
        out_fake_path = out_fake / image_id
        if out_real_path.exists() and out_fake_path.exists():
            processed += 1
            if processed % 50 == 0 or processed == count:
                print("PROGRESS", processed, "/", count, "image_id", image_id, "skipped", True, flush=True)
            continue

        pending_items.append(item)
        if len(pending_items) >= EVAL_BATCH_SIZE:
            processed = flush_batch(pending_items, processed)
            pending_items = []

    if pending_items:
        processed = flush_batch(pending_items, processed)

    print("SAVED", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
