import json
from pathlib import Path

import h5py
import torch
from PIL import Image

from .base_dataset import BaseDataset
from .relation_grounding_tokens import append_relation_grounding_tokens, parse_allowed_predicates
from .scene_graph_caption import (
    build_clean_scene_graph_caption,
    build_clean_primary_scene_graph_caption,
    build_natural_scene_graph_caption,
    build_scene_graph_caption,
)
from .scene_graph_conditioning import build_clean_scene_graph_condition
from .scene_graph_box_utils import transform_scene_graph_annotations


DEFAULT_GENERIC_OBJECTS = {
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "people",
    "tree",
    "building",
    "road",
    "street",
    "sky",
    "grass",
}

DEFAULT_PRIORITY_OBJECTS = {
    "ski",
    "skateboard",
    "pizza",
    "umbrella",
    "bench",
    "bus",
    "car",
    "dog",
    "horse",
}


def _box_area(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def compute_relation_geo_features(src_box, dst_box):
    src = src_box.float()
    dst = dst_box.float()

    src_cx = (src[0] + src[2]) * 0.5
    src_cy = (src[1] + src[3]) * 0.5
    dst_cx = (dst[0] + dst[2]) * 0.5
    dst_cy = (dst[1] + dst[3]) * 0.5

    src_w = (src[2] - src[0]).clamp(min=1e-6)
    src_h = (src[3] - src[1]).clamp(min=1e-6)
    dst_w = (dst[2] - dst[0]).clamp(min=1e-6)
    dst_h = (dst[3] - dst[1]).clamp(min=1e-6)

    inter_x1 = torch.maximum(src[0], dst[0])
    inter_y1 = torch.maximum(src[1], dst[1])
    inter_x2 = torch.minimum(src[2], dst[2])
    inter_y2 = torch.minimum(src[3], dst[3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    src_area = src_w * src_h
    dst_area = dst_w * dst_h
    union_area = (src_area + dst_area - inter_area).clamp(min=1e-6)
    iou = inter_area / union_area

    src_inside_dst = (
        (src[0] >= dst[0]) & (src[1] >= dst[1]) & (src[2] <= dst[2]) & (src[3] <= dst[3])
    ).float()
    dst_inside_src = (
        (dst[0] >= src[0]) & (dst[1] >= src[1]) & (dst[2] <= src[2]) & (dst[3] <= src[3])
    ).float()
    inside = torch.maximum(src_inside_dst, dst_inside_src)

    overlap_src = inter_area / src_area.clamp(min=1e-6)
    overlap_dst = inter_area / dst_area.clamp(min=1e-6)

    left = (src_cx < dst_cx).float()
    right = (src_cx > dst_cx).float()
    above = (src_cy < dst_cy).float()
    below = (src_cy > dst_cy).float()

    return torch.tensor(
        [
            float(dst_cx - src_cx),
            float(dst_cy - src_cy),
            float(torch.log(dst_w / src_w)),
            float(torch.log(dst_h / src_h)),
            float(iou),
            float(inside),
            float(overlap_src),
            float(overlap_dst),
            float(left),
            float(right),
            float(above),
            float(below),
        ],
        dtype=torch.float32,
    )


class VGSceneGraphDataset(BaseDataset):
    def __init__(
        self,
        image_root,
        h5_path,
        vocab_path,
        image_size=256,
        random_crop=False,
        random_flip=True,
        max_images=None,
        prob_use_caption=1.0,
        max_boxes_per_data=30,
        max_relations_per_data=30,
        box_transform_mode="gligen",
        min_box_size=0.0,
        selection_policy="first",
        conditioning_policy="legacy",
        clean_max_objects=6,
        clean_max_relations=1,
        clean_min_box_area=0.0025,
        clean_min_box_side=0.035,
        clean_relation_core_min_area=0.0015,
        clean_duplicate_iou_threshold=0.85,
        clean_relation_predicates=None,
        caption_policy="graph",
        caption_style_prefix="",
        caption_style_suffix="",
        enable_relation_grounding_tokens=False,
        max_relation_grounding_tokens=0,
        relation_grounding_template="{subject} {predicate} {object}",
        relation_grounding_allowed_predicates=None,
        deduplicate_relation_grounding_tokens=False,
        relation_grounding_mask_scale=1.0,
        **_,
    ):
        super().__init__(random_crop, random_flip, image_size)
        self.image_root = Path(image_root)
        self.h5_path = Path(h5_path)
        self.vocab_path = Path(vocab_path)
        self.max_images = max_images
        self.prob_use_caption = prob_use_caption
        self.max_boxes_per_data = max_boxes_per_data
        self.max_relations_per_data = max_relations_per_data
        self.box_transform_mode = str(box_transform_mode)
        self.min_box_size = float(min_box_size)
        self.selection_policy = str(selection_policy)
        self.conditioning_policy = str(conditioning_policy).strip().lower()
        self.clean_max_objects = int(clean_max_objects)
        self.clean_max_relations = int(clean_max_relations)
        self.clean_min_box_area = float(clean_min_box_area)
        self.clean_min_box_side = float(clean_min_box_side)
        self.clean_relation_core_min_area = float(clean_relation_core_min_area)
        self.clean_duplicate_iou_threshold = float(clean_duplicate_iou_threshold)
        self.clean_relation_predicates = parse_allowed_predicates(clean_relation_predicates)
        self.caption_policy = str(caption_policy).strip().lower()
        self.caption_style_prefix = str(caption_style_prefix).strip()
        self.caption_style_suffix = str(caption_style_suffix).strip()
        self.enable_relation_grounding_tokens = bool(enable_relation_grounding_tokens)
        self.max_relation_grounding_tokens = int(max_relation_grounding_tokens)
        self.relation_grounding_template = str(relation_grounding_template)
        self.relation_grounding_allowed_predicates = relation_grounding_allowed_predicates
        self.deduplicate_relation_grounding_tokens = bool(deduplicate_relation_grounding_tokens)
        self.relation_grounding_mask_scale = float(relation_grounding_mask_scale)
        if self.box_transform_mode not in {"gligen", "legacy_normalize"}:
            raise ValueError(
                "box_transform_mode must be 'gligen' or 'legacy_normalize', "
                f"got {self.box_transform_mode!r}"
            )
        if self.conditioning_policy not in {"legacy", "clean_spatial_v1"}:
            raise ValueError(
                "conditioning_policy must be 'legacy' or 'clean_spatial_v1', "
                f"got {self.conditioning_policy!r}"
            )
        if self.caption_policy not in {"graph", "natural", "clean", "clean_primary"}:
            raise ValueError(
                "caption_policy must be 'graph', 'natural', 'clean', or 'clean_primary', "
                f"got {self.caption_policy!r}"
            )

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.object_idx_to_name = vocab["object_idx_to_name"]
        self.pred_idx_to_name = vocab["pred_idx_to_name"]

        self.h5 = h5py.File(self.h5_path, "r")
        self.length = int(self.h5["image_ids"].shape[0])
        if self.max_images is not None:
            self.length = min(self.length, int(self.max_images))

    def __len__(self):
        return self.length

    def total_images(self):
        return self.length

    def _decode_path(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _caption_from_graph(self, object_texts, relation_edges, relation_masks, relation_texts):
        caption_policy = getattr(self, "caption_policy", "graph")
        conditioning_policy = getattr(self, "conditioning_policy", "legacy")
        if caption_policy == "clean_primary":
            return build_clean_primary_scene_graph_caption(
                object_texts,
                relation_edges,
                relation_masks,
                relation_texts,
                style_prefix=self.caption_style_prefix
                or "A full-color realistic DSLR photograph",
                style_suffix=self.caption_style_suffix
                or "vivid natural colors, realistic color photography, natural lighting",
            )
        if caption_policy == "clean" or conditioning_policy == "clean_spatial_v1":
            return build_clean_scene_graph_caption(
                object_texts,
                relation_edges,
                relation_masks,
                relation_texts,
                style_prefix=self.caption_style_prefix
                or "A full-color realistic DSLR photograph",
                style_suffix=self.caption_style_suffix
                or "vivid natural colors, realistic color photography, natural lighting",
            )
        if caption_policy == "natural":
            return build_natural_scene_graph_caption(
                object_texts,
                relation_edges,
                relation_masks,
                relation_texts,
                style_prefix=self.caption_style_prefix
                or "A realistic natural color photograph",
                style_suffix=self.caption_style_suffix
                or "natural lighting and realistic details",
            )
        return build_scene_graph_caption(
            object_texts,
            relation_edges,
            relation_masks,
            relation_texts,
            style_prefix=self.caption_style_prefix,
            style_suffix=self.caption_style_suffix,
        )

    def __getitem__(self, index):
        image_id = int(self.h5["image_ids"][index])
        rel_path = self._decode_path(self.h5["image_paths"][index])
        image_path = self.image_root / rel_path

        with Image.open(image_path).convert("RGB") as image:
            image_tensor, trans_info = self.transform_image(image)

        num_objects = int(self.h5["objects_per_image"][index])
        num_relations = int(self.h5["relationships_per_image"][index])
        max_boxes = min(int(self.h5["object_names"].shape[1]), self.max_boxes_per_data)
        max_relations = min(int(self.h5["relationship_predicates"].shape[1]), self.max_relations_per_data)

        boxes = torch.zeros(max_boxes, 4, dtype=torch.float32)
        masks = torch.zeros(max_boxes, dtype=torch.float32)
        object_texts = [""] * max_boxes

        object_names = self.h5["object_names"][index][:num_objects].tolist()
        object_boxes_xywh = self.h5["object_boxes"][index][:num_objects].tolist()
        rel_subjects = self.h5["relationship_subjects"][index][:num_relations].tolist()
        rel_predicates = self.h5["relationship_predicates"][index][:num_relations].tolist()
        rel_objects = self.h5["relationship_objects"][index][:num_relations].tolist()
        raw_relations = list(zip(rel_subjects, rel_predicates, rel_objects))

        if self.box_transform_mode == "gligen":
            transform_max_boxes = (
                int(self.h5["object_names"].shape[1])
                if self.conditioning_policy == "clean_spatial_v1"
                else max_boxes
            )
            transform_max_relations = (
                int(self.h5["relationship_predicates"].shape[1])
                if self.conditioning_policy == "clean_spatial_v1"
                else max_relations
            )
            annotations = transform_scene_graph_annotations(
                object_names=object_names,
                object_boxes_xywh=object_boxes_xywh,
                relations=raw_relations,
                trans_info=trans_info,
                image_size=self.image_size,
                min_box_size=self.min_box_size,
                max_boxes=transform_max_boxes,
                max_relations=transform_max_relations,
                selection_policy=(
                    "first"
                    if self.conditioning_policy == "clean_spatial_v1"
                    else self.selection_policy
                ),
            )
        else:
            legacy_boxes = []
            for x, y, width, height in object_boxes_xywh[:max_boxes]:
                legacy_boxes.append(
                    (
                        float(x) / max(trans_info["WW"], 1),
                        float(y) / max(trans_info["HH"], 1),
                        float(x + width) / max(trans_info["WW"], 1),
                        float(y + height) / max(trans_info["HH"], 1),
                    )
                )
            retained_objects = min(len(object_names), max_boxes)
            annotations = {
                "object_names": object_names[:retained_objects],
                "boxes": legacy_boxes,
                "relations": [
                    (int(src), int(predicate), int(dst))
                    for src, predicate, dst in raw_relations
                    if int(src) < retained_objects and int(dst) < retained_objects
                ][:max_relations],
            }

        if self.conditioning_policy == "clean_spatial_v1":
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
                max_objects=min(self.clean_max_objects, max_boxes),
                max_relations=min(self.clean_max_relations, max_relations),
                min_box_area=self.clean_min_box_area,
                min_box_side=self.clean_min_box_side,
                relation_core_min_area=self.clean_relation_core_min_area,
                duplicate_iou_threshold=self.clean_duplicate_iou_threshold,
                relation_predicates=self.clean_relation_predicates,
            )
            annotations = {
                "object_names": clean_condition.object_names,
                "boxes": clean_condition.boxes,
                "relations": clean_condition.relations,
            }

        for obj_idx, (name_idx, box) in enumerate(
            zip(annotations["object_names"], annotations["boxes"])
        ):
            boxes[obj_idx] = torch.tensor(box, dtype=torch.float32)
            masks[obj_idx] = 1.0
            object_texts[obj_idx] = str(
                self.object_idx_to_name[int(name_idx)]
            ).lower()

        relation_edges = torch.zeros(max_relations, 2, dtype=torch.float32)
        relation_masks = torch.zeros(max_relations, dtype=torch.float32)
        relation_geo_features = torch.zeros(max_relations, 12, dtype=torch.float32)
        relation_texts = [""] * max_relations
        relation_label_ids = torch.full((max_relations,), -1, dtype=torch.long)

        for rel_idx, (src, pred_idx, dst) in enumerate(annotations["relations"]):
            src = int(src)
            dst = int(dst)
            relation_edges[rel_idx] = torch.tensor([src, dst], dtype=torch.float32)
            relation_masks[rel_idx] = 1.0
            relation_label_ids[rel_idx] = int(pred_idx)
            relation_texts[rel_idx] = str(self.pred_idx_to_name[int(pred_idx)]).lower()
            relation_geo_features[rel_idx] = compute_relation_geo_features(boxes[src], boxes[dst])

        caption = self._caption_from_graph(object_texts, relation_edges, relation_masks, relation_texts)
        token_roles = [
            "object" if float(mask) > 0.5 else "padding"
            for mask in masks.detach().cpu().tolist()
        ]
        relation_token_source = torch.empty(0, dtype=torch.long)
        if self.enable_relation_grounding_tokens and self.max_relation_grounding_tokens > 0:
            relation_token_result = append_relation_grounding_tokens(
                boxes=boxes,
                masks=masks,
                object_texts=object_texts,
                relation_edges=relation_edges,
                relation_masks=relation_masks,
                relation_texts=relation_texts,
                max_relation_tokens=self.max_relation_grounding_tokens,
                phrase_template=self.relation_grounding_template,
                allowed_predicates=self.relation_grounding_allowed_predicates,
                deduplicate=self.deduplicate_relation_grounding_tokens,
                relation_mask_scale=self.relation_grounding_mask_scale,
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
            "image": image_tensor,
            "image_path": str(image_path),
            "caption": caption,
            "boxes": boxes,
            "masks": masks,
            "object_texts": object_texts,
            "grounding_token_roles": token_roles,
            "relation_token_source": relation_token_source,
            "relation_token_mask": relation_token_mask,
            "relation_edges": relation_edges,
            "relation_masks": relation_masks,
            "relation_geo_features": relation_geo_features,
            "relation_texts": relation_texts,
            "relation_label_ids": relation_label_ids,
        }
