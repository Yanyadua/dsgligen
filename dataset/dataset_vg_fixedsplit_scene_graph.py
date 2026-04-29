import json
import multiprocessing
from pathlib import Path

import h5py
import torch
from PIL import Image

from .base_dataset import BaseDataset, recalculate_box_and_verify_if_valid
from .dataset_vg_scene_graph import (
    DEFAULT_GENERIC_OBJECTS,
    DEFAULT_PRIORITY_OBJECTS,
    compute_relation_geo_features,
)


class VGFixedSplitSceneGraphDataset(BaseDataset):
    """Scene-graph dataset backed by SG2I/SGDiff-style fixed-split h5 files."""

    def __init__(
        self,
        h5_path,
        vocab_path,
        image_root,
        split_name="train",
        image_size=512,
        min_box_size=0.01,
        max_boxes_per_data=30,
        max_relations_per_data=64,
        max_images=None,
        random_crop=False,
        random_flip=True,
        min_objects=2,
        min_relations=1,
        generic_object_names=None,
        priority_object_names=None,
        max_caption_objects=8,
        max_caption_relations=4,
    ):
        super().__init__(random_crop=random_crop, random_flip=random_flip, image_size=image_size)
        self.h5_path = Path(h5_path)
        self.vocab_path = Path(vocab_path)
        self.image_root = Path(image_root)
        self.split_name = split_name
        self.min_box_size = min_box_size
        self.max_boxes_per_data = max_boxes_per_data
        self.max_relations_per_data = max_relations_per_data
        self.generic_object_names = set(generic_object_names or DEFAULT_GENERIC_OBJECTS)
        self.priority_object_names = set(priority_object_names or DEFAULT_PRIORITY_OBJECTS)
        self.max_caption_objects = max_caption_objects
        self.max_caption_relations = max_caption_relations
        self.min_objects = min_objects
        self.min_relations = min_relations
        self._h5_by_pid = {}

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.object_idx_to_name = vocab["object_idx_to_name"]
        self.pred_idx_to_name = vocab["pred_idx_to_name"]

        with h5py.File(self.h5_path, "r") as h5:
            total = int(h5["image_ids"].shape[0])
            self.indices = list(range(total))
        if max_images is not None:
            self.indices = self.indices[:max_images]

    def _get_h5(self):
        pid = multiprocessing.current_process().pid
        if pid not in self._h5_by_pid:
            self._h5_by_pid[pid] = h5py.File(self.h5_path, "r")
        return self._h5_by_pid[pid]

    def _decode_path(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def total_images(self):
        return len(self)

    def __len__(self):
        return len(self.indices)

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
        object_part = ", ".join(caption_object_texts[: self.max_caption_objects])
        relation_part = ". ".join(valid_relation_texts[: self.max_caption_relations])
        if object_part and relation_part:
            return f"A scene with {object_part}. {relation_part}."
        if object_part:
            return f"A scene with {object_part}."
        return "A scene with objects."

    def __getitem__(self, index):
        h5 = self._get_h5()
        dataset_index = self.indices[index]
        image_id = int(h5["image_ids"][dataset_index])
        image_rel_path = self._decode_path(h5["image_paths"][dataset_index])
        image_path = self.image_root / image_rel_path

        image = Image.open(image_path).convert("RGB")
        image_tensor, trans_info = self.transform_image(image)

        num_objects = int(h5["objects_per_image"][dataset_index])
        num_relations = int(h5["relationships_per_image"][dataset_index])
        max_boxes = min(int(h5["object_names"].shape[1]), self.max_boxes_per_data)
        max_relations = min(int(h5["relationship_predicates"].shape[1]), self.max_relations_per_data)

        object_names = h5["object_names"][dataset_index][:num_objects].tolist()
        object_boxes_xywh = h5["object_boxes"][dataset_index][:num_objects]
        rel_subjects = h5["relationship_subjects"][dataset_index][:num_relations].tolist()
        rel_predicates = h5["relationship_predicates"][dataset_index][:num_relations].tolist()
        rel_objects = h5["relationship_objects"][dataset_index][:num_relations].tolist()

        boxes = torch.zeros(self.max_boxes_per_data, 4, dtype=torch.float32)
        masks = torch.zeros(self.max_boxes_per_data, dtype=torch.float32)
        text_embeddings = torch.zeros(self.max_boxes_per_data, 768, dtype=torch.float32)
        image_embeddings = torch.zeros(self.max_boxes_per_data, 768, dtype=torch.float32)
        object_texts = [""] * self.max_boxes_per_data
        object_label_ids = torch.full((self.max_boxes_per_data,), -1, dtype=torch.long)

        related_object_indices = set(int(v) for pair in zip(rel_subjects, rel_objects) for v in pair)
        kept_objects = []
        for raw_obj_idx, (name_idx, xywh) in enumerate(zip(object_names, object_boxes_xywh)):
            if raw_obj_idx >= max_boxes:
                break
            x, y, w, h = [float(v) for v in xywh.tolist()]
            valid, (x0, y0, x1, y1) = recalculate_box_and_verify_if_valid(
                x, y, w, h, trans_info, self.image_size, self.min_box_size
            )
            if not valid:
                continue
            box = torch.tensor([x0, y0, x1, y1], dtype=torch.float32) / self.image_size
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            object_name = str(self.object_idx_to_name[int(name_idx)]).lower()
            is_generic = object_name in self.generic_object_names
            is_priority = object_name in self.priority_object_names
            is_related = raw_obj_idx in related_object_indices
            kept_objects.append((raw_obj_idx, int(name_idx), object_name, box, area, is_related, is_generic, is_priority))

        kept_objects.sort(
            key=lambda row: (row[7], row[5] and not row[6], not row[6], row[5], row[4]),
            reverse=True,
        )
        kept_objects = kept_objects[: self.max_boxes_per_data]

        raw_to_kept = {}
        for kept_idx, (raw_obj_idx, name_idx, object_name, box, _, _, _, _) in enumerate(kept_objects):
            boxes[kept_idx] = box
            masks[kept_idx] = 1.0
            object_texts[kept_idx] = object_name
            object_label_ids[kept_idx] = int(name_idx)
            raw_to_kept[raw_obj_idx] = kept_idx

        relation_edges = torch.zeros(self.max_relations_per_data, 2, dtype=torch.float32)
        relation_masks = torch.zeros(self.max_relations_per_data, dtype=torch.float32)
        relation_geo_features = torch.zeros(self.max_relations_per_data, 12, dtype=torch.float32)
        relation_texts = [""] * self.max_relations_per_data
        relation_label_ids = torch.full((self.max_relations_per_data,), -1, dtype=torch.long)

        rel_count = 0
        for src_raw, pred_idx, dst_raw in zip(rel_subjects, rel_predicates, rel_objects):
            src = raw_to_kept.get(int(src_raw))
            dst = raw_to_kept.get(int(dst_raw))
            if src is None or dst is None:
                continue
            if rel_count >= self.max_relations_per_data:
                break
            relation_edges[rel_count] = torch.tensor([src, dst], dtype=torch.float32)
            relation_masks[rel_count] = 1.0
            relation_texts[rel_count] = str(self.pred_idx_to_name[int(pred_idx)]).lower()
            relation_label_ids[rel_count] = int(pred_idx)
            relation_geo_features[rel_count] = compute_relation_geo_features(boxes[src], boxes[dst])
            rel_count += 1

        if masks.sum() < self.min_objects or relation_masks.sum() < self.min_relations:
            return self[(index + 1) % len(self)]

        caption = self._caption_from_graph(object_texts, relation_edges, relation_masks, relation_texts)

        return {
            "id": image_id,
            "image": image_tensor,
            "caption": caption,
            "boxes": boxes,
            "masks": masks,
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "image_masks": masks,
            "text_masks": masks,
            "object_texts": object_texts,
            "object_label_ids": object_label_ids,
            "relation_edges": relation_edges,
            "relation_masks": relation_masks,
            "relation_geo_features": relation_geo_features,
            "relation_texts": relation_texts,
            "relation_label_ids": relation_label_ids,
        }
