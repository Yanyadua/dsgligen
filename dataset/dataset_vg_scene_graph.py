import json
import os
import random

import torch
from PIL import Image

from .base_dataset import BaseDataset, recalculate_box_and_verify_if_valid


DEFAULT_GENERIC_OBJECTS = {
    "air", "background", "building", "carpet", "ceiling", "cloud", "curtain",
    "curtains", "drape", "floor", "grass", "ground", "road", "shade", "shadow",
    "sidewalk", "sky", "street", "wall", "window", "windows",
}

DEFAULT_PRIORITY_OBJECTS = {
    "airplane", "animal", "backpack", "bear", "bench", "bicycle", "bird", "boat",
    "bus", "car", "cat", "child", "couch", "cow", "desk", "dog", "elephant",
    "girl", "guy", "horse", "keyboard", "laptop", "man", "monitor", "motorcycle",
    "person", "sheep", "sofa", "table", "teddy bear", "train", "truck", "tv",
    "woman", "zebra",
}


def object_name(obj):
    names = obj.get("names") or [obj.get("name", "object")]
    return str(names[0]).lower()


def xywh(obj):
    return obj["x"], obj["y"], obj["w"], obj["h"]


def relation_object_id(rel, key):
    if key in rel:
        return rel[key]
    nested = rel.get(key.replace("_id", ""))
    if isinstance(nested, dict):
        return nested.get("object_id")
    return None


class VGSceneGraphDataset(BaseDataset):
    """Raw Visual Genome scene-graph dataset.

    This keeps GLIGEN's object-box grounding format while adding true scene
    graph relation tensors. Text embeddings are produced lazily by the trainer
    from object/relation strings so we do not need to rebuild a TSV upfront.
    """

    def __init__(
        self,
        image_root,
        scene_graphs_json,
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
        self.image_root = image_root
        self.scene_graphs_json = scene_graphs_json
        self.min_box_size = min_box_size
        self.max_boxes_per_data = max_boxes_per_data
        self.max_relations_per_data = max_relations_per_data
        self.generic_object_names = set(generic_object_names or DEFAULT_GENERIC_OBJECTS)
        self.priority_object_names = set(priority_object_names or DEFAULT_PRIORITY_OBJECTS)
        self.max_caption_objects = max_caption_objects
        self.max_caption_relations = max_caption_relations

        with open(scene_graphs_json, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.records = []
        for record in records:
            if len(record.get("objects", [])) < min_objects:
                continue
            if len(record.get("relationships", [])) < min_relations:
                continue
            image_path = self.resolve_image_path(record["image_id"])
            if image_path is None:
                continue
            record["_image_path"] = image_path
            self.records.append(record)
            if max_images is not None and len(self.records) >= max_images:
                break

    def resolve_image_path(self, image_id):
        filename = f"{image_id}.jpg"
        for subdir in ("VG_100K", "VG_100K_2", ""):
            path = os.path.join(self.image_root, subdir, filename)
            if os.path.exists(path):
                return path
        return None

    def total_images(self):
        return len(self)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["_image_path"]).convert("RGB")
        image_tensor, trans_info = self.transform_image(image)

        raw_relations = list(record.get("relationships", []))
        related_object_ids = set()
        for rel in raw_relations:
            for key in ("subject_id", "object_id"):
                object_id = relation_object_id(rel, key)
                if object_id is not None:
                    related_object_ids.add(object_id)

        objects = list(record.get("objects", []))

        boxes = torch.zeros(self.max_boxes_per_data, 4)
        masks = torch.zeros(self.max_boxes_per_data)
        text_embeddings = torch.zeros(self.max_boxes_per_data, 768)
        image_embeddings = torch.zeros(self.max_boxes_per_data, 768)
        object_texts = [""] * self.max_boxes_per_data
        object_id_to_idx = {}

        kept_objects = []
        for obj in objects:
            valid, (x0, y0, x1, y1) = recalculate_box_and_verify_if_valid(
                *xywh(obj), trans_info, self.image_size, self.min_box_size
            )
            if not valid:
                continue
            box = torch.tensor([x0, y0, x1, y1]) / self.image_size
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            name = object_name(obj)
            is_generic = name in self.generic_object_names
            is_priority = name in self.priority_object_names
            is_related = obj.get("object_id") in related_object_ids
            kept_objects.append((obj, box, area, is_related, is_generic, is_priority))

        # Prefer salient foreground objects over relation-heavy background
        # tokens such as wall/floor/street, then keep larger boxes.
        kept_objects.sort(key=lambda pair: (pair[5], pair[3] and not pair[4], not pair[4], pair[3], pair[2]), reverse=True)
        kept_objects = kept_objects[: self.max_boxes_per_data]
        for idx, (obj, box, _, _, _, _) in enumerate(kept_objects):
            boxes[idx] = box
            masks[idx] = 1
            text = object_name(obj)
            object_texts[idx] = text
            object_id_to_idx[obj["object_id"]] = idx

        rel_edges = torch.zeros(self.max_relations_per_data, 2)
        rel_masks = torch.zeros(self.max_relations_per_data)
        relation_texts = [""] * self.max_relations_per_data
        rel_count = 0
        for rel in raw_relations:
            subject_idx = object_id_to_idx.get(relation_object_id(rel, "subject_id"))
            object_idx = object_id_to_idx.get(relation_object_id(rel, "object_id"))
            if subject_idx is None or object_idx is None:
                continue
            if rel_count >= self.max_relations_per_data:
                break
            predicate = str(rel.get("predicate", "related to")).lower()
            rel_edges[rel_count] = torch.tensor([subject_idx, object_idx])
            rel_masks[rel_count] = 1
            relation_texts[rel_count] = predicate
            rel_count += 1

        valid_object_texts = [text for text in object_texts if text]
        valid_relation_texts = [
            f"{object_texts[int(src)]} {relation_texts[i]} {object_texts[int(dst)]}"
            for i, (src, dst) in enumerate(rel_edges.long().tolist())
            if rel_masks[i] > 0
        ]
        caption_object_texts = sorted(
            valid_object_texts,
            key=lambda text: (text in self.priority_object_names, text not in self.generic_object_names),
            reverse=True,
        )
        object_part = ", ".join(caption_object_texts[: self.max_caption_objects])
        relation_part = ". ".join(valid_relation_texts[: self.max_caption_relations])
        if object_part and relation_part:
            caption = f"A scene with {object_part}. {relation_part}."
        elif object_part:
            caption = f"A scene with {object_part}."
        else:
            caption = "A scene with objects."

        return {
            "id": record["image_id"],
            "image": image_tensor,
            "caption": caption,
            "boxes": boxes,
            "masks": masks,
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "image_masks": masks,
            "text_masks": masks,
            "object_texts": object_texts,
            "relation_edges": rel_edges,
            "relation_masks": rel_masks,
            "relation_texts": relation_texts,
        }
