from collections import OrderedDict, defaultdict
from pathlib import Path
import json
import os
import re

import h5py
import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor


FAKE_DIR = Path(os.environ.get("FAKE_DIR", "eval_outputs/vg_fixedsplit_ours_1000/fake"))
H5_PATH = Path(os.environ.get("H5_PATH", "/root/autodl-tmp/fixed_split_work/datasets/vg/test.h5"))
VOCAB_PATH = Path(os.environ.get("VOCAB_PATH", "/root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json"))
MODEL_NAME = os.environ.get("MODEL_NAME", "google/owlvit-base-patch32")
DETECT_THRESHOLD = float(os.environ.get("DETECT_THRESHOLD", "0.10"))
MAX_CAPTION_OBJECTS = int(os.environ.get("MAX_CAPTION_OBJECTS", "8"))
MAX_CAPTION_RELATIONS = int(os.environ.get("MAX_CAPTION_RELATIONS", "4"))
OUTPUT_JSON = os.environ.get("OUTPUT_JSON")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GEOMETRIC_RELATION_PATTERNS = OrderedDict(
    [
        ("left", re.compile(r"\b(left|left of|to the left of)\b")),
        ("right", re.compile(r"\b(right|right of|to the right of)\b")),
        ("above", re.compile(r"\b(above|over|on top of)\b")),
        ("below", re.compile(r"\b(below|under|beneath)\b")),
    ]
)

GENERIC_OBJECTS = {
    "air", "background", "building", "carpet", "ceiling", "cloud", "curtain",
    "curtains", "drape", "floor", "grass", "ground", "road", "shade", "shadow",
    "sidewalk", "sky", "street", "wall", "window", "windows",
}
PRIORITY_OBJECTS = {
    "airplane", "animal", "backpack", "bear", "bench", "bicycle", "bird", "boat",
    "bus", "car", "cat", "child", "couch", "cow", "desk", "dog", "elephant",
    "girl", "guy", "horse", "keyboard", "laptop", "man", "monitor", "motorcycle",
    "person", "sheep", "sofa", "table", "teddy bear", "train", "truck", "tv",
    "woman", "zebra",
}


def detect_relation_label(text):
    lowered = text.lower()
    for label, pattern in GEOMETRIC_RELATION_PATTERNS.items():
        if pattern.search(lowered):
            return label
    return None


def expected_relation(box_a, box_b):
    ax = 0.5 * (box_a[0] + box_a[2])
    ay = 0.5 * (box_a[1] + box_a[3])
    bx = 0.5 * (box_b[0] + box_b[2])
    by = 0.5 * (box_b[1] + box_b[3])
    dx = ax - bx
    dy = ay - by
    if abs(dx) >= abs(dy):
        return "left" if dx < 0 else "right"
    return "above" if dy < 0 else "below"


def dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


class VGFixedSplitIndex:
    def __init__(self, h5_path, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.object_idx_to_name = vocab["object_idx_to_name"]
        self.pred_idx_to_name = vocab["pred_idx_to_name"]
        self.h5 = h5py.File(h5_path, "r")
        self.image_id_to_index = {
            int(image_id): idx for idx, image_id in enumerate(self.h5["image_ids"][:].tolist())
        }

    def caption_objects(self, index):
        num_objects = int(self.h5["objects_per_image"][index])
        object_names = [
            str(self.object_idx_to_name[int(idx)]).lower()
            for idx in self.h5["object_names"][index][:num_objects].tolist()
        ]
        sorted_names = sorted(
            object_names,
            key=lambda text: (text in PRIORITY_OBJECTS, text not in GENERIC_OBJECTS),
            reverse=True,
        )
        return dedupe_keep_order(sorted_names[:MAX_CAPTION_OBJECTS])

    def caption_relations(self, index):
        num_relations = int(self.h5["relationships_per_image"][index])
        object_names = [
            str(self.object_idx_to_name[int(idx)]).lower()
            for idx in self.h5["object_names"][index].tolist()
        ]
        rels = []
        subjects = self.h5["relationship_subjects"][index][:num_relations].tolist()
        predicates = self.h5["relationship_predicates"][index][:num_relations].tolist()
        objects = self.h5["relationship_objects"][index][:num_relations].tolist()
        for s, p, o in zip(subjects, predicates, objects):
            predicate_text = str(self.pred_idx_to_name[int(p)]).lower()
            relation_label = detect_relation_label(predicate_text)
            if relation_label is None:
                continue
            rels.append(
                {
                    "subject_name": object_names[int(s)],
                    "object_name": object_names[int(o)],
                    "predicate_text": predicate_text,
                    "relation_label": relation_label,
                }
            )
            if len(rels) >= MAX_CAPTION_RELATIONS:
                break
        return rels


class OwlDetector:
    def __init__(self, model_name, threshold):
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(DEVICE).eval()
        self.threshold = threshold

    @torch.no_grad()
    def detect(self, image, queries):
        if not queries:
            return {}
        inputs = self.processor(text=[queries], images=image, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )[0]

        best = {}
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            query = queries[int(label)]
            score_value = float(score.item())
            box_value = [float(v) for v in box.tolist()]
            if query not in best or score_value > best[query]["score"]:
                best[query] = {"score": score_value, "box": box_value}
        return best


def run():
    if not FAKE_DIR.exists():
        raise FileNotFoundError(f"Missing FAKE_DIR: {FAKE_DIR}")
    if not H5_PATH.exists():
        raise FileNotFoundError(f"Missing H5_PATH: {H5_PATH}")
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Missing VOCAB_PATH: {VOCAB_PATH}")

    index = VGFixedSplitIndex(H5_PATH, VOCAB_PATH)
    detector = OwlDetector(MODEL_NAME, DETECT_THRESHOLD)

    image_paths = sorted(FAKE_DIR.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No generated png files found in {FAKE_DIR}")

    object_hits = 0
    object_total = 0
    relation_hits = 0
    relation_total = 0
    relation_cond_hits = 0
    relation_cond_total = 0

    per_image = []

    for image_path in image_paths:
        image_id = int(image_path.stem)
        if image_id not in index.image_id_to_index:
            continue
        sample_idx = index.image_id_to_index[image_id]
        object_queries = index.caption_objects(sample_idx)
        relation_specs = index.caption_relations(sample_idx)

        with Image.open(image_path).convert("RGB") as image:
            detections = detector.detect(image, object_queries)

        image_object_hits = sum(1 for obj in object_queries if obj in detections)
        object_hits += image_object_hits
        object_total += len(object_queries)

        image_relation_hits = 0
        image_relation_total = 0
        image_relation_cond_hits = 0
        image_relation_cond_total = 0

        for rel in relation_specs:
            image_relation_total += 1
            subj_det = detections.get(rel["subject_name"])
            obj_det = detections.get(rel["object_name"])
            if subj_det is not None and obj_det is not None:
                image_relation_cond_total += 1
                pred = expected_relation(subj_det["box"], obj_det["box"])
                if pred == rel["relation_label"]:
                    image_relation_hits += 1
                    image_relation_cond_hits += 1

        relation_hits += image_relation_hits
        relation_total += image_relation_total
        relation_cond_hits += image_relation_cond_hits
        relation_cond_total += image_relation_cond_total

        per_image.append(
            {
                "image_id": image_id,
                "num_object_queries": len(object_queries),
                "object_hits": image_object_hits,
                "num_relation_queries": image_relation_total,
                "relation_hits": image_relation_hits,
                "relation_cond_hits": image_relation_cond_hits,
                "relation_cond_total": image_relation_cond_total,
            }
        )

    object_occurrence_rate = 100.0 * object_hits / max(object_total, 1)
    relation_satisfaction_rate = 100.0 * relation_hits / max(relation_total, 1)
    relation_satisfaction_rate_cond = 100.0 * relation_cond_hits / max(relation_cond_total, 1)

    summary = {
        "protocol": "vg_fixed_split_spatial_subset",
        "fake_dir": str(FAKE_DIR),
        "h5_path": str(H5_PATH),
        "vocab_path": str(VOCAB_PATH),
        "detector_model": MODEL_NAME,
        "detector_threshold": DETECT_THRESHOLD,
        "num_images": len(per_image),
        "max_caption_objects": MAX_CAPTION_OBJECTS,
        "max_caption_relations": MAX_CAPTION_RELATIONS,
        "object_occurrence_rate": object_occurrence_rate,
        "relation_satisfaction_rate": relation_satisfaction_rate,
        "relation_satisfaction_rate_cond": relation_satisfaction_rate_cond,
        "object_total": object_total,
        "relation_total": relation_total,
        "relation_cond_total": relation_cond_total,
        "per_image": per_image,
    }

    print(
        "OOR={:.4f} RSR={:.4f} RSR_cond={:.4f} images={} object_total={} relation_total={} relation_cond_total={}".format(
            object_occurrence_rate,
            relation_satisfaction_rate,
            relation_satisfaction_rate_cond,
            len(per_image),
            object_total,
            relation_total,
            relation_cond_total,
        ),
        flush=True,
    )

    if OUTPUT_JSON:
        output_path = Path(OUTPUT_JSON)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {output_path}", flush=True)


if __name__ == "__main__":
    run()
