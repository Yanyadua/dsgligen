from collections import Counter, OrderedDict
from pathlib import Path
import json
import os
import re

import h5py
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn


FAKE_DIR = Path(os.environ.get("FAKE_DIR", "eval_outputs/vg_fixedsplit_ours_1000/fake"))
H5_PATH = Path(os.environ.get("H5_PATH", "/root/autodl-tmp/fixed_split_work/datasets/vg/test.h5"))
VOCAB_PATH = Path(os.environ.get("VOCAB_PATH", "/root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json"))
DETECT_THRESHOLD = float(os.environ.get("DETECT_THRESHOLD", "0.50"))
MAX_RELATION_EVALS = int(os.environ.get("MAX_RELATION_EVALS", "64"))
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

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

VG_TO_COCO = {
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "guy": "person",
    "child": "person",
    "person": "person",
    "people": "person",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "airplane": "airplane",
    "plane": "airplane",
    "boat": "boat",
    "bird": "bird",
    "cat": "cat",
    "dog": "dog",
    "horse": "horse",
    "sheep": "sheep",
    "cow": "cow",
    "elephant": "elephant",
    "bear": "bear",
    "zebra": "zebra",
    "giraffe": "giraffe",
    "bench": "bench",
    "backpack": "backpack",
    "umbrella": "umbrella",
    "handbag": "handbag",
    "tie": "tie",
    "suitcase": "suitcase",
    "ball": "sports ball",
    "bottle": "bottle",
    "wine bottle": "bottle",
    "glass": "wine glass",
    "wine glass": "wine glass",
    "cup": "cup",
    "bowl": "bowl",
    "banana": "banana",
    "apple": "apple",
    "orange": "orange",
    "pizza": "pizza",
    "cake": "cake",
    "donut": "donut",
    "chair": "chair",
    "couch": "couch",
    "sofa": "couch",
    "plant": "potted plant",
    "potted plant": "potted plant",
    "bed": "bed",
    "table": "dining table",
    "dining table": "dining table",
    "tv": "tv",
    "television": "tv",
    "monitor": "tv",
    "screen": "tv",
    "laptop": "laptop",
    "mouse": "mouse",
    "keyboard": "keyboard",
    "cell phone": "cell phone",
    "phone": "cell phone",
    "book": "book",
    "clock": "clock",
    "vase": "vase",
    "teddy bear": "teddy bear",
    "refrigerator": "refrigerator",
    "fridge": "refrigerator",
    "sink": "sink",
    "oven": "oven",
    "microwave": "microwave",
    "toilet": "toilet",
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

    def full_objects(self, index):
        num_objects = int(self.h5["objects_per_image"][index])
        return [
            str(self.object_idx_to_name[int(idx)]).lower()
            for idx in self.h5["object_names"][index][:num_objects].tolist()
        ]

    def canonical_objects(self, index):
        mapped = []
        for name in self.full_objects(index):
            coco_name = VG_TO_COCO.get(name)
            if coco_name is not None:
                mapped.append(coco_name)
        return mapped

    def relation_specs(self, index):
        num_relations = int(self.h5["relationships_per_image"][index])
        object_names = self.full_objects(index)
        rels = []
        subjects = self.h5["relationship_subjects"][index][:num_relations].tolist()
        predicates = self.h5["relationship_predicates"][index][:num_relations].tolist()
        objects = self.h5["relationship_objects"][index][:num_relations].tolist()
        for s, p, o in zip(subjects, predicates, objects):
            predicate_text = str(self.pred_idx_to_name[int(p)]).lower()
            relation_label = detect_relation_label(predicate_text)
            if relation_label is None:
                continue
            subject_name = VG_TO_COCO.get(object_names[int(s)])
            object_name = VG_TO_COCO.get(object_names[int(o)])
            if subject_name is None or object_name is None:
                continue
            rels.append(
                {
                    "subject_name": subject_name,
                    "object_name": object_name,
                    "predicate_text": predicate_text,
                    "relation_label": relation_label,
                }
            )
            if len(rels) >= MAX_RELATION_EVALS:
                break
        return rels


class FasterRCNNDetector:
    def __init__(self, threshold):
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(DEVICE).eval()
        self.threshold = threshold
        self.transform = transforms.ToTensor()

    @torch.no_grad()
    def detect(self, image, queries):
        if not queries:
            return {}
        tensor = self.transform(image).to(DEVICE)
        outputs = self.model([tensor])[0]

        grouped = {query: [] for query in queries}
        allowed = set(queries)
        for score, label, box in zip(outputs["scores"], outputs["labels"], outputs["boxes"]):
            score_value = float(score.item())
            if score_value < self.threshold:
                continue
            coco_name = COCO_INSTANCE_CATEGORY_NAMES[int(label)]
            if coco_name in allowed:
                grouped[coco_name].append({"score": score_value, "box": [float(v) for v in box.tolist()]})

        out = {}
        for query, dets in grouped.items():
            dets.sort(key=lambda item: item["score"], reverse=True)
            out[query] = {
                "count": len(dets),
                "best": dets[0] if dets else None,
                "detections": dets,
            }
        return out


def run():
    if not FAKE_DIR.exists():
        raise FileNotFoundError(f"Missing FAKE_DIR: {FAKE_DIR}")
    if not H5_PATH.exists():
        raise FileNotFoundError(f"Missing H5_PATH: {H5_PATH}")
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Missing VOCAB_PATH: {VOCAB_PATH}")

    index = VGFixedSplitIndex(H5_PATH, VOCAB_PATH)
    detector = FasterRCNNDetector(DETECT_THRESHOLD)

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
        gt_objects = index.canonical_objects(sample_idx)
        relation_specs = index.relation_specs(sample_idx)
        detection_queries = sorted(set(gt_objects))

        with Image.open(image_path).convert("RGB") as image:
            detections = detector.detect(image, detection_queries)

        gt_counter = Counter(gt_objects)
        image_object_hits = 0
        for obj_name, gt_count in gt_counter.items():
            det_count = detections.get(obj_name, {}).get("count", 0)
            image_object_hits += min(gt_count, det_count)
        object_hits += image_object_hits
        object_total += len(gt_objects)

        image_relation_hits = 0
        image_relation_total = 0
        image_relation_cond_hits = 0
        image_relation_cond_total = 0

        for rel in relation_specs:
            image_relation_total += 1
            subj_det = detections.get(rel["subject_name"], {}).get("best")
            obj_det = detections.get(rel["object_name"], {}).get("best")
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
                "num_canonical_gt_objects": len(gt_objects),
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
        "protocol": "vg_fixed_split_detectable_canonical_subset",
        "object_metric_variant": "LOCI-style object-instance recall on the COCO-overlap subset using Faster R-CNN",
        "relation_metric_variant": "VISOR-style left/right/above/below satisfaction on the COCO-overlap subset using Faster R-CNN detections",
        "fake_dir": str(FAKE_DIR),
        "h5_path": str(H5_PATH),
        "vocab_path": str(VOCAB_PATH),
        "detector_model": "torchvision::fasterrcnn_resnet50_fpn",
        "detector_threshold": DETECT_THRESHOLD,
        "num_images": len(per_image),
        "max_relation_evals": MAX_RELATION_EVALS,
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
