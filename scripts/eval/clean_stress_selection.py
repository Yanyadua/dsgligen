import argparse
from dataclasses import dataclass
import json
import re
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


WHOLE_OBJECTS = {
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "people",
    "bike",
    "bicycle",
    "motorcycle",
    "skateboard",
    "snowboard",
    "surfboard",
    "ski",
    "car",
    "bus",
    "truck",
    "train",
    "boat",
    "airplane",
    "dog",
    "cat",
    "horse",
    "cow",
    "sheep",
    "bird",
    "chair",
    "bench",
    "table",
    "bed",
    "pizza",
    "book",
    "umbrella",
    "bottle",
}

PART_OBJECTS = {
    "tire",
    "wheel",
    "handle",
    "leg",
    "arm",
    "hand",
    "head",
    "hair",
    "eye",
    "nose",
    "mouth",
    "shirt",
    "pant",
    "pants",
    "short",
    "shorts",
    "shoe",
    "window",
}

CLEAN_PREDICATES = {
    "on",
    "on top of",
    "under",
    "below",
    "above",
    "inside",
    "in",
    "in front of",
    "behind",
    "holding",
    "riding",
    "wearing",
    "sitting on",
    "standing on",
    "walking on",
    "carrying",
}


@dataclass
class CleanVerdict:
    accepted: bool
    score: float
    reasons: list


def normalize_label(value):
    value = str(value).strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", value)


def box_area(box):
    x1, y1, x2, y2 = [float(value) for value in box]
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def _object_texts(record):
    return [normalize_label(obj.get("text", "")) for obj in record.get("objects", [])]


def _object_area(record, index):
    objects = record.get("objects", [])
    if index < 0 or index >= len(objects):
        return 0.0
    return box_area(objects[index].get("box_xyxy", [0, 0, 0, 0]))


def classify_clean_record(
    record,
    min_primary_area=0.015,
    max_part_ratio=0.35,
):
    texts = _object_texts(record)
    active_texts = [text for text in texts if text]
    part_count = sum(1 for text in active_texts if text in PART_OBJECTS)
    whole_count = sum(1 for text in active_texts if text in WHOLE_OBJECTS)
    part_ratio = part_count / max(len(active_texts), 1)
    reasons = []

    clean_relations = []
    for relation in record.get("relations", []):
        predicate = normalize_label(relation.get("predicate", ""))
        if predicate not in CLEAN_PREDICATES:
            continue
        src = int(relation.get("subject", -1))
        dst = int(relation.get("object", -1))
        subject_text = normalize_label(relation.get("subject_text", texts[src] if 0 <= src < len(texts) else ""))
        object_text = normalize_label(relation.get("object_text", texts[dst] if 0 <= dst < len(texts) else ""))
        if subject_text in PART_OBJECTS or object_text in PART_OBJECTS:
            continue
        if subject_text not in WHOLE_OBJECTS and object_text not in WHOLE_OBJECTS:
            continue
        if max(_object_area(record, src), _object_area(record, dst)) < min_primary_area:
            continue
        clean_relations.append(relation)

    if not clean_relations:
        reasons.append("no_clean_relation")
    if whole_count < 2:
        reasons.append("not_enough_whole_objects")
    if part_ratio > max_part_ratio:
        reasons.append("part_dominant")

    accepted = not reasons
    score = 0.0
    if accepted:
        score = (
            len(clean_relations) * 3.0
            + whole_count * 1.0
            - part_count * 1.5
            + sum(_object_area(record, idx) for idx, text in enumerate(texts) if text in WHOLE_OBJECTS)
        )
    return CleanVerdict(accepted=accepted, score=float(score), reasons=reasons)


def select_clean_stress_records(records, limit=30):
    selected = []
    seen_indices = set()
    for record in records:
        index = int(record["index"])
        if index in seen_indices:
            continue
        verdict = classify_clean_record(record)
        if not verdict.accepted:
            continue
        enriched = dict(record)
        enriched["clean_score"] = verdict.score
        enriched["clean_reasons"] = verdict.reasons
        selected.append(enriched)
        seen_indices.add(index)
        if len(selected) >= int(limit):
            break
    return selected


def dataset_item_to_record(index, item):
    objects = []
    object_texts = list(item.get("object_texts", []))
    boxes = item["boxes"]
    masks = item["masks"]
    for obj_idx, text in enumerate(object_texts):
        if float(masks[obj_idx]) <= 0.5:
            continue
        objects.append(
            {
                "index": obj_idx,
                "text": str(text),
                "box_xyxy": [float(value) for value in boxes[obj_idx].tolist()],
            }
        )

    relations = []
    relation_edges = item.get("relation_edges")
    relation_masks = item.get("relation_masks")
    relation_texts = list(item.get("relation_texts", []))
    if relation_edges is not None and relation_masks is not None:
        for rel_idx, predicate in enumerate(relation_texts):
            if float(relation_masks[rel_idx]) <= 0.5:
                continue
            src = int(relation_edges[rel_idx][0].item())
            dst = int(relation_edges[rel_idx][1].item())
            relations.append(
                {
                    "index": rel_idx,
                    "subject": src,
                    "object": dst,
                    "predicate": str(predicate),
                    "subject_text": object_texts[src] if src < len(object_texts) else "",
                    "object_text": object_texts[dst] if dst < len(object_texts) else "",
                }
            )

    return {
        "index": int(index),
        "image_id": int(item.get("id", item.get("image_id", index))),
        "image_path": str(item.get("image_path", "")),
        "caption": str(item.get("caption", "")),
        "objects": objects,
        "relations": relations,
    }


def iter_dataset_records(h5_path, vocab_path, image_root, image_size=256, max_objects=10, max_relations=15):
    from dataset.dataset_vg_scene_graph import VGSceneGraphDataset

    dataset = VGSceneGraphDataset(
        image_root=image_root,
        h5_path=h5_path,
        vocab_path=vocab_path,
        image_size=image_size,
        random_crop=False,
        random_flip=False,
        box_transform_mode="gligen",
        min_box_size=0.0,
        max_boxes_per_data=max_objects,
        max_relations_per_data=max_relations,
        selection_policy="sg2im_relation_area",
        prob_use_caption=1.0,
    )
    for index in range(len(dataset)):
        yield dataset_item_to_record(index, dataset[index])


def draw_overlay(record, image_size=256):
    from PIL import Image, ImageDraw

    image_path = Path(record["image_path"])
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = image_size / max(min(width, height), 1)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    image = image.resize((resized_width, resized_height))
    crop_x = (resized_width - image_size) // 2
    crop_y = (resized_height - image_size) // 2
    image = image.crop((crop_x, crop_y, crop_x + image_size, crop_y + image_size))
    draw = ImageDraw.Draw(image)
    colors = [
        (255, 60, 60),
        (50, 170, 255),
        (255, 180, 30),
        (70, 220, 90),
        (210, 90, 255),
        (255, 90, 180),
    ]
    for obj in record.get("objects", []):
        x1, y1, x2, y2 = [float(v) * image_size for v in obj["box_xyxy"]]
        color = colors[int(obj["index"]) % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), f'{obj["index"]}:{obj["text"]}', fill=color)
    return image


def write_outputs(selected, out_path, overlay_dir=None):
    payload = {
        "status": "diagnostic_clean_stress_candidates",
        "sample_indices": [int(record["index"]) for record in selected],
        "image_ids": [int(record["image_id"]) for record in selected],
        "records": selected,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if overlay_dir:
        overlay_dir = Path(overlay_dir)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        for record in selected:
            draw_overlay(record).save(overlay_dir / f'{int(record["index"]):04d}_{int(record["image_id"])}.png')
    return payload


def main():
    parser = argparse.ArgumentParser(description="Select clean VG controllability stress samples.")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overlay-dir", default="")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--scan-limit", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=10)
    parser.add_argument("--max-relations", type=int, default=15)
    args = parser.parse_args()

    records = iter_dataset_records(
        h5_path=args.h5,
        vocab_path=args.vocab,
        image_root=args.image_root,
        image_size=args.image_size,
        max_objects=args.max_objects,
        max_relations=args.max_relations,
    )
    if args.scan_limit > 0:
        records = (record for idx, record in enumerate(records) if idx < args.scan_limit)
    selected = select_clean_stress_records(records, limit=args.limit)
    payload = write_outputs(
        selected,
        out_path=args.out,
        overlay_dir=args.overlay_dir or None,
    )
    print(json.dumps({k: payload[k] for k in ["status", "sample_indices", "image_ids"]}, indent=2))
    print("SAMPLE_INDICES=" + ",".join(str(index) for index in payload["sample_indices"]))


if __name__ == "__main__":
    main()
