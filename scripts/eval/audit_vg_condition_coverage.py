"""Build a reviewable annotation-coverage manifest for fixed VG samples.

This is deliberately an audit/template generator, not a generation or metric
script.  It reads one VG H5 split, records the raw graph and the compact clean
conditioning graph, and optionally compares them with a small manually reviewed
``expected`` JSON file.

Expected JSON format::

    {
      "284": {
        "primary_labels": ["person", "skateboard"],
        "relations": [
          {"subject": "person", "predicate": "on", "object": "skateboard"}
        ]
      }
    }

Keys may be dataset indices or image ids.  Without this file, records are
marked ``unreviewed`` rather than silently claiming that the graph is correct.
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import sys

import h5py
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.scene_graph_conditioning import build_clean_scene_graph_condition


DEFAULT_SAMPLE_INDICES = [
    284, 388, 513, 798, 919, 1008, 1048, 1277, 1760, 1859,
    1897, 1940, 1978, 2022, 2194, 2295, 2313, 2446, 2591, 2855,
    2942, 3360, 3530, 3544, 3651, 3787, 4545, 4742, 4786, 5000,
]


def normalize_text(value):
    value = str(value).strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", value)


def parse_indices(value, length):
    if value is None or not str(value).strip():
        indices = list(DEFAULT_SAMPLE_INDICES)
    else:
        indices = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(indices) != len(set(indices)):
        raise ValueError("sample indices must be unique")
    invalid = [index for index in indices if index < 0 or index >= length]
    if invalid:
        raise ValueError(f"sample indices out of range: {invalid}")
    return indices


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _xywh_to_xyxy(box, width, height):
    x, y, w, h = [float(value) for value in box]
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    return [
        max(0.0, min(1.0, x / width)),
        max(0.0, min(1.0, y / height)),
        max(0.0, min(1.0, (x + w) / width)),
        max(0.0, min(1.0, (y + h) / height)),
    ]


def box_area(box):
    x0, y0, x1, y1 = [float(value) for value in box]
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def _expected_for(expected, dataset_index, image_id):
    if not expected:
        return None
    value = expected.get(str(dataset_index))
    if value is None:
        value = expected.get(str(image_id))
    return value


def _relation_signature(subject, predicate, object_label):
    return (
        normalize_text(subject),
        normalize_text(predicate),
        normalize_text(object_label),
    )


def _coverage(raw_objects, raw_relations, expected):
    if expected is None:
        return {
            "annotation_status": "unreviewed",
            "expected_primary_labels": [],
            "supported_primary_labels": [],
            "missing_primary_labels": [],
            "expected_relations": [],
            "supported_relations": [],
            "missing_relations": [],
        }

    labels = {normalize_text(obj["label"]) for obj in raw_objects}
    expected_labels = [
        normalize_text(value) for value in expected.get("primary_labels", [])
    ]
    supported_labels = [value for value in expected_labels if value in labels]
    missing_labels = [value for value in expected_labels if value not in labels]

    graph_relations = {
        _relation_signature(
            relation["subject_text"],
            relation["predicate"],
            relation["object_text"],
        )
        for relation in raw_relations
    }
    expected_relations = []
    supported_relations = []
    missing_relations = []
    for relation in expected.get("relations", []):
        signature = _relation_signature(
            relation.get("subject", ""),
            relation.get("predicate", ""),
            relation.get("object", ""),
        )
        expected_relations.append(
            {
                "subject": signature[0],
                "predicate": signature[1],
                "object": signature[2],
            }
        )
        if signature in graph_relations:
            supported_relations.append(
                {
                    "subject": signature[0],
                    "predicate": signature[1],
                    "object": signature[2],
                }
            )
        else:
            missing_relations.append(
                {
                    "subject": signature[0],
                    "predicate": signature[1],
                    "object": signature[2],
                }
            )

    if missing_labels or missing_relations:
        status = "unsupported" if len(supported_labels) == 0 else "partial"
    else:
        status = "supported"
    return {
        "annotation_status": status,
        "expected_primary_labels": expected_labels,
        "supported_primary_labels": supported_labels,
        "missing_primary_labels": missing_labels,
        "expected_relations": expected_relations,
        "supported_relations": supported_relations,
        "missing_relations": missing_relations,
    }


def build_record(h5, vocab, image_root, index, expected=None, clean_args=None):
    object_names = vocab["object_idx_to_name"]
    predicate_names = vocab["pred_idx_to_name"]
    image_id = int(h5["image_ids"][index])
    rel_path = _decode(h5["image_paths"][index])
    image_path = Path(image_root) / rel_path
    with Image.open(image_path) as image:
        width, height = image.size

    object_count = int(h5["objects_per_image"][index])
    relation_count = int(h5["relationships_per_image"][index])
    raw_objects = []
    object_boxes = []
    object_texts = []
    for object_index in range(object_count):
        label_index = int(h5["object_names"][index, object_index])
        label = normalize_text(object_names[label_index])
        box = _xywh_to_xyxy(
            h5["object_boxes"][index, object_index],
            width,
            height,
        )
        object_boxes.append(box)
        object_texts.append(label)
        raw_objects.append(
            {
                "index": object_index,
                "label_id": label_index,
                "label": label,
                "box_xyxy": box,
                "area": box_area(box),
            }
        )

    raw_relations = []
    relation_tuples = []
    relation_texts = []
    for relation_index in range(relation_count):
        src = int(h5["relationship_subjects"][index, relation_index])
        predicate_id = int(h5["relationship_predicates"][index, relation_index])
        dst = int(h5["relationship_objects"][index, relation_index])
        predicate = normalize_text(predicate_names[predicate_id])
        relation_tuples.append((src, predicate_id, dst))
        relation_texts.append(predicate)
        raw_relations.append(
            {
                "index": relation_index,
                "subject": src,
                "subject_text": object_texts[src] if 0 <= src < len(object_texts) else "",
                "predicate_id": predicate_id,
                "predicate": predicate,
                "object": dst,
                "object_text": object_texts[dst] if 0 <= dst < len(object_texts) else "",
            }
        )

    clean = build_clean_scene_graph_condition(
        object_names=list(range(object_count)),
        object_texts=object_texts,
        boxes=object_boxes,
        relations=relation_tuples,
        relation_texts=relation_texts,
        **clean_args,
    )
    clean_objects = [
        {
            "index": int(index),
            "label": normalize_text(text),
            "box_xyxy": [float(value) for value in box],
            "area": box_area(box),
        }
        for index, text, box in zip(clean.trace["selected_original_object_indices"], clean.object_texts, clean.boxes)
    ]
    clean_relations = [
        {
            "subject": int(src),
            "predicate_id": int(predicate_id),
            "predicate": normalize_text(clean.relation_texts[relation_index]),
            "object": int(dst),
        }
        for relation_index, (src, predicate_id, dst) in enumerate(clean.relations)
    ]

    coverage = _coverage(raw_objects, raw_relations, expected)
    return {
        "dataset_index": int(index),
        "image_id": image_id,
        "image_path": str(image_path),
        "image_size": [int(width), int(height)],
        "raw_object_count": object_count,
        "raw_relation_count": relation_count,
        "raw_objects": raw_objects,
        "raw_relations": raw_relations,
        "clean_objects": clean_objects,
        "clean_relations": clean_relations,
        "clean_trace": clean.trace,
        "coverage": coverage,
    }


def write_csv(records, path):
    fields = [
        "dataset_index", "image_id", "annotation_status", "raw_object_count",
        "raw_relation_count", "clean_object_count", "clean_relation_count",
        "expected_primary_labels", "missing_primary_labels", "expected_relations",
        "missing_relations",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            coverage = record["coverage"]
            writer.writerow(
                {
                    "dataset_index": record["dataset_index"],
                    "image_id": record["image_id"],
                    "annotation_status": coverage["annotation_status"],
                    "raw_object_count": record["raw_object_count"],
                    "raw_relation_count": record["raw_relation_count"],
                    "clean_object_count": len(record["clean_objects"]),
                    "clean_relation_count": len(record["clean_relations"]),
                    "expected_primary_labels": ";".join(coverage["expected_primary_labels"]),
                    "missing_primary_labels": ";".join(coverage["missing_primary_labels"]),
                    "expected_relations": ";".join(
                        f"{r['subject']}|{r['predicate']}|{r['object']}"
                        for r in coverage["expected_relations"]
                    ),
                    "missing_relations": ";".join(
                        f"{r['subject']}|{r['predicate']}|{r['object']}"
                        for r in coverage["missing_relations"]
                    ),
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-out", default="")
    parser.add_argument("--template-out", default="")
    parser.add_argument("--split-name", default="unspecified")
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--expected-json", default="")
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--max-relations", type=int, default=1)
    parser.add_argument("--min-box-area", type=float, default=0.0025)
    parser.add_argument("--min-box-side", type=float, default=0.035)
    parser.add_argument("--relation-core-min-area", type=float, default=0.0015)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.85)
    args = parser.parse_args()

    with open(args.vocab, "r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    expected = None
    if args.expected_json:
        with open(args.expected_json, "r", encoding="utf-8") as handle:
            expected = json.load(handle)

    clean_args = {
        "max_objects": args.max_objects,
        "max_relations": args.max_relations,
        "min_box_area": args.min_box_area,
        "min_box_side": args.min_box_side,
        "relation_core_min_area": args.relation_core_min_area,
        "duplicate_iou_threshold": args.duplicate_iou_threshold,
    }
    with h5py.File(args.h5, "r") as h5:
        indices = parse_indices(args.sample_indices, int(h5["image_ids"].shape[0]))
        records = []
        for index in indices:
            image_id = int(h5["image_ids"][index])
            records.append(
                build_record(
                    h5,
                    vocab,
                    args.image_root,
                    index,
                    expected=_expected_for(expected, index, image_id),
                    clean_args=clean_args,
                )
            )
        protocol = {
            "protocol": "vg_condition_coverage_audit",
            "split_name": args.split_name,
            "h5_path": str(args.h5),
            "vocab_path": str(args.vocab),
            "image_root": str(args.image_root),
            "h5_length": int(h5["image_ids"].shape[0]),
            "sample_indices": indices,
            "sample_count": len(indices),
            "vocab_object_count": len(vocab["object_idx_to_name"]),
            "vocab_predicate_count": len(vocab["pred_idx_to_name"]),
            "clean_condition_parameters": clean_args,
            "expected_json": str(args.expected_json),
            "h5_sha256": sha256_file(args.h5),
            "vocab_sha256": sha256_file(args.vocab),
        }

    status_counts = {}
    for record in records:
        status = record["coverage"]["annotation_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    payload = {
        "protocol": protocol,
        "summary": {"annotation_status_counts": status_counts},
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(records, csv_path)
    if args.template_out:
        template = {
            str(record["dataset_index"]): {
                "primary_labels": [],
                "relations": [],
                "notes": "Fill from the reference image; use canonical VG labels where possible.",
            }
            for record in records
        }
        template_path = Path(args.template_out)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(
            json.dumps(template, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"out": str(out_path), "sample_count": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
