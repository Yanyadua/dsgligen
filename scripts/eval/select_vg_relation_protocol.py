"""Create disjoint VG relation-development candidates and a locked holdout.

This only reads the fixed H5 split.  It does not generate images, train, or
score a model.  Its purpose is to stop repeatedly tuning on the same manually
viewed test examples.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import h5py

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.scene_graph_conditioning import (  # noqa: E402
    DEFAULT_SPATIAL_PREDICATES,
    build_clean_scene_graph_condition,
)
from scripts.eval.audit_vg_condition_coverage import (  # noqa: E402
    _decode,
    _xywh_to_xyxy,
    box_area,
    sha256_file,
)


# These were manually reviewed during earlier GLIGEN diagnostics.  Do not use
# them to claim an unseen result, even if a new backbone did not see pixels.
DEFAULT_PREVIOUSLY_VIEWED = {
    284, 388, 513, 798, 919, 1008, 1048, 1277, 1760, 1859,
    1897, 1940, 1978, 2022, 2194, 2295, 2313, 2446, 2591, 2855,
    2942, 3360, 3530, 3544, 3651, 3787, 4545, 4742, 4786, 5000,
}

LOW_VALUE = {
    "background", "edge", "floor", "ground", "line", "light", "reflection",
    "shadow", "sky", "tile", "wall", "frame", "cloud", "grass", "face",
    "hair", "hand", "foot", "feet", "leg", "legs", "arm", "arms", "head",
    "wing", "wings", "paw", "paws", "pant", "pants", "shoe", "shoes",
    "shirt", "jacket", "top", "coat",
}

SEMANTIC_MAJOR = {
    "person", "man", "woman", "boy", "girl", "lady", "child", "dog", "cat",
    "bird", "horse", "elephant", "bear", "car", "bus", "truck", "bicycle",
    "motorcycle", "boat", "airplane", "train", "cup", "bottle", "book",
    "plant", "tree", "building", "road", "street", "cabinet", "table", "bed",
    "clock", "tire", "water", "snow", "food", "chair", "bench", "sign",
}


def parse_indices(value):
    if not value.strip():
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def stable_key(namespace, index):
    return hashlib.sha256(f"{namespace}:{index}".encode("utf-8")).hexdigest()


def candidate_record(h5, vocab, image_root, index):
    image_id = int(h5["image_ids"][index])
    image_path = Path(image_root) / _decode(h5["image_paths"][index])
    # PIL is intentionally avoided: the H5 dimensions are enough to normalize
    # raw boxes for this pre-selection.  Human review later checks actual crops.
    from PIL import Image
    with Image.open(image_path) as image:
        width, height = image.size
    num_objects = int(h5["objects_per_image"][index])
    num_relations = int(h5["relationships_per_image"][index])
    names = h5["object_names"][index][:num_objects].tolist()
    boxes = [_xywh_to_xyxy(box, width, height) for box in h5["object_boxes"][index][:num_objects]]
    labels = [str(vocab["object_idx_to_name"][int(name)]).strip().lower() for name in names]
    relations = list(zip(
        h5["relationship_subjects"][index][:num_relations].tolist(),
        h5["relationship_predicates"][index][:num_relations].tolist(),
        h5["relationship_objects"][index][:num_relations].tolist(),
    ))
    relation_texts = [str(vocab["pred_idx_to_name"][int(pred)]).strip().lower() for _, pred, _ in relations]
    clean = build_clean_scene_graph_condition(
        object_names=names,
        object_texts=labels,
        boxes=boxes,
        relations=relations,
        relation_texts=relation_texts,
        max_objects=8,
        max_relations=1,
    )
    if len(clean.relations) != 1:
        return None
    src, _, dst = clean.relations[0]
    subject, object_label = clean.object_texts[src], clean.object_texts[dst]
    subject_area, object_area = box_area(clean.boxes[src]), box_area(clean.boxes[dst])
    if subject in LOW_VALUE or object_label in LOW_VALUE:
        return None
    if min(subject_area, object_area) < 0.0125:
        return None
    if subject not in SEMANTIC_MAJOR and object_label not in SEMANTIC_MAJOR:
        return None
    predicate = clean.relation_texts[0]
    score = (
        4.0 * int(subject in SEMANTIC_MAJOR)
        + 4.0 * int(object_label in SEMANTIC_MAJOR)
        + min(subject_area, 0.20)
        + max(subject_area, object_area)
    )
    return {
        "dataset_index": int(index),
        "image_id": image_id,
        "image_path": str(image_path),
        "relation": {"subject": subject, "predicate": predicate, "object": object_label},
        "endpoint_boxes_xyxy": [list(clean.boxes[src]), list(clean.boxes[dst])],
        "endpoint_areas": [subject_area, object_area],
        "clean_object_texts": clean.object_texts,
        "clean_trace": clean.trace,
        "selection_score": score,
    }


def balanced_pick(candidates, limit, namespace):
    groups = defaultdict(list)
    for record in candidates:
        groups[record["relation"]["predicate"]].append(record)
    for predicate in groups:
        groups[predicate].sort(
            key=lambda r: (-r["selection_score"], stable_key(namespace, r["dataset_index"]))
        )
    selected = []
    while len(selected) < limit:
        progressed = False
        for predicate in sorted(groups):
            if not groups[predicate] or len(selected) >= limit:
                continue
            selected.append(groups[predicate].pop(0))
            progressed = True
        if not progressed:
            break
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h5", required=True, type=Path)
    parser.add_argument("--test-h5", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dev-candidate-count", type=int, default=40)
    parser.add_argument("--locked-holdout-count", type=int, default=200)
    parser.add_argument("--extra-excluded-indices", default="")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {args.out_dir}")
    with args.vocab.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    excluded = DEFAULT_PREVIOUSLY_VIEWED | parse_indices(args.extra_excluded_indices)
    with h5py.File(args.train_h5, "r") as train_h5, h5py.File(args.test_h5, "r") as test_h5:
        train_ids = set(int(value) for value in train_h5["image_ids"][:])
        test_ids = set(int(value) for value in test_h5["image_ids"][:])
        overlap = sorted(train_ids & test_ids)
        if overlap:
            raise RuntimeError(f"train/test image-id overlap detected: {len(overlap)}")
        candidates = []
        for index in range(len(test_h5["image_ids"])):
            if index in excluded:
                continue
            record = candidate_record(test_h5, vocab, args.image_root, index)
            if record:
                candidates.append(record)
    dev = balanced_pick(candidates, args.dev_candidate_count, "vg-relation-dev-v1")
    dev_indices = {record["dataset_index"] for record in dev}
    remaining = [record for record in candidates if record["dataset_index"] not in dev_indices]
    locked = sorted(remaining, key=lambda r: stable_key("vg-relation-locked-v1", r["dataset_index"]))[:args.locked_holdout_count]
    args.out_dir.mkdir(parents=True)
    common = {
        "status": "candidate protocol only; no generation, training, or metric",
        "test_h5": str(args.test_h5.resolve()),
        "test_h5_sha256": sha256_file(args.test_h5),
        "train_h5": str(args.train_h5.resolve()),
        "train_h5_sha256": sha256_file(args.train_h5),
        "vocab": str(args.vocab.resolve()),
        "vocab_sha256": sha256_file(args.vocab),
        "train_test_image_id_overlap": 0,
        "excluded_previously_viewed_indices": sorted(excluded),
        "candidate_count_after_filters": len(candidates),
        "filter": "clean_spatial_v1, one spatial relation, non-low-value endpoints, endpoint area >= 0.0125",
    }
    (args.out_dir / "dev_candidates_unreviewed.json").write_text(
        json.dumps({**common, "role": "development candidates; manual review required", "records": dev}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "locked_holdout_do_not_review.json").write_text(
        json.dumps({**common, "role": "locked holdout; do not inspect until condition/model freeze", "records": locked}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"candidate_count": len(candidates), "dev_count": len(dev), "locked_count": len(locked)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
