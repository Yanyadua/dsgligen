"""Export a protocol-pinned VG subset for instance-level layout generators.

This is intentionally an *input exporter*, not a training or metric script.
It applies the same deterministic GLIGEN center crop and clean-spatial object
selection used by the existing fixed-split diagnostic, then writes one JSON
file per sample in the InstanceDiffusion box-input schema.  Keeping this
boundary separate lets us test whether a new backbone actually obeys boxes
before adding any relation-specific module.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.scene_graph_box_utils import (
    compute_center_crop_transform,
    transform_scene_graph_annotations,
)
from dataset.scene_graph_caption import build_clean_primary_scene_graph_caption
from dataset.scene_graph_conditioning import build_clean_scene_graph_condition
from scripts.eval.sample_selection import parse_sample_indices


DEFAULT_STYLE_PREFIX = "A full-color realistic DSLR photograph"
DEFAULT_STYLE_SUFFIX = "vivid natural colors, realistic color photography, natural lighting"

# These labels are useful annotations but make poor independent generation
# targets.  A relation endpoint is exempt: e.g. ``water inside cup`` must keep
# water despite its contextual nature.
LOW_VALUE_INSTANCE_LABELS = frozenset(
    {
        "background", "edge", "floor", "ground", "line", "light", "reflection",
        "shadow", "sky", "tile", "wall", "frame", "cloud", "grass", "face",
        "hair", "hand", "foot", "feet", "leg", "legs", "arm", "arms", "head",
        "wing", "wings", "paw", "paws", "pant", "pants", "shoe", "shoes",
        "shirt", "jacket", "top", "coat",
    }
)

PRIMARY_INSTANCE_LABELS = frozenset(
    {
        "person", "man", "woman", "boy", "girl", "lady", "child", "dog", "cat",
        "bird", "horse", "elephant", "bear", "car", "bus", "truck", "bicycle",
        "motorcycle", "boat", "airplane", "train", "cup", "bottle", "book",
        "plant", "tree", "building", "road", "street", "cabinet", "table", "bed",
        "clock", "tire", "water", "snow", "food",
    }
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_path(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def xyxy_normalized_to_xywh_pixels(box, size):
    x0, y0, x1, y1 = (float(value) for value in box)
    x = max(0.0, min(float(size), x0 * size))
    y = max(0.0, min(float(size), y0 * size))
    right = max(x, min(float(size), x1 * size))
    bottom = max(y, min(float(size), y1 * size))
    # InstanceDiffusion documents integer-pixel [xmin, ymin, width, height].
    return [round(x, 4), round(y, 4), round(right - x, 4), round(bottom - y, 4)]


def box_area(box):
    x0, y0, x1, y1 = (float(value) for value in box)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def object_phrase(label):
    label = str(label).strip()
    article = "an" if label[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {label}"


def select_major_instance_layout(condition, max_major_objects):
    """Make a small per-instance layout bundle without breaking relation IDs."""
    relation_endpoints = {
        endpoint for relation in condition.relations for endpoint in (int(relation[0]), int(relation[2]))
    }
    all_indices = list(range(len(condition.object_texts)))
    ranked = sorted(
        all_indices,
        key=lambda index: (
            0 if index in relation_endpoints else 1,
            0 if condition.object_texts[index] in PRIMARY_INSTANCE_LABELS else 1,
            1 if condition.object_texts[index] in LOW_VALUE_INSTANCE_LABELS else 0,
            -box_area(condition.boxes[index]),
            index,
        ),
    )
    selected = []
    seen_noncore_labels = set()
    for index in ranked:
        label = condition.object_texts[index]
        is_relation_endpoint = index in relation_endpoints
        if not is_relation_endpoint and label in LOW_VALUE_INSTANCE_LABELS:
            continue
        # A box-conditioned generator treats every retained phrase as an
        # explicit target.  Beyond the relation endpoints, do not fill spare
        # slots with arbitrary VG parts/attributes such as ear or windshield.
        if not is_relation_endpoint and label not in PRIMARY_INSTANCE_LABELS:
            continue
        # Repeated category nouns give a box model ambiguous text bindings.
        # Keep repeats only when a relation explicitly requires the instance.
        if not is_relation_endpoint and label in seen_noncore_labels:
            continue
        selected.append(index)
        seen_noncore_labels.add(label)
        if len(selected) >= max_major_objects:
            break
    selected.sort()
    old_to_new = {old: new for new, old in enumerate(selected)}
    relations = [
        (old_to_new[int(src)], int(predicate), old_to_new[int(dst)])
        for src, predicate, dst in condition.relations
        if int(src) in old_to_new and int(dst) in old_to_new
    ]
    relation_texts = [
        text for relation, text in zip(condition.relations, condition.relation_texts)
        if int(relation[0]) in old_to_new and int(relation[2]) in old_to_new
    ]
    return {
        "object_texts": [condition.object_texts[index] for index in selected],
        "boxes": [condition.boxes[index] for index in selected],
        "relations": relations,
        "relation_texts": relation_texts,
        "selected_clean_object_indices": selected,
    }


def build_export_record(h5, vocab, image_root, index, image_size, max_objects, max_relations, max_major_objects):
    image_id = int(h5["image_ids"][index])
    image_path = image_root / decode_path(h5["image_paths"][index])
    with Image.open(image_path).convert("RGB") as image:
        width, height = image.size

    num_objects = int(h5["objects_per_image"][index])
    num_relations = int(h5["relationships_per_image"][index])
    object_names = h5["object_names"][index][:num_objects].tolist()
    object_boxes_xywh = h5["object_boxes"][index][:num_objects].tolist()
    raw_relations = list(
        zip(
            h5["relationship_subjects"][index][:num_relations].tolist(),
            h5["relationship_predicates"][index][:num_relations].tolist(),
            h5["relationship_objects"][index][:num_relations].tolist(),
        )
    )
    annotations = transform_scene_graph_annotations(
        object_names=object_names,
        object_boxes_xywh=object_boxes_xywh,
        relations=raw_relations,
        trans_info=compute_center_crop_transform(width, height, image_size),
        image_size=image_size,
        min_box_size=0.0,
        max_boxes=int(h5["object_names"].shape[1]),
        max_relations=int(h5["relationship_predicates"].shape[1]),
        selection_policy="first",
    )
    object_texts = [str(vocab["object_idx_to_name"][int(name)]).lower() for name in annotations["object_names"]]
    relation_texts = [
        str(vocab["pred_idx_to_name"][int(predicate)]).lower()
        for _, predicate, _ in annotations["relations"]
    ]
    condition = build_clean_scene_graph_condition(
        object_names=annotations["object_names"],
        object_texts=object_texts,
        boxes=annotations["boxes"],
        relations=annotations["relations"],
        relation_texts=relation_texts,
        max_objects=max_objects,
        max_relations=max_relations,
    )
    layout = select_major_instance_layout(condition, max_major_objects)
    relation_edges = [[int(src), int(dst)] for src, _, dst in layout["relations"]]
    # The first probe isolates box/instance control.  Relations remain in the
    # manifest for scoring, but are intentionally not verbalized in the global
    # prompt (nor in individual phrases).
    caption = build_clean_primary_scene_graph_caption(
        layout["object_texts"],
        [],
        [],
        [],
        style_prefix=DEFAULT_STYLE_PREFIX,
        style_suffix=DEFAULT_STYLE_SUFFIX,
    )
    annos = [
        {
            "bbox": xyxy_normalized_to_xywh_pixels(box, image_size),
            "mask": [],
            "category_name": "",
            # Instance text is deliberately relation-free.  The boxes carry
            # spatial structure in this first probe.
            "caption": object_phrase(text),
        }
        for text, box in zip(layout["object_texts"], layout["boxes"])
    ]
    return {
        "caption": caption,
        "width": image_size,
        "height": image_size,
        "annos": annos,
    }, {
        "test_index": index,
        "image_id": image_id,
        "source_image_path": str(image_path),
        "selected_object_texts": layout["object_texts"],
        "selected_boxes_xyxy_normalized": [list(box) for box in layout["boxes"]],
        "selected_relation_texts": layout["relation_texts"],
        "selected_relation_edges": relation_edges,
        "selected_clean_object_indices": layout["selected_clean_object_indices"],
        "conditioning_trace": condition.trace,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sample-indices", required=True)
    parser.add_argument("--image-size", default=512, type=int)
    parser.add_argument("--max-objects", default=6, type=int)
    parser.add_argument("--max-relations", default=1, type=int)
    parser.add_argument("--max-major-objects", default=4, type=int)
    args = parser.parse_args()
    if min(args.image_size, args.max_objects, args.max_major_objects) <= 0 or args.max_relations < 0:
        raise ValueError("image-size/max-objects/max-major-objects must be positive; max-relations must be nonnegative")

    with args.vocab.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with h5py.File(args.h5, "r") as h5:
        indices = parse_sample_indices(args.sample_indices, len(h5["image_ids"]))
        if not indices:
            raise ValueError("sample-indices must contain at least one fixed test index")
        samples = []
        for index in indices:
            record, trace = build_export_record(
                h5, vocab, args.image_root, index, args.image_size,
                args.max_objects, args.max_relations, args.max_major_objects,
            )
            path = args.out_dir / f"{index:05d}_{trace['image_id']}.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            samples.append({"input_json": path.name, **trace})

    manifest = {
        "purpose": "exploratory fixed-10 instance-layout probe; not FID/IS",
        "protocol": "VG fixed test only; deterministic center crop; clean_spatial_v1 selection",
        "h5_path": str(args.h5.resolve()),
        "h5_sha256": sha256_file(args.h5),
        "vocab_path": str(args.vocab.resolve()),
        "vocab_sha256": sha256_file(args.vocab),
        "image_root": str(args.image_root.resolve()),
        "sample_indices": indices,
        "image_size": args.image_size,
        "max_objects": args.max_objects,
        "max_relations": args.max_relations,
        "max_major_objects": args.max_major_objects,
        "global_caption_policy": "clean_primary positive color style",
        "instance_caption_policy": "relation-free object noun phrase",
        "samples": samples,
    }
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"EXPORTED {len(samples)} fixed test samples to {args.out_dir}")


if __name__ == "__main__":
    main()
