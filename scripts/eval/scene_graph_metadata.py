import json
from pathlib import Path


def _to_python(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _as_float_list(value):
    return [float(item) for item in _to_python(value)]


def _is_active(mask_value):
    # Relation grounding deliberately uses fractional masks (for example 0.5)
    # to soften its PositionNet input. Metadata must still expose those active
    # tokens instead of silently treating them as padding.
    return float(_to_python(mask_value)) > 0.0


def _safe_text(values, index):
    if values is None or index < 0 or index >= len(values):
        return ""
    return str(values[index])


def build_sample_metadata(item, dataset_index, generation=None):
    boxes = item["boxes"]
    masks = item["masks"]
    object_texts = item.get("object_texts", [])
    token_roles = item.get("grounding_token_roles", [])
    object_categories = item.get("object_categories") or []

    objects = []
    for obj_idx in range(len(masks)):
        if not _is_active(masks[obj_idx]):
            continue
        objects.append(
            {
                "index": int(obj_idx),
                "text": _safe_text(object_texts, obj_idx),
                "role": _safe_text(token_roles, obj_idx) or "object",
                "category": _safe_text(object_categories, obj_idx),
                "box_xyxy": _as_float_list(boxes[obj_idx]),
            }
        )

    relation_edges = item.get("relation_edges")
    relation_masks = item.get("relation_masks")
    relation_texts = item.get("relation_texts", [])
    relation_geo_features = item.get("relation_geo_features")
    relations = []
    if relation_edges is not None and relation_masks is not None:
        for rel_idx in range(len(relation_masks)):
            if not _is_active(relation_masks[rel_idx]):
                continue
            edge = [int(round(value)) for value in _to_python(relation_edges[rel_idx])]
            src = edge[0] if len(edge) > 0 else -1
            dst = edge[1] if len(edge) > 1 else -1
            relation = {
                "index": int(rel_idx),
                "subject": int(src),
                "object": int(dst),
                "subject_text": _safe_text(object_texts, src),
                "object_text": _safe_text(object_texts, dst),
                "predicate": _safe_text(relation_texts, rel_idx),
            }
            if relation_geo_features is not None:
                relation["geo_features"] = _as_float_list(
                    relation_geo_features[rel_idx]
                )
            relations.append(relation)

    metadata = {
        "dataset_index": int(dataset_index),
        "image_id": int(item["id"]),
        "image_path": str(item.get("image_path", "")),
        "caption": str(item.get("caption", "")),
        "generation": dict(generation or {}),
        "objects": objects,
        "relations": relations,
        "relation_token_source": _to_python(
            item.get("relation_token_source", [])
        ),
    }
    if item.get("conditioning_trace") is not None:
        metadata["conditioning_trace"] = _to_python(item["conditioning_trace"])
    return metadata


def write_sample_metadata(out_dir, item, dataset_index, generation=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_sample_metadata(
        item,
        dataset_index=dataset_index,
        generation=generation,
    )
    out_path = out_dir / f"{metadata['image_id']}.json"
    out_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path
