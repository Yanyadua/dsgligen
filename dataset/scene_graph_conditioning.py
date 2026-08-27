"""Deterministic, shared scene-graph conditioning selection.

The legacy VG path truncates objects and relationships largely in annotation
order.  This module is deliberately framework-free so that training and
fixed-split evaluation can make the exact same conditioning decision.
"""

from dataclasses import dataclass
import math
import re


DEFAULT_SPATIAL_PREDICATES = frozenset(
    {
        "on",
        "on top of",
        "under",
        "below",
        "above",
        "inside",
        "in",
        "near",
        "next to",
        "in front of",
        "behind",
    }
)

_PREDICATE_PRIORITY = {
    "inside": 4.0,
    "in": 4.0,
    "on top of": 3.8,
    "on": 3.4,
    "above": 3.2,
    "below": 3.2,
    "under": 3.2,
    "next to": 1.2,
    "near": 1.0,
    "in front of": 1.6,
    "behind": 1.6,
}

_PRIMARY_OBJECT_LABELS = frozenset(
    {
        "person",
        "man",
        "woman",
        "boy",
        "girl",
        "child",
        "dog",
        "cat",
        "bird",
        "horse",
        "elephant",
        "car",
        "bus",
        "truck",
        "bicycle",
        "motorcycle",
    }
)

_SUPPORT_OBJECT_LABELS = frozenset(
    {
        "road",
        "street",
        "sidewalk",
        "table",
        "chair",
        "bench",
        "bed",
        "desk",
        "counter",
        "snow",
        "water",
        "sand",
        "field",
        "mountain",
        "building",
        "window",
        "door",
        "sign",
        "fence",
        "tree",
        "plant",
        "grass",
    }
)

_GENERIC_OBJECT_LABELS = frozenset(
    {
        "background",
        "edge",
        "floor",
        "ground",
        "line",
        "light",
        "reflection",
        "shadow",
        "sky",
        "tile",
        "wall",
        "frame",
        "cloud",
        "grass",
    }
)

# VG often records part/attribute relations such as "pant on floor". They are
# valid annotations but poor global controls and can dominate a short caption.
_LOW_LEVEL_RELATION_SUBJECTS = frozenset(
    {
        "pant",
        "pants",
        "shoe",
        "shoes",
        "foot",
        "feet",
        "hand",
        "hair",
        "shirt",
        "jacket",
        "towel",
        "tile",
    }
)


def normalize_conditioning_text(value):
    value = str(value).strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", value)


def _object_priority(text, use_support=False):
    text = normalize_conditioning_text(text)
    if text in _PRIMARY_OBJECT_LABELS:
        return 3
    if use_support and text in _SUPPORT_OBJECT_LABELS:
        return 1
    if text in _GENERIC_OBJECT_LABELS:
        return -1
    return 0


def _object_category(text):
    text = normalize_conditioning_text(text)
    if text in _PRIMARY_OBJECT_LABELS:
        return "foreground"
    if text in _SUPPORT_OBJECT_LABELS:
        return "support"
    if text in _GENERIC_OBJECT_LABELS:
        return "background"
    return "other"


def _area(box):
    x0, y0, x1, y1 = [float(value) for value in box]
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def _short_side(box):
    x0, y0, x1, y1 = [float(value) for value in box]
    return min(max(x1 - x0, 0.0), max(y1 - y0, 0.0))


def _iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = [float(value) for value in box_a]
    bx0, by0, bx1, by1 = [float(value) for value in box_b]
    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter = max(inter_x1 - inter_x0, 0.0) * max(inter_y1 - inter_y0, 0.0)
    union = _area(box_a) + _area(box_b) - inter
    return inter / union if union > 0 else 0.0


def _center(box):
    x0, y0, x1, y1 = [float(value) for value in box]
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _intersection_over_subject(subject_box, object_box):
    sx0, sy0, sx1, sy1 = [float(value) for value in subject_box]
    ox0, oy0, ox1, oy1 = [float(value) for value in object_box]
    inter = max(min(sx1, ox1) - max(sx0, ox0), 0.0) * max(
        min(sy1, oy1) - max(sy0, oy0), 0.0
    )
    return inter / max(_area(subject_box), 1e-8)


def relation_geometry_is_consistent(predicate, subject_box, object_box):
    """Use conservative 2-D checks; uncertain predicates are not promoted."""
    predicate = normalize_conditioning_text(predicate)
    sx, sy = _center(subject_box)
    ox, oy = _center(object_box)
    if predicate in {"above", "on", "on top of"}:
        return sy <= oy + 0.20
    if predicate in {"below", "under"}:
        return sy >= oy - 0.05
    if predicate in {"inside", "in"}:
        return _intersection_over_subject(subject_box, object_box) >= 0.60
    if predicate in {"near", "next to"}:
        return math.hypot(sx - ox, sy - oy) <= 0.80
    if predicate in {"in front of", "behind"}:
        return math.hypot(sx - ox, sy - oy) <= 0.85
    return False


@dataclass
class CleanSceneGraphCondition:
    object_names: list
    object_texts: list
    boxes: list
    relations: list
    relation_texts: list
    object_categories: list
    object_mask_scales: list
    trace: dict


def build_clean_scene_graph_condition(
    *,
    object_names,
    object_texts,
    boxes,
    relations,
    relation_texts,
    max_objects=6,
    max_relations=1,
    min_box_area=0.0025,
    min_box_side=0.035,
    relation_core_min_area=0.0015,
    duplicate_iou_threshold=0.85,
    spatial_predicates=DEFAULT_SPATIAL_PREDICATES,
    relation_predicates=None,
    policy="clean_spatial_v1",
    foreground_mask_scale=1.0,
    support_mask_scale=0.8,
    background_mask_scale=0.4,
    other_mask_scale=0.8,
):
    """Select a compact, spatially meaningful VG conditioning subgraph.

    ``relations`` and ``relation_texts`` have matching order. Relation tuples
    retain their predicate IDs, while this function only uses the text to rank
    and filter them.
    """
    policy = normalize_conditioning_text(policy)
    if policy in {"vg conditioning v2", "clean spatial v2"}:
        policy = "clean spatial v2"
    if policy in {
        "vg conditioning v2.1",
        "vg conditioning v2 1",
        "vg conditioning v21",
        "clean spatial v2.1",
        "clean spatial v2 1",
        "clean spatial v21",
    }:
        policy = "clean spatial v2.1"
    if policy not in {"clean spatial v1", "clean spatial v2", "clean spatial v2.1"}:
        raise ValueError(
            "policy must be 'clean_spatial_v1', 'clean_spatial_v2', "
            "or 'clean_spatial_v2_1', "
            f"got {policy!r}"
        )
    policy = policy.replace(" ", "_").replace(".", "_")
    is_v2 = policy == "clean_spatial_v2"
    is_v21 = policy == "clean_spatial_v2_1"
    use_category_ranking = is_v2 or is_v21
    max_objects = max(int(max_objects), 0)
    max_relations = max(int(max_relations), 0)
    spatial_predicates = {
        normalize_conditioning_text(predicate) for predicate in spatial_predicates
    }
    relation_predicates = (
        set(spatial_predicates)
        if relation_predicates is None
        else {
            normalize_conditioning_text(predicate)
            for predicate in relation_predicates
        }
    )
    object_texts = [str(text).strip().lower() for text in object_texts]
    boxes = [tuple(float(value) for value in box) for box in boxes]

    valid_relations = []
    all_relation_degree = [0] * len(boxes)
    spatial_relation_degree = [0] * len(boxes)
    spatial_endpoints = set()
    for rel_idx, relation in enumerate(relations):
        if len(relation) != 3 or rel_idx >= len(relation_texts):
            continue
        src, _, dst = [int(value) for value in relation]
        if src < 0 or dst < 0 or src >= len(boxes) or dst >= len(boxes) or src == dst:
            continue
        predicate = normalize_conditioning_text(relation_texts[rel_idx])
        if not predicate:
            continue
        valid_relations.append((rel_idx, src, dst, predicate))
        all_relation_degree[src] += 1
        all_relation_degree[dst] += 1
        if predicate in spatial_predicates:
            spatial_relation_degree[src] += 1
            spatial_relation_degree[dst] += 1
            spatial_endpoints.update((src, dst))

    retained = set()
    dropped_small = []
    for index, box in enumerate(boxes):
        area = _area(box)
        side = _short_side(box)
        is_regular = area >= float(min_box_area) and side >= float(min_box_side)
        is_relation_core = index in spatial_endpoints and area >= float(relation_core_min_area)
        if is_regular or is_relation_core:
            retained.add(index)
        else:
            dropped_small.append(index)

    # Only deduplicate identical labels. Cross-class overlap often represents a
    # valid hierarchy such as person/shirt and must remain available.
    duplicate_of = {}
    by_label = {}
    for index in sorted(retained):
        by_label.setdefault(normalize_conditioning_text(object_texts[index]), []).append(index)
    for label_indices in by_label.values():
        priority = sorted(
            label_indices,
            key=lambda index: (
                -spatial_relation_degree[index],
                -all_relation_degree[index],
                -_area(boxes[index]),
                index,
            ),
        )
        kept = []
        for index in priority:
            duplicate = next(
                (other for other in kept if _iou(boxes[index], boxes[other]) >= duplicate_iou_threshold),
                None,
            )
            if duplicate is None:
                kept.append(index)
            else:
                retained.discard(index)
                duplicate_of[index] = duplicate

    candidates = []
    for rel_idx, src, dst, predicate in valid_relations:
        if src not in retained or dst not in retained:
            continue
        if predicate not in spatial_predicates:
            continue
        if normalize_conditioning_text(object_texts[src]) in _LOW_LEVEL_RELATION_SUBJECTS:
            continue
        if not relation_geometry_is_consistent(predicate, boxes[src], boxes[dst]):
            continue
        endpoint_priority = _object_priority(
            object_texts[src],
            use_support=is_v21,
        ) + _object_priority(
            object_texts[dst],
            use_support=is_v21,
        )
        score = (
            _PREDICATE_PRIORITY.get(predicate, 0.0)
            + (0.45 * endpoint_priority if use_category_ranking else 0.0)
            + 0.25 * (spatial_relation_degree[src] + spatial_relation_degree[dst])
            + 0.10 * math.sqrt(_area(boxes[src]) + _area(boxes[dst]))
        )
        candidates.append((score, rel_idx, src, dst, predicate))
    candidates.sort(key=lambda value: (-value[0], value[1], value[2], value[3]))
    relation_candidates = [
        candidate for candidate in candidates if candidate[4] in relation_predicates
    ]

    selected_old = []
    seed_relation_count = max_relations if use_category_ranking else 1
    if max_objects:
        for _, _, src, dst, _ in (relation_candidates or candidates)[:seed_relation_count]:
            for endpoint in (src, dst):
                if endpoint not in selected_old and len(selected_old) < max_objects:
                    selected_old.append(endpoint)

    if use_category_ranking:
        remaining = sorted(
            retained.difference(selected_old),
            key=lambda index: (
                -_object_priority(object_texts[index], use_support=is_v21),
                -spatial_relation_degree[index],
                -all_relation_degree[index],
                -_area(boxes[index]),
                index,
            ),
        )
    else:
        remaining = sorted(
            retained.difference(selected_old),
            key=lambda index: (
                -spatial_relation_degree[index],
                -all_relation_degree[index],
                -_area(boxes[index]),
                index,
            ),
        )
    selected_old.extend(remaining[: max(0, max_objects - len(selected_old))])
    selected_old = selected_old[:max_objects]
    selected_old.sort(
        key=lambda index: (
            -_object_priority(object_texts[index], use_support=is_v21),
            -_area(boxes[index]),
            -spatial_relation_degree[index],
            -all_relation_degree[index],
            index,
        )
    )
    old_to_new = {old_index: new_index for new_index, old_index in enumerate(selected_old)}

    selected_relations = []
    selected_relation_texts = []
    selected_relation_sources = []
    for _, rel_idx, src, dst, predicate in relation_candidates:
        if len(selected_relations) >= max_relations:
            break
        if src not in old_to_new or dst not in old_to_new:
            continue
        _, predicate_id, _ = relations[rel_idx]
        selected_relations.append((old_to_new[src], int(predicate_id), old_to_new[dst]))
        selected_relation_texts.append(predicate)
        selected_relation_sources.append(int(rel_idx))

    trace = {
        "policy": policy,
        "parameters": {
            "max_objects": max_objects,
            "max_relations": max_relations,
            "min_box_area": float(min_box_area),
            "min_box_side": float(min_box_side),
            "relation_core_min_area": float(relation_core_min_area),
            "duplicate_iou_threshold": float(duplicate_iou_threshold),
            "spatial_predicates": sorted(spatial_predicates),
            "relation_predicates": sorted(relation_predicates),
            "primary_object_labels": sorted(_PRIMARY_OBJECT_LABELS) if use_category_ranking else [],
            "support_object_labels": sorted(_SUPPORT_OBJECT_LABELS) if is_v21 else [],
            "generic_object_labels": sorted(_GENERIC_OBJECT_LABELS) if use_category_ranking else [],
            "foreground_mask_scale": float(foreground_mask_scale),
            "support_mask_scale": float(support_mask_scale),
            "background_mask_scale": float(background_mask_scale),
            "other_mask_scale": float(other_mask_scale),
        },
        "selected_original_object_indices": selected_old,
        "dropped_small_object_indices": dropped_small,
        "duplicate_of": {str(key): int(value) for key, value in duplicate_of.items()},
        "candidate_relation_sources": [int(value[1]) for value in candidates],
        "relation_candidate_sources": [int(value[1]) for value in relation_candidates],
        "selected_relation_sources": selected_relation_sources,
    }
    selected_categories = [_object_category(object_texts[index]) for index in selected_old]
    category_to_scale = {
        "foreground": float(foreground_mask_scale),
        "support": float(support_mask_scale),
        "background": float(background_mask_scale),
        "other": float(other_mask_scale),
    }
    if not is_v21:
        category_to_scale = {key: 1.0 for key in category_to_scale}

    return CleanSceneGraphCondition(
        object_names=[object_names[index] for index in selected_old],
        object_texts=[object_texts[index] for index in selected_old],
        boxes=[boxes[index] for index in selected_old],
        relations=selected_relations,
        relation_texts=selected_relation_texts,
        object_categories=selected_categories,
        object_mask_scales=[category_to_scale[category] for category in selected_categories],
        trace=trace,
    )
