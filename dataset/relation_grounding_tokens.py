from dataclasses import dataclass
import re

import torch


@dataclass
class RelationGroundingTokenResult:
    boxes: torch.Tensor
    masks: torch.Tensor
    object_texts: list
    token_roles: list
    relation_token_source: torch.Tensor


def _is_active(mask_value):
    return float(mask_value) > 0.0


def _safe_text(texts, index):
    if index < 0 or index >= len(texts):
        return ""
    return str(texts[index]).strip()


def normalize_predicate(value):
    value = str(value).strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", value)


def parse_allowed_predicates(value):
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        values = value.split(",")
    else:
        values = value
    normalized = {
        normalize_predicate(item)
        for item in values
        if str(item).strip()
    }
    return normalized or None


def _union_box(src_box, dst_box):
    return torch.stack(
        [
            torch.minimum(src_box[0], dst_box[0]),
            torch.minimum(src_box[1], dst_box[1]),
            torch.maximum(src_box[2], dst_box[2]),
            torch.maximum(src_box[3], dst_box[3]),
        ]
    )


def _relation_phrase(subject, predicate, obj, template):
    if template:
        return template.format(subject=subject, predicate=predicate, object=obj).strip()
    return f"{subject} {predicate} {obj}".strip()


def append_relation_grounding_tokens(
    *,
    boxes,
    masks,
    object_texts,
    relation_edges,
    relation_masks,
    relation_texts,
    max_relation_tokens,
    phrase_template="{subject} {predicate} {object}",
    allowed_predicates=None,
    deduplicate=False,
    relation_mask_scale=1.0,
):
    max_relation_tokens = int(max_relation_tokens)
    allowed_predicates = parse_allowed_predicates(allowed_predicates)
    relation_mask_scale = float(relation_mask_scale)
    base_roles = [
        "object" if _is_active(mask) else "padding"
        for mask in masks.detach().cpu().tolist()
    ]
    if max_relation_tokens <= 0:
        return RelationGroundingTokenResult(
            boxes=boxes,
            masks=masks,
            object_texts=list(object_texts),
            token_roles=base_roles,
            relation_token_source=torch.empty(0, dtype=torch.long, device=masks.device),
        )

    extra_boxes = torch.zeros(
        max_relation_tokens,
        boxes.shape[-1],
        dtype=boxes.dtype,
        device=boxes.device,
    )
    extra_masks = torch.zeros(max_relation_tokens, dtype=masks.dtype, device=masks.device)
    extra_texts = [""] * max_relation_tokens
    extra_roles = ["padding"] * max_relation_tokens
    relation_token_source = torch.full(
        (max_relation_tokens,),
        -1,
        dtype=torch.long,
        device=masks.device,
    )

    write_idx = 0
    max_objects = boxes.shape[0]
    seen_relation_keys = set()
    for rel_idx in range(int(relation_edges.shape[0])):
        if write_idx >= max_relation_tokens:
            break
        if rel_idx >= len(relation_texts) or not _is_active(relation_masks[rel_idx]):
            continue

        src = int(relation_edges[rel_idx, 0].item())
        dst = int(relation_edges[rel_idx, 1].item())
        if src < 0 or dst < 0 or src >= max_objects or dst >= max_objects:
            continue
        if not _is_active(masks[src]) or not _is_active(masks[dst]):
            continue

        subject = _safe_text(object_texts, src)
        obj = _safe_text(object_texts, dst)
        predicate = _safe_text(relation_texts, rel_idx)
        if not subject or not obj or not predicate:
            continue
        normalized_predicate = normalize_predicate(predicate)
        if allowed_predicates is not None and normalized_predicate not in allowed_predicates:
            continue
        relation_key = (src, dst, normalized_predicate)
        if deduplicate and relation_key in seen_relation_keys:
            continue
        seen_relation_keys.add(relation_key)

        extra_boxes[write_idx] = _union_box(boxes[src], boxes[dst])
        extra_masks[write_idx] = relation_mask_scale
        extra_texts[write_idx] = _relation_phrase(
            subject,
            predicate,
            obj,
            phrase_template,
        )
        extra_roles[write_idx] = "relation"
        relation_token_source[write_idx] = int(rel_idx)
        write_idx += 1

    return RelationGroundingTokenResult(
        boxes=torch.cat([boxes, extra_boxes], dim=0),
        masks=torch.cat([masks, extra_masks], dim=0),
        object_texts=list(object_texts) + extra_texts,
        token_roles=base_roles + extra_roles,
        relation_token_source=relation_token_source,
    )
