def _as_list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _active_relation_phrases(object_texts, relation_edges, relation_masks, relation_texts):
    phrases = []
    for rel_idx, (src, dst) in enumerate(_as_list(relation_edges)):
        if float(relation_masks[rel_idx]) <= 0:
            continue
        src = int(src)
        dst = int(dst)
        subj = object_texts[src] if src < len(object_texts) else ""
        obj = object_texts[dst] if dst < len(object_texts) else ""
        rel = relation_texts[rel_idx]
        if subj and obj and rel:
            phrases.append(f"{subj} {rel} {obj}")
    return phrases


def build_scene_graph_caption(
    object_texts,
    relation_edges,
    relation_masks,
    relation_texts,
    max_objects=8,
    max_relations=4,
    style_prefix="",
    style_suffix="",
):
    valid_object_texts = [text for text in object_texts if text]
    valid_relation_texts = _active_relation_phrases(
        object_texts,
        relation_edges,
        relation_masks,
        relation_texts,
    )

    object_part = ", ".join(valid_object_texts[:max_objects])
    relation_part = ". ".join(valid_relation_texts[:max_relations])
    if object_part and relation_part:
        caption = f"a scene with {object_part}. {relation_part}."
    elif object_part:
        caption = f"a scene with {object_part}."
    else:
        caption = "a scene with objects."

    style_prefix = str(style_prefix).strip()
    style_suffix = str(style_suffix).strip()
    if style_prefix:
        caption = f"{style_prefix} {caption}"
    else:
        caption = caption[:1].upper() + caption[1:]
    if style_suffix:
        caption = f"{caption} {style_suffix.rstrip('.')}."
    return caption


def build_natural_scene_graph_caption(
    object_texts,
    relation_edges,
    relation_masks,
    relation_texts,
    max_objects=8,
    max_relations=4,
    style_prefix="A realistic natural color photograph",
    style_suffix="natural lighting and realistic details",
):
    valid_object_texts = []
    seen_objects = set()
    for text in object_texts:
        text = str(text).strip()
        if not text or text in seen_objects:
            continue
        seen_objects.add(text)
        valid_object_texts.append(text)

    relation_phrases = _active_relation_phrases(
        object_texts,
        relation_edges,
        relation_masks,
        relation_texts,
    )

    objects = valid_object_texts[:max_objects]
    relations = relation_phrases[:max_relations]

    if objects:
        if len(objects) == 1:
            object_part = objects[0]
        elif len(objects) == 2:
            object_part = f"{objects[0]} and {objects[1]}"
        else:
            object_part = ", ".join(objects[:-1]) + f", and {objects[-1]}"
        caption = f"{style_prefix} showing {object_part}"
    else:
        caption = f"{style_prefix} of a scene"

    if relations:
        if len(relations) == 1:
            relation_part = relations[0]
        elif len(relations) == 2:
            relation_part = f"{relations[0]} and {relations[1]}"
        else:
            relation_part = ", ".join(relations[:-1]) + f", and {relations[-1]}"
        caption = f"{caption}, with {relation_part}"

    style_suffix = str(style_suffix).strip()
    if style_suffix:
        caption = f"{caption}, {style_suffix.rstrip('.')}"
    return f"{caption}."


def build_clean_scene_graph_caption(
    object_texts,
    relation_edges,
    relation_masks,
    relation_texts,
    style_prefix="A full-color realistic DSLR photograph",
    style_suffix="vivid natural colors, realistic color photography, natural lighting",
):
    """Short positive caption used by the clean spatial conditioning policy."""
    objects = []
    seen = set()
    for text in object_texts:
        text = str(text).strip()
        normalized = text.lower()
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        objects.append(text)
    if not objects:
        subject = "a scene"
    elif len(objects) == 1:
        subject = objects[0]
    elif len(objects) == 2:
        subject = f"{objects[0]} and {objects[1]}"
    else:
        subject = ", ".join(objects[:-1]) + f", and {objects[-1]}"
    caption = f"{str(style_prefix).strip()} featuring {subject}"
    relations = _active_relation_phrases(
        object_texts,
        relation_edges,
        relation_masks,
        relation_texts,
    )
    low_level_subjects = {
        "pant", "pants", "shoe", "shoes", "foot", "feet", "hand",
        "hair", "shirt", "jacket", "towel", "tile",
    }
    relations = [
        phrase for phrase in relations
        if not phrase.split(" ", 1)[0].strip().lower() in low_level_subjects
    ]
    if relations:
        caption = f"{caption}. {relations[0].capitalize()}"
    suffix = str(style_suffix).strip().rstrip(".")
    if suffix:
        caption = f"{caption}. {suffix}"
    return f"{caption}."


_PRIMARY_CAPTION_OBJECTS = {
    "person", "man", "woman", "boy", "girl", "child", "lady",
    "dog", "cat", "bird", "horse", "bear", "elephant", "car", "bus",
    "truck", "bicycle", "bike", "motorcycle", "ski", "skier",
    "skateboard", "boat", "airplane", "train", "chair", "bed", "table",
    "cabinet", "plant", "tree", "building", "road", "street", "snow",
    "water", "food", "cup", "bottle", "book", "umbrella", "pizza",
    "clock", "sign", "window", "door",
}

_GENERIC_CAPTION_OBJECTS = {
    "background", "edge", "floor", "ground", "line", "light", "reflection",
    "shadow", "sky", "tile", "wall", "frame", "cloud", "grass",
}


def build_clean_primary_scene_graph_caption(
    object_texts,
    relation_edges,
    relation_masks,
    relation_texts,
    max_objects=4,
    style_prefix="A full-color realistic DSLR photograph",
    style_suffix="vivid natural colors, realistic color photography, natural lighting",
):
    """Keep global caption semantics focused on major objects.

    Box tokens still carry the full compact object set.  This caption variant
    deliberately excludes generic context labels and only verbalizes relations
    whose endpoints remain in the caption, reducing CLIP sequence leakage.
    """
    unique = []
    seen = set()
    for text in object_texts:
        normalized = str(text).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)

    primary = [text for text in unique if text in _PRIMARY_CAPTION_OBJECTS]
    contextual = [
        text for text in unique
        if text not in _PRIMARY_CAPTION_OBJECTS and text not in _GENERIC_CAPTION_OBJECTS
    ]
    selected = (primary + contextual)[: max(int(max_objects), 1)]
    if not selected:
        selected = ["a scene"]

    if len(selected) == 1:
        subject = selected[0]
    elif len(selected) == 2:
        subject = f"{selected[0]} and {selected[1]}"
    else:
        subject = ", ".join(selected[:-1]) + f", and {selected[-1]}"
    caption = f"{str(style_prefix).strip()} featuring {subject}"

    selected_set = set(selected)
    relations = []
    for phrase in _active_relation_phrases(
        object_texts, relation_edges, relation_masks, relation_texts
    ):
        parts = phrase.split(" ")
        if len(parts) < 3:
            continue
        subject_text = parts[0].lower()
        object_text = parts[-1].lower()
        if subject_text in selected_set and object_text in selected_set:
            relations.append(phrase)
    if relations:
        caption = f"{caption}. {relations[0].capitalize()}"
    suffix = str(style_suffix).strip().rstrip(".")
    if suffix:
        caption = f"{caption}. {suffix}"
    return f"{caption}."
