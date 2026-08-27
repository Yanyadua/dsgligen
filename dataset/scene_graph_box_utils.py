def compute_center_crop_transform(width, height, image_size):
    """Return the geometry used by GLIGEN's deterministic center crop."""
    width = int(width)
    height = int(height)
    image_size = int(image_size)
    scale = image_size / min(width, height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    return {
        "performed_scale": scale,
        "crop_x": (resized_width - image_size) // 2,
        "crop_y": (resized_height - image_size) // 2,
        "performed_flip": False,
        "WW": width,
        "HH": height,
        "resized_width": resized_width,
        "resized_height": resized_height,
    }


def transform_box_xywh(
    box_xywh,
    trans_info,
    image_size,
    min_box_size=0.0,
):
    """Apply the same resize, center-crop, and flip used for the image."""
    x, y, width, height = [float(value) for value in box_xywh]
    scale = float(trans_info["performed_scale"])
    crop_x = float(trans_info["crop_x"])
    crop_y = float(trans_info["crop_y"])
    image_size = float(image_size)

    x0 = x * scale - crop_x
    y0 = y * scale - crop_y
    x1 = (x + width) * scale - crop_x
    y1 = (y + height) * scale - crop_y

    if x0 >= image_size or y0 >= image_size or x1 <= 0 or y1 <= 0:
        return None

    x0 = max(x0, 0.0)
    y0 = max(y0, 0.0)
    x1 = min(x1, image_size)
    y1 = min(y1, image_size)
    area_ratio = ((x1 - x0) * (y1 - y0)) / (image_size * image_size)
    if area_ratio < float(min_box_size):
        return None

    if trans_info.get("performed_flip", False):
        x0, x1 = image_size - x1, image_size - x0

    return (
        x0 / image_size,
        y0 / image_size,
        x1 / image_size,
        y1 / image_size,
    )


def _box_area(box):
    x0, y0, x1, y1 = [float(value) for value in box]
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def _select_first_objects(num_objects, max_boxes):
    return list(range(min(num_objects, int(max_boxes))))


def _select_sg2im_relation_area_objects(boxes, relations, max_boxes):
    """Select a compact SG2IM-style subgraph deterministically.

    SG2IM samples a relation-connected subgraph during training. We use a
    deterministic variant for reproducible diffusion experiments: prefer large
    objects that participate in relationships, then fill remaining slots with
    large orphan objects.
    """
    max_boxes = int(max_boxes)
    if max_boxes <= 0:
        return []

    related = set()
    for src, _, dst in relations:
        related.add(int(src))
        related.add(int(dst))

    def sort_key(index):
        return (-_box_area(boxes[index]), index)

    related_sorted = sorted(related, key=sort_key)
    selected = related_sorted[:max_boxes]
    selected_set = set(selected)

    if len(selected) < max_boxes:
        orphan_sorted = sorted(
            (idx for idx in range(len(boxes)) if idx not in selected_set),
            key=sort_key,
        )
        selected.extend(orphan_sorted[: max_boxes - len(selected)])

    return selected


def select_scene_graph_subgraph(
    object_names,
    boxes,
    relations,
    max_boxes,
    max_relations,
    selection_policy="first",
    original_indices=None,
):
    """Compact objects and relations while keeping relation indices valid."""
    if original_indices is None:
        original_indices = list(range(len(object_names)))

    if selection_policy == "first":
        selected_old_indices = _select_first_objects(len(object_names), max_boxes)
    elif selection_policy == "sg2im_relation_area":
        selected_old_indices = _select_sg2im_relation_area_objects(
            boxes,
            relations,
            max_boxes,
        )
    else:
        raise ValueError(
            "selection_policy must be 'first' or 'sg2im_relation_area', "
            f"got {selection_policy!r}"
        )

    old_to_new = {
        original_indices[old_idx]: new_idx
        for new_idx, old_idx in enumerate(selected_old_indices)
    }
    local_to_new = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(selected_old_indices)
    }
    selected_names = [object_names[idx] for idx in selected_old_indices]
    selected_boxes = [boxes[idx] for idx in selected_old_indices]

    selected_relations = []
    for src, predicate, dst in relations:
        src = int(src)
        dst = int(dst)
        if src not in local_to_new or dst not in local_to_new:
            continue
        selected_relations.append(
            (local_to_new[src], int(predicate), local_to_new[dst])
        )
        if len(selected_relations) >= int(max_relations):
            break

    return {
        "object_names": selected_names,
        "boxes": selected_boxes,
        "relations": selected_relations,
        "old_to_new": old_to_new,
    }


def transform_scene_graph_annotations(
    object_names,
    object_boxes_xywh,
    relations,
    trans_info,
    image_size,
    min_box_size,
    max_boxes,
    max_relations,
    selection_policy="first",
):
    """Transform visible objects and keep relation indices consistent."""
    transformed_names = []
    transformed_boxes = []
    visible_original_indices = []
    old_to_new = {}

    for old_idx, (name, box_xywh) in enumerate(
        zip(object_names, object_boxes_xywh)
    ):
        box = transform_box_xywh(
            box_xywh,
            trans_info=trans_info,
            image_size=image_size,
            min_box_size=min_box_size,
        )
        if box is None:
            continue
        old_to_new[old_idx] = len(transformed_names)
        visible_original_indices.append(old_idx)
        transformed_names.append(name)
        transformed_boxes.append(box)

    transformed_relations = []
    for src, predicate, dst in relations:
        src = int(src)
        dst = int(dst)
        if src not in old_to_new or dst not in old_to_new:
            continue
        transformed_relations.append(
            (old_to_new[src], int(predicate), old_to_new[dst])
        )

    return select_scene_graph_subgraph(
        object_names=transformed_names,
        boxes=transformed_boxes,
        relations=transformed_relations,
        max_boxes=max_boxes,
        max_relations=max_relations,
        selection_policy=selection_policy,
        original_indices=visible_original_indices,
    )
