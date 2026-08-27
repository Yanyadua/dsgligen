class GroundingCheckpointError(RuntimeError):
    pass


class BaseCheckpointCompatibilityError(RuntimeError):
    pass


class ProtocolMismatchError(RuntimeError):
    pass


def normalize_saved_config(saved_config):
    if (
        isinstance(saved_config, dict)
        and isinstance(saved_config.get("_content"), dict)
    ):
        return saved_config["_content"]
    return saved_config


REQUIRED_GROUNDING_PREFIXES = (
    "position_net.gat_layers.",
    "position_net.graph_gate",
)

HISTORICAL_LOSS_WEIGHTS = {
    "diffusion_loss_weight": 1.0,
    "object_align_loss_weight": 0.05,
    "spatial_consistency_loss_weight": 0.05,
    "relation_geo_prediction_loss_weight": 0.05,
    "relation_visual_align_loss_weight": 0.0,
    "graph_image_align_loss_weight": 0.0,
    "masked_relation_loss_weight": 0.0,
}

GLIGEN_POSITION_BASE_KEYS = (
    "position_net.linears.0.weight",
    "position_net.linears.0.bias",
    "position_net.linears.2.weight",
    "position_net.linears.2.bias",
    "position_net.linears.4.weight",
    "position_net.linears.4.bias",
    "position_net.null_positive_feature",
    "position_net.null_position_feature",
)

SCENE_GRAPH_ONLY_PREFIXES = (
    "position_net.gat_layers.",
    "position_net.graph_gate",
    "position_net.graph_adapter.",
    "position_net.relation_geo_predictor.",
    "position_net.relation_visual_predictor.",
    "position_net.relation_predicate_predictor.",
    "position_net.graph_visual_projector.",
    "position_net.triplet_fuser.",
    "position_net.triplet_gate",
)


def validate_grounding_state(
    model_state,
    checkpoint_state,
    required_prefixes=None,
):
    if not checkpoint_state:
        raise GroundingCheckpointError("grounding checkpoint contains no model tensors")

    # Infer the minimal required branch from the configured model.  A
    # fuser-only model deliberately has no GAT tensors and keeps graph_gate
    # frozen, so requiring the historical GAT pair would reject its valid
    # lightweight checkpoint.
    if required_prefixes is None:
        required_prefixes = []
        has_gat = any(key.startswith("position_net.gat_layers.") for key in model_state)
        if has_gat:
            required_prefixes.extend(("position_net.gat_layers.", "position_net.graph_gate"))
        if "position_net.triplet_gate" in model_state:
            required_prefixes.extend(("position_net.triplet_fuser.", "position_net.triplet_gate"))

    missing_required = [
        prefix
        for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in checkpoint_state)
    ]
    if missing_required:
        raise GroundingCheckpointError(
            "grounding checkpoint is missing required scene-graph tensors: "
            + ", ".join(missing_required)
        )

    unknown = sorted(key for key in checkpoint_state if key not in model_state)
    if unknown:
        raise GroundingCheckpointError(
            "grounding checkpoint has tensors absent from the configured model: "
            + ", ".join(unknown[:10])
        )

    shape_mismatches = sorted(
        key
        for key, value in checkpoint_state.items()
        if model_state[key].shape != value.shape
    )
    if shape_mismatches:
        raise GroundingCheckpointError(
            "grounding checkpoint shape mismatch for: "
            + ", ".join(shape_mismatches[:10])
        )

    compatible = dict(checkpoint_state)
    report = {
        "loaded": len(compatible),
        "skipped": [],
        "required_prefixes": list(required_prefixes),
    }
    return compatible, report


def validate_checkpoint_trainable_manifest(checkpoint_state, trainable_names):
    if not trainable_names:
        raise GroundingCheckpointError(
            "checkpoint has no trainable_names manifest"
        )
    state_names = set(checkpoint_state)
    manifest_names = set(trainable_names)
    if state_names != manifest_names:
        missing_state = sorted(manifest_names - state_names)
        unlisted_state = sorted(state_names - manifest_names)
        raise GroundingCheckpointError(
            "checkpoint trainable manifest mismatch: "
            f"missing_state={missing_state[:10]}, "
            f"unlisted_state={unlisted_state[:10]}"
        )


def validate_base_grounding_compatibility(model_state, base_state):
    missing_model = [key for key in GLIGEN_POSITION_BASE_KEYS if key not in model_state]
    if missing_model:
        raise BaseCheckpointCompatibilityError(
            "configured grounding model is not GLIGEN-compatible; missing linears tensors: "
            + ", ".join(missing_model)
        )

    missing_base = [key for key in GLIGEN_POSITION_BASE_KEYS if key not in base_state]
    if missing_base:
        raise BaseCheckpointCompatibilityError(
            "base checkpoint is missing GLIGEN position tensors: "
            + ", ".join(missing_base)
        )

    shape_mismatches = [
        key
        for key in GLIGEN_POSITION_BASE_KEYS
        if model_state[key].shape != base_state[key].shape
    ]
    if shape_mismatches:
        raise BaseCheckpointCompatibilityError(
            "base checkpoint shape mismatch for GLIGEN position tensors: "
            + ", ".join(shape_mismatches)
        )

    return {"compatible_base_tensor_count": len(GLIGEN_POSITION_BASE_KEYS)}


def validate_base_model_compatibility(model_state, base_state):
    validate_base_grounding_compatibility(model_state, base_state)

    unknown_base = sorted(key for key in base_state if key not in model_state)
    if unknown_base:
        raise BaseCheckpointCompatibilityError(
            "base checkpoint contains tensors absent from the configured model: "
            + ", ".join(unknown_base[:10])
        )

    shape_mismatches = sorted(
        key
        for key, value in base_state.items()
        if model_state[key].shape != value.shape
    )
    if shape_mismatches:
        raise BaseCheckpointCompatibilityError(
            "base checkpoint shape mismatch for: "
            + ", ".join(shape_mismatches[:10])
        )

    missing_pretrained = sorted(
        key
        for key in model_state
        if key not in base_state
        and not any(key.startswith(prefix) for prefix in SCENE_GRAPH_ONLY_PREFIXES)
    )
    if missing_pretrained:
        raise BaseCheckpointCompatibilityError(
            "configured model is missing pretrained base tensors for: "
            + ", ".join(missing_pretrained[:10])
        )

    new_scene_graph_tensors = sorted(
        key for key in model_state if key not in base_state
    )
    return dict(base_state), {
        "loaded": len(base_state),
        "new_scene_graph_tensors": new_scene_graph_tensors,
    }


def validate_resume_metadata(existing, expected):
    mismatches = {
        key: (existing.get(key), expected_value)
        for key, expected_value in expected.items()
        if existing.get(key) != expected_value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: existing={old!r}, expected={new!r}"
            for key, (old, new) in sorted(mismatches.items())
        )
        raise ProtocolMismatchError(
            "existing generated images use a different sampling protocol: " + details
        )


def validate_split_ids(train_ids, test_ids, expected_train, expected_test):
    train_ids = [int(value) for value in train_ids]
    test_ids = [int(value) for value in test_ids]
    if len(train_ids) != expected_train:
        raise ProtocolMismatchError(
            f"unexpected train image count: {len(train_ids)} != {expected_train}"
        )
    if len(test_ids) != expected_test:
        raise ProtocolMismatchError(
            f"unexpected test image count: {len(test_ids)} != {expected_test}"
        )

    overlap = sorted(set(train_ids).intersection(test_ids))
    if overlap:
        raise ProtocolMismatchError(
            f"train/test image-id overlap detected: {len(overlap)} images"
        )
    return {
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "overlap_count": 0,
    }


def validate_historical_loss_weights(config):
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in HISTORICAL_LOSS_WEIGHTS.items()
        if float(config.get(key, 0.0)) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}: actual={actual!r}, expected={expected!r}"
            for key, (actual, expected) in sorted(mismatches.items())
        )
        raise ProtocolMismatchError(
            "historical three-loss configuration mismatch: " + details
        )


def validate_box_transform_config(config):
    try:
        dataset_config = config["train_dataset_names"]["VGSceneGraph"]
    except KeyError as error:
        raise ProtocolMismatchError(
            "missing train_dataset_names.VGSceneGraph configuration"
        ) from error

    mode = dataset_config.get("box_transform_mode")
    if mode != "gligen":
        raise ProtocolMismatchError(
            "box_transform_mode must be 'gligen' so boxes follow image "
            f"resize/crop/flip; got {mode!r}"
        )


def validate_image_directories(real_dir, fake_dir, expected_count):
    real_names = {path.name for path in real_dir.glob("*.png")}
    fake_names = {path.name for path in fake_dir.glob("*.png")}
    if real_names != fake_names:
        missing_fake = sorted(real_names - fake_names)
        stale_fake = sorted(fake_names - real_names)
        raise ProtocolMismatchError(
            "real/fake filename mismatch: "
            f"missing_fake={missing_fake[:10]}, stale_fake={stale_fake[:10]}"
        )
    if len(real_names) != expected_count:
        raise ProtocolMismatchError(
            f"unexpected metric image count: {len(real_names)} != {expected_count}"
        )
    return {"count": len(real_names), "filenames_match": True}
