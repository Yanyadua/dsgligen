import torch


FUSER_RESOLUTION_PREFIXES = {
    64: (
        "input_blocks.1.",
        "input_blocks.2.",
        "output_blocks.9.",
        "output_blocks.10.",
        "output_blocks.11.",
    ),
    32: (
        "input_blocks.4.",
        "input_blocks.5.",
        "output_blocks.6.",
        "output_blocks.7.",
        "output_blocks.8.",
    ),
    16: (
        "input_blocks.7.",
        "input_blocks.8.",
        "output_blocks.3.",
        "output_blocks.4.",
        "output_blocks.5.",
    ),
    8: ("middle_block.1.",),
}


def restore_base_fuser_state(model_state, base_state):
    restored = 0
    for name, value in base_state.items():
        if ".fuser." not in name or name not in model_state:
            continue
        model_state[name] = value
        restored += 1
    return restored


def apply_graph_gate_override(model, value):
    position_net = getattr(model, "position_net", None)
    if position_net is None:
        raise AttributeError("model has no position_net for graph-gate ablation")
    if value is None:
        position_net.clear_graph_gate_override()
    else:
        position_net.set_graph_gate_override(value)


def _iter_fuser_modules(model):
    for name, module in model.named_modules():
        if name.endswith(".fuser"):
            yield name, module


def parse_fuser_alpha_profile(value):
    if value is None:
        return {}
    value = str(value).strip()
    if not value:
        return {}
    profile = {}
    for piece in value.split(","):
        key, separator, raw_multiplier = piece.strip().partition(":")
        if not separator:
            raise ValueError(
                "Fuser alpha profile entries must use 'key:value', "
                f"got {piece!r}"
            )
        key = key.strip().lower()
        if key == "all":
            profile["all"] = float(raw_multiplier)
            continue
        resolution = int(key)
        if resolution not in FUSER_RESOLUTION_PREFIXES:
            raise ValueError(
                "Fuser alpha profile layer keys must be one of "
                f"{sorted(FUSER_RESOLUTION_PREFIXES)} or 'all', got {key!r}"
            )
        profile[resolution] = float(raw_multiplier)
    return profile


def infer_fuser_resolution(name):
    for resolution, prefixes in FUSER_RESOLUTION_PREFIXES.items():
        if any(name.startswith(prefix) for prefix in prefixes):
            return resolution
    return None


def resolve_fuser_multiplier(name, profile):
    if not profile:
        return 1.0
    resolution = infer_fuser_resolution(name)
    if resolution in profile:
        return float(profile[resolution])
    if "all" in profile:
        return float(profile["all"])
    return 1.0


def summarize_fuser_alpha(model):
    attn_values = []
    dense_values = []
    for _, module in _iter_fuser_modules(model):
        alpha_attn = getattr(module, "alpha_attn", None)
        alpha_dense = getattr(module, "alpha_dense", None)
        if alpha_attn is not None:
            attn_values.append(torch.tanh(alpha_attn.detach()).abs().float())
        if alpha_dense is not None:
            dense_values.append(torch.tanh(alpha_dense.detach()).abs().float())

    def summarize(values):
        if not values:
            return 0.0, 0.0
        stacked = torch.stack([value.reshape(()) for value in values])
        return float(stacked.mean().cpu()), float(stacked.max().cpu())

    mean_attn, max_attn = summarize(attn_values)
    mean_dense, max_dense = summarize(dense_values)
    return {
        "count": len(attn_values),
        "mean_abs_tanh_alpha_attn": mean_attn,
        "max_abs_tanh_alpha_attn": max_attn,
        "mean_abs_tanh_alpha_dense": mean_dense,
        "max_abs_tanh_alpha_dense": max_dense,
    }


def apply_fuser_alpha_multiplier(model, attn_multiplier=1.0, dense_multiplier=1.0):
    before = summarize_fuser_alpha(model)
    updated = 0
    with torch.no_grad():
        for _, module in _iter_fuser_modules(model):
            alpha_attn = getattr(module, "alpha_attn", None)
            alpha_dense = getattr(module, "alpha_dense", None)
            if alpha_attn is None:
                continue
            alpha_attn.mul_(float(attn_multiplier))
            if alpha_dense is not None:
                alpha_dense.mul_(float(dense_multiplier))
            updated += 1
    after = summarize_fuser_alpha(model)
    return {
        "updated": updated,
        "attn_multiplier": float(attn_multiplier),
        "dense_multiplier": float(dense_multiplier),
        "before": before,
        "after": after,
    }


def apply_fuser_alpha_profile(model, attn_profile=None, dense_profile=None):
    attn_profile = attn_profile or {}
    dense_profile = dense_profile or {}
    before = summarize_fuser_alpha(model)
    updated = 0
    layers = []
    with torch.no_grad():
        for name, module in _iter_fuser_modules(model):
            alpha_attn = getattr(module, "alpha_attn", None)
            alpha_dense = getattr(module, "alpha_dense", None)
            if alpha_attn is None:
                continue
            attn_multiplier = resolve_fuser_multiplier(name, attn_profile)
            dense_multiplier = resolve_fuser_multiplier(name, dense_profile)
            alpha_attn.mul_(attn_multiplier)
            if alpha_dense is not None:
                alpha_dense.mul_(dense_multiplier)
            updated += 1
            layers.append({
                "name": name,
                "resolution": infer_fuser_resolution(name),
                "attn_multiplier": float(attn_multiplier),
                "dense_multiplier": float(dense_multiplier),
            })
    after = summarize_fuser_alpha(model)
    return {
        "updated": updated,
        "attn_profile": dict(attn_profile),
        "dense_profile": dict(dense_profile),
        "layers": layers,
        "before": before,
        "after": after,
    }
