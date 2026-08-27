import math

import numpy as np


def parse_grounding_alpha_type(value):
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",")]
        if len(pieces) != 3:
            raise ValueError("GROUNDING_ALPHA_TYPE must contain three comma-separated ratios")
        ratios = tuple(float(piece) for piece in pieces)
    else:
        ratios = tuple(float(piece) for piece in value)
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError("Grounding alpha ratios must be three non-negative values")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Grounding alpha ratios must sum to 1, got {ratios}")
    return ratios


def build_grounding_alpha_schedule(length, alpha_type):
    alpha_type = parse_grounding_alpha_type(alpha_type)
    full_length = int(alpha_type[0] * length)
    decay_length = int(alpha_type[1] * length)
    off_length = length - full_length - decay_length
    if decay_length:
        decay = np.arange(start=0, stop=1, step=1 / decay_length)[::-1].tolist()
    else:
        decay = []
    schedule = [1.0] * full_length + decay + [0.0] * off_length
    if len(schedule) != length:
        raise RuntimeError(f"Invalid grounding alpha schedule length: {len(schedule)} != {length}")
    return schedule


def set_grounding_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense

    for module in model.modules():
        if isinstance(module, (GatedCrossAttentionDense, GatedSelfAttentionDense)):
            module.scale = alpha_scale
