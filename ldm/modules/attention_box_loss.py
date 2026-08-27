import math

import torch

from scripts.eval.inference_ablation import infer_fuser_resolution


def parse_attention_box_layer_weights(value):
    if value is None:
        return {}
    value = str(value).strip()
    if not value:
        return {}
    weights = {}
    for piece in value.split(","):
        key, separator, raw_weight = piece.strip().partition(":")
        if not separator:
            raise ValueError(
                "Attention-box layer weights must use 'key:value', "
                f"got {piece!r}"
            )
        key = key.strip().lower()
        if key == "all":
            weights["all"] = float(raw_weight)
        else:
            resolution = int(key)
            if resolution not in {8, 16, 32, 64}:
                raise ValueError(
                    "Attention-box layer weight keys must be 8, 16, 32, 64, or all"
                )
            weights[resolution] = float(raw_weight)
    return weights


def _infer_square_grid(num_visual_tokens):
    size = int(math.sqrt(int(num_visual_tokens)))
    if size * size != int(num_visual_tokens):
        return None
    return size, size


def _boxes_to_token_masks(boxes, grid_h, grid_w, masks):
    device = boxes.device
    dtype = boxes.dtype
    ys = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) / float(grid_h)
    xs = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) / float(grid_w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx.reshape(1, 1, grid_h * grid_w)
    yy = yy.reshape(1, 1, grid_h * grid_w)
    x0, y0, x1, y1 = boxes.clamp(0.0, 1.0).unbind(dim=-1)
    token_masks = (
        (xx >= x0.unsqueeze(-1))
        & (xx <= x1.unsqueeze(-1))
        & (yy >= y0.unsqueeze(-1))
        & (yy <= y1.unsqueeze(-1))
    )
    return token_masks & (masks > 0.5).unsqueeze(-1)


def compute_attention_box_loss_from_attention(
    visual_to_grounding_attention,
    boxes,
    masks,
    target_inside_ratio=0.5,
):
    if visual_to_grounding_attention.dim() != 4:
        raise ValueError("visual_to_grounding_attention must have shape [B, H, V, G]")
    batch, _, num_visual, num_grounding = visual_to_grounding_attention.shape
    if boxes.shape[:2] != (batch, num_grounding):
        boxes = boxes[:, :num_grounding, :]
        masks = masks[:, :num_grounding]
    grid = _infer_square_grid(num_visual)
    if grid is None:
        return visual_to_grounding_attention.new_tensor(0.0)

    token_masks = _boxes_to_token_masks(boxes, grid[0], grid[1], masks)
    attention = visual_to_grounding_attention.clamp_min(0.0).mean(dim=1).transpose(1, 2)
    inside = (attention * token_masks.to(attention.dtype)).sum(dim=-1)
    total = attention.sum(dim=-1).clamp_min(1e-12)
    inside_ratio = inside / total
    valid = masks.to(dtype=torch.bool) & (total > 0)
    if not valid.any():
        return visual_to_grounding_attention.new_tensor(0.0)
    per_token_loss = torch.relu(float(target_inside_ratio) - inside_ratio)
    return per_token_loss[valid].mean()


def set_fuser_attention_recording(model, enabled=True, detach=True):
    updated = 0
    for module in model.modules():
        if hasattr(module, "set_attention_recording") and hasattr(module, "get_visual_to_grounding_attention"):
            module.set_attention_recording(enabled, detach=detach)
            updated += 1
    return updated


def collect_fuser_attention_box_loss(
    model,
    boxes,
    masks,
    target_inside_ratio=0.5,
    layer_weights=None,
):
    layer_weights = layer_weights or {}
    losses = []
    weights = []
    for name, module in model.named_modules():
        if not hasattr(module, "get_visual_to_grounding_attention"):
            continue
        attention = module.get_visual_to_grounding_attention()
        if attention is None:
            continue
        resolution = infer_fuser_resolution(name)
        weight = layer_weights.get(
            resolution,
            layer_weights.get("all", 1.0),
        )
        if weight <= 0:
            continue
        loss = compute_attention_box_loss_from_attention(
            attention,
            boxes.to(device=attention.device, dtype=attention.dtype),
            masks.to(device=attention.device, dtype=attention.dtype),
            target_inside_ratio=target_inside_ratio,
        )
        losses.append(loss * float(weight))
        weights.append(float(weight))
    if not losses:
        device = boxes.device
        return torch.tensor(0.0, device=device, dtype=boxes.dtype)
    return torch.stack(losses).sum() / max(sum(weights), 1e-12)
