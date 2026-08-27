import math

import torch


def infer_square_grid(num_visual_tokens):
    size = int(math.sqrt(int(num_visual_tokens)))
    if size * size != int(num_visual_tokens):
        return None
    return size, size


def boxes_to_token_masks(boxes, grid_h, grid_w, valid_mask=None):
    """Map normalized xyxy boxes to flattened visual-token masks."""
    if boxes.dim() != 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [batch, num_boxes, 4]")

    device = boxes.device
    dtype = boxes.dtype
    ys = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) / float(grid_h)
    xs = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) / float(grid_w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx.reshape(1, 1, grid_h * grid_w)
    yy = yy.reshape(1, 1, grid_h * grid_w)

    x0, y0, x1, y1 = boxes.clamp(0.0, 1.0).unbind(dim=-1)
    masks = (
        (xx >= x0.unsqueeze(-1))
        & (xx <= x1.unsqueeze(-1))
        & (yy >= y0.unsqueeze(-1))
        & (yy <= y1.unsqueeze(-1))
    )
    if valid_mask is not None:
        masks = masks & (valid_mask > 0.5).unsqueeze(-1)
    return masks


def attention_inside_box_ratio(visual_to_grounding_attention, boxes, grounding_mask=None):
    """Return per-grounding-token attention mass inside its box.

    Args:
        visual_to_grounding_attention: tensor [B, H, V, G], where V is a
            square visual grid and G is grounding-token count.
        boxes: normalized xyxy tensor [B, G, 4].
        grounding_mask: optional tensor [B, G].

    Returns:
        ratio: [B, G], attention mass inside box divided by total mass for the
            corresponding grounding token. Invalid tokens are NaN.
        area_ratio: [B, G], fraction of visual tokens covered by each box.
    """
    if visual_to_grounding_attention.dim() != 4:
        raise ValueError("visual_to_grounding_attention must have shape [B, H, V, G]")
    batch, _, num_visual, num_grounding = visual_to_grounding_attention.shape
    if boxes.shape[:2] != (batch, num_grounding):
        raise ValueError("boxes must have shape [B, G, 4] matching attention")

    grid = infer_square_grid(num_visual)
    if grid is None:
        raise ValueError(f"visual token count must be square, got {num_visual}")
    token_masks = boxes_to_token_masks(boxes, grid[0], grid[1], grounding_mask)

    attn = visual_to_grounding_attention.clamp_min(0.0).mean(dim=1).transpose(1, 2)
    inside = (attn * token_masks.to(attn.dtype)).sum(dim=-1)
    total = attn.sum(dim=-1)

    valid = total > 0
    if grounding_mask is not None:
        valid = valid & (grounding_mask > 0.5)
    ratio = torch.full_like(total, float("nan"))
    ratio[valid] = inside[valid] / total[valid].clamp_min(1e-12)

    area_ratio = token_masks.to(attn.dtype).mean(dim=-1)
    area_ratio = torch.where(valid, area_ratio, torch.full_like(area_ratio, float("nan")))
    return ratio, area_ratio
