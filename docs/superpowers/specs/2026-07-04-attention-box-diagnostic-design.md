# Attention-Box Diagnostic Design

## Goal

Add a low-risk diagnostic path that measures whether GLIGEN grounding tokens attend to their own bounding boxes on the clean18 Visual Genome stress set.

This is a diagnostic prototype only. It must not change default training or sampling behavior, and it must not be reported as a formal VG metric.

## Motivation

The clean18 three-way probe showed:

```text
graph_on vs graph_off mean_mad: 0.1315 / 255
relation_token_v2 vs graph_on mean_mad: 10.3119 / 255
```

This suggests the current GAT graph residual is visually negligible, while explicit relation tokens are visible but not reliably spatially correct. Before adding another training loss, we need to inspect whether fuser attention connects grounding tokens to the intended image regions.

## Approach

1. Add an optional attention recorder to `SelfAttention`.
   - Default: disabled.
   - When enabled: store the latest attention matrix.
   - Default outputs and checkpoint compatibility remain unchanged.

2. Add lightweight helpers to `GatedSelfAttentionDense`.
   - Enable/disable recording on its internal self-attention.
   - Return the visual-query to grounding-key attention slice.

3. Add a diagnostic script.
   - Runs a normal eval forward/sampling path with recording enabled.
   - Computes object-level inside-box attention ratios.
   - Writes JSON summaries for clean18.

## Success Criteria

The implementation is acceptable if:

- Existing default fuser outputs are unchanged when recording is off.
- A test can verify the recorded attention shape and visual-to-grounding slice shape.
- The diagnostic can run on the clean18 protocol without changing generated images.
- Output includes per-object `attention_inside_box_ratio`, `object_text`, `box_xyxy`, and sample id.

## Non-Goals

- Do not add an attention-box training loss yet.
- Do not change fuser alpha gates or graph gate behavior.
- Do not use this diagnostic as FID/IS/OOR evidence.
