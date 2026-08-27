# Layer-Aware Fuser Gate Design

## Goal

Add a safe, evaluation-first layer-aware fuser gate profile for GLIGEN/VG scene
graph generation. The purpose is to test whether selective fuser strengthening
can improve controllability without the quality collapse seen with global
`FUSER_ALPHA_ATTN_MULTIPLIER=3` or `6`.

## Context

Recent diagnostics showed:

```text
64x64 fuser layers: attention-box lift ~= 1.0, close to area-random
16x16 fuser layers: attention-box lift can reach ~= 5-6
global x1.5/x2 alpha sweep: visible control but style drift
global x3/x6 alpha sweep: repeated objects / texture collapse
```

This suggests the control path is not absent. It is weak and poorly distributed
across U-Net layers. A uniform multiplier is too blunt.

## Recommended Approach

Implement an eval-only layer profile parser and applicator:

```text
FUSER_ALPHA_ATTN_PROFILE="64:1.0,32:1.3,16:1.8,8:1.5"
FUSER_ALPHA_DENSE_PROFILE="all:1.0"
```

The profile is applied after checkpoint loading and before sampling. It only
modifies in-memory `alpha_attn` / `alpha_dense` values and never writes a new
checkpoint.

## Layer Resolution Mapping

Use fuser module names already observed in diagnostics:

```text
input_blocks.1, input_blocks.2, output_blocks.9, output_blocks.10, output_blocks.11 -> 64
input_blocks.4, input_blocks.5, output_blocks.6, output_blocks.7, output_blocks.8 -> 32
input_blocks.7, input_blocks.8, output_blocks.3, output_blocks.4, output_blocks.5 -> 16
middle_block.1 -> 8
```

Unknown fuser names should use `all` if present, otherwise `1.0`.

## Alternatives Considered

1. Global multiplier only.
   Rejected as the main path because x3/x6 quickly collapse quality.

2. Train a new learnable gate immediately.
   Deferred because we first need an eval-only proof that a selective profile has
   a better quality/control tradeoff.

3. Layer-aware eval profile first.
   Chosen because it is low-risk, reversible, metadata-tracked, and directly
   tests the current failure mode.

## Evaluation Protocol

Run diagnostic-only sampling first:

```text
split: VG fixed test
sample set: clean6, then clean18 if clean6 is promising
sampler: DDIM
steps: 50
guidance: 5.0
checkpoint: compact style-gate 1k checkpoint
```

Compare:

```text
x1 baseline
global x1.5
layer profile conservative: 64:1.0,32:1.2,16:1.5,8:1.3
layer profile stronger: 64:1.0,32:1.3,16:1.8,8:1.5
```

Acceptance criteria:

```text
1. More visible graph/control changes than x1.
2. Less texture collapse than global x3.
3. Less style drift than global x2.
4. Metadata records the exact profile.
```

## Implementation Boundary

Modify only:

```text
scripts/eval/inference_ablation.py
scripts/eval/generate_vg_fixedsplit_eval.py
tests/test_inference_ablation.py
docs/controlled_generation_code_audit_20260704.md
```

No training code change in this first step. If the eval-only profile helps, the
next step can turn the profile into a trainable or scheduled gate.

## Guardrails

This is not a formal FID/IS result. It is an exploratory controllability
diagnostic on fixed split samples. It must not be reported as a clean main-table
metric.
