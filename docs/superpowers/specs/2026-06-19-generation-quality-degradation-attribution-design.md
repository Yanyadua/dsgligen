# Generation Quality Degradation Attribution

## Goal

Identify whether the observed color and texture degradation comes from trained
GLIGEN fuser weights, scene-graph residual tokens, or their interaction.

## Protocol

Use VG fixed test samples `10, 113, 126, 16` under one protocol: DDIM 50,
guidance 5, seed 20260429, 256px, center-crop preprocessing.

Evaluate four inference-only variants from the same 1k checkpoint:

| Variant | Fuser | Graph residual |
|---|---|---|
| F0G0 | Official base | Off |
| F0G1 | Official base | On |
| F1G0 | Trained 1k | Off |
| F1G1 | Trained 1k | On |

## Implementation

Add two evaluation-only environment controls:

- `RESTORE_BASE_FUSER`: after loading the training checkpoint, restore every
  `.fuser.` tensor from the verified official GLIGEN base state.
- `GRAPH_GATE_OVERRIDE`: call the scene-graph PositionNet override API after
  checkpoint loading; `0` disables the residual exactly.

Record both controls in `meta.txt` so output directories cannot be resumed with
different ablation settings.

## Interpretation

- F1G0 degrades: full-fuser training is the primary cause.
- F0G1 degrades: graph token distribution or strength is the primary cause.
- Only F1G1 degrades: interaction between trained fuser and graph residual.
- F0G0 degrades: scene-graph base-path compatibility or evaluation protocol is
  wrong; stop model optimization and repair the baseline first.

## Acceptance

All four variants load the same base/checkpoint hashes, save four images, and
produce a labeled comparison grid. No training state is modified.
