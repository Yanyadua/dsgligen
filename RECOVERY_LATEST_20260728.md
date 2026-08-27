# Latest Recovery Target: 2026-07-28 Conditioning Experiments

## Authority and scope

This file supersedes `RECOVERY_STATUS.md` when deciding what to restore or
run next. `RECOVERY_STATUS.md` documents a June repaired-training line. It is
useful historical evidence, but it is not the last experimental state.

The latest recoverable work was the July 28 conditioning study. Its purpose
was to diagnose and improve how Visual Genome scene graphs are converted into
the text-plus-box conditions accepted by the official GLIGEN base model. It
did not depend on a newly trained GAT or relation-geometry checkpoint.

## Latest code artifacts

- `dataset/scene_graph_conditioning.py`: legacy and compact scene-graph
  condition builders, including `clean_spatial_v2`.
- `dataset/relation_grounding_tokens.py`: optional conservative relation-token
  construction. The latest main A/C comparisons keep it disabled.
- `scripts/eval/generate_vg_fixedsplit_eval.py`: deterministic fixed-split
  generation with saved sample metadata.
- `scripts/eval/clip_crop_grounding_score.py`: crop-aware CLIP grounding
  diagnostic, separate from FID/IS.
- `run_vg_conditioning_v2_clean30.sh`: 30-example diagnostic arms A-D.
- `run_vg_conditioning_ac500_eval.sh`: 500-example A/C comparison.
- `run_vg_conditioning_c_v2_graphcap500_eval.sh`: caption-isolation arm.

## Reproducible July A/C protocol

All A/C results are exploratory diagnostics, not main-paper FID/IS claims.

| Setting | Value |
| --- | --- |
| Split | first 500 items of the fixed SG2IM/VG test H5, start index 0 |
| Base model | official `gligen/gligen-generation-text-box` UNet checkpoint |
| Learned grounding checkpoint | none (`GROUNDING_CKPT` unset) |
| Sampler | DDIM, 50 steps |
| Guidance | 3.0 |
| Seed | 20260728 |
| Image size / batch | 256 / 4 |
| Relation grounding tokens | disabled |
| A arm | `CONDITIONING_POLICY=legacy`, `CAPTION_POLICY=graph` |
| C arm | `CONDITIONING_POLICY=clean_spatial_v2`, max 8 objects, max 2 relations |

The `C_v2_graphcap` arm retains the compact `clean_spatial_v2` grounding but
uses `CAPTION_POLICY=graph`. It isolates caption wording from object/relation
selection.

## What is and is not recoverable

Recoverable:

- The complete GLIGEN base source tree and the July 28 uncommitted patch layer.
- The July generation/evaluation launchers and their protocol settings.
- A historical 500-step July 13 fuser checkpoint, which is not a current
  default and must not be presented as the July 28 result.

Not recoverable from the local materials:

- The exact long-run 2k/10k trained checkpoints from released instances.
- A definitive final FID/IS result for the July 28 `C_v2_graphcap` run. Only
  generated samples and metadata were preserved.

Do not reconstruct missing numerical results from logs, screenshots, or
memory. Re-run them under a newly audited data and metric protocol.

## Data and evaluation guardrails

Before any new run, validate the exact H5 image counts, vocabulary and
zero train/test image-id overlap. Record the resolved config, checkpoint SHA,
real/fake counts, preprocessing, sampler, seed and metric implementation.

Do not compare first-500 diagnostics with full-test FID/IS, and do not report
either as a paper-level result without the same protocol for every compared
method.
