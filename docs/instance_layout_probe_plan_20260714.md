# Instance-level layout probe: pre-registered next experiment

## Decision

Do not extend the GLIGEN triplet fuser.  Its 500-step clean-split diagnostic
produced only a small pixel perturbation and no stable subject or relation
gain, including when the learned gate was amplified.  The next falsifiable
hypothesis is narrower: **an instance-native control path can place the
cleaned VG objects before we attempt relation-specific learning.**

The candidate is InstanceDiffusion (SD1.5), not a direct SDXL/FLUX replacement.
It accepts one text phrase and one box per instance, so the condition channel
matches the useful part of a VG graph.  Its official repository was cloned for
read-only compatibility inspection at `/root/autodl-tmp/InstanceDiffusion`,
commit `722900ef03e3579fb693e65a1550333a4a162581`.  No weights or dependencies
have been downloaded or installed yet.

## Fixed input contract

Exporter:
`scripts/eval/export_vg_instance_layout_json.py`

Materialized remote inputs:
`/root/autodl-tmp/GLIGEN/eval_outputs/clean10_instance_layout_major4_20260714`

Local copy:
`artifacts/clean10_instance_layout_major4_20260714/clean10_instance_layout_major4_20260714`

The manifest pins:

- test H5 SHA-256: `880899a547b24b893ee985ac6985b308a346f0178291dbda3c3f4e931231a273`
- vocab SHA-256: `0205fae37829e763805aed30b6df623d65e844982bbab26c49196bb3842d04cf`
- fixed test indices: `1008,1048,1978,2022,2942,3530,3544,3651,4786,5000`
- 512px deterministic GLIGEN center crop, then clean spatial selection
- at most one spatial relation and at most four major instance controls

The first probe intentionally has **no relation text token**.  The global
caption carries only clean positive photographic style plus major nouns; each
instance phrase is a relation-free noun phrase; boxes carry placement.  A
relation stays only in the manifest as an evaluation target.  This prevents a
good result from being misattributed to relation language when it came from
layout.

The major-object pass always preserves the selected relation endpoints, drops
background/part labels where possible, removes redundant non-core category
nouns, and then caps the layout at four objects.  For example, sample 3544 is
now `{elephant, man, building}` with `man near elephant`; sample 4786 is
`{bus, boy, road, tire}` with `tire under bus`.

## Execution order (no training)

1. Create an isolated environment for the official diffusers InstanceDiffusion
   port.  Do not alter the GLIGEN Conda environment.  Record the exact branch
   commit, Python/Torch/diffusers versions, and every downloaded weight hash.
2. Download the officially published `kyeongry/instancediffusion_sd15` model
   only after checking total disk footprint.  The server currently has about
   14 GB free, so cache location and final artifact sizes must be checked before
   download.
3. Smoke-test the official demo once.  This is an installation check, not a VG
   result.
4. Generate the fixed ten inputs at two fixed seeds, with the paper/demo-style
   sampler settings recorded in each output manifest (steps, guidance,
   instance gating/scheduled sampling settings, seed, model revision).
5. Score manually and independently for each seed: (a) required subject is
   visible, (b) subject is substantially inside/overlaps its target box,
   (c) the selected spatial relation is visually correct, (d) image remains
   color-realistic.  Save a grid with target boxes overlaid.

## Decision rules and rollback

- **Advance:** clear, repeated object-count/box-placement improvement over the
  existing GLIGEN clean-caption baseline on the same ten inputs, without a
  serious quality collapse.  Only then add a single relation-aware component:
  localized pair attention or a pair-region map, not a global residual graph
  token.
- **Stop / rollback:** if box placement is not visibly better on both seeds,
  do not train adapters or add relation loss.  The problem is then model choice
  or input semantics, not a missing graph loss.
- This is an exploratory fixed-10 diagnostic, not FID/IS and not a main-table
  result.  No first-k/full-split comparison is permitted.

## Compatibility finding

The legacy InstanceDiffusion CLI uses the exporter schema directly: pixel
`[xmin, ymin, width, height]`.  It targets an old dependency stack, while the
official README also documents a newer diffusers implementation.  The current
GLIGEN environment has PyTorch 2.0.0+cu118 and Transformers 4.46.3 but lacks
both diffusers and xformers; an isolated setup is therefore required.
