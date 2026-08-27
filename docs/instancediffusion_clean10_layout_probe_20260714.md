# InstanceDiffusion VG clean10 layout-only probe (exploratory)

## Status

**Pass as a layout-control diagnostic; not a paper result and not FID/IS.**

This is the first model switch after the GLIGEN triplet-fuser negative result.
It tests only whether clean VG object boxes can control an instance-native
generator.  It does not train, use a relation token, or report a dataset-wide
metric.

## Pinned protocol

- Test source: `/root/autodl-tmp/standard_sg2im_fresh_h5/test.h5`
  (`sha256=880899a547b24b893ee985ac6985b308a346f0178291dbda3c3f4e931231a273`).
- Vocab: `/root/autodl-tmp/standard_sg2im_fresh_h5/vocab.json`
  (`sha256=0205fae37829e763805aed30b6df623d65e844982bbab26c49196bb3842d04cf`).
- Training source: none.  This is inference with the published
  `kyeongry/instancediffusion_sd15` fp16 model, so it cannot be compared as a
  clean trained VG main result.
- Fixed test indices (10): `1008,1048,1978,2022,2942,3530,3544,3651,4786,5000`.
- Input selection: deterministic GLIGEN center crop; clean spatial graph;
  one spatial relation retained for *scoring only*; at most four major object
  instances.  Relation endpoints are retained; low-value background/part
  labels are removed where possible.
- Global caption: positive color/style plus object nouns only.  It contains no
  relation phrase.  Instance phrases are relation-free object noun phrases.
- Model snapshot: `kyeongry/instancediffusion_sd15`, revision
  `8fb12b54d893acbe333a5e2195f197bec1829f01`, fp16 files:
  `unet=26305ecece83dca73cc72801c3e0c364754065a45c40e20cce17e738d20936f6`,
  `text_encoder=660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd`,
  `vae=4fbcf0ebe55a0984f5a5e00d8c4521d52359af7229bb4d81890039d2aa16dd7c`.
- Runtime: official InstanceDiffusion diffusers branch commit
  `d377f574007a2cb0dc31d768bf61a9d56cbfc7ad`, `diffusers 0.32.0.dev0`,
  PyTorch `2.0.0+cu118`; PNDM scheduler; 512px; 50 steps; guidance `7.5`;
  InstanceDiffusion alpha `0.8`, beta `0.36`; no negative prompt.
- Seeds: two fixed base seeds `20260714` and `20260715`; per-sample seed is
  `base_seed + fixed_test_index`.
- Shared input-manifest SHA-256:
  `757de8422741b5f4ececfce12e654357e12f13db86ee30f6e0d601e8a6b08475`.

## Visual finding (two-seed manual audit)

Stable and visually plausible in both seeds: `bird above water`, `plant on
cabinet`, `tree next to road`, `building next to street`, `man near elephant`,
`clock near tree`, `tire under bus`, and `person in snow` (8/10 relation
targets).  The subjects and their requested regions are visibly represented in
these cases despite relations being withheld from the prompt.

The two weak cases are `water inside cup` (the food/table composition appears,
but cup/water are not unambiguous) and `grass next to tree` (the bear dominates
and the requested contextual pair is not reliably distinct).  These are not
evidence for a relation module yet; they first indicate that the selected VG
objects can still be semantically weak for an instance generator.

This is qualitatively stronger and more stable than the earlier GLIGEN
triplet-fuser diagnostic, whose learned residual barely changed images even
after gate amplification.  It supports the architectural inference that
instance-/region-level injection is a viable base, whereas a small global
graph-token residual is not an effective control entrance.

## Artifacts

- Seed 20260714 raw grid:
  `artifacts/instancediffusion_clean10_layout_norel_v2_seed20260714/raw_grid.png`
- Seed 20260715 raw grid:
  `artifacts/instancediffusion_clean10_layout_norel_v2_seed20260715/raw_grid.png`
- Each artifact directory includes raw images, target-box overlays, and a
  machine-readable `meta.json` with individual image hashes.

## Next decision

Do **not** train yet.  Freeze this model/input configuration as the layout
baseline.  Next, audit and replace only the two weak inputs with semantically
cleaner relation pairs, then run a pre-registered *pair-local* ablation:
layout-only versus exactly one union-region relation phrase/feature.  If that
improves the two hard cases without damaging the eight stable cases across two
seeds, then and only then implement a lightweight frozen-backbone pair adapter.
