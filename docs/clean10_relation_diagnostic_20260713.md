# Clean10 relation diagnostic — 2026-07-13

This is a small controllability diagnostic, not a FID/IS/OOR evaluation and not a training run.

## Protocol

- Test source: `/root/autodl-tmp/standard_sg2im_fresh_h5/test.h5`
- Vocabulary: the matching `vocab.json` (179 object labels, 46 predicates)
- Fixed dataset indices: `1008,1048,1978,2022,2942,3530,3544,3651,4786,5000`
- Samples: 10
- Base checkpoint: `gligen_checkpoints/diffusion_pytorch_model.bin`
- Sampler: DDIM, 50 steps, guidance 3.0
- Seed: 20260713
- Conditioning: `clean_spatial_v1`, max 6 objects, max 1 relation, relation mask scale 0.5
- Arms: C = no relation grounding token; D = one spatial relation grounding token
- No learned grounding checkpoint and no graph gate override

## Findings

1. The current synced code and the local code have identical SHA256 hashes for the key dataset, caption, grounding, evaluation, and audit files.
2. The old `clean30_d` outputs are stale with respect to the current low-level-relation filter: their `2338873` metadata still contains `pant on floor`, while the current dataset returns no active relation for that sample. Those old images must not be used as the latest protocol result.
3. On the ten supported relation samples, C and D are often semantically similar. D changes pixels, but the relation token does not consistently improve object identity or relation fidelity.
4. `water inside cup` still produces a pizza-like image; this is a semantic/object failure, not a missing relation token.
5. `clean_primary` removes generic caption words such as `edge`, `reflection`, `floor`, `wall`, and only verbalizes relations whose endpoints remain in the caption. In this ten-sample probe it changes outputs modestly, but does not establish a consistent improvement over `clean`.

## Current decision

Do not sweep relation scale or start training yet. The next architectural step should be an explicit object/triplet adapter or fuser with object-to-relation binding. Caption cleanup alone is insufficient, and relation tokens alone are not a reliable semantic controller.

Artifacts:

- `artifacts/clean10_rel_current_20260713/clean10_current_relation_comparison_20260713.png`
- `artifacts/clean10_rel_current_20260713/clean10_primary_caption_comparison_20260713.png`
- `artifacts/clean30_condition_coverage_audit_20260712/clean30_condition_coverage_review_v2_20260713.json`
