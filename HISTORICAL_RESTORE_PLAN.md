# Historical Strong-Run Restore Plan

This note tracks the stricter restoration target inferred from surviving local
artifacts, not from later approximate restarts.

## Historical Anchors

- Qualitative sample metadata:
  - `gligen_eval_fixed/qual_sggig_like_best_10k/meta.txt`
- Historical eval output naming:
  - `gligen_eval_fixed/rsr_fail_visuals/rsr_fail_records.json`

These surviving artifacts consistently point to:

- config: `configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml`
- training output root: `OUTPUT_FIXED_CLEAN_GEOPRED/vg_fixedsplit_geopred_clean_full_10k`
- eval output root: `eval_outputs/vg_fixedsplit_geopred_clean_10k_full_b4`
- dataset root: `/root/autodl-tmp/fixed_split_work/datasets/vg`

## What Is Not Yet Recovered

- The original remote directory `/root/autodl-tmp/fixed_split_work/datasets/vg`
- The original output tree `/root/autodl-tmp/GLIGEN/OUTPUT_FIXED_CLEAN_GEOPRED`
- The original eval tree `/root/autodl-tmp/GLIGEN/eval_outputs/vg_fixedsplit_geopred_clean_10k_full_b4`
- The exact historical metric summary file containing the remembered strong
  FID/IS values

## Temporary Compatibility Strategy

Until the original fixed-split directory is recovered, mirror the currently
restored clean split into the historical path layout using symlinks:

- source: `/root/autodl-tmp/standard_sg2im_fresh_h5`
- target: `/root/autodl-tmp/fixed_split_work/datasets/vg`

This does not prove equivalence to the original strong run. It only restores
the expected path semantics so we can separate path/protocol drift from model
behavior drift.

## Execution Order

1. Run `prepare_historical_fixedsplit_layout.sh`
2. Verify the historical dataset paths resolve correctly
3. Run a historical-name 1k smoke from the 10k launcher template
4. Sample a small preview with the historical eval launcher
5. Only after those checks, consider a longer historical-name training run
