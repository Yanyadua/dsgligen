# GLIGEN Recovery Status

## 2026-06-10 verified root cause

The poor restored 10k run is not a valid continuation checkpoint:

- the official GLIGEN checkpoint stores the object/box grounding base as
  `position_net.linears.*`;
- the failed restored run instantiated `position_net.node_in.*`;
- `freeze_position_base=true` then froze that randomly initialized, unloaded
  object/box path;
- the failed run also normalized boxes against the raw image while training
  images were center-cropped and randomly flipped.

The repaired line now:

- restores the exact official three-layer `position_net.linears.*` layout;
- requires all 966 pretrained GLIGEN model tensors to load with matching shapes;
- allows only the 39 new scene-graph tensors to be absent from the base;
- transforms boxes with the same resize/crop/flip applied to images and remaps
  relation indices after filtering;
- saves a plain resolved config dictionary and a checked trainable manifest;
- pins the clean split at 62,565 train / 5,096 test with zero image-id overlap;
- uses DDIM 50, guidance 5.0, and seed 20260429 for historical evaluation.

Rollback archive on the current instance:

```text
/root/autodl-tmp/GLIGEN_before_recovery_20260610.tgz
```

Recovered code on the new AutoDL instance:

- `main.py`
- `trainer.py`
- `scripts/eval/generate_vg_fixedsplit_eval.py`
- `grounding_input/scene_graph_grounding_tokenizer_input.py`
- `dataset/dataset_vg_scene_graph.py`
- `dataset/catalog.py`
- `ldm/modules/diffusionmodules/scene_graph_grounding_net.py`
- `configs/vg_standard_sg2im_scene_graph_geopred_clean_full.yaml`
- `configs/vg_raw_scene_graph_compatible_spatial_gat_geo_prediction_loss.yaml`
- `configs/vg_text_box_baseline.yaml`
- `configs/vg_scene_graph_mlp.yaml`
- `configs/vg_scene_graph_gat.yaml`
- `configs/vg_scene_graph_gat_residual.yaml`
- `sample_residual_horse_compare.py`
- `sample_residual_horse_compare_align.py`
- `run_standard_sg2im_geopred_clean_10k.sh`
- `run_standard_sg2im_ddim50_eval.sh`

Smoke status:

- `main.py --help` passes
- `import trainer` passes
- `import scripts.eval.generate_vg_fixedsplit_eval` passes
- Scene-graph grounding modules instantiate successfully
- `configs/vg_scene_graph_gat_residual.yaml` instantiates with a live `graph_adapter` branch
- Old-style `VGGrounding` config aliases now pass through `ConCatDataset`
  and reach the recovered h5-based scene-graph dataset path before failing on
  missing data files, which confirms the remaining blocker there is data, not code

 Known legacy gaps:

- The recovered repo now includes the verified residual branch config
  `configs/vg_scene_graph_gat_residual.yaml`, plus the historical `mlp/gat`
  compatibility configs and the standard and baseline configs needed by the
  restored training/eval entry points.

Recovered compatibility layer:

- `dataset/concat_dataset.py` now accepts old `VGGrounding` config entries when
  they carry h5/vocab/image-root parameters, and redirects them to the recovered
  scene-graph dataset implementation.
- This lets old local helper scripts get past config/dataset wiring and fail only
  on missing non-code resources when VG data is absent.

Current instance resource status:

- Visual Genome standard split data restored
  - path: `/root/autodl-tmp/standard_sg2im_fresh_h5`
- GLIGEN base checkpoint restored
  - path: `/root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin`
- Standard-split training outputs present
  - root: `/root/autodl-tmp/GLIGEN/OUTPUT_STANDARD_SG2IM_GEOPRED`
- Historical fixedsplit recovery outputs present
  - root: `/root/autodl-tmp/GLIGEN/OUTPUT_FIXED_CLEAN_GEOPRED`

Future fresh-instance blockers:

- A brand-new instance still needs these non-code resources restored before
  training or evaluation can run:
  - `/root/autodl-tmp/standard_sg2im_fresh_h5`
  - `/root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin`
