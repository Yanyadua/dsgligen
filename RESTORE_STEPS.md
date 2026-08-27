# GLIGEN Recovery Steps

This bundle is the code-side recovery seed for the GLIGEN + scene-graph branch.

## 1. Prepare a fresh GLIGEN repo

Clone the official base repo to a target path, for example:

```bash
git clone <GLIGEN_REPO_URL> /root/autodl-tmp/GLIGEN
```

## 2. Apply this bundle

Run:

```bash
bash apply_recovery.sh /root/autodl-tmp/GLIGEN
```

This restores:

- `main.py`
- `trainer.py`
- scene-graph grounding files
- recovered configs
- historical comparison helpers
- eval launchers
- recovery smoke script
- bundle verification script

If you already have a partially restored repo on a remote machine and only need
to sync the files in this bundle, see:

```bash
bash sync_to_remote_example.sh <ssh-port> /root/autodl-tmp/GLIGEN
```

## 3. Run the code smoke test

From the target repo root:

```bash
/root/miniconda3/bin/python scripts/smoke_recovery.py
```

Expected behavior:

- It should print `SMOKE ok`
- It should instantiate:
  - `vg_standard_sg2im_scene_graph_geopred_clean_full.yaml`
  - `vg_raw_scene_graph_compatible_spatial_gat_geo_prediction_loss.yaml`
  - `vg_text_box_baseline.yaml`
  - `vg_scene_graph_mlp.yaml`
  - `vg_scene_graph_gat.yaml`
  - `vg_scene_graph_gat_residual.yaml`

## 4. What this bundle does not restore by itself

This is a code recovery bundle, not a full experiment snapshot. You still need:

- Visual Genome processed split data
  - expected root: `/root/autodl-tmp/standard_sg2im_fresh_h5`
- GLIGEN base checkpoint
  - expected path: `/root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin`
- optional trained checkpoints
  - expected root: `/root/autodl-tmp/GLIGEN/OUTPUT_STANDARD_SG2IM_GEOPRED`

On the current restored AutoDL instance, these resources are already present.
This section is mainly a warning for future fresh-instance restores.

## 5. Historical comparison configs

The recovered bundle now includes the historical comparison configs referenced by
old local scripts:

- `configs/vg_scene_graph_mlp.yaml`
- `configs/vg_scene_graph_gat.yaml`
- `configs/vg_scene_graph_gat_residual.yaml`

It also includes the historical comparison helper scripts:

- `sample_residual_horse_compare.py`
- `sample_residual_horse_compare_align.py`
