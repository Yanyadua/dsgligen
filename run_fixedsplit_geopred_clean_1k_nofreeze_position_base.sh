#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

mkdir -p /root/autodl-tmp/GLIGEN/OUTPUT_FIXED_CLEAN_GEOPRED_REPRO
export HF_ENDPOINT=https://hf-mirror.com

exec /root/miniconda3/bin/python main.py \
  --yaml_file configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml \
  --DATA_ROOT /root/autodl-tmp \
  --OUTPUT_ROOT /root/autodl-tmp/GLIGEN/OUTPUT_FIXED_CLEAN_GEOPRED_REPRO \
  --name vg_fixedsplit_geopred_clean_full_1k_nofreeze_position_base_20260604 \
  --batch_size 2 \
  --workers 0 \
  --total_iters 1000 \
  --save_every_iters 500 \
  --disable_inference_in_training true \
  --save_trainable_only true \
  --freeze_fuser true \
  --init_from_gligen_ckpt /root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin
