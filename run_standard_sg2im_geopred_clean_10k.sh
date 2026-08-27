#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

mkdir -p /root/autodl-tmp/GLIGEN/OUTPUT_STANDARD_SG2IM_GEOPRED
export HF_ENDPOINT=https://hf-mirror.com

exec /root/miniconda3/bin/python main.py \
  --yaml_file configs/vg_standard_sg2im_scene_graph_geopred_clean_full.yaml \
  --DATA_ROOT /root/autodl-tmp \
  --OUTPUT_ROOT /root/autodl-tmp/GLIGEN/OUTPUT_STANDARD_SG2IM_GEOPRED \
  --name vg_standard_sg2im_geopred_clean_full_10k \
  --batch_size 2 \
  --workers 0 \
  --total_iters 10000 \
  --save_every_iters 1000 \
  --disable_inference_in_training true \
  --save_trainable_only true \
  --freeze_fuser true \
  --freeze_position_base true \
  --init_from_gligen_ckpt /root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin
