#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

mkdir -p /root/autodl-tmp/GLIGEN/OUTPUT_FIXED_CLEAN_GEOPRED
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

/root/miniconda3/bin/python scripts/eval/audit_historical_vg_run.py \
  --config configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml \
  --train-h5 /root/autodl-tmp/fixed_split_work/datasets/vg/train.h5 \
  --test-h5 /root/autodl-tmp/fixed_split_work/datasets/vg/test.h5 \
  --vocab /root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json \
  --base-checkpoint /root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin

exec /root/miniconda3/bin/python main.py \
  --yaml_file configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml \
  --DATA_ROOT /root/autodl-tmp \
  --OUTPUT_ROOT /root/autodl-tmp/GLIGEN/OUTPUT_FIXED_CLEAN_GEOPRED \
  --name vg_fixedsplit_geopred_clean_full_10k \
  --batch_size 2 \
  --workers 0 \
  --total_iters 10000 \
  --save_every_iters 1000 \
  --disable_inference_in_training true \
  --save_trainable_only true \
  --freeze_fuser true \
  --freeze_position_base true \
  --init_from_gligen_ckpt /root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin
