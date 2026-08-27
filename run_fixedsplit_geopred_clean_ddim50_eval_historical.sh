#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

mkdir -p /root/autodl-tmp/GLIGEN/eval_outputs/vg_fixedsplit_geopred_clean_10k_full_b4

export HF_ENDPOINT=https://hf-mirror.com
export MODEL_YAML=configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml
export DATA_YAML=configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml
export GROUNDING_CKPT=/root/autodl-tmp/GLIGEN/OUTPUT_FIXED_CLEAN_GEOPRED/vg_fixedsplit_geopred_clean_full_10k/tag00/checkpoint_00010000.pth
export H5_PATH=/root/autodl-tmp/fixed_split_work/datasets/vg/test.h5
export VOCAB_PATH=/root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json
export IMAGE_ROOT=/root/autodl-tmp/fixed_split_work/datasets/vg/images
export OUT_DIR=/root/autodl-tmp/GLIGEN/eval_outputs/vg_fixedsplit_geopred_clean_10k_full_b4
export NUM_SAMPLES=5096
export START_INDEX=0
export SPLIT_NAME=test
export SAMPLER=ddim
export STEPS=50
export GUIDANCE=5.0
export SAVE_SIZE=256
export EVAL_BATCH_SIZE=4
export SEED=20260429
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

/root/miniconda3/bin/python scripts/eval/audit_historical_vg_run.py \
  --config "$MODEL_YAML" \
  --train-h5 /root/autodl-tmp/fixed_split_work/datasets/vg/train.h5 \
  --test-h5 "$H5_PATH" \
  --vocab "$VOCAB_PATH" \
  --base-checkpoint /root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin \
  --grounding-checkpoint "$GROUNDING_CKPT"

exec /root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py
