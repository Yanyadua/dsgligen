#!/usr/bin/env bash
set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf_cache}"
cd /root/autodl-tmp/GLIGEN
mkdir -p "$HUGGINGFACE_HUB_CACHE"
mkdir -p /root/autodl-tmp/GLIGEN/eval_outputs/vg_standard_sg2im_geopred_clean_full_10k_ddim50_full_b4
export MODEL_YAML=configs/vg_standard_sg2im_scene_graph_geopred_clean_full.yaml
export DATA_YAML=configs/vg_standard_sg2im_scene_graph_geopred_clean_full.yaml
export GROUNDING_CKPT=/root/autodl-tmp/GLIGEN/OUTPUT_STANDARD_SG2IM_GEOPRED/vg_standard_sg2im_geopred_clean_full_10k/tag00/checkpoint_00010000.pth
export H5_PATH=/root/autodl-tmp/standard_sg2im_fresh_h5/test.h5
export VOCAB_PATH=/root/autodl-tmp/standard_sg2im_fresh_h5/vocab.json
export IMAGE_ROOT=/root/autodl-tmp/standard_sg2im_fresh_h5/images
export OUT_DIR=/root/autodl-tmp/GLIGEN/eval_outputs/vg_standard_sg2im_geopred_clean_full_10k_ddim50_full_b4
export NUM_SAMPLES=5096
export START_INDEX=0
export SPLIT_NAME=test
export SAMPLER=ddim
export STEPS=50
export GUIDANCE=5.0
export SAVE_SIZE=256
export EVAL_BATCH_SIZE=4
export SEED=20260508
exec /root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py
