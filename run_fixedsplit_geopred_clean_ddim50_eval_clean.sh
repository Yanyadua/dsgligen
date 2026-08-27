#!/usr/bin/env bash
set -euo pipefail

source /etc/network_turbo >/dev/null 2>&1 || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/hf_cache}"

cd /root/autodl-tmp/GLIGEN

mkdir -p "$HUGGINGFACE_HUB_CACHE"

export MODEL_YAML="${MODEL_YAML:-configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml}"
export DATA_YAML="${DATA_YAML:-configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml}"
export GROUNDING_CKPT="${GROUNDING_CKPT:?Set GROUNDING_CKPT to the clean fixed-split checkpoint path}"
export H5_PATH="${H5_PATH:-/root/autodl-tmp/fixed_split_work/datasets/vg/test.h5}"
export VOCAB_PATH="${VOCAB_PATH:-/root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json}"
export IMAGE_ROOT="${IMAGE_ROOT:-/root/autodl-tmp/fixed_split_work/datasets/vg/images}"
export OUT_DIR="${OUT_DIR:-/root/autodl-tmp/GLIGEN/eval_outputs/vg_fixedsplit_geopred_clean_ddim50_full_b4}"
export NUM_SAMPLES="${NUM_SAMPLES:-5096}"
export START_INDEX="${START_INDEX:-0}"
export SPLIT_NAME="${SPLIT_NAME:-test}"
export SAMPLER="${SAMPLER:-ddim}"
export STEPS="${STEPS:-50}"
export GUIDANCE="${GUIDANCE:-5.0}"
export GROUNDING_ALPHA_TYPE="${GROUNDING_ALPHA_TYPE:-1,0,0}"
export SAVE_SIZE="${SAVE_SIZE:-256}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export SEED="${SEED:-20260508}"
export EVAL_TRANSFORM_MODE="${EVAL_TRANSFORM_MODE:-gligen_center_crop}"

exec /root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py
