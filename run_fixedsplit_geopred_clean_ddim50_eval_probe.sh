#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MODEL_YAML="${MODEL_YAML:-configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml}"
export DATA_YAML="${DATA_YAML:-configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml}"
export GROUNDING_CKPT="${GROUNDING_CKPT:-/root/autodl-tmp/GLIGEN/OUTPUT_FIXED_CLEAN_GEOPRED/vg_fixedsplit_geopred_clean_full_1k_restoreprobe_20260607/tag00/checkpoint_00001000.pth}"
export H5_PATH="${H5_PATH:-/root/autodl-tmp/fixed_split_work/datasets/vg/test.h5}"
export VOCAB_PATH="${VOCAB_PATH:-/root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json}"
export IMAGE_ROOT="${IMAGE_ROOT:-/root/autodl-tmp/fixed_split_work/datasets/vg/images}"
export OUT_DIR="${OUT_DIR:-/root/autodl-tmp/GLIGEN/eval_outputs/vg_fixedsplit_geopred_clean_1k_restoreprobe_ddim50_b1}"
export NUM_SAMPLES="${NUM_SAMPLES:-1}"
export START_INDEX="${START_INDEX:-3208}"
export SPLIT_NAME="${SPLIT_NAME:-test}"
export SAMPLER="${SAMPLER:-ddim}"
export STEPS="${STEPS:-50}"
export GUIDANCE="${GUIDANCE:-5.0}"
export SAVE_SIZE="${SAVE_SIZE:-256}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export SEED="${SEED:-20260429}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

exec /root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py
