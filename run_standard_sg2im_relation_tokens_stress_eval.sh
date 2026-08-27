#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

GROUNDING_CKPT="${GROUNDING_CKPT:?Set GROUNDING_CKPT to a compact-style checkpoint}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/GLIGEN/eval_outputs/stress_relation_tokens_ddim50}"
GUIDANCE="${GUIDANCE:-3.0}"

export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export BASE_CKPT="${BASE_CKPT:-/root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin}"
export MODEL_YAML="${MODEL_YAML:-configs/vg_standard_sg2im_scene_graph_relation_tokens.yaml}"
export DATA_YAML="${DATA_YAML:-configs/vg_standard_sg2im_scene_graph_relation_tokens.yaml}"
export GROUNDING_CKPT
export H5_PATH="${H5_PATH:-/root/autodl-tmp/fixed_split_work/datasets/vg/test.h5}"
export VOCAB_PATH="${VOCAB_PATH:-/root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json}"
export IMAGE_ROOT="${IMAGE_ROOT:-/root/autodl-tmp/fixed_split_work/datasets/vg/images}"
export OUT_DIR
export SAMPLE_INDICES="${SAMPLE_INDICES:-6,14,15,23,77,87,144,177,28,42,53,4,8,9,13,56,59,174}"
export NUM_SAMPLES="${NUM_SAMPLES:-18}"
export START_INDEX="${START_INDEX:-0}"
export SAMPLER="${SAMPLER:-ddim}"
export STEPS="${STEPS:-50}"
export GUIDANCE
export GROUNDING_ALPHA_TYPE="${GROUNDING_ALPHA_TYPE:-1,0,0}"
export SEED="${SEED:-20260704}"
export SAVE_SIZE="${SAVE_SIZE:-256}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export SPLIT_NAME="${SPLIT_NAME:-test}"
export EVAL_TRANSFORM_MODE="${EVAL_TRANSFORM_MODE:-gligen_center_crop}"
export MAX_EVAL_OBJECTS="${MAX_EVAL_OBJECTS:-10}"
export MAX_EVAL_RELATIONS="${MAX_EVAL_RELATIONS:-15}"
export EVAL_SELECTION_POLICY="${EVAL_SELECTION_POLICY:-sg2im_relation_area}"
export SAVE_SAMPLE_METADATA="${SAVE_SAMPLE_METADATA:-1}"
export ENABLE_RELATION_GROUNDING_TOKENS="${ENABLE_RELATION_GROUNDING_TOKENS:-1}"
export MAX_RELATION_GROUNDING_TOKENS="${MAX_RELATION_GROUNDING_TOKENS:-5}"
if [ -z "${RELATION_GROUNDING_TEMPLATE:-}" ]; then
  export RELATION_GROUNDING_TEMPLATE='{subject} {predicate} {object}'
else
  export RELATION_GROUNDING_TEMPLATE
fi

exec /root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py
