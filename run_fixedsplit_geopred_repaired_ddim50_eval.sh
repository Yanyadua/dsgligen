#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

: "${GROUNDING_CKPT:?Set GROUNDING_CKPT to the repaired checkpoint path}"
EXPECTED_ITERS="${EXPECTED_ITERS:-1000}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
START_INDEX="${START_INDEX:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/GLIGEN/eval_outputs/vg_fixedsplit_linears_boxfix_probe}"

export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MODEL_YAML=configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml
export DATA_YAML=configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml
export GROUNDING_CKPT
export H5_PATH=/root/autodl-tmp/fixed_split_work/datasets/vg/test.h5
export VOCAB_PATH=/root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json
export IMAGE_ROOT=/root/autodl-tmp/fixed_split_work/datasets/vg/images
export OUT_DIR
export NUM_SAMPLES
export START_INDEX
export SPLIT_NAME=test
export SAMPLER=ddim
export STEPS=50
export GUIDANCE=5.0
export SAVE_SIZE=256
export EVAL_BATCH_SIZE
export SEED=20260429

/root/miniconda3/bin/python scripts/eval/audit_historical_vg_run.py \
  --config "$MODEL_YAML" \
  --train-h5 /root/autodl-tmp/fixed_split_work/datasets/vg/train.h5 \
  --test-h5 "$H5_PATH" \
  --vocab "$VOCAB_PATH" \
  --base-checkpoint /root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin \
  --grounding-checkpoint "$GROUNDING_CKPT" \
  --expected-iters "$EXPECTED_ITERS"

exec /root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py
