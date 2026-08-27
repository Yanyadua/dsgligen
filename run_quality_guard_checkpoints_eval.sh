#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

TRAIN_ROOT="${TRAIN_ROOT:-/root/autodl-tmp/GLIGEN/OUTPUT_FIXED_QUALITY_GUARD/vg_fixedsplit_quality_guard_300_20260620/tag00}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/GLIGEN/eval_outputs/quality_guard_300_checkpoints_20260620}"

export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export BASE_CKPT=/root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin
export MODEL_YAML=configs/vg_fixedsplit_scene_graph_quality_guard.yaml
export DATA_YAML=configs/vg_fixedsplit_scene_graph_quality_guard.yaml
export H5_PATH=/root/autodl-tmp/fixed_split_work/datasets/vg/test.h5
export VOCAB_PATH=/root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json
export IMAGE_ROOT=/root/autodl-tmp/fixed_split_work/datasets/vg/images
export SAMPLE_INDICES=0,4,5,1
export SPLIT_NAME=test
export SAMPLER=ddim
export STEPS=50
export GUIDANCE=5.0
export SAVE_SIZE=256
export EVAL_BATCH_SIZE=4
export SEED=20260429
export EVAL_TRANSFORM_MODE=gligen_center_crop
export RESTORE_BASE_FUSER=0

run_checkpoint() {
  local checkpoint="$1"
  local expected_iters="$2"
  local label="$3"

  export GROUNDING_CKPT="$checkpoint"
  /root/miniconda3/bin/python scripts/eval/audit_historical_vg_run.py \
    --config "$MODEL_YAML" \
    --train-h5 /root/autodl-tmp/fixed_split_work/datasets/vg/train.h5 \
    --test-h5 "$H5_PATH" \
    --vocab "$VOCAB_PATH" \
    --base-checkpoint "$BASE_CKPT" \
    --grounding-checkpoint "$GROUNDING_CKPT" \
    --expected-iters "$expected_iters" \
    --expected-freeze-fuser false

  export GRAPH_GATE_OVERRIDE=0
  export OUT_DIR="$OUT_ROOT/${label}_graph_off"
  /root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py

  unset GRAPH_GATE_OVERRIDE
  export OUT_DIR="$OUT_ROOT/${label}_graph_on"
  /root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py
}

run_checkpoint "$TRAIN_ROOT/checkpoint_00000101.pth" 101 step101
run_checkpoint "$TRAIN_ROOT/checkpoint_00000201.pth" 201 step201
run_checkpoint "$TRAIN_ROOT/checkpoint_00000300.pth" 300 step300
