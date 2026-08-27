#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

ROOT="${ROOT:-/root/autodl-tmp/GLIGEN/eval_outputs/normmatch_fair_compare_20260620}"
CKPT="${CKPT:-/root/autodl-tmp/GLIGEN/OUTPUT_FIXED_NORMMATCH/vg_fixedsplit_normmatch_300_20260620/tag00/checkpoint_00000300.pth}"

export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export BASE_CKPT=/root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin
export DATA_YAML=configs/vg_fixedsplit_scene_graph_normmatch.yaml
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

unset GROUNDING_CKPT GRAPH_GATE_OVERRIDE
export MODEL_YAML=configs/vg_text_box_baseline.yaml
export OUT_DIR="$ROOT/official"
/root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py

export GROUNDING_CKPT="$CKPT"
export MODEL_YAML=configs/vg_fixedsplit_scene_graph_normmatch.yaml
/root/miniconda3/bin/python scripts/eval/audit_historical_vg_run.py \
  --config "$MODEL_YAML" \
  --train-h5 /root/autodl-tmp/fixed_split_work/datasets/vg/train.h5 \
  --test-h5 "$H5_PATH" \
  --vocab "$VOCAB_PATH" \
  --base-checkpoint "$BASE_CKPT" \
  --grounding-checkpoint "$GROUNDING_CKPT" \
  --expected-iters 300 \
  --expected-freeze-fuser true

export GRAPH_GATE_OVERRIDE=0
export OUT_DIR="$ROOT/step300_graph_off"
/root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py

unset GRAPH_GATE_OVERRIDE
export OUT_DIR="$ROOT/step300_graph_on"
/root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py
