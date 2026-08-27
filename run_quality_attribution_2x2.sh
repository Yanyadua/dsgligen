#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

ROOT="${ROOT:-/root/autodl-tmp/GLIGEN/eval_outputs/quality_attribution_2x2_20260619}"
CKPT="${CKPT:-/root/autodl-tmp/GLIGEN/OUTPUT_FIXED_GRAPHDISTILL_FUSER/vg_fixedsplit_graphdistill_fuser_1k_20260619/tag00/checkpoint_00001000.pth}"

export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export BASE_CKPT=/root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin
export MODEL_YAML=configs/vg_fixedsplit_scene_graph_graphdistill_fuser.yaml
export DATA_YAML=configs/vg_fixedsplit_scene_graph_graphdistill_fuser.yaml
export GROUNDING_CKPT="$CKPT"
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

/root/miniconda3/bin/python scripts/eval/audit_historical_vg_run.py \
  --config "$MODEL_YAML" \
  --train-h5 /root/autodl-tmp/fixed_split_work/datasets/vg/train.h5 \
  --test-h5 "$H5_PATH" \
  --vocab "$VOCAB_PATH" \
  --base-checkpoint "$BASE_CKPT" \
  --grounding-checkpoint "$GROUNDING_CKPT" \
  --expected-iters 1000 \
  --expected-freeze-fuser false

run_variant() {
  local name="$1"
  local restore_fuser="$2"
  local graph_gate="$3"
  export OUT_DIR="$ROOT/$name"
  export RESTORE_BASE_FUSER="$restore_fuser"
  if [[ "$graph_gate" == "default" ]]; then
    unset GRAPH_GATE_OVERRIDE
  else
    export GRAPH_GATE_OVERRIDE="$graph_gate"
  fi
  /root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py
}

run_variant f0g0 1 0
run_variant f0g1 1 default
run_variant f1g0 0 0
run_variant f1g1 0 default
