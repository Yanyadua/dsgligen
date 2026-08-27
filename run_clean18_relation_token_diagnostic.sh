#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/GLIGEN

export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export BASE_CKPT=/root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin
export H5_PATH=/root/autodl-tmp/standard_sg2im_fresh_h5/test.h5
export VOCAB_PATH=/root/autodl-tmp/standard_sg2im_fresh_h5/vocab.json
export IMAGE_ROOT=/root/autodl-tmp/standard_sg2im_fresh_h5/images
export SAMPLE_INDICES=1,5,14,18,31,36,44,53,56,59,61,62,75,78,124,160,174,185
export NUM_SAMPLES=18
export START_INDEX=0
export SAMPLER=ddim
export STEPS=50
export GUIDANCE=3.0
export GROUNDING_ALPHA_TYPE=1,0,0
export SEED=20260704
export SAVE_SIZE=256
export EVAL_BATCH_SIZE=4
export SPLIT_NAME=test
export EVAL_TRANSFORM_MODE=gligen_center_crop
export MAX_EVAL_OBJECTS=10
export MAX_EVAL_RELATIONS=15
export EVAL_SELECTION_POLICY=sg2im_relation_area
export SAVE_SAMPLE_METADATA=1
export RESTORE_BASE_FUSER=0
export CAPTION_POLICY=graph
export ENABLE_RELATION_GROUNDING_TOKENS=1
export MAX_RELATION_GROUNDING_TOKENS=3
export RELATION_GROUNDING_ALLOWED_PREDICATES="on,on top of,under,below,above,inside,in front of,behind,holding,riding,wearing,sitting on,standing on,walking on,carrying"
export DEDUP_RELATION_GROUNDING_TOKENS=1
export RELATION_GROUNDING_TEMPLATE="spatial relation: {subject} {predicate} {object}"

COMPACT_CKPT=/root/autodl-tmp/GLIGEN/OUTPUT_STANDARD_SG2IM_COMPACT_STYLE_GATE/vg_standard_sg2im_compact_style_gate_1k_20260627/tag00/checkpoint_latest.pth

run_one() {
  local name="$1"
  local model_yaml="$2"
  local grounding_ckpt="$3"
  local graph_gate="$4"

  export MODEL_YAML="$model_yaml"
  export DATA_YAML="$model_yaml"
  export OUT_DIR="/root/autodl-tmp/GLIGEN/eval_outputs/${name}"

  if [[ "$grounding_ckpt" == "none" ]]; then
    unset GROUNDING_CKPT || true
  else
    export GROUNDING_CKPT="$grounding_ckpt"
  fi

  if [[ "$graph_gate" == "default" ]]; then
    unset GRAPH_GATE_OVERRIDE || true
  else
    export GRAPH_GATE_OVERRIDE="$graph_gate"
  fi

  rm -rf "$OUT_DIR"
  echo "===== START $name $(date +%F\ %T) ====="
  /root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py
  echo "===== DONE $name $(date +%F\ %T) ====="
}

run_one \
  clean18_diag_E_official_relation_tokens_ddim50_20260708 \
  configs/vg_text_box_baseline.yaml \
  none \
  default

run_one \
  clean18_diag_F_compact_relation_tokens_ddim50_20260708 \
  configs/vg_standard_sg2im_scene_graph_relation_tokens_v2.yaml \
  "$COMPACT_CKPT" \
  default
