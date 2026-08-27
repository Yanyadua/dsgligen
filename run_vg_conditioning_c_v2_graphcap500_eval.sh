#!/usr/bin/env bash
set -euo pipefail

# Exploratory 500-image isolation run:
# keep clean_spatial_v2 compact grounding, but restore the legacy graph caption.
# This tests whether the IS drop in C_v2 was mainly caused by clean_primary
# caption rewriting rather than object/relation filtering itself.
cd /root/autodl-tmp/GLIGEN

export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export BASE_CKPT=/root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin
export MODEL_YAML=configs/vg_text_box_baseline.yaml
export DATA_YAML=configs/vg_text_box_baseline.yaml
unset GROUNDING_CKPT || true
unset GRAPH_GATE_OVERRIDE || true
unset TRIPLET_GATE_OVERRIDE || true

export H5_PATH=/root/autodl-tmp/standard_sg2im_fresh_h5/test.h5
export VOCAB_PATH=/root/autodl-tmp/standard_sg2im_fresh_h5/vocab.json
export IMAGE_ROOT=/root/autodl-tmp/standard_sg2im_fresh_h5/images
export SPLIT_NAME=test
unset SAMPLE_INDICES || true
export NUM_SAMPLES=500
export START_INDEX=0
export SAMPLER=ddim
export STEPS=50
export GUIDANCE=3.0
export GROUNDING_ALPHA_TYPE=1,0,0
export SEED=20260728
export SAVE_SIZE=256
export EVAL_BATCH_SIZE=4
export EVAL_TRANSFORM_MODE=gligen_center_crop
export MAX_EVAL_OBJECTS=10
export MAX_EVAL_RELATIONS=15
export EVAL_SELECTION_POLICY=sg2im_relation_area
export SAVE_SAMPLE_METADATA=1
export RESTORE_BASE_FUSER=0
export MAX_CAPTION_OBJECTS=4
export MAX_CAPTION_RELATIONS=2
export ENABLE_RELATION_GROUNDING_TOKENS=0
export MAX_RELATION_GROUNDING_TOKENS=0

export OUT_DIR=/root/autodl-tmp/GLIGEN/eval_outputs/cond_ac500_c_v2_graph_caption_20260728
if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT_DIR}" >&2
  exit 2
fi

export CONDITIONING_POLICY=clean_spatial_v2
export CAPTION_POLICY=graph
unset CAPTION_STYLE_PREFIX || true
unset CAPTION_STYLE_SUFFIX || true
export CLEAN_MAX_OBJECTS=8
export CLEAN_MAX_RELATIONS=2
export CLEAN_MIN_BOX_AREA=0.0025
export CLEAN_MIN_BOX_SIDE=0.035
export CLEAN_RELATION_CORE_MIN_AREA=0.0015
export CLEAN_DUPLICATE_IOU_THRESHOLD=0.85
export CLEAN_RELATION_PREDICATES='on,on top of,under,below,above,inside,in,near,next to,in front of,behind'

echo "===== START cond_ac500_c_v2_graph_caption_20260728 $(date '+%F %T') ====="
/root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py
echo "===== DONE cond_ac500_c_v2_graph_caption_20260728 $(date '+%F %T') ====="
