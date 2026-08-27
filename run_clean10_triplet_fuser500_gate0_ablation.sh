#!/usr/bin/env bash
set -euo pipefail

# Diagnostic only: reopen the learned triplet residual at inference. This is
# not a trained checkpoint and must never be compared as a main result.
cd /root/autodl-tmp/GLIGEN
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export BASE_CKPT=/root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin
export MODEL_YAML=configs/vg_standard_sg2im_triplet_fuser_clean_v1.yaml
export DATA_YAML=configs/vg_standard_sg2im_triplet_fuser_clean_v1.yaml
export GROUNDING_CKPT=/root/autodl-tmp/GLIGEN/OUTPUT_STANDARD_SG2IM_TRIPLET_FUSER/vg_standard_sg2im_triplet_fuser_500step_20260713/tag00/checkpoint_00000500.pth
export TRIPLET_GATE_OVERRIDE=0.0
export H5_PATH=/root/autodl-tmp/standard_sg2im_fresh_h5/test.h5
export VOCAB_PATH=/root/autodl-tmp/standard_sg2im_fresh_h5/vocab.json
export IMAGE_ROOT=/root/autodl-tmp/standard_sg2im_fresh_h5/images
export SPLIT_NAME=test
export SAMPLE_INDICES=1008,1048,1978,2022,2942,3530,3544,3651,4786,5000
export SAMPLER=ddim
export STEPS=50
export GUIDANCE=3.0
export GROUNDING_ALPHA_TYPE=1,0,0
export SEED=20260713
export SAVE_SIZE=256
export EVAL_BATCH_SIZE=4
export EVAL_TRANSFORM_MODE=gligen_center_crop
export MAX_EVAL_OBJECTS=10
export MAX_EVAL_RELATIONS=15
export EVAL_SELECTION_POLICY=sg2im_relation_area
export SAVE_SAMPLE_METADATA=1
export RESTORE_BASE_FUSER=0
export CONDITIONING_POLICY=clean_spatial_v1
export CAPTION_POLICY=clean
export CLEAN_MAX_OBJECTS=6
export CLEAN_MAX_RELATIONS=1
export CLEAN_MIN_BOX_AREA=0.0025
export CLEAN_MIN_BOX_SIDE=0.035
export CLEAN_RELATION_CORE_MIN_AREA=0.0015
export CLEAN_DUPLICATE_IOU_THRESHOLD=0.85
export ENABLE_RELATION_GROUNDING_TOKENS=0
export MAX_RELATION_GROUNDING_TOKENS=0
export OUT_DIR="${OUT_DIR:-/root/autodl-tmp/GLIGEN/eval_outputs/clean10_triplet_fuser500_gate0_ablation_20260713}"

if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite existing output: $OUT_DIR" >&2
  exit 2
fi
/root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py
