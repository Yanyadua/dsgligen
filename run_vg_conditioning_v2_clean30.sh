#!/usr/bin/env bash
set -euo pipefail

# Exploratory diagnostic only: fixed VG test examples, no FID/IS claims.
# The arms isolate whether failures come from caption wording, noisy VG boxes,
# or relation-token injection. It is intentionally short and reproducible.
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
export SAMPLE_INDICES=284,388,513,798,919,1008,1048,1277,1760,1859,1897,1940,1978,2022,2194,2295,2313,2446,2591,2855,2942,3360,3530,3544,3651,3787,4545,4742,4786,5000
export NUM_SAMPLES=30
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

run_one() {
  local name="$1"
  if [[ -e "/root/autodl-tmp/GLIGEN/eval_outputs/${name}" ]]; then
    echo "Refusing to overwrite existing output: ${name}" >&2
    exit 2
  fi
  export OUT_DIR="/root/autodl-tmp/GLIGEN/eval_outputs/${name}"
  echo "===== START ${name} $(date '+%F %T') ====="
  /root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py
  echo "===== DONE ${name} $(date '+%F %T') ====="
}

# A: old VG-to-GLIGEN condition. This exposes the original object-list caption.
export CONDITIONING_POLICY=legacy
export CAPTION_POLICY=graph
unset CAPTION_STYLE_PREFIX || true
unset CAPTION_STYLE_SUFFIX || true
export ENABLE_RELATION_GROUNDING_TOKENS=0
export MAX_RELATION_GROUNDING_TOKENS=0
run_one condv2_clean30_a_legacy_graph_20260728

# B: caption-only intervention. Boxes stay legacy; wording becomes photo-like.
export CONDITIONING_POLICY=legacy
export CAPTION_POLICY=clean_primary
export ENABLE_RELATION_GROUNDING_TOKENS=0
export MAX_RELATION_GROUNDING_TOKENS=0
run_one condv2_clean30_b_caption_only_20260728

# C: caption plus clean_spatial_v2 compact object/relation subgraph.
export CONDITIONING_POLICY=clean_spatial_v2
export CAPTION_POLICY=clean_primary
export CLEAN_MAX_OBJECTS=8
export CLEAN_MAX_RELATIONS=2
export CLEAN_MIN_BOX_AREA=0.0025
export CLEAN_MIN_BOX_SIDE=0.035
export CLEAN_RELATION_CORE_MIN_AREA=0.0015
export CLEAN_DUPLICATE_IOU_THRESHOLD=0.85
export CLEAN_RELATION_PREDICATES='on,on top of,under,below,above,inside,in,near,next to,in front of,behind'
export ENABLE_RELATION_GROUNDING_TOKENS=0
export MAX_RELATION_GROUNDING_TOKENS=0
run_one condv2_clean30_c_v2_no_relation_token_20260728

# D: v2 condition plus one conservative relation grounding token.
export ENABLE_RELATION_GROUNDING_TOKENS=1
export MAX_RELATION_GROUNDING_TOKENS=1
export RELATION_GROUNDING_TEMPLATE='{subject} {predicate} {object}'
export RELATION_GROUNDING_ALLOWED_PREDICATES="$CLEAN_RELATION_PREDICATES"
export DEDUP_RELATION_GROUNDING_TOKENS=1
export RELATION_GROUNDING_MASK_SCALE=0.5
run_one condv2_clean30_d_v2_top1_relation_s050_20260728
