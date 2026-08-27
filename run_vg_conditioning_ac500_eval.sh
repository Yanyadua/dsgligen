#!/usr/bin/env bash
set -euo pipefail

# Exploratory 500-image A/C protocol for testing whether compact VG scene-graph
# conditioning scales beyond the 100-image diagnostic.
#
# Status: exploratory, not paper-ready by itself. It uses first 500 samples of
# the fixed SG2I/VG test split and keeps all generation knobs identical between
# the two groups.
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

run_one() {
  local name="$1"
  local out_dir="/root/autodl-tmp/GLIGEN/eval_outputs/${name}"
  if [[ -e "${out_dir}" ]]; then
    echo "Refusing to overwrite existing output: ${name}" >&2
    exit 2
  fi
  mkdir -p "${out_dir}"
  cat > "${out_dir}/meta.txt" <<META
status=exploratory_first500
split=test
h5_path=${H5_PATH}
vocab_path=${VOCAB_PATH}
image_root=${IMAGE_ROOT}
base_ckpt=${BASE_CKPT}
num_samples=${NUM_SAMPLES}
start_index=${START_INDEX}
sampler=${SAMPLER}
steps=${STEPS}
guidance=${GUIDANCE}
batch_size=${EVAL_BATCH_SIZE}
seed=${SEED}
conditioning_policy=${CONDITIONING_POLICY}
caption_policy=${CAPTION_POLICY}
relation_grounding_tokens=${ENABLE_RELATION_GROUNDING_TOKENS}
META
  export OUT_DIR="${out_dir}"
  echo "===== START ${name} $(date '+%F %T') ====="
  /root/miniconda3/bin/python scripts/eval/generate_vg_fixedsplit_eval.py
  echo "===== DONE ${name} $(date '+%F %T') ====="
}

# A: official GLIGEN-VG legacy baseline.
export CONDITIONING_POLICY=legacy
export CAPTION_POLICY=graph
unset CAPTION_STYLE_PREFIX || true
unset CAPTION_STYLE_SUFFIX || true
run_one cond_ac500_a_legacy_graph_20260728

# C: compact scene-graph conditioning, no relation token.
export CONDITIONING_POLICY=clean_spatial_v2
export CAPTION_POLICY=clean_primary
export CLEAN_MAX_OBJECTS=8
export CLEAN_MAX_RELATIONS=2
export CLEAN_MIN_BOX_AREA=0.0025
export CLEAN_MIN_BOX_SIDE=0.035
export CLEAN_RELATION_CORE_MIN_AREA=0.0015
export CLEAN_DUPLICATE_IOU_THRESHOLD=0.85
export CLEAN_RELATION_PREDICATES='on,on top of,under,below,above,inside,in,near,next to,in front of,behind'
run_one cond_ac500_c_v2_no_relation_token_20260728
