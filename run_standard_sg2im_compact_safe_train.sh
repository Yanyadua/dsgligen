#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

EXPERIMENT_NAME="${EXPERIMENT_NAME:-vg_standard_sg2im_compact_safe_1k_20260627}"
TOTAL_ITERS="${TOTAL_ITERS:-1000}"
SAVE_EVERY_ITERS="${SAVE_EVERY_ITERS:-500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/GLIGEN/OUTPUT_STANDARD_SG2IM_COMPACT_SAFE}"
YAML_FILE="${YAML_FILE:-configs/vg_standard_sg2im_scene_graph_compact_safe.yaml}"
EXPERIMENT_ROOT="$OUTPUT_ROOT/$EXPERIMENT_NAME"
ALLOW_RESUME="${ALLOW_RESUME:-0}"

if find "$EXPERIMENT_ROOT" -name checkpoint_latest.pth -print -quit 2>/dev/null | grep -q .; then
  if [[ "$ALLOW_RESUME" != "1" ]]; then
    echo "Existing checkpoint found in $EXPERIMENT_ROOT. Set ALLOW_RESUME=1 to resume."
    exit 2
  fi
fi

mkdir -p "$OUTPUT_ROOT"
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

exec /root/miniconda3/bin/python main.py \
  --yaml_file "$YAML_FILE" \
  --DATA_ROOT /root/autodl-tmp \
  --OUTPUT_ROOT "$OUTPUT_ROOT" \
  --name "$EXPERIMENT_NAME" \
  --seed 123 \
  --batch_size 2 \
  --workers 0 \
  --total_iters "$TOTAL_ITERS" \
  --save_every_iters "$SAVE_EVERY_ITERS" \
  --disable_inference_in_training true \
  --save_trainable_only true \
  --freeze_fuser true \
  --fuser_train_mode frozen \
  --freeze_position_base true \
  --graph_gate_lr_multiplier 1.0 \
  --init_from_gligen_ckpt /root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin
