#!/usr/bin/env bash
set -euo pipefail

# Controlled VG fixed-train run. Defaults to one optimizer step; increase only
# after its checkpoint manifest and log have been inspected.
cd /root/autodl-tmp/GLIGEN

EXPERIMENT_NAME="${EXPERIMENT_NAME:-vg_standard_sg2im_triplet_fuser_smoke_20260713}"
TOTAL_ITERS="${TOTAL_ITERS:-1}"
SAVE_EVERY_ITERS="${SAVE_EVERY_ITERS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/GLIGEN/OUTPUT_STANDARD_SG2IM_TRIPLET_FUSER}"
EXPERIMENT_ROOT="$OUTPUT_ROOT/$EXPERIMENT_NAME"

if [[ -e "$EXPERIMENT_ROOT" ]]; then
  echo "Refusing existing experiment directory: $EXPERIMENT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

exec /root/miniconda3/bin/python main.py \
  --yaml_file configs/vg_standard_sg2im_triplet_fuser_clean_v1.yaml \
  --DATA_ROOT /root/autodl-tmp \
  --OUTPUT_ROOT "$OUTPUT_ROOT" \
  --name "$EXPERIMENT_NAME" \
  --seed 20260713 \
  --batch_size 2 \
  --workers 0 \
  --total_iters "$TOTAL_ITERS" \
  --save_every_iters "$SAVE_EVERY_ITERS" \
  --base_learning_rate 5e-5 \
  --graph_lr_multiplier 5.0 \
  --disable_inference_in_training true \
  --save_trainable_only true \
  --freeze_fuser true \
  --fuser_train_mode frozen \
  --freeze_position_base true \
  --init_from_gligen_ckpt /root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin
