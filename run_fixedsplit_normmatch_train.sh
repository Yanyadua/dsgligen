#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

EXPERIMENT_NAME="${EXPERIMENT_NAME:-vg_fixedsplit_normmatch_300_20260620}"
TOTAL_ITERS="${TOTAL_ITERS:-300}"
SAVE_EVERY_ITERS="${SAVE_EVERY_ITERS:-100}"
ALLOW_RESUME="${ALLOW_RESUME:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/GLIGEN/OUTPUT_FIXED_NORMMATCH}"
EXPERIMENT_ROOT="$OUTPUT_ROOT/$EXPERIMENT_NAME"

if find "$EXPERIMENT_ROOT" -name checkpoint_latest.pth -print -quit 2>/dev/null | grep -q .; then
  if [[ "$ALLOW_RESUME" != "1" ]]; then
    echo "Existing checkpoint found in $EXPERIMENT_ROOT."
    exit 2
  fi
fi

mkdir -p "$OUTPUT_ROOT"
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

/root/miniconda3/bin/python scripts/eval/audit_historical_vg_run.py \
  --config configs/vg_fixedsplit_scene_graph_normmatch.yaml \
  --train-h5 /root/autodl-tmp/fixed_split_work/datasets/vg/train.h5 \
  --test-h5 /root/autodl-tmp/fixed_split_work/datasets/vg/test.h5 \
  --vocab /root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json \
  --base-checkpoint /root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin \
  --expected-freeze-fuser true

exec /root/miniconda3/bin/python main.py \
  --yaml_file configs/vg_fixedsplit_scene_graph_normmatch.yaml \
  --DATA_ROOT /root/autodl-tmp \
  --OUTPUT_ROOT "$OUTPUT_ROOT" \
  --name "$EXPERIMENT_NAME" \
  --seed 123 \
  --batch_size 2 \
  --workers 0 \
  --total_iters "$TOTAL_ITERS" \
  --save_every_iters "$SAVE_EVERY_ITERS" \
  --graph_gate_lr_multiplier 1.0 \
  --disable_inference_in_training true \
  --save_trainable_only true \
  --freeze_fuser true \
  --fuser_train_mode frozen \
  --freeze_position_base true \
  --init_from_gligen_ckpt /root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin
