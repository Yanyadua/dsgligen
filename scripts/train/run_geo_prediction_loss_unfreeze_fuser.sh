#!/bin/sh
set -eu

ROOT="${1:-/root/autodl-tmp/GLIGEN}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
YAML_FILE="${YAML_FILE:-$ROOT/configs/vg_raw_scene_graph_compatible_spatial_gat_geo_prediction_loss.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/OUTPUT_RAW_VG_GEOPRED_FUSER}"
EXP_NAME="${EXP_NAME:-vg_raw_scene_graph_geo_prediction_loss_fuser_open_300}"
TOTAL_ITERS="${TOTAL_ITERS:-300}"

cd "$ROOT"

exec "$PYTHON_BIN" main.py \
  --yaml_file "$YAML_FILE" \
  --name "$EXP_NAME" \
  --OUTPUT_ROOT "$OUTPUT_ROOT" \
  --init_from_gligen_ckpt "$ROOT/gligen_checkpoints/diffusion_pytorch_model.bin" \
  --total_iters "$TOTAL_ITERS" \
  --freeze_fuser False \
  --freeze_position_base True \
  --save_trainable_only True \
  --disable_inference_in_training True
