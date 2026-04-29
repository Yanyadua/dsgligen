#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/GLIGEN}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
YAML_FILE="${YAML_FILE:-$ROOT/configs/vg_fixedsplit_scene_graph_sggig_style.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/GLIGEN/OUTPUT_SGGIG}"
EXP_NAME="${EXP_NAME:-vg_fixedsplit_scene_graph_sggig_style_300}"
TOTAL_ITERS="${TOTAL_ITERS:-300}"
GROUNDING_CKPT="${GROUNDING_CKPT:-/root/autodl-tmp/GLIGEN/OUTPUT_SGGIG/vg_fixedsplit_scene_graph_sggig_stage_a_300/tag00/checkpoint_latest.pth}"
GLIGEN_CKPT="${GLIGEN_CKPT:-/root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin}"

cd "$ROOT"

"$PYTHON_BIN" main.py \
  --yaml_file "$YAML_FILE" \
  --name "$EXP_NAME" \
  --OUTPUT_ROOT "$OUTPUT_ROOT" \
  --total_iters "$TOTAL_ITERS" \
  --grounding_ckpt "$GROUNDING_CKPT" \
  --init_from_gligen_ckpt "$GLIGEN_CKPT" \
  --freeze_fuser True \
  --freeze_position_base True \
  --save_trainable_only True \
  --disable_inference_in_training True
