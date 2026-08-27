#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

EVAL_DIR="${EVAL_DIR:-/root/autodl-tmp/GLIGEN/eval_outputs/vg_fixedsplit_geopred_clean_10k_full_b4}"

exec /root/miniconda3/bin/python scripts/eval/compute_vg_fid_is.py \
  --real "$EVAL_DIR/real" \
  --fake "$EVAL_DIR/fake" \
  --expected-count 5096 \
  --backend both \
  --batch-size 50 \
  --device cuda:0 \
  --output "$EVAL_DIR/fid_is_metrics.json"
