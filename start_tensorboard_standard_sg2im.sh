#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-6006}"
LOGDIR="${2:-/root/autodl-tmp/GLIGEN/OUTPUT_STANDARD_SG2IM_GEOPRED}"

echo "Starting TensorBoard"
echo "  port:   ${PORT}"
echo "  logdir: ${LOGDIR}"

cd /root/autodl-tmp/GLIGEN
nohup /root/miniconda3/bin/python -m tensorboard.main \
  --logdir "${LOGDIR}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --load_fast=false \
  > /root/autodl-tmp/GLIGEN/run_logs/tensorboard_${PORT}.log 2>&1 < /dev/null &

echo $!
