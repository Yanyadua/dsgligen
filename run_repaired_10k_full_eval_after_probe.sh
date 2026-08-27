#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

TRAIN_PID="${TRAIN_PID:-8756}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/GLIGEN/OUTPUT_FIXED_CLEAN_GEOPRED_REPAIRED/vg_fixedsplit_linears_boxfix_clean_seed123_20260610/tag01/checkpoint_00010000.pth}"
PROBE_DIR="${PROBE_DIR:-/root/autodl-tmp/GLIGEN/eval_outputs/vg_fixedsplit_linears_boxfix_clean_10k_ddim50_seed123}"
FULL_DIR="${FULL_DIR:-/root/autodl-tmp/GLIGEN/eval_outputs/vg_fixedsplit_linears_boxfix_clean_10k_ddim50_full5096_seed20260429}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-10800}"

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for training PID $TRAIN_PID." >&2
    exit 2
  fi
  sleep 60
done

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Expected final checkpoint does not exist: $CHECKPOINT" >&2
  exit 3
fi

/root/miniconda3/bin/python scripts/eval/audit_historical_vg_run.py \
  --config configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml \
  --train-h5 /root/autodl-tmp/fixed_split_work/datasets/vg/train.h5 \
  --test-h5 /root/autodl-tmp/fixed_split_work/datasets/vg/test.h5 \
  --vocab /root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json \
  --base-checkpoint /root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin \
  --grounding-checkpoint "$CHECKPOINT" \
  --expected-iters 10000

# The independent post-training guard creates this probe first. Do not launch the
# expensive full split until all probe files and its pinned protocol metadata exist.
deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while true; do
  fake_count=0
  real_count=0
  if [[ -d "$PROBE_DIR/fake" ]]; then
    fake_count="$(find "$PROBE_DIR/fake" -maxdepth 1 -type f -name '*.png' | wc -l)"
  fi
  if [[ -d "$PROBE_DIR/real" ]]; then
    real_count="$(find "$PROBE_DIR/real" -maxdepth 1 -type f -name '*.png' | wc -l)"
  fi
  if [[ "$fake_count" -eq 8 && "$real_count" -eq 8 && -f "$PROBE_DIR/meta.txt" ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for the audited 8-image probe." >&2
    exit 4
  fi
  sleep 30
done

grep -Fx "SAMPLER=ddim" "$PROBE_DIR/meta.txt"
grep -Fx "STEPS=50" "$PROBE_DIR/meta.txt"
grep -Fx "GUIDANCE=5.0" "$PROBE_DIR/meta.txt"
grep -Fx "SEED=20260429" "$PROBE_DIR/meta.txt"
grep -Fx "NUM_SAMPLES=8" "$PROBE_DIR/meta.txt"

GROUNDING_CKPT="$CHECKPOINT" \
EXPECTED_ITERS=10000 \
NUM_SAMPLES=5096 \
START_INDEX=0 \
EVAL_BATCH_SIZE=4 \
OUT_DIR="$FULL_DIR" \
bash run_fixedsplit_geopred_repaired_ddim50_eval.sh

fake_count="$(find "$FULL_DIR/fake" -maxdepth 1 -type f -name '*.png' | wc -l)"
real_count="$(find "$FULL_DIR/real" -maxdepth 1 -type f -name '*.png' | wc -l)"
if [[ "$fake_count" -ne 5096 || "$real_count" -ne 5096 ]]; then
  echo "Full evaluation count mismatch: real=$real_count fake=$fake_count" >&2
  exit 5
fi

grep -Fx "SAMPLER=ddim" "$FULL_DIR/meta.txt"
grep -Fx "STEPS=50" "$FULL_DIR/meta.txt"
grep -Fx "GUIDANCE=5.0" "$FULL_DIR/meta.txt"
grep -Fx "SEED=20260429" "$FULL_DIR/meta.txt"
grep -Fx "NUM_SAMPLES=5096" "$FULL_DIR/meta.txt"

EVAL_DIR="$FULL_DIR" bash run_fixedsplit_geopred_clean_metrics_historical.sh
