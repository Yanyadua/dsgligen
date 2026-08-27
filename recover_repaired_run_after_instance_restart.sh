#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/GLIGEN

EXPERIMENT_NAME="vg_fixedsplit_linears_boxfix_clean_seed123_20260610"
EXPERIMENT_ROOT="/root/autodl-tmp/GLIGEN/OUTPUT_FIXED_CLEAN_GEOPRED_REPAIRED/$EXPERIMENT_NAME"

wait_for_active_training() {
  while pgrep -f "python main.py.*--name $EXPERIMENT_NAME" >/dev/null; do
    echo "Existing training process is still active; waiting."
    sleep 60
  done
}

find_final_checkpoint() {
  find "$EXPERIMENT_ROOT" -type f -name checkpoint_00010000.pth | sort | tail -n 1
}

wait_for_active_training
final_checkpoint="$(find_final_checkpoint)"

if [[ -z "$final_checkpoint" ]]; then
  echo "No verified 10k checkpoint found; resuming the latest saved tag."
  ALLOW_RESUME=1 \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  TOTAL_ITERS=10000 \
  SAVE_EVERY_ITERS=1000 \
  bash run_fixedsplit_geopred_repaired_train.sh
  final_checkpoint="$(find_final_checkpoint)"
fi

if [[ -z "$final_checkpoint" || ! -f "$final_checkpoint" ]]; then
  echo "Training exited without a 10k checkpoint." >&2
  exit 2
fi

/root/miniconda3/bin/python scripts/eval/audit_historical_vg_run.py \
  --config configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml \
  --train-h5 /root/autodl-tmp/fixed_split_work/datasets/vg/train.h5 \
  --test-h5 /root/autodl-tmp/fixed_split_work/datasets/vg/test.h5 \
  --vocab /root/autodl-tmp/fixed_split_work/datasets/vg/vocab.json \
  --base-checkpoint /root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin \
  --grounding-checkpoint "$final_checkpoint" \
  --expected-iters 10000

checkpoint_hash="$(sha256sum "$final_checkpoint" | awk '{print substr($1,1,12)}')"
probe_dir="/root/autodl-tmp/GLIGEN/eval_outputs/vg_fixedsplit_linears_boxfix_clean_10k_${checkpoint_hash}_ddim50_probe8"
full_dir="/root/autodl-tmp/GLIGEN/eval_outputs/vg_fixedsplit_linears_boxfix_clean_10k_${checkpoint_hash}_ddim50_full5096_seed20260429"

GROUNDING_CKPT="$final_checkpoint" \
EXPECTED_ITERS=10000 \
NUM_SAMPLES=8 \
START_INDEX=0 \
EVAL_BATCH_SIZE=4 \
OUT_DIR="$probe_dir" \
bash run_fixedsplit_geopred_repaired_ddim50_eval.sh

probe_fake_count="$(find "$probe_dir/fake" -maxdepth 1 -type f -name '*.png' | wc -l)"
probe_real_count="$(find "$probe_dir/real" -maxdepth 1 -type f -name '*.png' | wc -l)"
if [[ "$probe_fake_count" -ne 8 || "$probe_real_count" -ne 8 ]]; then
  echo "Probe count mismatch: real=$probe_real_count fake=$probe_fake_count" >&2
  exit 3
fi

grep -Fx "SAMPLER=ddim" "$probe_dir/meta.txt"
grep -Fx "STEPS=50" "$probe_dir/meta.txt"
grep -Fx "GUIDANCE=5.0" "$probe_dir/meta.txt"
grep -Fx "SEED=20260429" "$probe_dir/meta.txt"
grep -Fx "NUM_SAMPLES=8" "$probe_dir/meta.txt"

GROUNDING_CKPT="$final_checkpoint" \
EXPECTED_ITERS=10000 \
NUM_SAMPLES=5096 \
START_INDEX=0 \
EVAL_BATCH_SIZE=4 \
OUT_DIR="$full_dir" \
bash run_fixedsplit_geopred_repaired_ddim50_eval.sh

fake_count="$(find "$full_dir/fake" -maxdepth 1 -type f -name '*.png' | wc -l)"
real_count="$(find "$full_dir/real" -maxdepth 1 -type f -name '*.png' | wc -l)"
if [[ "$fake_count" -ne 5096 || "$real_count" -ne 5096 ]]; then
  echo "Full evaluation count mismatch: real=$real_count fake=$fake_count" >&2
  exit 4
fi

grep -Fx "SAMPLER=ddim" "$full_dir/meta.txt"
grep -Fx "STEPS=50" "$full_dir/meta.txt"
grep -Fx "GUIDANCE=5.0" "$full_dir/meta.txt"
grep -Fx "SEED=20260429" "$full_dir/meta.txt"
grep -Fx "NUM_SAMPLES=5096" "$full_dir/meta.txt"

source /etc/network_turbo >/dev/null 2>&1 || true
/root/miniconda3/bin/pip install -q pytorch-fid torch-fidelity
EVAL_DIR="$full_dir" bash run_fixedsplit_geopred_clean_metrics_historical.sh
