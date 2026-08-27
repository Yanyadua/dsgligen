#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-root@region-9.autodl.pro}"
REMOTE_PORT="${REMOTE_PORT:-30710}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/GLIGEN}"
RETRIES="${RETRIES:-8}"
RETRY_SLEEP="${RETRY_SLEEP:-5}"
TOTAL_ITERS="${TOTAL_ITERS:-1000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-vg_fixedsplit_graphdistill_fuser_20260619}"

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=no
  -o ConnectTimeout=20
  -p "$REMOTE_PORT"
)

FILES=(
  "ldm/modules/diffusionmodules/scene_graph_grounding_net.py"
  "trainer.py"
  "configs/vg_fixedsplit_scene_graph_graphdistill_fuser.yaml"
  "run_fixedsplit_graphdistill_fuser_train.sh"
)

retry_cmd() {
  local label="$1"
  shift
  local attempt
  for attempt in $(seq 1 "$RETRIES"); do
    echo "[$label] attempt $attempt/$RETRIES"
    if "$@"; then
      return 0
    fi
    sleep "$RETRY_SLEEP"
  done
  return 1
}

run_remote() {
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$@"
}

copy_file() {
  local src="$1"
  local dst="$2"
  local remote_dir
  remote_dir="$(dirname "$dst")"
  run_remote "mkdir -p '$remote_dir' && cat > '${dst}.tmp' && mv '${dst}.tmp' '$dst'" < "$src"
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

retry_cmd ssh_probe run_remote "echo OK && cd '$REMOTE_ROOT' && pwd"

for file in "${FILES[@]}"; do
  retry_cmd "scp:$file" copy_file "$SCRIPT_DIR/$file" "$REMOTE_ROOT/$file"
done

retry_cmd remote_compile run_remote \
  "cd '$REMOTE_ROOT' && chmod +x run_fixedsplit_graphdistill_fuser_train.sh && /root/miniconda3/bin/python -m py_compile trainer.py ldm/modules/diffusionmodules/scene_graph_grounding_net.py main.py"

retry_cmd remote_launch run_remote \
  "cd '$REMOTE_ROOT' && nohup env TOTAL_ITERS='$TOTAL_ITERS' EXPERIMENT_NAME='$EXPERIMENT_NAME' bash run_fixedsplit_graphdistill_fuser_train.sh > '${REMOTE_ROOT}/logs_${EXPERIMENT_NAME}.txt' 2>&1 < /dev/null & echo STARTED"

echo "Remote graph-distill fuser training launch requested: $EXPERIMENT_NAME"
