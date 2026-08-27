#!/usr/bin/env bash
set -euo pipefail

HOST="36.140.33.200"
PORT="30710"
REMOTE_ROOT="/root/autodl-tmp/GLIGEN"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${WATCH_LOG:-/Users/yaoduanyang/Desktop/src/gligen_instance_recovery_watch.log}"

while true; do
  printf '%s attempting SSH reconnect\n' "$(date '+%F %T')" >> "$LOG"
  if ssh \
    -o HostKeyAlias=region-9.autodl.pro \
    -o BatchMode=yes \
    -o ConnectTimeout=10 \
    -p "$PORT" \
    "root@$HOST" \
    "echo connected" >> "$LOG" 2>&1; then
    printf '%s SSH restored\n' "$(date '+%F %T')" >> "$LOG"
    scp \
      -o HostKeyAlias=region-9.autodl.pro \
      -P "$PORT" \
      "$SCRIPT_DIR/recover_repaired_run_after_instance_restart.sh" \
      "root@$HOST:$REMOTE_ROOT/recover_repaired_run_after_instance_restart.sh" >> "$LOG" 2>&1

    ssh \
      -o HostKeyAlias=region-9.autodl.pro \
      -p "$PORT" \
      "root@$HOST" \
      "cd $REMOTE_ROOT && chmod +x recover_repaired_run_after_instance_restart.sh && \
       if pgrep -f '[r]un_repaired_10k_full_eval_after_probe|[r]ecover_repaired_run_after_instance_restart.sh' >/dev/null; then \
         echo 'An evaluation/recovery pipeline is already active.'; \
       else \
         nohup bash recover_repaired_run_after_instance_restart.sh > recovered_after_restart_pipeline.log 2>&1 < /dev/null & \
         echo RECOVERY_PID=\$!; \
       fi" >> "$LOG" 2>&1
    printf '%s recovery handoff complete\n' "$(date '+%F %T')" >> "$LOG"
    exit 0
  fi
  sleep 60
done
