#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <ssh-port> <remote-root>"
  echo "Example: $0 30710 /root/autodl-tmp/GLIGEN"
  exit 1
fi

PORT="$1"
REMOTE_ROOT="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

scp -P "$PORT" "$SCRIPT_DIR/main.py" root@region-9.autodl.pro:"$REMOTE_ROOT/main.py"
scp -P "$PORT" "$SCRIPT_DIR/trainer.py" root@region-9.autodl.pro:"$REMOTE_ROOT/trainer.py"
scp -P "$PORT" "$SCRIPT_DIR/RECOVERY_STATUS.md" root@region-9.autodl.pro:"$REMOTE_ROOT/RECOVERY_STATUS.md"
scp -P "$PORT" "$SCRIPT_DIR/verify_recovery_bundle.py" root@region-9.autodl.pro:"$REMOTE_ROOT/verify_recovery_bundle.py"
scp -P "$PORT" "$SCRIPT_DIR/run_standard_sg2im_geopred_clean_10k.sh" root@region-9.autodl.pro:"$REMOTE_ROOT/run_standard_sg2im_geopred_clean_10k.sh"
scp -P "$PORT" "$SCRIPT_DIR/run_standard_sg2im_ddim50_eval.sh" root@region-9.autodl.pro:"$REMOTE_ROOT/run_standard_sg2im_ddim50_eval.sh"

scp -P "$PORT" "$SCRIPT_DIR/scripts/smoke_recovery.py" root@region-9.autodl.pro:"$REMOTE_ROOT/scripts/smoke_recovery.py"
scp -P "$PORT" "$SCRIPT_DIR/scripts/eval/generate_vg_fixedsplit_eval.py" root@region-9.autodl.pro:"$REMOTE_ROOT/scripts/eval/generate_vg_fixedsplit_eval.py"
scp -P "$PORT" "$SCRIPT_DIR/scripts/eval/recovery_checks.py" root@region-9.autodl.pro:"$REMOTE_ROOT/scripts/eval/recovery_checks.py"
scp -P "$PORT" "$SCRIPT_DIR/scripts/eval/audit_historical_vg_run.py" root@region-9.autodl.pro:"$REMOTE_ROOT/scripts/eval/audit_historical_vg_run.py"
scp -P "$PORT" "$SCRIPT_DIR/scripts/eval/compute_vg_fid_is.py" root@region-9.autodl.pro:"$REMOTE_ROOT/scripts/eval/compute_vg_fid_is.py"

scp -P "$PORT" "$SCRIPT_DIR/dataset/catalog.py" root@region-9.autodl.pro:"$REMOTE_ROOT/dataset/catalog.py"
scp -P "$PORT" "$SCRIPT_DIR/dataset/concat_dataset.py" root@region-9.autodl.pro:"$REMOTE_ROOT/dataset/concat_dataset.py"
scp -P "$PORT" "$SCRIPT_DIR/dataset/dataset_vg_scene_graph.py" root@region-9.autodl.pro:"$REMOTE_ROOT/dataset/dataset_vg_scene_graph.py"
scp -P "$PORT" "$SCRIPT_DIR/dataset/scene_graph_box_utils.py" root@region-9.autodl.pro:"$REMOTE_ROOT/dataset/scene_graph_box_utils.py"

scp -P "$PORT" "$SCRIPT_DIR/grounding_input/scene_graph_grounding_tokenizer_input.py" root@region-9.autodl.pro:"$REMOTE_ROOT/grounding_input/scene_graph_grounding_tokenizer_input.py"

scp -P "$PORT" "$SCRIPT_DIR/ldm/modules/diffusionmodules/scene_graph_grounding_net.py" root@region-9.autodl.pro:"$REMOTE_ROOT/ldm/modules/diffusionmodules/scene_graph_grounding_net.py"

scp -P "$PORT" "$SCRIPT_DIR/configs/vg_standard_sg2im_scene_graph_geopred_clean_full.yaml" root@region-9.autodl.pro:"$REMOTE_ROOT/configs/vg_standard_sg2im_scene_graph_geopred_clean_full.yaml"
scp -P "$PORT" "$SCRIPT_DIR/configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml" root@region-9.autodl.pro:"$REMOTE_ROOT/configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml"
scp -P "$PORT" "$SCRIPT_DIR/configs/vg_raw_scene_graph_compatible_spatial_gat_geo_prediction_loss.yaml" root@region-9.autodl.pro:"$REMOTE_ROOT/configs/vg_raw_scene_graph_compatible_spatial_gat_geo_prediction_loss.yaml"
scp -P "$PORT" "$SCRIPT_DIR/configs/vg_text_box_baseline.yaml" root@region-9.autodl.pro:"$REMOTE_ROOT/configs/vg_text_box_baseline.yaml"
scp -P "$PORT" "$SCRIPT_DIR/configs/vg_scene_graph_mlp.yaml" root@region-9.autodl.pro:"$REMOTE_ROOT/configs/vg_scene_graph_mlp.yaml"
scp -P "$PORT" "$SCRIPT_DIR/configs/vg_scene_graph_gat.yaml" root@region-9.autodl.pro:"$REMOTE_ROOT/configs/vg_scene_graph_gat.yaml"
scp -P "$PORT" "$SCRIPT_DIR/configs/vg_scene_graph_gat_residual.yaml" root@region-9.autodl.pro:"$REMOTE_ROOT/configs/vg_scene_graph_gat_residual.yaml"
scp -P "$PORT" "$SCRIPT_DIR/sample_residual_horse_compare.py" root@region-9.autodl.pro:"$REMOTE_ROOT/sample_residual_horse_compare.py"
scp -P "$PORT" "$SCRIPT_DIR/sample_residual_horse_compare_align.py" root@region-9.autodl.pro:"$REMOTE_ROOT/sample_residual_horse_compare_align.py"
scp -P "$PORT" "$SCRIPT_DIR/prepare_historical_fixedsplit_layout.sh" root@region-9.autodl.pro:"$REMOTE_ROOT/prepare_historical_fixedsplit_layout.sh"
scp -P "$PORT" "$SCRIPT_DIR/run_fixedsplit_geopred_clean_1k_historical_probe.sh" root@region-9.autodl.pro:"$REMOTE_ROOT/run_fixedsplit_geopred_clean_1k_historical_probe.sh"
scp -P "$PORT" "$SCRIPT_DIR/run_fixedsplit_geopred_clean_10k_historical.sh" root@region-9.autodl.pro:"$REMOTE_ROOT/run_fixedsplit_geopred_clean_10k_historical.sh"
scp -P "$PORT" "$SCRIPT_DIR/run_fixedsplit_geopred_clean_10k_resume_from_probe.sh" root@region-9.autodl.pro:"$REMOTE_ROOT/run_fixedsplit_geopred_clean_10k_resume_from_probe.sh"
scp -P "$PORT" "$SCRIPT_DIR/run_fixedsplit_geopred_clean_ddim50_eval_probe.sh" root@region-9.autodl.pro:"$REMOTE_ROOT/run_fixedsplit_geopred_clean_ddim50_eval_probe.sh"
scp -P "$PORT" "$SCRIPT_DIR/run_fixedsplit_geopred_clean_ddim50_eval_historical.sh" root@region-9.autodl.pro:"$REMOTE_ROOT/run_fixedsplit_geopred_clean_ddim50_eval_historical.sh"
scp -P "$PORT" "$SCRIPT_DIR/run_fixedsplit_geopred_clean_metrics_historical.sh" root@region-9.autodl.pro:"$REMOTE_ROOT/run_fixedsplit_geopred_clean_metrics_historical.sh"
scp -P "$PORT" "$SCRIPT_DIR/run_fixedsplit_geopred_repaired_train.sh" root@region-9.autodl.pro:"$REMOTE_ROOT/run_fixedsplit_geopred_repaired_train.sh"
scp -P "$PORT" "$SCRIPT_DIR/run_fixedsplit_geopred_repaired_ddim50_eval.sh" root@region-9.autodl.pro:"$REMOTE_ROOT/run_fixedsplit_geopred_repaired_ddim50_eval.sh"
scp -P "$PORT" "$SCRIPT_DIR/RESTORE_STEPS.md" root@region-9.autodl.pro:"$REMOTE_ROOT/RESTORE_STEPS.md"
scp -P "$PORT" "$SCRIPT_DIR/tests/test_recovery_checks.py" root@region-9.autodl.pro:"$REMOTE_ROOT/tests/test_recovery_checks.py"
scp -P "$PORT" "$SCRIPT_DIR/tests/test_scene_graph_box_transform.py" root@region-9.autodl.pro:"$REMOTE_ROOT/tests/test_scene_graph_box_transform.py"

scp -P "$PORT" "$SCRIPT_DIR/apply_recovery.sh" root@region-9.autodl.pro:"$REMOTE_ROOT/apply_recovery.sh"
ssh -p "$PORT" root@region-9.autodl.pro "chmod +x '$REMOTE_ROOT/apply_recovery.sh' '$REMOTE_ROOT'/run_*historical.sh '$REMOTE_ROOT'/run_*probe.sh '$REMOTE_ROOT'/run_*repaired*.sh '$REMOTE_ROOT'/run_standard_sg2im_*.sh '$REMOTE_ROOT/prepare_historical_fixedsplit_layout.sh'"

echo "SYNC_DONE"
