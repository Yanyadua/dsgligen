#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 /path/to/GLIGEN"
  exit 1
fi

TARGET_ROOT="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p \
  "$TARGET_ROOT/scripts" \
  "$TARGET_ROOT/scripts/eval" \
  "$TARGET_ROOT/grounding_input" \
  "$TARGET_ROOT/dataset" \
  "$TARGET_ROOT/ldm/modules/diffusionmodules" \
  "$TARGET_ROOT/configs"

cp "$SCRIPT_DIR/main.py" "$TARGET_ROOT/main.py"
cp "$SCRIPT_DIR/trainer.py" "$TARGET_ROOT/trainer.py"
cp "$SCRIPT_DIR/scripts/eval/generate_vg_fixedsplit_eval.py" "$TARGET_ROOT/scripts/eval/generate_vg_fixedsplit_eval.py"
cp "$SCRIPT_DIR/scripts/eval/recovery_checks.py" "$TARGET_ROOT/scripts/eval/recovery_checks.py"
cp "$SCRIPT_DIR/scripts/eval/audit_historical_vg_run.py" "$TARGET_ROOT/scripts/eval/audit_historical_vg_run.py"
cp "$SCRIPT_DIR/scripts/eval/compute_vg_fid_is.py" "$TARGET_ROOT/scripts/eval/compute_vg_fid_is.py"
cp "$SCRIPT_DIR/grounding_input/scene_graph_grounding_tokenizer_input.py" "$TARGET_ROOT/grounding_input/scene_graph_grounding_tokenizer_input.py"
cp "$SCRIPT_DIR/dataset/dataset_vg_scene_graph.py" "$TARGET_ROOT/dataset/dataset_vg_scene_graph.py"
cp "$SCRIPT_DIR/dataset/scene_graph_box_utils.py" "$TARGET_ROOT/dataset/scene_graph_box_utils.py"
cp "$SCRIPT_DIR/dataset/catalog.py" "$TARGET_ROOT/dataset/catalog.py"
cp "$SCRIPT_DIR/dataset/concat_dataset.py" "$TARGET_ROOT/dataset/concat_dataset.py"
cp "$SCRIPT_DIR/ldm/modules/diffusionmodules/scene_graph_grounding_net.py" "$TARGET_ROOT/ldm/modules/diffusionmodules/scene_graph_grounding_net.py"
cp "$SCRIPT_DIR/configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml" "$TARGET_ROOT/configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml"
cp "$SCRIPT_DIR/configs/vg_standard_sg2im_scene_graph_geopred_clean_full.yaml" "$TARGET_ROOT/configs/vg_standard_sg2im_scene_graph_geopred_clean_full.yaml"
cp "$SCRIPT_DIR/configs/vg_raw_scene_graph_compatible_spatial_gat_geo_prediction_loss.yaml" "$TARGET_ROOT/configs/vg_raw_scene_graph_compatible_spatial_gat_geo_prediction_loss.yaml"
cp "$SCRIPT_DIR/configs/vg_text_box_baseline.yaml" "$TARGET_ROOT/configs/vg_text_box_baseline.yaml"
cp "$SCRIPT_DIR/configs/vg_scene_graph_mlp.yaml" "$TARGET_ROOT/configs/vg_scene_graph_mlp.yaml"
cp "$SCRIPT_DIR/configs/vg_scene_graph_gat.yaml" "$TARGET_ROOT/configs/vg_scene_graph_gat.yaml"
cp "$SCRIPT_DIR/configs/vg_scene_graph_gat_residual.yaml" "$TARGET_ROOT/configs/vg_scene_graph_gat_residual.yaml"
cp "$SCRIPT_DIR/sample_residual_horse_compare.py" "$TARGET_ROOT/sample_residual_horse_compare.py"
cp "$SCRIPT_DIR/sample_residual_horse_compare_align.py" "$TARGET_ROOT/sample_residual_horse_compare_align.py"
cp "$SCRIPT_DIR/run_standard_sg2im_geopred_clean_10k.sh" "$TARGET_ROOT/run_standard_sg2im_geopred_clean_10k.sh"
cp "$SCRIPT_DIR/run_standard_sg2im_ddim50_eval.sh" "$TARGET_ROOT/run_standard_sg2im_ddim50_eval.sh"
cp "$SCRIPT_DIR/prepare_historical_fixedsplit_layout.sh" "$TARGET_ROOT/prepare_historical_fixedsplit_layout.sh"
cp "$SCRIPT_DIR/run_fixedsplit_geopred_clean_1k_historical_probe.sh" "$TARGET_ROOT/run_fixedsplit_geopred_clean_1k_historical_probe.sh"
cp "$SCRIPT_DIR/run_fixedsplit_geopred_clean_10k_historical.sh" "$TARGET_ROOT/run_fixedsplit_geopred_clean_10k_historical.sh"
cp "$SCRIPT_DIR/run_fixedsplit_geopred_clean_10k_resume_from_probe.sh" "$TARGET_ROOT/run_fixedsplit_geopred_clean_10k_resume_from_probe.sh"
cp "$SCRIPT_DIR/run_fixedsplit_geopred_clean_ddim50_eval_probe.sh" "$TARGET_ROOT/run_fixedsplit_geopred_clean_ddim50_eval_probe.sh"
cp "$SCRIPT_DIR/run_fixedsplit_geopred_clean_ddim50_eval_historical.sh" "$TARGET_ROOT/run_fixedsplit_geopred_clean_ddim50_eval_historical.sh"
cp "$SCRIPT_DIR/run_fixedsplit_geopred_clean_metrics_historical.sh" "$TARGET_ROOT/run_fixedsplit_geopred_clean_metrics_historical.sh"
cp "$SCRIPT_DIR/run_fixedsplit_geopred_repaired_train.sh" "$TARGET_ROOT/run_fixedsplit_geopred_repaired_train.sh"
cp "$SCRIPT_DIR/run_fixedsplit_geopred_repaired_ddim50_eval.sh" "$TARGET_ROOT/run_fixedsplit_geopred_repaired_ddim50_eval.sh"
cp "$SCRIPT_DIR/scripts/smoke_recovery.py" "$TARGET_ROOT/scripts/smoke_recovery.py"
mkdir -p "$TARGET_ROOT/tests"
cp "$SCRIPT_DIR/tests/test_recovery_checks.py" "$TARGET_ROOT/tests/test_recovery_checks.py"
cp "$SCRIPT_DIR/tests/test_scene_graph_box_transform.py" "$TARGET_ROOT/tests/test_scene_graph_box_transform.py"
cp "$SCRIPT_DIR/verify_recovery_bundle.py" "$TARGET_ROOT/verify_recovery_bundle.py"
cp "$SCRIPT_DIR/RECOVERY_STATUS.md" "$TARGET_ROOT/RECOVERY_STATUS.md"
cp "$SCRIPT_DIR/RESTORE_STEPS.md" "$TARGET_ROOT/RESTORE_STEPS.md"

chmod +x \
  "$TARGET_ROOT/run_standard_sg2im_geopred_clean_10k.sh" \
  "$TARGET_ROOT/run_standard_sg2im_ddim50_eval.sh" \
  "$TARGET_ROOT/prepare_historical_fixedsplit_layout.sh" \
  "$TARGET_ROOT/run_fixedsplit_geopred_clean_1k_historical_probe.sh" \
  "$TARGET_ROOT/run_fixedsplit_geopred_clean_10k_historical.sh" \
  "$TARGET_ROOT/run_fixedsplit_geopred_clean_10k_resume_from_probe.sh" \
  "$TARGET_ROOT/run_fixedsplit_geopred_clean_ddim50_eval_probe.sh" \
  "$TARGET_ROOT/run_fixedsplit_geopred_clean_ddim50_eval_historical.sh" \
  "$TARGET_ROOT/run_fixedsplit_geopred_clean_metrics_historical.sh"
chmod +x \
  "$TARGET_ROOT/run_fixedsplit_geopred_repaired_train.sh" \
  "$TARGET_ROOT/run_fixedsplit_geopred_repaired_ddim50_eval.sh"

echo "Recovery bundle applied to $TARGET_ROOT"
