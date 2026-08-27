from pathlib import Path
import subprocess
import sys


REQUIRED_FILES = [
    "main.py",
    "trainer.py",
    "dataset/catalog.py",
    "dataset/concat_dataset.py",
    "dataset/dataset_vg_scene_graph.py",
    "dataset/scene_graph_box_utils.py",
    "grounding_input/scene_graph_grounding_tokenizer_input.py",
    "ldm/modules/diffusionmodules/scene_graph_grounding_net.py",
    "scripts/eval/generate_vg_fixedsplit_eval.py",
    "scripts/eval/recovery_checks.py",
    "scripts/eval/audit_historical_vg_run.py",
    "scripts/eval/compute_vg_fid_is.py",
    "scripts/smoke_recovery.py",
    "tests/test_recovery_checks.py",
    "tests/test_scene_graph_box_transform.py",
    "configs/vg_fixedsplit_scene_graph_geopred_clean_full.yaml",
    "configs/vg_standard_sg2im_scene_graph_geopred_clean_full.yaml",
    "configs/vg_raw_scene_graph_compatible_spatial_gat_geo_prediction_loss.yaml",
    "configs/vg_text_box_baseline.yaml",
    "configs/vg_scene_graph_mlp.yaml",
    "configs/vg_scene_graph_gat.yaml",
    "configs/vg_scene_graph_gat_residual.yaml",
    "sample_residual_horse_compare.py",
    "sample_residual_horse_compare_align.py",
    "run_standard_sg2im_geopred_clean_10k.sh",
    "run_standard_sg2im_ddim50_eval.sh",
    "prepare_historical_fixedsplit_layout.sh",
    "run_fixedsplit_geopred_clean_1k_historical_probe.sh",
    "run_fixedsplit_geopred_clean_10k_historical.sh",
    "run_fixedsplit_geopred_clean_10k_resume_from_probe.sh",
    "run_fixedsplit_geopred_clean_ddim50_eval_probe.sh",
    "run_fixedsplit_geopred_clean_ddim50_eval_historical.sh",
    "run_fixedsplit_geopred_clean_metrics_historical.sh",
    "run_fixedsplit_geopred_repaired_train.sh",
    "run_fixedsplit_geopred_repaired_ddim50_eval.sh",
    "run_repaired_10k_full_eval_after_probe.sh",
    "recover_repaired_run_after_instance_restart.sh",
    "watch_instance_and_recover.sh",
    "RECOVERY_STATUS.md",
    "REPAIRED_RUN_20260610.md",
    "RESTORE_STEPS.md",
]

HISTORICAL_EVAL_TOKENS = (
    "NUM_SAMPLES=5096",
    "SAMPLER=ddim",
    "STEPS=50",
    "GUIDANCE=5.0",
    "EVAL_BATCH_SIZE=4",
    "SEED=20260429",
)


def main():
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    missing = [rel for rel in REQUIRED_FILES if not (root / rel).exists()]
    if missing:
        print("MISSING")
        for rel in missing:
            print(rel)
        raise SystemExit(1)

    eval_launcher = (root / "run_fixedsplit_geopred_clean_ddim50_eval_historical.sh").read_text()
    missing_tokens = [token for token in HISTORICAL_EVAL_TOKENS if token not in eval_launcher]
    if missing_tokens:
        print("HISTORICAL_EVAL_PROTOCOL_MISMATCH")
        for token in missing_tokens:
            print(token)
        raise SystemExit(1)

    for launcher in (
        "run_repaired_10k_full_eval_after_probe.sh",
        "recover_repaired_run_after_instance_restart.sh",
        "watch_instance_and_recover.sh",
    ):
        subprocess.run(["bash", "-n", launcher], cwd=root, check=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests/test_recovery_checks.py",
            "tests/test_scene_graph_box_transform.py",
        ],
        cwd=root,
        check=True,
    )
    print("BUNDLE_OK")
    for rel in REQUIRED_FILES:
        print(rel)


if __name__ == "__main__":
    main()
