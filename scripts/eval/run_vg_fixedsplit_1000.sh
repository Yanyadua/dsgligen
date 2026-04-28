#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/GLIGEN}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASE_CKPT="${BASE_CKPT:-/root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin}"
OURS_CKPT="${OURS_CKPT:-/root/autodl-tmp/GLIGEN/OUTPUT_RAW_VG_GEOPRED/vg_raw_scene_graph_geo_prediction_loss_1k/tag02/checkpoint_00001000.pth}"
FIXED_ROOT="${FIXED_ROOT:-/root/autodl-tmp/fixed_split_work/datasets/vg}"
H5_PATH="${H5_PATH:-${FIXED_ROOT}/test.h5}"
VOCAB_PATH="${VOCAB_PATH:-${FIXED_ROOT}/vocab.json}"
IMAGE_ROOT="${IMAGE_ROOT:-${FIXED_ROOT}/images}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
START_INDEX="${START_INDEX:-0}"
SAVE_SIZE="${SAVE_SIZE:-256}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-5.0}"
SEED="${SEED:-20260429}"
IS_SPLITS="${IS_SPLITS:-5}"

BASE_OUT="${BASE_OUT:-${ROOT_DIR}/eval_outputs/vg_fixedsplit_baseline_1000}"
OURS_OUT="${OURS_OUT:-${ROOT_DIR}/eval_outputs/vg_fixedsplit_ours_1000}"

mkdir -p "${BASE_OUT}" "${OURS_OUT}"

cd "${ROOT_DIR}"

echo "[1/6] Generate baseline on fixed test split"
OUT_DIR="${BASE_OUT}" \
SAVE_SIZE="${SAVE_SIZE}" \
NUM_SAMPLES="${NUM_SAMPLES}" \
START_INDEX="${START_INDEX}" \
STEPS="${STEPS}" \
GUIDANCE="${GUIDANCE}" \
SEED="${SEED}" \
H5_PATH="${H5_PATH}" \
VOCAB_PATH="${VOCAB_PATH}" \
IMAGE_ROOT="${IMAGE_ROOT}" \
BASE_CKPT="${BASE_CKPT}" \
"${PYTHON_BIN}" scripts/eval/generate_vg_fixedsplit_eval.py | tee "${BASE_OUT}/generate.log"

echo "[2/6] Generate ours on fixed test split"
OUT_DIR="${OURS_OUT}" \
SAVE_SIZE="${SAVE_SIZE}" \
NUM_SAMPLES="${NUM_SAMPLES}" \
START_INDEX="${START_INDEX}" \
STEPS="${STEPS}" \
GUIDANCE="${GUIDANCE}" \
SEED="${SEED}" \
H5_PATH="${H5_PATH}" \
VOCAB_PATH="${VOCAB_PATH}" \
IMAGE_ROOT="${IMAGE_ROOT}" \
BASE_CKPT="${BASE_CKPT}" \
DATA_YAML="configs/vg_raw_scene_graph_compatible_spatial_gat_geo_prediction_loss.yaml" \
GROUNDING_CKPT="${OURS_CKPT}" \
"${PYTHON_BIN}" scripts/eval/generate_vg_fixedsplit_eval.py | tee "${OURS_OUT}/generate.log"

echo "[3/6] Baseline FID"
REAL_DIR="${BASE_OUT}/real" \
FAKE_DIR="${BASE_OUT}/fake" \
"${PYTHON_BIN}" scripts/eval/compute_fid.py | tee "${BASE_OUT}/fid.txt"

echo "[4/6] Baseline IS"
FAKE_DIR="${BASE_OUT}/fake" \
SPLITS="${IS_SPLITS}" \
"${PYTHON_BIN}" scripts/eval/compute_is.py | tee "${BASE_OUT}/is.txt"

echo "[5/6] Ours FID"
REAL_DIR="${OURS_OUT}/real" \
FAKE_DIR="${OURS_OUT}/fake" \
"${PYTHON_BIN}" scripts/eval/compute_fid.py | tee "${OURS_OUT}/fid.txt"

echo "[6/6] Ours IS"
FAKE_DIR="${OURS_OUT}/fake" \
SPLITS="${IS_SPLITS}" \
"${PYTHON_BIN}" scripts/eval/compute_is.py | tee "${OURS_OUT}/is.txt"

echo "DONE fixed-split 1000 evaluation"
