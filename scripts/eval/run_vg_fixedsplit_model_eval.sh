#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/GLIGEN}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASE_CKPT="${BASE_CKPT:-/root/autodl-tmp/GLIGEN/gligen_checkpoints/diffusion_pytorch_model.bin}"
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

MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL is required}"
MODEL_YAML="${MODEL_YAML:?MODEL_YAML is required}"
GROUNDING_CKPT="${GROUNDING_CKPT:?GROUNDING_CKPT is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"

mkdir -p "${OUT_DIR}"

cd "${ROOT_DIR}"

echo "[1/4] Generate ${MODEL_LABEL} on fixed test split"
OUT_DIR="${OUT_DIR}" \
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
DATA_YAML="${MODEL_YAML}" \
GROUNDING_CKPT="${GROUNDING_CKPT}" \
"${PYTHON_BIN}" scripts/eval/generate_vg_fixedsplit_eval.py | tee "${OUT_DIR}/generate.log"

echo "[2/4] ${MODEL_LABEL} FID"
REAL_DIR="${OUT_DIR}/real" \
FAKE_DIR="${OUT_DIR}/fake" \
"${PYTHON_BIN}" scripts/eval/compute_fid.py | tee "${OUT_DIR}/fid.txt"

echo "[3/4] ${MODEL_LABEL} IS"
FAKE_DIR="${OUT_DIR}/fake" \
SPLITS="${IS_SPLITS}" \
"${PYTHON_BIN}" scripts/eval/compute_is.py | tee "${OUT_DIR}/is.txt"

echo "[4/4] ${MODEL_LABEL} spatial metrics"
FAKE_DIR="${OUT_DIR}/fake" \
H5_PATH="${H5_PATH}" \
VOCAB_PATH="${VOCAB_PATH}" \
"${PYTHON_BIN}" scripts/eval/compute_spatial_metrics.py | tee "${OUT_DIR}/spatial_metrics.txt"

echo "DONE ${MODEL_LABEL} fixed-split evaluation"
