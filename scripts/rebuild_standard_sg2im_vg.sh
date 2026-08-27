#!/usr/bin/env bash
set -euo pipefail

# Rebuild the canonical SG2IM-style Visual Genome H5 files from public sources.
# This is intentionally separate from model code and never overwrites an
# existing H5/vocab output. Run it inside screen on a fresh AutoDL instance.

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp}"
PROJECT_DIR="${PROJECT_DIR:-${ROOT_DIR}/GLIGEN}"
REFERENCE_DIR="${REFERENCE_DIR:-${ROOT_DIR}/sg2im-reference-v2}"
RAW_DIR="${RAW_DIR:-${ROOT_DIR}/vg_sg2im_raw}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/standard_sg2im_fresh_h5}"
GITHUB_PROXY="${GITHUB_PROXY:-https://gh-proxy.com/https://github.com/}"
VG_MIRROR="${VG_MIRROR:-https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset}"

mkdir -p "${RAW_DIR}" "${OUT_DIR}"

for output in train.h5 val.h5 test.h5 vocab.json; do
  if [[ -e "${OUT_DIR}/${output}" ]]; then
    echo "Refusing to overwrite existing ${OUT_DIR}/${output}" >&2
    exit 2
  fi
done

if [[ ! -f "${REFERENCE_DIR}/scripts/preprocess_vg.py" ]]; then
  git -c url."${GITHUB_PROXY}".insteadOf=https://github.com/ \
    clone --depth 1 https://github.com/google/sg2im.git "${REFERENCE_DIR}"
fi

download() {
  local url="$1"
  local target="$2"
  if [[ -s "${target}" ]]; then
    echo "Using existing $(basename "${target}")"
    return
  fi
  curl -fL --retry 5 --retry-all-errors --continue-at - \
    --output "${target}" "${url}"
}

cd "${RAW_DIR}"
download "${VG_MIRROR}/objects.json.zip" objects.json.zip
download "${VG_MIRROR}/attributes.json.zip" attributes.json.zip
download "${VG_MIRROR}/relationships.json.zip" relationships.json.zip
download "${VG_MIRROR}/image_data.json.zip" image_data.json.zip
download "${VG_MIRROR}/object_alias.txt" object_alias.txt
download "${VG_MIRROR}/relationship_alias.txt" relationship_alias.txt
download "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip" images.zip
download "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip" images2.zip

unzip -n objects.json.zip
unzip -n attributes.json.zip
unzip -n relationships.json.zip
unzip -n image_data.json.zip
mkdir -p images
unzip -n images.zip -d images
unzip -n images2.zip -d images

"${PYTHON:-python}" "${PROJECT_DIR}/scripts/run_official_sg2im_preprocess.py" \
  "${REFERENCE_DIR}/scripts/preprocess_vg.py" \
  --splits_json "${REFERENCE_DIR}/sg2im/data/vg_splits.json" \
  --images_json "${RAW_DIR}/image_data.json" \
  --objects_json "${RAW_DIR}/objects.json" \
  --attributes_json "${RAW_DIR}/attributes.json" \
  --object_aliases "${RAW_DIR}/object_alias.txt" \
  --relationship_aliases "${RAW_DIR}/relationship_alias.txt" \
  --relationships_json "${RAW_DIR}/relationships.json" \
  --output_h5_dir "${OUT_DIR}" \
  --output_vocab_json "${OUT_DIR}/vocab.json"

ln -s "${RAW_DIR}/images" "${OUT_DIR}/images"

"${PYTHON:-python}" "${PROJECT_DIR}/scripts/eval/validate_standard_sg2im_h5.py" \
  --h5-root "${OUT_DIR}"
