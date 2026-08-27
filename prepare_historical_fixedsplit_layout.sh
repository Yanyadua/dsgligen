#!/usr/bin/env bash
set -euo pipefail

# Recreate the historical fixed-split path layout expected by the old strong-run
# metadata, while using the currently restored clean split files as a temporary
# compatibility mirror.

SRC_ROOT="${1:-/root/autodl-tmp/standard_sg2im_fresh_h5}"
DST_ROOT="${2:-/root/autodl-tmp/fixed_split_work/datasets/vg}"

mkdir -p "${DST_ROOT}"

for name in train.h5 val.h5 test.h5 vocab.json images; do
  src="${SRC_ROOT}/${name}"
  dst="${DST_ROOT}/${name}"
  if [ ! -e "${src}" ]; then
    echo "MISSING_SOURCE ${src}" >&2
    exit 1
  fi
  if [ -L "${dst}" ] || [ -e "${dst}" ]; then
    rm -rf "${dst}"
  fi
  ln -s "${src}" "${dst}"
  echo "LINKED ${dst} -> ${src}"
done

echo "HISTORICAL_FIXEDSPLIT_LAYOUT_READY ${DST_ROOT}"
