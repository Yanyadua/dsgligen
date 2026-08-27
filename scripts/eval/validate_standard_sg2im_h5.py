#!/usr/bin/env python3
"""Fail closed unless a rebuilt VG H5 root matches the SG2IM protocol."""

import argparse
import json
from pathlib import Path

import h5py


EXPECTED_COUNTS = {"train": 62565, "val": 5062, "test": 5096}
EXPECTED_VOCAB_SIZES = {"object_idx_to_name": 179, "pred_idx_to_name": 46}


def image_ids(h5_path):
    with h5py.File(h5_path, "r") as handle:
        required = {
            "image_ids",
            "object_names",
            "object_boxes",
            "relationship_subjects",
            "relationship_predicates",
            "relationship_objects",
        }
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise RuntimeError(f"{h5_path} is missing required datasets: {missing}")
        return [int(value) for value in handle["image_ids"][:]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-root", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    root = Path(args.h5_root)
    split_ids = {}
    for split, expected_count in EXPECTED_COUNTS.items():
        h5_path = root / f"{split}.h5"
        if not h5_path.is_file():
            raise FileNotFoundError(h5_path)
        split_ids[split] = image_ids(h5_path)
        if len(split_ids[split]) != expected_count:
            raise RuntimeError(
                f"{split} image count {len(split_ids[split])} != {expected_count}"
            )

    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(split_ids[left]).intersection(split_ids[right])
        if overlap:
            raise RuntimeError(f"{left}/{right} overlap: {len(overlap)} image ids")
        overlaps[f"{left}_{right}"] = 0

    vocab_path = root / "vocab.json"
    with vocab_path.open() as handle:
        vocab = json.load(handle)
    for key, expected_size in EXPECTED_VOCAB_SIZES.items():
        actual_size = len(vocab.get(key, []))
        if actual_size != expected_size:
            raise RuntimeError(f"{key} size {actual_size} != {expected_size}")

    image_root = root / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"missing image root: {image_root}")

    report = {
        "status": "clean_sg2im_protocol",
        "h5_root": str(root.resolve()),
        "counts": {split: len(ids) for split, ids in split_ids.items()},
        "overlap_count": overlaps,
        "vocab_sizes": {key: len(vocab[key]) for key in EXPECTED_VOCAB_SIZES},
        "image_root": str(image_root.resolve()),
    }
    report_path = Path(args.report) if args.report else root / "protocol_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
