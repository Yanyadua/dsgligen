from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_vg_scene_graph import compute_relation_geo_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ObjectBank / RelationBank manifests from SG2I-style fixed-split VG h5 files."
    )
    parser.add_argument(
        "--h5_path",
        type=Path,
        required=True,
        help="Path to train.h5 / val.h5 / test.h5 from the SG2I-style fixed split.",
    )
    parser.add_argument(
        "--vocab_path",
        type=Path,
        required=True,
        help="Path to vocab.json associated with the fixed split.",
    )
    parser.add_argument(
        "--image_root",
        type=Path,
        required=True,
        help="Root directory containing VG images referenced by image_paths in the h5 file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where manifests and summary json will be written.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        help="Human-readable split name stored in each record.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional cap on number of images to export for quick experiments.",
    )
    parser.add_argument(
        "--min_object_size_px",
        type=float,
        default=8.0,
        help="Filter objects whose width or height in pixels is smaller than this threshold.",
    )
    parser.add_argument(
        "--skip_pseudo_labels",
        action="store_true",
        help="Skip SG2I pseudo labels such as __image__ / __in_image__.",
    )
    return parser.parse_args()


def decode_maybe_bytes(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def xywh_to_xyxy(box_xywh: np.ndarray) -> list[float]:
    x, y, w, h = [float(v) for v in box_xywh.tolist()]
    return [x, y, x + w, y + h]


def normalize_xyxy(box_xyxy: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box_xyxy
    return [
        x1 / max(width, 1),
        y1 / max(height, 1),
        x2 / max(width, 1),
        y2 / max(height, 1),
    ]


def union_xyxy(box_a: list[float], box_b: list[float]) -> list[float]:
    return [
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    ]


def bbox_area(box_xyxy: list[float]) -> float:
    return max(0.0, box_xyxy[2] - box_xyxy[0]) * max(0.0, box_xyxy[3] - box_xyxy[1])


def maybe_write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def maybe_write_parquet(records: list[dict[str, Any]], path: Path) -> bool:
    try:
        import pandas as pd
    except Exception:
        return False
    try:
        pd.DataFrame.from_records(records).to_parquet(path, index=False)
    except Exception:
        return False
    return True


def image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path).convert("RGB") as image:
        return image.size


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.vocab_path.open("r", encoding="utf-8") as fp:
        vocab = json.load(fp)

    object_idx_to_name = vocab["object_idx_to_name"]
    pred_idx_to_name = vocab["pred_idx_to_name"]
    attr_idx_to_name = vocab.get("attribute_idx_to_name", [])

    object_records: list[dict[str, Any]] = []
    relation_records: list[dict[str, Any]] = []

    with h5py.File(args.h5_path, "r") as h5:
        num_images = int(h5["image_ids"].shape[0])
        if args.max_images is not None:
            num_images = min(num_images, args.max_images)

        object_offsets = np.zeros(num_images + 1, dtype=np.int64)
        object_offsets[1:] = np.cumsum(h5["objects_per_image"][:num_images], dtype=np.int64)

        for image_index in range(num_images):
            image_id = int(h5["image_ids"][image_index])
            relative_path = decode_maybe_bytes(h5["image_paths"][image_index])
            image_path = args.image_root / relative_path
            width, height = image_size(image_path)

            num_objects = int(h5["objects_per_image"][image_index])
            num_relations = int(h5["relationships_per_image"][image_index])

            object_names = h5["object_names"][image_index][:num_objects].tolist()
            object_boxes_xywh = h5["object_boxes"][image_index][:num_objects]
            object_ids = h5["object_ids"][image_index][:num_objects].tolist()
            object_attr_counts = h5["attributes_per_object"][image_index][:num_objects].tolist()
            object_attr_values = h5["object_attributes"][
                object_offsets[image_index] : object_offsets[image_index + 1]
            ]

            kept_local_to_record_id: dict[int, int] = {}
            kept_local_to_box_norm: dict[int, list[float]] = {}

            for obj_local_idx, (name_idx, box_xywh, object_id, attr_count) in enumerate(
                zip(object_names, object_boxes_xywh, object_ids, object_attr_counts)
            ):
                obj_name = decode_maybe_bytes(object_idx_to_name[int(name_idx)]).lower()
                if args.skip_pseudo_labels and obj_name.startswith("__"):
                    continue

                xyxy = xywh_to_xyxy(box_xywh)
                w_px = xyxy[2] - xyxy[0]
                h_px = xyxy[3] - xyxy[1]
                if min(w_px, h_px) < args.min_object_size_px:
                    continue

                xyxy_norm = normalize_xyxy(xyxy, width, height)
                attr_names: list[str] = []
                attr_row = object_attr_values[obj_local_idx]
                for attr_idx in attr_row[: int(attr_count)].tolist():
                    if 0 <= int(attr_idx) < len(attr_idx_to_name):
                        attr_name = decode_maybe_bytes(attr_idx_to_name[int(attr_idx)]).lower()
                        if attr_name:
                            attr_names.append(attr_name)

                record_id = len(object_records)
                kept_local_to_record_id[obj_local_idx] = record_id
                kept_local_to_box_norm[obj_local_idx] = xyxy_norm
                object_records.append(
                    {
                        "id": record_id,
                        "split": args.split_name,
                        "image_index": image_index,
                        "image_id": image_id,
                        "image_rel_path": relative_path,
                        "image_width": width,
                        "image_height": height,
                        "obj_local_idx": obj_local_idx,
                        "object_id": int(object_id),
                        "obj_name": obj_name,
                        "attributes": attr_names,
                        "bbox_xyxy": xyxy,
                        "bbox_xyxy_norm": xyxy_norm,
                        "area_px": float(bbox_area(xyxy)),
                        "area_ratio": float(bbox_area(xyxy) / max(width * height, 1)),
                        "tight_crop_rel_path": f"object/tight/{record_id:09d}.jpg",
                        "context_crop_rel_path": f"object/context/{record_id:09d}.jpg",
                    }
                )

            rel_subjects = h5["relationship_subjects"][image_index][:num_relations].tolist()
            rel_predicates = h5["relationship_predicates"][image_index][:num_relations].tolist()
            rel_objects = h5["relationship_objects"][image_index][:num_relations].tolist()
            relation_ids = h5["relationship_ids"][image_index][:num_relations].tolist()

            for rel_local_idx, (subject_local_idx, pred_idx, object_local_idx, relation_id) in enumerate(
                zip(rel_subjects, rel_predicates, rel_objects, relation_ids)
            ):
                subject_local_idx = int(subject_local_idx)
                object_local_idx = int(object_local_idx)
                if (
                    subject_local_idx not in kept_local_to_record_id
                    or object_local_idx not in kept_local_to_record_id
                ):
                    continue

                predicate = decode_maybe_bytes(pred_idx_to_name[int(pred_idx)]).lower()
                if args.skip_pseudo_labels and predicate.startswith("__"):
                    continue

                subject_record = object_records[kept_local_to_record_id[subject_local_idx]]
                object_record = object_records[kept_local_to_record_id[object_local_idx]]
                subject_xyxy = subject_record["bbox_xyxy"]
                object_xyxy = object_record["bbox_xyxy"]
                union_box_xyxy = union_xyxy(subject_xyxy, object_xyxy)
                union_box_norm = normalize_xyxy(union_box_xyxy, width, height)
                geo_feat = compute_relation_geo_features(
                    torch.tensor(kept_local_to_box_norm[subject_local_idx], dtype=torch.float32),
                    torch.tensor(kept_local_to_box_norm[object_local_idx], dtype=torch.float32),
                ).tolist()

                relation_record_id = len(relation_records)
                relation_records.append(
                    {
                        "id": relation_record_id,
                        "split": args.split_name,
                        "image_index": image_index,
                        "image_id": image_id,
                        "image_rel_path": relative_path,
                        "image_width": width,
                        "image_height": height,
                        "rel_local_idx": rel_local_idx,
                        "relation_id": int(relation_id),
                        "subject_local_idx": subject_local_idx,
                        "object_local_idx": object_local_idx,
                        "subject_record_id": subject_record["id"],
                        "object_record_id": object_record["id"],
                        "subj_name": subject_record["obj_name"],
                        "predicate": predicate,
                        "obj_name": object_record["obj_name"],
                        "subj_bbox_xyxy": subject_xyxy,
                        "obj_bbox_xyxy": object_xyxy,
                        "union_bbox_xyxy": union_box_xyxy,
                        "subj_bbox_xyxy_norm": kept_local_to_box_norm[subject_local_idx],
                        "obj_bbox_xyxy_norm": kept_local_to_box_norm[object_local_idx],
                        "union_bbox_xyxy_norm": union_box_norm,
                        "geo_feat_12d": geo_feat,
                        "subj_crop_rel_path": f"relation/subj/{relation_record_id:09d}.jpg",
                        "obj_crop_rel_path": f"relation/obj/{relation_record_id:09d}.jpg",
                        "union_crop_rel_path": f"relation/union/{relation_record_id:09d}.jpg",
                    }
                )

    object_jsonl = args.output_dir / f"{args.split_name}_object_manifest.jsonl"
    relation_jsonl = args.output_dir / f"{args.split_name}_relation_manifest.jsonl"
    object_parquet = args.output_dir / f"{args.split_name}_object_manifest.parquet"
    relation_parquet = args.output_dir / f"{args.split_name}_relation_manifest.parquet"

    maybe_write_jsonl(object_records, object_jsonl)
    maybe_write_jsonl(relation_records, relation_jsonl)
    object_parquet_ok = maybe_write_parquet(object_records, object_parquet)
    relation_parquet_ok = maybe_write_parquet(relation_records, relation_parquet)

    summary = {
        "split": args.split_name,
        "h5_path": str(args.h5_path),
        "vocab_path": str(args.vocab_path),
        "image_root": str(args.image_root),
        "num_images_processed": num_images,
        "num_object_records": len(object_records),
        "num_relation_records": len(relation_records),
        "wrote_object_jsonl": str(object_jsonl),
        "wrote_relation_jsonl": str(relation_jsonl),
        "wrote_object_parquet": str(object_parquet) if object_parquet_ok else None,
        "wrote_relation_parquet": str(relation_parquet) if relation_parquet_ok else None,
        "min_object_size_px": args.min_object_size_px,
        "skip_pseudo_labels": bool(args.skip_pseudo_labels),
    }
    with (args.output_dir / f"{args.split_name}_manifest_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
