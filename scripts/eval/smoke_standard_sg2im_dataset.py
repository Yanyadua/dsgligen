#!/usr/bin/env python3
"""Read representative clean-SG2IM samples and validate loader invariants."""

import argparse
import json

import torch

from dataset.dataset_vg_scene_graph import VGSceneGraphDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-root", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    dataset = VGSceneGraphDataset(
        image_root=f"{args.h5_root}/images",
        h5_path=f"{args.h5_root}/train.h5",
        vocab_path=f"{args.h5_root}/vocab.json",
        image_size=args.image_size,
        random_crop=False,
        random_flip=False,
        box_transform_mode="gligen",
    )
    report = []
    for index in (0, 123, len(dataset) - 1):
        item = dataset[index]
        active_objects = int(item["masks"].sum().item())
        active_relations = int(item["relation_masks"].sum().item())
        assert item["image"].shape == (3, args.image_size, args.image_size)
        assert active_objects >= 3 and active_relations >= 1
        assert torch.all(item["boxes"] >= 0) and torch.all(item["boxes"] <= 1)
        active_edges = item["relation_edges"][:active_relations]
        assert int(active_edges.min()) >= 0
        assert int(active_edges.max()) < active_objects
        report.append(
            {
                "index": index,
                "image_id": int(item["id"]),
                "active_objects": active_objects,
                "active_relations": active_relations,
                "caption": item["caption"],
            }
        )

    print(json.dumps({"status": "DATASET_SMOKE_OK", "samples": report}, indent=2))


if __name__ == "__main__":
    main()
