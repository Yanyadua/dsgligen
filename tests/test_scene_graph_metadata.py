import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval.scene_graph_metadata import (
    build_sample_metadata,
    write_sample_metadata,
)


class SceneGraphMetadataTest(unittest.TestCase):
    def test_builds_active_objects_and_relations_only(self):
        item = {
            "id": 123,
            "image_path": "/tmp/vg/123.jpg",
            "caption": "a person on a skateboard",
            "boxes": torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.2, 0.5, 0.7, 0.8],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.7, 0.8],
                ]
            ),
            "masks": torch.tensor([1.0, 1.0, 0.0, 0.5]),
            "object_texts": ["person", "skateboard", "", "person on skateboard"],
            "grounding_token_roles": ["object", "object", "padding", "relation"],
            "relation_token_source": torch.tensor([0]),
            "relation_edges": torch.tensor([[0, 1], [1, 2]]),
            "relation_masks": torch.tensor([1.0, 0.0]),
            "relation_texts": ["on", "near"],
            "relation_geo_features": torch.tensor(
                [
                    [0.0] * 12,
                    [1.0] * 12,
                ]
            ),
        }

        metadata = build_sample_metadata(
            item,
            dataset_index=7,
            generation={
                "sampler": "ddim",
                "steps": 50,
                "guidance": 5.0,
                "seed": 20260508,
            },
        )

        self.assertEqual(metadata["dataset_index"], 7)
        self.assertEqual(metadata["image_id"], 123)
        self.assertEqual(metadata["generation"]["sampler"], "ddim")
        self.assertEqual(
            [obj["text"] for obj in metadata["objects"]],
            ["person", "skateboard", "person on skateboard"],
        )
        self.assertEqual(
            [obj["role"] for obj in metadata["objects"]],
            ["object", "object", "relation"],
        )
        self.assertEqual(metadata["relation_token_source"], [0])
        self.assertEqual(len(metadata["relations"]), 1)
        self.assertEqual(metadata["relations"][0]["predicate"], "on")
        self.assertEqual(metadata["relations"][0]["subject_text"], "person")
        self.assertEqual(metadata["relations"][0]["object_text"], "skateboard")

    def test_writes_one_json_file_per_image(self):
        item = {
            "id": 456,
            "image_path": "/tmp/vg/456.jpg",
            "caption": "a cup on a table",
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "masks": torch.tensor([1.0]),
            "object_texts": ["cup"],
            "relation_edges": torch.zeros(0, 2),
            "relation_masks": torch.zeros(0),
            "relation_texts": [],
            "relation_geo_features": torch.zeros(0, 12),
        }

        with tempfile.TemporaryDirectory() as tmp:
            out_path = write_sample_metadata(
                Path(tmp),
                item,
                dataset_index=3,
                generation={"sampler": "plms"},
            )
            payload = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(out_path.name, "456.json")
        self.assertEqual(payload["image_id"], 456)
        self.assertEqual(payload["objects"][0]["text"], "cup")


if __name__ == "__main__":
    unittest.main()
