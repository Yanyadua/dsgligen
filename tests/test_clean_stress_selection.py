import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval.clean_stress_selection import (
    classify_clean_record,
    dataset_item_to_record,
    select_clean_stress_records,
)


class CleanStressSelectionTest(unittest.TestCase):
    def test_accepts_whole_object_spatial_relation(self):
        record = {
            "index": 7,
            "image_id": 1007,
            "objects": [
                {"text": "person", "box_xyxy": [0.1, 0.1, 0.4, 0.8]},
                {"text": "bike", "box_xyxy": [0.2, 0.5, 0.8, 0.9]},
                {"text": "road", "box_xyxy": [0.0, 0.6, 1.0, 1.0]},
            ],
            "relations": [
                {
                    "subject": 0,
                    "object": 1,
                    "predicate": "riding",
                    "subject_text": "person",
                    "object_text": "bike",
                }
            ],
        }

        verdict = classify_clean_record(record)

        self.assertTrue(verdict.accepted)
        self.assertGreater(verdict.score, 0)

    def test_rejects_part_only_bicycle_case(self):
        record = {
            "index": 210,
            "image_id": 210,
            "objects": [
                {"text": "sky", "box_xyxy": [0.0, 0.0, 1.0, 0.6]},
                {"text": "hill", "box_xyxy": [0.0, 0.5, 1.0, 1.0]},
                {"text": "man", "box_xyxy": [0.2, 0.2, 0.35, 0.55]},
                {"text": "tire", "box_xyxy": [0.2, 0.4, 0.3, 0.5]},
                {"text": "tire", "box_xyxy": [0.3, 0.4, 0.4, 0.5]},
                {"text": "tire", "box_xyxy": [0.4, 0.4, 0.5, 0.5]},
                {"text": "tire", "box_xyxy": [0.5, 0.4, 0.6, 0.5]},
            ],
            "relations": [],
        }

        verdict = classify_clean_record(record)

        self.assertFalse(verdict.accepted)
        self.assertIn("no_clean_relation", verdict.reasons)
        self.assertIn("part_dominant", verdict.reasons)

    def test_selects_high_scoring_unique_records(self):
        records = [
            {
                "index": 1,
                "image_id": 11,
                "objects": [
                    {"text": "person", "box_xyxy": [0.1, 0.1, 0.5, 0.8]},
                    {"text": "skateboard", "box_xyxy": [0.2, 0.7, 0.7, 0.9]},
                ],
                "relations": [
                    {
                        "subject": 0,
                        "object": 1,
                        "predicate": "riding",
                        "subject_text": "person",
                        "object_text": "skateboard",
                    }
                ],
            },
            {
                "index": 2,
                "image_id": 12,
                "objects": [
                    {"text": "person", "box_xyxy": [0.1, 0.1, 0.5, 0.8]},
                    {"text": "tire", "box_xyxy": [0.2, 0.7, 0.3, 0.8]},
                ],
                "relations": [],
            },
            {
                "index": 1,
                "image_id": 11,
                "objects": [
                    {"text": "person", "box_xyxy": [0.1, 0.1, 0.5, 0.8]},
                    {"text": "bike", "box_xyxy": [0.2, 0.7, 0.7, 0.9]},
                ],
                "relations": [
                    {
                        "subject": 0,
                        "object": 1,
                        "predicate": "riding",
                        "subject_text": "person",
                        "object_text": "bike",
                    }
                ],
            },
        ]

        selected = select_clean_stress_records(records, limit=5)

        self.assertEqual([record["index"] for record in selected], [1])

    def test_dataset_item_to_record_uses_real_image_id(self):
        item = {
            "id": 4204,
            "image_path": "/tmp/example.jpg",
            "caption": "A scene with person and bike.",
            "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.8, 0.9]]),
            "masks": torch.tensor([1.0, 1.0]),
            "object_texts": ["person", "bike"],
            "relation_edges": torch.tensor([[0.0, 1.0]]),
            "relation_masks": torch.tensor([1.0]),
            "relation_texts": ["riding"],
        }

        record = dataset_item_to_record(17, item)

        self.assertEqual(record["index"], 17)
        self.assertEqual(record["image_id"], 4204)
        self.assertEqual(record["image_path"], "/tmp/example.jpg")
        self.assertEqual(
            [
                (rel["subject_text"], rel["predicate"], rel["object_text"])
                for rel in record["relations"]
            ],
            [("person", "riding", "bike")],
        )


if __name__ == "__main__":
    unittest.main()
