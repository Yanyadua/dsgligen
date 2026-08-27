import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.relation_grounding_tokens import (
    append_relation_grounding_tokens,
    normalize_predicate,
    parse_allowed_predicates,
)


class RelationGroundingTokensTest(unittest.TestCase):
    def test_appends_union_box_relation_phrase_and_role(self):
        boxes = torch.tensor(
            [
                [0.10, 0.20, 0.30, 0.40],
                [0.25, 0.35, 0.80, 0.90],
            ],
            dtype=torch.float32,
        )
        masks = torch.tensor([1.0, 1.0])
        object_texts = ["person", "skateboard"]
        relation_edges = torch.tensor([[0, 1]], dtype=torch.float32)
        relation_masks = torch.tensor([1.0])
        relation_texts = ["on"]

        result = append_relation_grounding_tokens(
            boxes=boxes,
            masks=masks,
            object_texts=object_texts,
            relation_edges=relation_edges,
            relation_masks=relation_masks,
            relation_texts=relation_texts,
            max_relation_tokens=2,
        )

        self.assertEqual(result.object_texts, ["person", "skateboard", "person on skateboard", ""])
        self.assertEqual(result.token_roles, ["object", "object", "relation", "padding"])
        torch.testing.assert_close(
            result.boxes[2],
            torch.tensor([0.10, 0.20, 0.80, 0.90]),
        )
        self.assertEqual(result.masks.tolist(), [1.0, 1.0, 1.0, 0.0])
        self.assertEqual(result.relation_token_source.tolist(), [0, -1])

    def test_skips_invalid_or_padded_relations_but_keeps_fixed_capacity(self):
        boxes = torch.tensor(
            [
                [0.0, 0.0, 0.2, 0.2],
                [0.4, 0.4, 0.6, 0.6],
            ],
            dtype=torch.float32,
        )
        masks = torch.tensor([1.0, 0.0])
        object_texts = ["cup", ""]
        relation_edges = torch.tensor([[0, 1], [0, 99]], dtype=torch.float32)
        relation_masks = torch.tensor([1.0, 1.0])
        relation_texts = ["on", "near"]

        result = append_relation_grounding_tokens(
            boxes=boxes,
            masks=masks,
            object_texts=object_texts,
            relation_edges=relation_edges,
            relation_masks=relation_masks,
            relation_texts=relation_texts,
            max_relation_tokens=2,
        )

        self.assertEqual(result.object_texts, ["cup", "", "", ""])
        self.assertEqual(result.token_roles, ["object", "padding", "padding", "padding"])
        self.assertEqual(result.masks.tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(result.relation_token_source.tolist(), [-1, -1])

    def test_zero_capacity_returns_original_tensors_and_roles(self):
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
        masks = torch.tensor([1.0])

        result = append_relation_grounding_tokens(
            boxes=boxes,
            masks=masks,
            object_texts=["table"],
            relation_edges=torch.zeros(0, 2),
            relation_masks=torch.zeros(0),
            relation_texts=[],
            max_relation_tokens=0,
        )

        self.assertIs(result.boxes, boxes)
        self.assertIs(result.masks, masks)
        self.assertEqual(result.object_texts, ["table"])
        self.assertEqual(result.token_roles, ["object"])
        self.assertEqual(result.relation_token_source.numel(), 0)

    def test_relation_mask_scale_controls_appended_relation_strength(self):
        boxes = torch.tensor(
            [
                [0.10, 0.20, 0.30, 0.40],
                [0.25, 0.35, 0.80, 0.90],
            ],
            dtype=torch.float32,
        )
        masks = torch.tensor([1.0, 1.0])

        result = append_relation_grounding_tokens(
            boxes=boxes,
            masks=masks,
            object_texts=["person", "skateboard"],
            relation_edges=torch.tensor([[0, 1]], dtype=torch.float32),
            relation_masks=torch.tensor([1.0]),
            relation_texts=["on"],
            max_relation_tokens=1,
            relation_mask_scale=0.25,
        )

        self.assertEqual(result.object_texts[2], "person on skateboard")
        self.assertEqual(result.token_roles[2], "relation")
        torch.testing.assert_close(result.masks, torch.tensor([1.0, 1.0, 0.25]))

    def test_fractional_base_mask_is_still_active(self):
        boxes = torch.tensor(
            [
                [0.10, 0.20, 0.30, 0.40],
                [0.00, 0.50, 1.00, 0.90],
            ],
            dtype=torch.float32,
        )
        masks = torch.tensor([1.0, 0.4])

        result = append_relation_grounding_tokens(
            boxes=boxes,
            masks=masks,
            object_texts=["car", "road"],
            relation_edges=torch.tensor([[0, 1]], dtype=torch.float32),
            relation_masks=torch.tensor([1.0]),
            relation_texts=["on"],
            max_relation_tokens=1,
            relation_mask_scale=0.5,
        )

        self.assertEqual(result.token_roles, ["object", "object", "relation"])
        self.assertEqual(result.object_texts[2], "car on road")
        torch.testing.assert_close(result.masks, torch.tensor([1.0, 0.4, 0.5]))

    def test_filters_deduplicates_and_limits_relation_tokens(self):
        boxes = torch.tensor(
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.1, 0.3, 0.9, 0.9],
                [0.5, 0.1, 0.9, 0.5],
            ],
            dtype=torch.float32,
        )
        masks = torch.ones(3)
        object_texts = ["car", "road", "person"]
        relation_edges = torch.tensor(
            [
                [0, 1],
                [0, 1],
                [1, 2],
                [2, 0],
            ],
            dtype=torch.float32,
        )
        relation_masks = torch.ones(4)
        relation_texts = ["On_Top_Of", "on top of", "has", "holding"]

        result = append_relation_grounding_tokens(
            boxes=boxes,
            masks=masks,
            object_texts=object_texts,
            relation_edges=relation_edges,
            relation_masks=relation_masks,
            relation_texts=relation_texts,
            max_relation_tokens=2,
            phrase_template="spatial relation: {subject} {predicate} {object}",
            allowed_predicates=["on top of", "holding"],
            deduplicate=True,
        )

        self.assertEqual(
            result.object_texts,
            [
                "car",
                "road",
                "person",
                "spatial relation: car On_Top_Of road",
                "spatial relation: person holding car",
            ],
        )
        self.assertEqual(result.token_roles, ["object", "object", "object", "relation", "relation"])
        self.assertEqual(result.relation_token_source.tolist(), [0, 3])

    def test_parse_and_normalize_predicates(self):
        self.assertEqual(normalize_predicate(" On_Top_Of  "), "on top of")
        self.assertEqual(
            parse_allowed_predicates("on, on top of,holding"),
            {"on", "on top of", "holding"},
        )
        self.assertIsNone(parse_allowed_predicates(""))


if __name__ == "__main__":
    unittest.main()
