import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnostics.attention_box_metrics import (
    attention_inside_box_ratio,
    boxes_to_token_masks,
    infer_square_grid,
)


class AttentionBoxMetricsTest(unittest.TestCase):
    def test_infer_square_grid_rejects_non_square_counts(self):
        self.assertEqual(infer_square_grid(16), (4, 4))
        self.assertIsNone(infer_square_grid(18))

    def test_boxes_to_token_masks_uses_token_centers(self):
        boxes = torch.tensor([[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]])
        masks = boxes_to_token_masks(boxes, grid_h=2, grid_w=2)

        self.assertEqual(tuple(masks.shape), (1, 2, 4))
        self.assertEqual(masks[0, 0].tolist(), [True, False, False, False])
        self.assertEqual(masks[0, 1].tolist(), [False, False, False, True])

    def test_attention_inside_box_ratio_scores_localized_grounding_attention(self):
        boxes = torch.tensor([[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]])
        attention = torch.zeros(1, 2, 4, 2)
        attention[:, :, 0, 0] = 0.9
        attention[:, :, 3, 0] = 0.1
        attention[:, :, 0, 1] = 0.2
        attention[:, :, 3, 1] = 0.8

        ratio, area_ratio = attention_inside_box_ratio(attention, boxes)

        self.assertTrue(torch.allclose(ratio, torch.tensor([[0.9, 0.8]]), atol=1e-6))
        self.assertTrue(torch.allclose(area_ratio, torch.tensor([[0.25, 0.25]]), atol=1e-6))

    def test_attention_inside_box_ratio_masks_invalid_grounding_tokens(self):
        boxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]])
        attention = torch.ones(1, 1, 4, 2)
        ratio, area_ratio = attention_inside_box_ratio(
            attention,
            boxes,
            grounding_mask=torch.tensor([[1.0, 0.0]]),
        )

        self.assertEqual(float(ratio[0, 0]), 1.0)
        self.assertTrue(torch.isnan(ratio[0, 1]))
        self.assertTrue(torch.isnan(area_ratio[0, 1]))


if __name__ == "__main__":
    unittest.main()
