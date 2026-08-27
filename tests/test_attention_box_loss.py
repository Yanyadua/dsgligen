import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ldm.modules.attention_box_loss import (
    compute_attention_box_loss_from_attention,
    parse_attention_box_layer_weights,
)


class AttentionBoxLossTest(unittest.TestCase):
    def test_concentrated_attention_has_lower_loss_than_uniform_attention(self):
        boxes = torch.tensor([[[0.0, 0.0, 0.5, 0.5]]])
        masks = torch.tensor([[1.0]])
        good_attention = torch.zeros(1, 1, 4, 1)
        good_attention[..., 0, 0] = 1.0
        uniform_attention = torch.full((1, 1, 4, 1), 0.25)

        good_loss = compute_attention_box_loss_from_attention(
            good_attention,
            boxes,
            masks,
            target_inside_ratio=0.75,
        )
        uniform_loss = compute_attention_box_loss_from_attention(
            uniform_attention,
            boxes,
            masks,
            target_inside_ratio=0.75,
        )

        self.assertLess(float(good_loss), float(uniform_loss))
        self.assertAlmostEqual(float(good_loss), 0.0)

    def test_attention_box_loss_is_differentiable(self):
        boxes = torch.tensor([[[0.0, 0.0, 0.5, 0.5]]])
        masks = torch.tensor([[1.0]])
        attention = torch.full((1, 1, 4, 1), 0.25, requires_grad=True)

        loss = compute_attention_box_loss_from_attention(
            attention,
            boxes,
            masks,
            target_inside_ratio=0.75,
        )
        loss.backward()

        self.assertIsNotNone(attention.grad)

    def test_parse_attention_box_layer_weights_accepts_resolution_keys(self):
        weights = parse_attention_box_layer_weights("64:0.0,32:0.5,16:1.0,8:0.8,all:0.1")

        self.assertEqual(weights[64], 0.0)
        self.assertEqual(weights[32], 0.5)
        self.assertEqual(weights[16], 1.0)
        self.assertEqual(weights[8], 0.8)
        self.assertEqual(weights["all"], 0.1)


if __name__ == "__main__":
    unittest.main()
