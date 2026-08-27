import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import einops  # noqa: F401
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("einops is required for ldm.modules.attention tests") from exc

from ldm.modules.attention import GatedSelfAttentionDense, SelfAttention


class AttentionRecorderTest(unittest.TestCase):
    def test_self_attention_records_attention_without_changing_output(self):
        torch.manual_seed(123)
        layer = SelfAttention(query_dim=8, heads=2, dim_head=4)
        layer.eval()
        x = torch.randn(2, 5, 8)

        with torch.no_grad():
            expected = layer(x)
            self.assertIsNone(layer.get_last_attention())

            layer.set_attention_recording(True)
            actual = layer(x)
            recorded = layer.get_last_attention()

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertEqual(tuple(recorded.shape), (2, 2, 5, 5))
        self.assertTrue(torch.allclose(recorded.sum(dim=-1), torch.ones(2, 2, 5), atol=1e-5))

        layer.clear_last_attention()
        self.assertIsNone(layer.get_last_attention())

    def test_self_attention_can_record_differentiable_attention(self):
        torch.manual_seed(124)
        layer = SelfAttention(query_dim=8, heads=2, dim_head=4)
        x = torch.randn(2, 5, 8, requires_grad=True)

        layer.set_attention_recording(True, detach=False)
        output = layer(x)
        recorded = layer.get_last_attention()

        self.assertTrue(recorded.requires_grad)
        (output.mean() + recorded.mean()).backward()
        self.assertIsNotNone(x.grad)

    def test_gated_self_attention_exposes_visual_to_grounding_attention(self):
        torch.manual_seed(456)
        layer = GatedSelfAttentionDense(query_dim=8, context_dim=8, n_heads=2, d_head=4)
        layer.eval()
        layer.set_attention_recording(True)
        x = torch.randn(1, 4, 8)
        objs = torch.randn(1, 3, 8)

        with torch.no_grad():
            _ = layer(x, objs)
            visual_to_grounding = layer.get_visual_to_grounding_attention()

        self.assertEqual(tuple(visual_to_grounding.shape), (1, 2, 4, 3))
        self.assertGreater(float(visual_to_grounding.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
