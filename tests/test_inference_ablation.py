import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval.inference_ablation import (
    apply_fuser_alpha_multiplier,
    apply_fuser_alpha_profile,
    apply_graph_gate_override,
    infer_fuser_resolution,
    parse_fuser_alpha_profile,
    restore_base_fuser_state,
    summarize_fuser_alpha,
)


class FakePositionNet:
    def __init__(self):
        self.override = None

    def set_graph_gate_override(self, value):
        self.override = float(value)

    def clear_graph_gate_override(self):
        self.override = None


class FakeModel:
    def __init__(self):
        self.position_net = FakePositionNet()

    def named_modules(self):
        return []


class FakeFuser:
    def __init__(self):
        self.alpha_attn = torch.nn.Parameter(torch.tensor(0.25))
        self.alpha_dense = torch.nn.Parameter(torch.tensor(0.5))


class FakeFuserModel:
    def __init__(self):
        self.fuser = FakeFuser()
        self.not_fuser = FakeFuser()

    def named_modules(self):
        return [
            ("block.fuser", self.fuser),
            ("block.not_fuser", self.not_fuser),
        ]


class FakeLayeredFuserModel:
    def __init__(self):
        self.fuser64 = FakeFuser()
        self.fuser32 = FakeFuser()
        self.fuser16 = FakeFuser()
        self.fuser8 = FakeFuser()
        self.unknown = FakeFuser()

    def named_modules(self):
        return [
            ("input_blocks.1.1.transformer_blocks.0.fuser", self.fuser64),
            ("input_blocks.4.1.transformer_blocks.0.fuser", self.fuser32),
            ("output_blocks.4.1.transformer_blocks.0.fuser", self.fuser16),
            ("middle_block.1.transformer_blocks.0.fuser", self.fuser8),
            ("custom_blocks.0.transformer_blocks.0.fuser", self.unknown),
        ]


class InferenceAblationTest(unittest.TestCase):
    def test_restore_base_fuser_state_only_replaces_fuser_tensors(self):
        model_state = {
            "block.fuser.weight": "trained-fuser",
            "position_net.graph_gate": "trained-graph",
        }
        base_state = {
            "block.fuser.weight": "official-fuser",
            "position_net.graph_gate": "official-graph",
        }

        restored = restore_base_fuser_state(model_state, base_state)

        self.assertEqual(restored, 1)
        self.assertEqual(model_state["block.fuser.weight"], "official-fuser")
        self.assertEqual(
            model_state["position_net.graph_gate"],
            "trained-graph",
        )

    def test_apply_graph_gate_override_uses_position_net_api(self):
        model = FakeModel()

        apply_graph_gate_override(model, 0.0)
        self.assertEqual(model.position_net.override, 0.0)

        apply_graph_gate_override(model, None)
        self.assertIsNone(model.position_net.override)

    def test_apply_fuser_alpha_multiplier_only_changes_requested_fuser_alpha(self):
        model = FakeFuserModel()

        report = apply_fuser_alpha_multiplier(
            model,
            attn_multiplier=4.0,
            dense_multiplier=1.0,
        )

        self.assertEqual(report["updated"], 1)
        self.assertAlmostEqual(float(model.fuser.alpha_attn), 1.0)
        self.assertAlmostEqual(float(model.fuser.alpha_dense), 0.5)
        self.assertAlmostEqual(float(model.not_fuser.alpha_attn), 0.25)

    def test_summarize_fuser_alpha_reports_tanh_magnitudes(self):
        model = FakeFuserModel()

        summary = summarize_fuser_alpha(model)

        self.assertEqual(summary["count"], 1)
        self.assertIn("mean_abs_tanh_alpha_attn", summary)
        self.assertIn("max_abs_tanh_alpha_attn", summary)

    def test_parse_fuser_alpha_profile_accepts_layer_keys_and_all(self):
        profile = parse_fuser_alpha_profile("64:1.0,32:1.2,16:1.5,8:1.3,all:1.0")

        self.assertEqual(profile[64], 1.0)
        self.assertEqual(profile[32], 1.2)
        self.assertEqual(profile[16], 1.5)
        self.assertEqual(profile[8], 1.3)
        self.assertEqual(profile["all"], 1.0)

    def test_infer_fuser_resolution_from_gligen_module_name(self):
        self.assertEqual(
            infer_fuser_resolution("input_blocks.2.1.transformer_blocks.0.fuser"),
            64,
        )
        self.assertEqual(
            infer_fuser_resolution("output_blocks.7.1.transformer_blocks.0.fuser"),
            32,
        )
        self.assertEqual(
            infer_fuser_resolution("input_blocks.8.1.transformer_blocks.0.fuser"),
            16,
        )
        self.assertEqual(
            infer_fuser_resolution("middle_block.1.transformer_blocks.0.fuser"),
            8,
        )
        self.assertIsNone(infer_fuser_resolution("custom_blocks.0.fuser"))

    def test_apply_fuser_alpha_profile_uses_layer_specific_multipliers(self):
        model = FakeLayeredFuserModel()

        report = apply_fuser_alpha_profile(
            model,
            attn_profile=parse_fuser_alpha_profile("64:1,32:2,16:3,8:4,all:5"),
            dense_profile=parse_fuser_alpha_profile("all:1"),
        )

        self.assertEqual(report["updated"], 5)
        self.assertAlmostEqual(float(model.fuser64.alpha_attn), 0.25)
        self.assertAlmostEqual(float(model.fuser32.alpha_attn), 0.5)
        self.assertAlmostEqual(float(model.fuser16.alpha_attn), 0.75)
        self.assertAlmostEqual(float(model.fuser8.alpha_attn), 1.0)
        self.assertAlmostEqual(float(model.unknown.alpha_attn), 1.25)
        self.assertAlmostEqual(float(model.fuser32.alpha_dense), 0.5)


if __name__ == "__main__":
    unittest.main()
