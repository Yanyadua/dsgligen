import unittest

import torch

from trainer import (
    build_frozen_quality_teacher,
    is_trainable_fuser_parameter,
)


class TinyPositionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.override = None

    def set_graph_gate_override(self, value):
        self.override = float(value)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.position_net = TinyPositionNet()
        self.linear = torch.nn.Linear(1, 1)


class QualityTeacherTrainingTest(unittest.TestCase):
    def test_gates_only_selects_only_fuser_alpha_parameters(self):
        self.assertTrue(
            is_trainable_fuser_parameter(
                "input_blocks.1.fuser.alpha_attn",
                "gates_only",
            )
        )
        self.assertTrue(
            is_trainable_fuser_parameter(
                "output_blocks.1.fuser.alpha_dense",
                "gates_only",
            )
        )
        self.assertFalse(
            is_trainable_fuser_parameter(
                "input_blocks.1.fuser.attn.to_q.weight",
                "gates_only",
            )
        )
        self.assertFalse(
            is_trainable_fuser_parameter(
                "input_blocks.1.fuser.linear.weight",
                "gates_only",
            )
        )
        self.assertTrue(
            is_trainable_fuser_parameter(
                "input_blocks.1.fuser.linear.weight",
                "gates_and_linear",
            )
        )
        self.assertTrue(
            is_trainable_fuser_parameter(
                "input_blocks.1.fuser.linear.bias",
                "gates_and_linear",
            )
        )
        self.assertTrue(
            is_trainable_fuser_parameter(
                "input_blocks.1.fuser.alpha_attn",
                "gates_and_linear",
            )
        )
        self.assertFalse(
            is_trainable_fuser_parameter(
                "input_blocks.1.fuser.attn.to_q.weight",
                "gates_and_linear",
            )
        )
        self.assertFalse(
            is_trainable_fuser_parameter(
                "input_blocks.1.fuser.ff.net.2.weight",
                "gates_and_linear",
            )
        )
        self.assertTrue(
            is_trainable_fuser_parameter(
                "input_blocks.1.fuser.linear.weight",
                "full",
            )
        )
        self.assertFalse(
            is_trainable_fuser_parameter(
                "input_blocks.1.fuser.alpha_attn",
                "frozen",
            )
        )

    def test_frozen_teacher_is_independent_and_disables_graph_residual(self):
        student = TinyModel()

        teacher = build_frozen_quality_teacher(student)

        self.assertIsNot(teacher, student)
        self.assertFalse(teacher.training)
        self.assertEqual(teacher.position_net.override, 0.0)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))

        with torch.no_grad():
            student.linear.weight.add_(1.0)
        self.assertFalse(
            torch.equal(student.linear.weight, teacher.linear.weight)
        )


if __name__ == "__main__":
    unittest.main()
