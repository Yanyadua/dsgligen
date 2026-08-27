import unittest

from scripts.eval.grounding_alpha import (
    build_grounding_alpha_schedule,
    parse_grounding_alpha_type,
)


class GroundingAlphaTest(unittest.TestCase):
    def test_parse_three_stage_ratios(self):
        self.assertEqual(parse_grounding_alpha_type("0.7,0,0.3"), (0.7, 0.0, 0.3))

    def test_rejects_ratios_that_do_not_sum_to_one(self):
        with self.assertRaises(ValueError):
            parse_grounding_alpha_type("0.7,0.1,0.3")

    def test_builds_official_hard_cutoff_schedule(self):
        self.assertEqual(
            build_grounding_alpha_schedule(10, (0.7, 0.0, 0.3)),
            [1.0] * 7 + [0.0] * 3,
        )

    def test_builds_linear_decay_schedule(self):
        schedule = build_grounding_alpha_schedule(10, (0.6, 0.2, 0.2))
        self.assertEqual(schedule[:6], [1.0] * 6)
        self.assertEqual(schedule[-2:], [0.0] * 2)
        self.assertEqual(schedule[6:8], [0.5, 0.0])


if __name__ == "__main__":
    unittest.main()
