import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval.stress_selection import (
    DEFAULT_STRESS_PREDICATES,
    normalize_predicate,
    select_stress_indices,
)


class StressSelectionTest(unittest.TestCase):
    def test_normalizes_predicate_spelling(self):
        self.assertEqual(normalize_predicate("In Front Of"), "in front of")
        self.assertEqual(normalize_predicate("on_side_of"), "on side of")

    def test_selects_indices_by_predicate_group(self):
        records = [
            {"index": 0, "predicates": ["wearing", "near"]},
            {"index": 1, "predicates": ["on", "holding"]},
            {"index": 2, "predicates": ["inside"]},
            {"index": 3, "predicates": ["on top of"]},
            {"index": 4, "predicates": ["behind"]},
        ]

        selected = select_stress_indices(
            records,
            predicate_groups={
                "support": ["on", "on top of"],
                "containment": ["inside"],
            },
            per_group=1,
        )

        self.assertEqual(selected["support"], [1])
        self.assertEqual(selected["containment"], [2])

    def test_default_predicates_include_common_spatial_cases(self):
        for key in ("support", "containment", "vertical", "depth", "interaction"):
            self.assertIn(key, DEFAULT_STRESS_PREDICATES)


if __name__ == "__main__":
    unittest.main()
