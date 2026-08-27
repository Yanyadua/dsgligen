import unittest

from scripts.eval.sample_selection import parse_sample_indices


class SampleSelectionTest(unittest.TestCase):
    def test_parses_fixed_noncontiguous_indices(self):
        self.assertEqual(
            parse_sample_indices("4977,3918,1215", dataset_length=5096),
            [4977, 3918, 1215],
        )

    def test_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            parse_sample_indices("1,1", dataset_length=10)

    def test_rejects_out_of_range_indices(self):
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            parse_sample_indices("0,10", dataset_length=10)


if __name__ == "__main__":
    unittest.main()
