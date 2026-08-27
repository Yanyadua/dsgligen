import unittest

import torch

from scripts.eval.deterministic_noise import build_per_sample_noise


class DeterministicEvalNoiseTest(unittest.TestCase):
    def test_same_image_ids_produce_same_noise_across_batch_order(self):
        first = build_per_sample_noise(
            image_ids=[10, 113],
            sample_shape=(4, 8, 8),
            base_seed=20260429,
        )
        second = build_per_sample_noise(
            image_ids=[113, 10],
            sample_shape=(4, 8, 8),
            base_seed=20260429,
        )

        torch.testing.assert_close(first[0], second[1])
        torch.testing.assert_close(first[1], second[0])

    def test_different_image_ids_produce_different_noise(self):
        noise = build_per_sample_noise(
            image_ids=[10, 11],
            sample_shape=(4, 8, 8),
            base_seed=123,
        )

        self.assertFalse(torch.equal(noise[0], noise[1]))


if __name__ == "__main__":
    unittest.main()
