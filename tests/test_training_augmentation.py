import unittest

import torch

from signal_fusion.training.augmentation import add_random_complex_awgn


class TrainingAugmentationTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(17)
        self.inputs = torch.randn(8, 2, 128, generator=generator)

    def test_zero_probability_returns_clean_batch_unchanged(self):
        augmented = add_random_complex_awgn(
            self.inputs,
            probability=0.0,
            snr_min_db=5.0,
            snr_max_db=20.0,
        )

        self.assertIs(augmented, self.inputs)
        torch.testing.assert_close(augmented, self.inputs)

    def test_full_probability_is_reproducible_and_does_not_mutate_input(self):
        original = self.inputs.clone()
        torch.manual_seed(44)
        first = add_random_complex_awgn(
            self.inputs,
            probability=1.0,
            snr_min_db=5.0,
            snr_max_db=20.0,
        )
        torch.manual_seed(44)
        second = add_random_complex_awgn(
            self.inputs,
            probability=1.0,
            snr_min_db=5.0,
            snr_max_db=20.0,
        )

        torch.testing.assert_close(self.inputs, original)
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, self.inputs))

    def test_noisy_windows_are_dc_removed_and_complex_rms_normalized(self):
        torch.manual_seed(44)
        augmented = add_random_complex_awgn(
            self.inputs,
            probability=1.0,
            snr_min_db=10.0,
            snr_max_db=10.0,
        )

        dc = augmented.mean(dim=2)
        rms = torch.sqrt(augmented.square().sum(dim=1).mean(dim=1))
        torch.testing.assert_close(dc, torch.zeros_like(dc), atol=1e-6, rtol=0)
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-6, rtol=0)

    def test_invalid_configuration_is_rejected(self):
        invalid_options = [
            {"probability": -0.1, "snr_min_db": 5.0, "snr_max_db": 20.0},
            {"probability": 1.1, "snr_min_db": 5.0, "snr_max_db": 20.0},
            {"probability": 0.5, "snr_min_db": 20.0, "snr_max_db": 5.0},
        ]
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                add_random_complex_awgn(self.inputs, **options)


if __name__ == "__main__":
    unittest.main()
