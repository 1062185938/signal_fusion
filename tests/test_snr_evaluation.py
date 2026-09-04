import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from signal_fusion import PreparedDataset
from signal_fusion.evaluation import (
    add_complex_awgn,
    classification_metrics,
    evaluate_snr_robustness,
    standardize_iq_windows,
)
from signal_fusion.io import write_prepared_dataset


class SnrNumericalTests(unittest.TestCase):
    def test_standardization_removes_dc_and_sets_complex_rms(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((4, 2, 32)).astype(np.float32)
        x[:, 0, :] += 3.0
        x[:, 1, :] -= 2.0

        standardized = standardize_iq_windows(x)

        np.testing.assert_allclose(
            standardized.mean(axis=2), 0.0, atol=2e-7
        )
        rms = np.sqrt(
            np.mean(np.square(standardized).sum(axis=1), axis=1)
        )
        np.testing.assert_allclose(rms, 1.0, atol=2e-7)

    def test_awgn_is_reproducible_and_hits_requested_incremental_snr(self):
        x = standardize_iq_windows(
            np.random.default_rng(3).standard_normal((5, 2, 64))
        )
        first, first_achieved = add_complex_awgn(
            x, 5.0, rng=np.random.default_rng(44)
        )
        second, second_achieved = add_complex_awgn(
            x, 5.0, rng=np.random.default_rng(44)
        )

        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first_achieved, 5.0, atol=1e-12)
        np.testing.assert_array_equal(first_achieved, second_achieved)
        np.testing.assert_allclose(first.mean(axis=2), 0.0, atol=2e-7)
        rms = np.sqrt(np.mean(np.square(first).sum(axis=1), axis=1))
        np.testing.assert_allclose(rms, 1.0, atol=2e-7)

    def test_metrics_distinguish_window_errors_from_correct_group_votes(self):
        probabilities = np.asarray(
            [[0.9, 0.1], [0.4, 0.6], [0.6, 0.4], [0.1, 0.9]],
            dtype=np.float64,
        )
        result = classification_metrics(
            probabilities,
            labels=np.asarray([0, 0, 1, 1]),
            group_ids=np.asarray([10, 10, 20, 20]),
            source_ids=np.asarray(["source_a", "source_a", "source_b", "source_b"]),
            label_map={"0": "A", "1": "B"},
        )

        self.assertEqual(result["window"]["accuracy_percent"], 50.0)
        self.assertEqual(result["group"]["accuracy_percent"], 100.0)
        self.assertEqual(result["group"]["sample_count"], 2)
        self.assertEqual(result["group"]["vote_method"], "mean_probability")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
class SnrRunnerTests(unittest.TestCase):
    def test_runner_writes_json_and_csv_without_retraining(self):
        import torch

        from signal_fusion.modeling import build_model

        rng = np.random.default_rng(11)
        x = standardize_iq_windows(rng.standard_normal((4, 2, 128)))
        dataset = PreparedDataset(
            X=x,
            y=np.asarray([0, 0, 1, 1], dtype=np.int64),
            source_id="snr_test:test",
            meta={
                "seq_len": 128,
                "split": "test",
                "dataset_id": "snr_test",
                "label_map_json": json.dumps({"0": "A", "1": "B"}),
                "group_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
                "sample_source_id": np.asarray(
                    ["source_a", "source_a", "source_b", "source_b"]
                ),
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            dataset_dir = base / "dataset"
            dataset_dir.mkdir()
            write_prepared_dataset(dataset, dataset_dir / "test.npz")
            model_path = base / "model.pth"
            torch.save(build_model("deepconvnet_1d", 2, 2, 128).state_dict(), model_path)
            output_dir = base / "evaluation"

            result = evaluate_snr_robustness(
                dataset_dir=dataset_dir,
                model_path=model_path,
                output_dir=output_dir,
                snr_db_values=[5.0],
                trials=2,
                seed=9,
                batch_size=2,
                device="cpu",
                plot=False,
            )

            self.assertEqual(result["evaluation_type"], "incremental_complex_awgn_snr_sweep")
            self.assertEqual(len(result["conditions"]), 2)
            self.assertEqual(result["conditions"][1]["aggregate"]["trial_count"], 2)
            self.assertTrue((output_dir / "snr_evaluation.json").is_file())
            self.assertTrue((output_dir / "snr_accuracy.csv").is_file())
            self.assertFalse((output_dir / "snr_accuracy_curve.png").exists())


if __name__ == "__main__":
    unittest.main()
