import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from signal_fusion import PreparedDataset
from signal_fusion.benchmarks.technology_recognition_periodicity import (
    run_location_fold_evaluation,
    select_reliability_margin,
    summarize_periodicity_scores,
)
from signal_fusion.io import write_prepared_dataset


class PeriodicityThresholdTests(unittest.TestCase):
    def test_threshold_rejects_every_validation_error(self):
        scores = {
            "y": np.asarray([0, 0, 2, 2, 1]),
            "prediction": np.asarray([0, 2, 2, 0, 0]),
            "margin": np.asarray([0.4, 0.1, 0.3, 0.2, 0.7]),
        }

        threshold = select_reliability_margin(scores)
        summary = summarize_periodicity_scores(scores, threshold)

        self.assertGreater(threshold, 0.2)
        self.assertEqual(summary["reliable_error_count"], 0)
        self.assertEqual(summary["reliable_count"], 2)
        self.assertEqual(summary["wifi_counterfactual_reliable_count"], 1)


class PeriodicityFoldEvaluationTests(unittest.TestCase):
    @staticmethod
    def _signal(period, length=1024, seed=0):
        rng = np.random.default_rng(seed)
        block = (
            rng.standard_normal(period) + 1j * rng.standard_normal(period)
        ).astype(np.complex64)
        return np.tile(block, int(np.ceil(length / period)))[:length]

    def _write_split(self, path, location, split):
        lte = self._signal(67, seed=1)
        wifi = (
            np.random.default_rng(2).standard_normal(1024)
            + 1j * np.random.default_rng(3).standard_normal(1024)
        ).astype(np.complex64)
        dvbt = self._signal(896, seed=4)
        complex_x = np.stack((lte, wifi, dvbt))
        x = np.stack((complex_x.real, complex_x.imag), axis=1).astype(np.float32)
        source_ids = np.asarray(
            [
                f"tr_{location}_lte_r01",
                f"tr_{location}_wifi_r01",
                f"tr_{location}_dvbt_r01",
            ]
        )
        dataset = PreparedDataset(
            X=x,
            y=np.asarray([0, 1, 2]),
            source_id=f"{location}_{split}",
            meta={
                "group_id": np.arange(3),
                "sample_source_id": source_ids,
                "sample_rate": np.full(3, 1_000_000.0),
            },
        )
        write_prepared_dataset(dataset, path)

    def test_writes_compact_phase_p0_p2_artifacts(self):
        locations = ("gentbrugge", "merelbeke", "rabot", "reep")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold_root = root / "folds"
            for fold_index, location in enumerate(locations, start=1):
                for split in ("train", "validation", "test"):
                    self._write_split(
                        fold_root / f"fold{fold_index}" / f"{split}.npz",
                        location,
                        split,
                    )
            baseline = root / "baseline.json"
            baseline.write_text(
                json.dumps(
                    {
                        "labels": ["LTE", "WiFi", "DVB-T"],
                        "region_count": 10,
                        "source_count": 2,
                        "ensemble": {
                            "metrics": {
                                "region": {
                                    "correct_count": 9,
                                    "error_count": 1,
                                    "accuracy_percent": 90.0,
                                }
                            }
                        },
                        "risk_gate": {
                            "accept": {"region_count": 8, "error_count": 0},
                            "review_required": {
                                "region_count": 2,
                                "error_count": 1,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = run_location_fold_evaluation(
                fold_root,
                baseline,
                root / "output",
                batch_size=2,
            )

            report = result["report"]
            self.assertEqual(
                report["result_type"],
                "technology_recognition_periodicity_phase_p0_p2",
            )
            self.assertEqual(len(report["phase_p2_location_isolated_folds"]), 4)
            self.assertTrue(Path(result["report_path"]).is_file())
            self.assertTrue(Path(result["source_csv_path"]).is_file())
            prediction_path = Path(result["predictions_path"])
            self.assertTrue(prediction_path.is_file())
            with np.load(prediction_path, allow_pickle=False) as predictions:
                self.assertEqual(predictions["correlations"].shape, (12, 3))
                self.assertEqual(predictions["peak_lags"].shape, (12, 3))


if __name__ == "__main__":
    unittest.main()
