import unittest

import numpy as np

from signal_fusion import PreparedDataset
from signal_fusion.model_inference import (
    ACCEPT,
    REVIEW_REQUIRED,
    ModelInferenceResult,
    RegionEnsembleInferenceService,
    aggregate_region_predictions,
    evaluate_ensemble_results,
)


LABELS = ("LTE", "WiFi", "DVB-T")


def _result(model_id, probabilities, sample_indices=None, labels=LABELS):
    values = np.asarray(probabilities, dtype=np.float32)
    indices = (
        np.arange(values.shape[0], dtype=np.int64)
        if sample_indices is None
        else np.asarray(sample_indices, dtype=np.int64)
    )
    return ModelInferenceResult(
        source_id="capture",
        model_id=model_id,
        labels=labels,
        logits=np.log(values).astype(np.float32),
        probabilities=values,
        auxiliary_outputs={},
        sample_indices=indices,
        provider="CPU",
    )


class _StaticService:
    def __init__(self, model_id, probabilities):
        self.model_id = model_id
        self.probabilities = probabilities

    def predict(
        self,
        dataset,
        *,
        source_id,
        sample_indices,
        batch_size,
    ):
        del dataset, batch_size
        result = _result(
            self.model_id,
            self.probabilities,
            sample_indices=sample_indices,
        )
        result.source_id = source_id
        return result


class RegionEnsembleAggregationTests(unittest.TestCase):
    def test_unanimous_member_top1_is_accepted(self):
        results = [
            _result("seed44", [[0.8, 0.1, 0.1], [0.6, 0.2, 0.2]]),
            _result("seed45", [[0.7, 0.2, 0.1], [0.9, 0.05, 0.05]]),
            _result("seed46", [[0.6, 0.3, 0.1], [0.8, 0.1, 0.1]]),
        ]

        result = aggregate_region_predictions(9, results)
        payload = result.to_dict()

        self.assertEqual(result.decision_status, ACCEPT)
        self.assertTrue(payload["risk_gate"]["unanimous"])
        self.assertEqual(payload["ensemble"]["region_top3"][0]["label"], "LTE")
        self.assertEqual(len(payload["members"]), 3)
        self.assertEqual(len(payload["members"][0]["window_predictions"]), 2)
        np.testing.assert_allclose(
            result.member_region_probabilities,
            np.asarray(
                [
                    [0.7, 0.15, 0.15],
                    [0.8, 0.125, 0.075],
                    [0.7, 0.2, 0.1],
                ]
            ),
        )
        np.testing.assert_allclose(
            result.ensemble_probabilities,
            np.asarray([0.7333333333, 0.1583333333, 0.1083333333]),
        )

    def test_window_disagreement_does_not_trigger_member_risk_gate(self):
        results = [
            _result("seed44", [[0.9, 0.05, 0.05], [0.2, 0.7, 0.1]]),
            _result("seed45", [[0.8, 0.1, 0.1], [0.5, 0.4, 0.1]]),
            _result("seed46", [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]]),
        ]

        payload = aggregate_region_predictions(9, results).to_dict()

        self.assertEqual(payload["decision_status"], ACCEPT)
        self.assertTrue(payload["risk_gate"]["unanimous"])
        self.assertEqual(
            payload["members"][0]["window_agreement"]["agreeing_windows"],
            1,
        )

    def test_member_top1_disagreement_requires_review(self):
        results = [
            _result("seed44", [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]]),
            _result("seed45", [[0.1, 0.1, 0.8], [0.2, 0.1, 0.7]]),
            _result("seed46", [[0.7, 0.1, 0.2], [0.6, 0.1, 0.3]]),
        ]

        result = aggregate_region_predictions(9, results)
        payload = result.to_dict()

        self.assertEqual(result.decision_status, REVIEW_REQUIRED)
        self.assertFalse(payload["risk_gate"]["unanimous"])
        self.assertEqual(
            payload["risk_gate"]["reason"],
            "member_region_top1_disagreement",
        )
        self.assertEqual(
            payload["risk_gate"]["unique_top1_labels"],
            ["LTE", "DVB-T"],
        )

    def test_members_must_share_labels_and_windows(self):
        first = _result("seed44", [[0.8, 0.1, 0.1]], sample_indices=[2])
        wrong_labels = _result(
            "seed45",
            [[0.8, 0.1, 0.1]],
            sample_indices=[2],
            labels=("DVB-T", "WiFi", "LTE"),
        )
        with self.assertRaisesRegex(ValueError, "label order"):
            aggregate_region_predictions(9, [first, wrong_labels])

        wrong_window = _result(
            "seed45",
            [[0.8, 0.1, 0.1]],
            sample_indices=[3],
        )
        with self.assertRaisesRegex(ValueError, "same windows"):
            aggregate_region_predictions(9, [first, wrong_window])

    def test_members_must_have_unique_ids_and_source(self):
        first = _result("seed44", [[0.8, 0.1, 0.1]])
        duplicate = _result("seed44", [[0.7, 0.2, 0.1]])
        with self.assertRaisesRegex(ValueError, "model_id values must be unique"):
            aggregate_region_predictions(9, [first, duplicate])

        wrong_source = _result("seed45", [[0.7, 0.2, 0.1]])
        wrong_source.source_id = "another-capture"
        with self.assertRaisesRegex(ValueError, "same source_id"):
            aggregate_region_predictions(9, [first, wrong_source])

    def test_single_class_models_have_defined_uncertainty(self):
        results = [
            _result("seed44", [[1.0]], labels=("LoRa",)),
            _result("seed45", [[1.0]], labels=("LoRa",)),
        ]

        uncertainty = aggregate_region_predictions(9, results).to_dict()[
            "ensemble"
        ]["uncertainty"]

        self.assertEqual(uncertainty["top1_probability"], 1.0)
        self.assertEqual(uncertainty["top1_top2_margin"], 1.0)
        self.assertEqual(uncertainty["normalized_entropy"], 0.0)

    def test_exact_probability_tie_uses_same_label_as_risk_gate(self):
        results = [
            _result("seed44", [[0.45, 0.45, 0.1]]),
            _result("seed45", [[0.45, 0.45, 0.1]]),
            _result("seed46", [[0.45, 0.45, 0.1]]),
        ]

        payload = aggregate_region_predictions(9, results).to_dict()

        self.assertEqual(payload["risk_gate"]["unique_top1_labels"], ["LTE"])
        self.assertEqual(payload["ensemble"]["region_top3"][0]["label"], "LTE")
        self.assertEqual(payload["members"][0]["region_top3"][0]["label"], "LTE")


class RegionEnsembleServiceTests(unittest.TestCase):
    def test_predict_group_selects_all_group_windows(self):
        dataset = PreparedDataset(
            X=np.zeros((3, 2, 4), dtype=np.float32),
            source_id="fixture:test",
            meta={
                "seq_len": 4,
                "group_id": np.asarray([7, 7, 8], dtype=np.int64),
                "sample_source_id": np.asarray(
                    ["capture", "capture", "other"]
                ),
                "window_start_sample": np.asarray([0, 4, 0]),
                "window_end_sample": np.asarray([4, 8, 4]),
                "region_start_sample": np.asarray([0, 0, 0]),
                "region_end_sample": np.asarray([8, 8, 4]),
            },
        )
        services = [
            _StaticService("seed44", [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]]),
            _StaticService("seed45", [[0.6, 0.3, 0.1], [0.7, 0.2, 0.1]]),
        ]

        result = RegionEnsembleInferenceService(services).predict_group(
            dataset,
            group_id=7,
        )

        np.testing.assert_array_equal(
            result.member_results[0].sample_indices,
            np.asarray([0, 1], dtype=np.int64),
        )
        self.assertEqual(result.source_id, "capture")
        self.assertEqual(result.decision_status, ACCEPT)

    def test_missing_group_is_rejected(self):
        dataset = PreparedDataset(
            X=np.zeros((1, 2, 4), dtype=np.float32),
            source_id="fixture:test",
            meta={
                "seq_len": 4,
                "group_id": np.asarray([1], dtype=np.int64),
                "window_start_sample": np.asarray([0]),
                "window_end_sample": np.asarray([4]),
                "region_start_sample": np.asarray([0]),
                "region_end_sample": np.asarray([4]),
            },
        )
        service = RegionEnsembleInferenceService(
            [
                _StaticService("seed44", [[0.8, 0.1, 0.1]]),
                _StaticService("seed45", [[0.7, 0.2, 0.1]]),
            ]
        )

        with self.assertRaisesRegex(ValueError, "does not exist"):
            service.predict_group(dataset, group_id=99)

    def test_group_with_multiple_sources_is_rejected(self):
        dataset = PreparedDataset(
            X=np.zeros((2, 2, 4), dtype=np.float32),
            source_id="fixture:test",
            meta={
                "group_id": np.asarray([7, 7]),
                "sample_source_id": np.asarray(["first", "second"]),
                "window_start_sample": np.asarray([0, 4]),
                "window_end_sample": np.asarray([4, 8]),
                "region_start_sample": np.asarray([0, 0]),
                "region_end_sample": np.asarray([8, 8]),
            },
        )
        service = RegionEnsembleInferenceService(
            [
                _StaticService("seed44", [[0.8, 0.1, 0.1]] * 2),
                _StaticService("seed45", [[0.7, 0.2, 0.1]] * 2),
            ]
        )

        with self.assertRaisesRegex(ValueError, "multiple sample_source_id"):
            service.predict_group(dataset, group_id=7)

    def test_incomplete_region_is_rejected(self):
        dataset = PreparedDataset(
            X=np.zeros((2, 2, 4), dtype=np.float32),
            source_id="fixture:test",
            meta={
                "group_id": np.asarray([7, 7]),
                "window_start_sample": np.asarray([0, 5]),
                "window_end_sample": np.asarray([4, 9]),
                "region_start_sample": np.asarray([0, 0]),
                "region_end_sample": np.asarray([9, 9]),
            },
        )
        service = RegionEnsembleInferenceService(
            [
                _StaticService("seed44", [[0.8, 0.1, 0.1]] * 2),
                _StaticService("seed45", [[0.7, 0.2, 0.1]] * 2),
            ]
        )

        with self.assertRaisesRegex(ValueError, "complete non-overlapping"):
            service.predict_group(dataset, group_id=7)


class RegionEnsembleEvaluationTests(unittest.TestCase):
    def test_evaluates_regions_sources_and_unanimous_gate(self):
        dataset = PreparedDataset(
            X=np.zeros((6, 2, 4), dtype=np.float32),
            y=np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64),
            source_id="fixture:test",
            meta={
                "seq_len": 4,
                "group_id": np.asarray([0, 0, 1, 1, 2, 2]),
                "sample_source_id": np.asarray(
                    ["lte", "lte", "wifi", "wifi", "wifi", "wifi"]
                ),
            },
        )
        first = _result(
            "seed44",
            [
                [0.9, 0.05, 0.05],
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.7, 0.2],
                [0.2, 0.7, 0.1],
                [0.2, 0.6, 0.2],
            ],
        )
        second = _result(
            "seed45",
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.2, 0.6, 0.2],
                [0.7, 0.2, 0.1],
                [0.6, 0.3, 0.1],
            ],
        )

        result = evaluate_ensemble_results(dataset, [first, second])

        self.assertEqual(result["region_count"], 3)
        self.assertEqual(result["source_count"], 2)
        self.assertAlmostEqual(
            result["ensemble"]["metrics"]["region"]["accuracy_percent"],
            100.0,
        )
        self.assertEqual(result["risk_gate"]["accept"]["region_count"], 2)
        self.assertEqual(
            result["risk_gate"]["review_required"]["region_count"], 1
        )
        self.assertEqual(result["risk_gate"]["review_group_ids"], [2])
        self.assertEqual(
            result["ensemble"]["per_source"]["wifi"]["region_count"], 2
        )

    def test_requires_complete_in_order_member_predictions(self):
        dataset = PreparedDataset(
            X=np.zeros((2, 2, 4), dtype=np.float32),
            y=np.asarray([0, 0], dtype=np.int64),
            source_id="fixture:test",
            meta={
                "group_id": np.asarray([0, 0]),
                "sample_source_id": np.asarray(["lte", "lte"]),
            },
        )
        first = _result("seed44", [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])
        second = _result(
            "seed45",
            [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]],
            sample_indices=[1, 0],
        )

        with self.assertRaisesRegex(ValueError, "cover the dataset in order"):
            evaluate_ensemble_results(dataset, [first, second])


if __name__ == "__main__":
    unittest.main()
