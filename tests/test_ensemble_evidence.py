import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from signal_fusion import PreparedDataset
from signal_fusion.evaluation.snr import standardize_iq_windows
from signal_fusion.feature_extraction import (
    FEATURE_COUNT,
    PERIODICITY_SCHEMA_ID,
    feature_code_names,
    load_feature_map,
)
from signal_fusion.fusion import (
    REFERENCE_DOCUMENTS,
    build_evidence_records,
    rebuild_continuous_region_dataset,
)
from signal_fusion.fusion.periodicity_gate import (
    TECHNOLOGY_PERIODICITY_CANDIDATES,
)
from signal_fusion.fusion.references import technology_reference_path
from signal_fusion.io import load_prepared_dataset, write_prepared_dataset
from signal_fusion.model_inference import (
    ModelInferenceResult,
    RegionEnsembleInferenceService,
)


LABELS = ("LTE", "WiFi", "DVB-T")
FEATURE_NAMES = feature_code_names(load_feature_map())


def _iq_x(windows):
    values = np.asarray(windows, dtype=np.complex64)
    return np.stack((values.real, values.imag), axis=1).astype(np.float32)


def _all_keys(value):
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_all_keys(item))
        return keys
    if isinstance(value, list):
        keys = set()
        for item in value:
            keys.update(_all_keys(item))
        return keys
    return set()


def _write_gate_manifest(path, reliability_margin=0.01):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "result_type": "technology_recognition_periodicity_gate",
                "labels": list(LABELS),
                "sample_rate_hz": 1_000_000,
                "region_sample_count": 4096,
                "periodicity_schema_id": PERIODICITY_SCHEMA_ID,
                "candidates": [
                    {
                        "candidate_id": item.candidate_id,
                        "period_seconds": item.period_seconds,
                    }
                    for item in TECHNOLOGY_PERIODICITY_CANDIDATES
                ],
                "frozen_rule": {
                    "reliability_margin": reliability_margin,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _feature_identity(dataset):
    return {
        "source_dataset_id": dataset.source_id,
        "group_id": np.asarray(dataset.meta["group_id"]),
        "sample_source_id": np.asarray(dataset.meta["sample_source_id"]),
        "source_region_id": np.asarray(dataset.meta["source_region_id"]),
    }


class _StaticService:
    def __init__(self, model_id, probabilities):
        self.model_id = model_id
        self.probabilities = np.asarray(probabilities, dtype=np.float32)

    def predict(
        self,
        dataset,
        *,
        source_id=None,
        batch_size=64,
        max_samples=None,
        sample_indices=None,
    ):
        del batch_size, max_samples
        indices = (
            np.arange(dataset.num_samples, dtype=np.int64)
            if sample_indices is None
            else np.asarray(sample_indices, dtype=np.int64)
        )
        probabilities = self.probabilities[indices]
        return ModelInferenceResult(
            source_id=source_id or dataset.source_id,
            model_id=self.model_id,
            labels=LABELS,
            logits=np.log(probabilities).astype(np.float32),
            probabilities=probabilities,
            auxiliary_outputs={},
            sample_indices=indices,
            provider="fixture",
        )


class ContinuousRegionDatasetTests(unittest.TestCase):
    def test_rebuilds_raw_windows_then_standardizes_whole_region_once(self):
        region_0 = np.asarray(
            [1 + 1j, 2 + 1j, 3 + 2j, 4 + 2j, 5 + 3j, 6 + 3j, 7 + 4j, 8 + 4j],
            dtype=np.complex64,
        )
        region_1 = np.asarray(
            [2 + 8j, 3 + 7j, 5 + 6j, 7 + 5j, 11 + 4j, 13 + 3j, 17 + 2j, 19 + 1j],
            dtype=np.complex64,
        )
        source_x = _iq_x(
            [region_0[:4], region_0[4:], region_1[:4], region_1[4:]]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared_root = root / "prepared"
            prepared = PreparedDataset(
                X=source_x,
                y=np.asarray([0, 0, 1, 1], dtype=np.int64),
                source_id="capture",
                meta={
                    "normalization": "none",
                    "remove_dc": False,
                    "coordinate_schema": "dual_rate_v1",
                    "region_id": np.asarray([0, 0, 1, 1]),
                    "target_window_start_sample": np.asarray([0, 4, 8, 12]),
                    "target_window_end_sample": np.asarray([4, 8, 12, 16]),
                },
            )
            write_prepared_dataset(prepared, prepared_root / "site" / "capture.npz")

            assembled = PreparedDataset(
                X=np.zeros((4, 2, 4), dtype=np.float32),
                y=np.asarray([0, 0, 1, 1], dtype=np.int64),
                source_id="assembled",
                meta={
                    "group_id": np.asarray([3, 3, 8, 8]),
                    "sample_source_id": np.asarray(["capture"] * 4),
                    "source_region_id": np.asarray([0, 0, 1, 1]),
                    "target_window_start_sample": np.asarray([0, 4, 8, 12]),
                    "target_window_end_sample": np.asarray([4, 8, 12, 16]),
                    "target_region_start_sample": np.asarray([0, 0, 8, 8]),
                    "target_region_end_sample": np.asarray([8, 8, 16, 16]),
                    "target_sample_rate": np.full(4, 1_000_000.0),
                },
            )
            dataset_path = write_prepared_dataset(
                assembled, root / "test.npz"
            )
            output_path = root / "regions.npz"

            report = rebuild_continuous_region_dataset(
                dataset_path,
                prepared_root,
                output_path,
            )
            rebuilt = load_prepared_dataset(output_path)

        raw = _iq_x([region_0, region_1])
        expected = standardize_iq_windows(raw)
        np.testing.assert_allclose(rebuilt.X, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(rebuilt.y, np.asarray([0, 1]))
        np.testing.assert_array_equal(rebuilt.meta["group_id"], np.asarray([3, 8]))
        self.assertEqual(report["shape"], [2, 2, 8])
        self.assertEqual(rebuilt.meta["normalization_scope"], "complete_region")

    def test_rejects_a_prepared_source_that_was_already_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = PreparedDataset(
                X=np.ones((2, 2, 4), dtype=np.float32),
                source_id="capture",
                meta={
                    "normalization": "rms",
                    "remove_dc": False,
                    "coordinate_schema": "dual_rate_v1",
                    "region_id": np.asarray([0, 0]),
                    "target_window_start_sample": np.asarray([0, 4]),
                    "target_window_end_sample": np.asarray([4, 8]),
                },
            )
            write_prepared_dataset(prepared, root / "prepared" / "capture.npz")
            assembled = PreparedDataset(
                X=np.zeros((2, 2, 4), dtype=np.float32),
                source_id="assembled",
                meta={
                    "group_id": np.asarray([0, 0]),
                    "sample_source_id": np.asarray(["capture", "capture"]),
                    "source_region_id": np.asarray([0, 0]),
                    "target_window_start_sample": np.asarray([0, 4]),
                    "target_window_end_sample": np.asarray([4, 8]),
                    "target_region_start_sample": np.asarray([0, 0]),
                    "target_region_end_sample": np.asarray([8, 8]),
                    "target_sample_rate": np.full(2, 1_000_000.0),
                },
            )
            dataset_path = write_prepared_dataset(assembled, root / "test.npz")

            with self.assertRaisesRegex(ValueError, "unnormalized"):
                rebuild_continuous_region_dataset(
                    dataset_path,
                    root / "prepared",
                    root / "regions.npz",
                )


class EnsembleEvidenceTests(unittest.TestCase):
    def test_reference_documents_include_feature_definitions_and_class_context(self):
        self.assertEqual(
            REFERENCE_DOCUMENTS,
            (
                "time_domain_iq_features.md",
                "frequency_domain_iq_features.md",
                "time_frequency_iq_features.md",
                "technology_reference_lte_wifi_dvbt.md",
            ),
        )
        reference = technology_reference_path()
        self.assertTrue(reference.is_file())
        content = reference.read_text(encoding="utf-8")
        self.assertIn("## 2. 当前证据的观测边界", content)
        self.assertIn("## 7. 禁止的推理方式", content)

    def test_blind_records_use_ensemble_and_one_global_feature_vector(self):
        window_dataset = PreparedDataset(
            X=np.zeros((4, 2, 2048), dtype=np.float32),
            y=np.asarray([0, 0, 1, 1], dtype=np.int64),
            source_id="window-test",
            meta={
                "group_id": np.asarray([3, 3, 8, 8]),
                "sample_source_id": np.asarray(["secret_a", "secret_a", "secret_b", "secret_b"]),
                "source_region_id": np.asarray([10, 10, 20, 20]),
                "window_start_sample": np.asarray([0, 2048, 0, 2048]),
                "window_end_sample": np.asarray([2048, 4096, 2048, 4096]),
                "region_start_sample": np.asarray([0, 0, 0, 0]),
                "region_end_sample": np.asarray([4096, 4096, 4096, 4096]),
            },
        )
        region_dataset = PreparedDataset(
            X=np.zeros((2, 2, 4096), dtype=np.float32),
            y=np.asarray([0, 1], dtype=np.int64),
            source_id="region-test",
            meta={
                "group_id": np.asarray([3, 8]),
                "sample_source_id": np.asarray(["secret_a", "secret_b"]),
                "source_region_id": np.asarray([10, 20]),
                "sample_rate": np.asarray([1_000_000.0, 1_000_000.0]),
                "sample_location": np.asarray(["site_a", "site_b"]),
                "gain_db": np.asarray([0, 30]),
                "center_frequency": np.asarray([806e6, 2.4e9]),
                "remove_dc": np.asarray(True),
                "rms_normalize": np.asarray(True),
            },
        )
        probabilities = [
            [0.90, 0.05, 0.05],
            [0.80, 0.10, 0.10],
            [0.05, 0.90, 0.05],
            [0.10, 0.80, 0.10],
        ]
        service = RegionEnsembleInferenceService(
            [
                _StaticService("private_seed_44", probabilities),
                _StaticService("private_seed_45", probabilities),
                _StaticService("private_seed_46", probabilities),
            ]
        )
        features = np.arange(2 * FEATURE_COUNT, dtype=np.float32).reshape(2, -1)
        names = FEATURE_NAMES

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_gate_manifest(
                Path(directory) / "periodicity_gate_manifest.json"
            )
            blind, audit = build_evidence_records(
                window_dataset,
                region_dataset,
                features,
                names,
                _feature_identity(region_dataset),
                service,
                manifest_path,
            )

        self.assertEqual(len(blind), 2)
        self.assertEqual(len(audit), 2)
        self.assertEqual(blind[0]["schema_version"], 3)
        self.assertEqual(blind[0]["bundle_type"], "hermes_signal_fusion_input")
        self.assertEqual(blind[0]["analysis_id"], "case_0001")
        self.assertEqual(
            blind[0]["iq_ensemble_evidence"]["ensemble"]["region_top3"][0][
                "label"
            ],
            "LTE",
        )
        self.assertEqual(
            blind[0]["global_feature_evidence"]["extraction_count"], 1
        )
        self.assertEqual(
            len(blind[0]["global_feature_evidence"]["values"]), FEATURE_COUNT
        )
        self.assertEqual(
            blind[0]["global_feature_evidence"]["reference_documents"],
            list(REFERENCE_DOCUMENTS),
        )
        reference_paths = blind[0]["global_feature_evidence"][
            "reference_document_paths"
        ]
        self.assertEqual(set(reference_paths), set(REFERENCE_DOCUMENTS))
        self.assertTrue(all(Path(path).is_file() for path in reference_paths.values()))
        self.assertTrue(
            blind[0]["decision_contract"]["result_is_frozen"]
        )
        self.assertEqual(
            blind[0]["deterministic_fusion_result"]["decision_status"],
            "accept",
        )
        self.assertEqual(
            blind[0]["deterministic_fusion_result"]["final_label"], "LTE"
        )
        self.assertIsNone(
            blind[0]["deterministic_fusion_result"]["provisional_label"]
        )
        periodicity_reference = Path(
            blind[0]["periodicity_evidence"]["reference_document_path"]
        )
        self.assertTrue(periodicity_reference.is_file())
        self.assertEqual(
            blind[0]["periodicity_evidence"]["frozen_gate"][
                "reliability_margin"
            ],
            0.01,
        )
        self.assertIn("required_output", blind[0])
        self.assertEqual(
            [member["member_id"] for member in blind[0]["iq_ensemble_evidence"]["members"]],
            ["iq_member_1", "iq_member_2", "iq_member_3"],
        )
        forbidden = {
            "group_id",
            "source_id",
            "source_path",
            "true_label",
            "true_class_index",
            "location",
            "gain_db",
            "center_frequency",
            "center_frequency_hz",
        }
        self.assertTrue(forbidden.isdisjoint(_all_keys(blind)))
        serialized = json.dumps(blind)
        self.assertNotIn("secret_a", serialized)
        self.assertNotIn("site_a", serialized)
        self.assertEqual(audit[0]["source_id"], "secret_a")
        self.assertEqual(audit[0]["true_label"], "LTE")

    def test_periodicity_can_resolve_lte_dvbt_but_not_wifi_conflicts(self):
        rng = np.random.default_rng(7)
        block = (
            rng.standard_normal(67) + 1j * rng.standard_normal(67)
        ).astype(np.complex64)
        periodic_signal = np.tile(block, 62)[:4096]
        window_dataset = PreparedDataset(
            X=np.zeros((4, 2, 2048), dtype=np.float32),
            source_id="window-test",
            meta={
                "group_id": np.asarray([3, 3, 8, 8]),
                "sample_source_id": np.asarray(["a", "a", "b", "b"]),
                "source_region_id": np.asarray([1, 1, 2, 2]),
                "window_start_sample": np.asarray([0, 2048, 0, 2048]),
                "window_end_sample": np.asarray([2048, 4096, 2048, 4096]),
                "region_start_sample": np.asarray([0, 0, 0, 0]),
                "region_end_sample": np.asarray([4096, 4096, 4096, 4096]),
            },
        )
        region_dataset = PreparedDataset(
            X=_iq_x([periodic_signal, periodic_signal]),
            source_id="region-test",
            meta={
                "group_id": np.asarray([3, 8]),
                "sample_source_id": np.asarray(["a", "b"]),
                "source_region_id": np.asarray([1, 2]),
                "sample_rate": np.full(2, 1_000_000.0),
                "remove_dc": np.asarray(True),
                "rms_normalize": np.asarray(True),
            },
        )
        dvbt = [[0.05, 0.05, 0.90]] * 4
        lte = [[0.90, 0.05, 0.05]] * 4
        wifi = [[0.05, 0.90, 0.05]] * 4
        service = RegionEnsembleInferenceService(
            [
                _StaticService("m1", dvbt),
                _StaticService("m2", lte[:2] + wifi[2:]),
                _StaticService("m3", dvbt[:2] + wifi[2:]),
            ]
        )
        features = np.zeros((2, FEATURE_COUNT), dtype=np.float32)
        names = FEATURE_NAMES

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_gate_manifest(
                Path(directory) / "periodicity_gate_manifest.json",
                reliability_margin=0.01,
            )
            blind, _ = build_evidence_records(
                window_dataset,
                region_dataset,
                features,
                names,
                _feature_identity(region_dataset),
                service,
                manifest_path,
            )

        changed = blind[0]["deterministic_fusion_result"]
        self.assertEqual(changed["ensemble_label"], "DVB-T")
        self.assertEqual(changed["resolution"], "periodicity_changed_label")
        self.assertEqual(changed["decision_status"], "accept")
        self.assertEqual(changed["final_label"], "LTE")

        review = blind[1]["deterministic_fusion_result"]
        self.assertEqual(review["ensemble_label"], "WiFi")
        self.assertEqual(review["resolution"], "unresolved_review")
        self.assertEqual(review["decision_status"], "review_required")
        self.assertIsNone(review["final_label"])
        self.assertEqual(review["provisional_label"], "WiFi")
        self.assertFalse(
            blind[1]["periodicity_evidence"]["frozen_gate"]["eligible"]
        )

    def test_rejects_misaligned_feature_rows_and_noncanonical_names(self):
        window_dataset = PreparedDataset(
            X=np.zeros((2, 2, 2048), dtype=np.float32),
            source_id="windows",
            meta={
                "group_id": np.asarray([3, 3]),
                "sample_source_id": np.asarray(["capture", "capture"]),
                "source_region_id": np.asarray([7, 7]),
                "window_start_sample": np.asarray([0, 2048]),
                "window_end_sample": np.asarray([2048, 4096]),
                "region_start_sample": np.asarray([0, 0]),
                "region_end_sample": np.asarray([4096, 4096]),
            },
        )
        region_dataset = PreparedDataset(
            X=np.zeros((1, 2, 4096), dtype=np.float32),
            source_id="regions",
            meta={
                "group_id": np.asarray([3]),
                "sample_source_id": np.asarray(["capture"]),
                "source_region_id": np.asarray([7]),
                "sample_rate": np.asarray([1_000_000.0]),
                "remove_dc": np.asarray(True),
                "rms_normalize": np.asarray(True),
            },
        )
        probabilities = [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]]
        service = RegionEnsembleInferenceService(
            [_StaticService(f"m{index}", probabilities) for index in range(3)]
        )
        features = np.zeros((1, FEATURE_COUNT), dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_gate_manifest(
                Path(directory) / "periodicity_gate_manifest.json"
            )
            with self.assertRaisesRegex(ValueError, "canonical"):
                build_evidence_records(
                    window_dataset,
                    region_dataset,
                    features,
                    tuple(reversed(FEATURE_NAMES)),
                    _feature_identity(region_dataset),
                    service,
                    manifest_path,
                )
            wrong_identity = _feature_identity(region_dataset)
            wrong_identity["source_region_id"] = np.asarray([99])
            with self.assertRaisesRegex(ValueError, "source_region_id"):
                build_evidence_records(
                    window_dataset,
                    region_dataset,
                    features,
                    FEATURE_NAMES,
                    wrong_identity,
                    service,
                    manifest_path,
                )
            two_member_service = RegionEnsembleInferenceService(service.members[:2])
            with self.assertRaisesRegex(ValueError, "exactly three"):
                build_evidence_records(
                    window_dataset,
                    region_dataset,
                    features,
                    FEATURE_NAMES,
                    _feature_identity(region_dataset),
                    two_member_service,
                    manifest_path,
                )


if __name__ == "__main__":
    unittest.main()
