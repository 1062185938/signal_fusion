import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from signal_fusion import Evidence, PreparedDataset
from signal_fusion.feature_extraction import (
    EXPECTED_GROUP_COUNTS,
    FEATURE_COUNT,
    FEATURE_SCHEMA_ID,
    FeatureExtractionService,
    FeatureResult,
    IQFeatureCtypesBackend,
    c_api_header_path,
    default_feature_map_path,
    default_library_dir,
    feature_code_names,
    load_feature_map,
)
from signal_fusion.feature_extraction.cli import write_feature_result


class FakeFeatureBackend:
    def extract_features(self, i_data, q_data, sample_rate):
        base = float(np.mean(i_data) + np.mean(q_data))
        return (np.arange(FEATURE_COUNT, dtype=np.float32) + base).astype(
            np.float32
        )


class FeatureResultTests(unittest.TestCase):
    def test_contract_projects_one_sample_to_json_friendly_evidence(self):
        names = tuple(f"feature_{index}" for index in range(FEATURE_COUNT))
        result = FeatureResult(
            source_id="feature_contract",
            features=np.arange(FEATURE_COUNT, dtype=np.float32).reshape(1, -1),
            feature_names=names,
            sample_rate=1_000_000,
            seq_len=128,
        )

        evidence = result.evidence_for_sample(0)
        serialized = json.dumps(evidence.to_dict(), allow_nan=False)

        self.assertIsInstance(evidence, Evidence)
        self.assertEqual(evidence.kind, "iq_features")
        self.assertEqual(evidence.producer, FEATURE_SCHEMA_ID)
        self.assertEqual(len(evidence.payload["values"]), FEATURE_COUNT)
        self.assertIn('"feature_63": 63.0', serialized)

    def test_contract_rejects_wrong_feature_width(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            FeatureResult(
                source_id="wrong_width",
                features=np.zeros((1, FEATURE_COUNT - 1), dtype=np.float32),
                feature_names=tuple(
                    f"feature_{index}" for index in range(FEATURE_COUNT)
                ),
                sample_rate=1_000_000,
                seq_len=128,
            )

    def test_written_artifact_identifies_the_current_schema(self):
        result = FeatureResult(
            source_id="artifact",
            features=np.zeros((1, FEATURE_COUNT), dtype=np.float32),
            feature_names=tuple(
                f"feature_{index}" for index in range(FEATURE_COUNT)
            ),
            sample_rate=1_000_000,
            seq_len=128,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = write_feature_result(result, Path(directory) / "features.npz")
            with np.load(output, allow_pickle=False) as artifact:
                schema_id = str(artifact["feature_schema_id"].reshape(()).item())
                feature_count = int(artifact["feature_count"].reshape(()).item())
                source_dataset_id = str(
                    artifact["source_dataset_id"].reshape(()).item()
                )

        self.assertEqual(schema_id, FEATURE_SCHEMA_ID)
        self.assertEqual(feature_count, FEATURE_COUNT)
        self.assertEqual(source_dataset_id, "artifact")


class FeatureMapTests(unittest.TestCase):
    def test_current_feature_map_contract_is_explicit(self):
        mapping = load_feature_map()
        names = feature_code_names(mapping)
        counts = {
            group: sum(item["group"] == group for item in mapping["features"])
            for group in EXPECTED_GROUP_COUNTS
        }

        self.assertEqual(mapping["feature_count"], FEATURE_COUNT)
        self.assertEqual(mapping["schema_id"], FEATURE_SCHEMA_ID)
        self.assertEqual(counts, EXPECTED_GROUP_COUNTS)
        self.assertEqual(names[0], "time_rms_amplitude")
        self.assertEqual(
            names[-1], "time_frequency_wsst_ridge_energy_ratio"
        )

        self.assertIn("time_envelope_power_autocorrelation_peak", names)
        self.assertIn("time_frequency_stft_frame_energy_cv", names)

    def test_packaged_reference_documents_are_resolvable(self):
        references = default_feature_map_path().parent
        self.assertIn(
            "signal_fusion/feature_extraction/assets",
            references.as_posix(),
        )
        for filename in (
            "frequency_domain_iq_features.md",
            "time_domain_iq_features.md",
            "time_frequency_iq_features.md",
        ):
            self.assertTrue((references / filename).is_file())


class FeatureNativeAssetTests(unittest.TestCase):
    def test_packaged_header_and_linux_library_use_64_feature_contract(self):
        header = c_api_header_path().read_text(encoding="utf-8")
        library = default_library_dir() / "libextractAllFeatures.so"

        self.assertIn("64 维特征", header)
        self.assertIn("长度至少为 64", header)
        self.assertTrue(library.is_file())
        self.assertGreater(library.stat().st_size, 0)

    def test_linux_library_extracts_one_finite_64_feature_vector(self):
        sample_count = 512
        time = np.arange(sample_count, dtype=np.float64) / sample_count
        amplitude = 0.7 + 0.3 * np.sin(2.0 * np.pi * 3.0 * time)
        phase = 2.0 * np.pi * (17.0 * time + 11.0 * time**2)
        samples = amplitude * np.exp(1j * phase)

        with IQFeatureCtypesBackend() as backend:
            features = backend.extract_features(
                samples.real,
                samples.imag,
                1_000_000.0,
            )

        self.assertEqual(features.shape, (FEATURE_COUNT,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(features)))
        centered_envelope = np.abs(samples - np.mean(samples))
        expected_envelope_cv = float(
            np.std(centered_envelope, ddof=1) / np.mean(centered_envelope)
        )
        self.assertAlmostEqual(float(features[4]), expected_envelope_cv, places=5)


class FeatureExtractionServiceTests(unittest.TestCase):
    def test_service_accepts_prepared_dataset_and_injected_backend(self):
        x = np.arange(3 * 2 * 32, dtype=np.float32).reshape(3, 2, 32)
        dataset = PreparedDataset(
            X=x,
            source_id="service_fixture",
            meta={"sample_rate": np.asarray(2_000_000.0)},
        )

        result = FeatureExtractionService().extract(
            dataset,
            max_samples=2,
            backend=FakeFeatureBackend(),
            progress_every=0,
        )

        self.assertEqual(result.source_id, "service_fixture")
        self.assertEqual(result.features.shape, (2, FEATURE_COUNT))
        self.assertEqual(result.features.dtype, np.float32)
        self.assertEqual(result.sample_rate, 2_000_000.0)
        self.assertEqual(result.seq_len, 32)
        self.assertEqual(result.metadata["input_num_samples"], 3)
        self.assertEqual(result.metadata["selected_num_samples"], 2)

    def test_service_requires_readable_source_id(self):
        dataset = PreparedDataset(
            X=np.zeros((1, 2, 32), dtype=np.float32),
            meta={"sample_rate": 1_000_000},
        )
        with self.assertRaisesRegex(ValueError, "source_id"):
            FeatureExtractionService().extract(
                dataset,
                backend=FakeFeatureBackend(),
                progress_every=0,
            )


if __name__ == "__main__":
    unittest.main()
