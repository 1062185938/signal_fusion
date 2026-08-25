import importlib.util
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
    native_resource_dir,
)
from signal_fusion.feature_extraction.cli import extract_features_from_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WIFI_FIXTURE = (
    PROJECT_ROOT
    / "data/raw/wifi/"
    "WIFI_5_batch;Freq=5230 MHz;Span=80 MHz;Rate=100.0 MHz;0005.mat"
)
FEATURE_GOLDEN = (
    PROJECT_ROOT / "data/processed/extraction/wifi_5_features_4096_smoke.npz"
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import compatibility wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        self.assertIn('"feature_61": 61.0', serialized)

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


class FeatureMapTests(unittest.TestCase):
    def test_phase0_feature_map_contract_is_explicit(self):
        mapping = load_feature_map()
        names = feature_code_names(mapping)
        counts = {
            group: sum(item["group"] == group for item in mapping["features"])
            for group in EXPECTED_GROUP_COUNTS
        }

        self.assertEqual(mapping["feature_count"], FEATURE_COUNT)
        self.assertEqual(counts, EXPECTED_GROUP_COUNTS)
        self.assertEqual(names[0], "time_rms_amplitude")
        self.assertEqual(
            names[-1], "time_frequency_wsst_ridge_energy_ratio"
        )

    def test_packaged_feature_map_and_reference_documents_are_exact_copies(self):
        legacy_references = PROJECT_ROOT / "iq_feature_extraction/references"
        packaged_references = default_feature_map_path().parent
        filenames = (
            "feature_map.json",
            "frequency_domain_iq_features.md",
            "time_domain_iq_features.md",
            "time_frequency_iq_features.md",
        )

        self.assertIn(
            "signal_fusion/feature_extraction/assets",
            packaged_references.as_posix(),
        )
        for filename in filenames:
            self.assertEqual(
                (packaged_references / filename).read_bytes(),
                (legacy_references / filename).read_bytes(),
            )

        legacy_mapping = load_feature_map(legacy_references / "feature_map.json")
        packaged_mapping = load_feature_map()
        self.assertEqual(
            feature_code_names(packaged_mapping),
            feature_code_names(legacy_mapping),
        )


class FeatureNativeAssetTests(unittest.TestCase):
    def test_packaged_header_is_exact_copy_of_both_legacy_headers(self):
        packaged_header = c_api_header_path().read_bytes()

        self.assertEqual(
            packaged_header,
            (PROJECT_ROOT / "iq_feature_extraction/native/linux/iq_feature_c_api.h")
            .read_bytes(),
        )
        self.assertEqual(
            packaged_header,
            (PROJECT_ROOT / "iq_feature_extraction/native/windows/iq_feature_c_api.h")
            .read_bytes(),
        )

    def test_packaged_native_bundles_are_exact_copies(self):
        legacy_native = PROJECT_ROOT / "iq_feature_extraction/native"
        linux_dir = native_resource_dir("linux")
        windows_dir = native_resource_dir("windows")

        self.assertEqual(default_library_dir(), linux_dir)
        self.assertEqual(
            (linux_dir / "libextractAllFeatures.so").read_bytes(),
            (legacy_native / "linux/libextractAllFeatures.so").read_bytes(),
        )
        for filename in (
            "extractAllFeatures.dll",
            "libgcc_s_seh-1.dll",
            "libgomp-1.dll",
            "libstdc++-6.dll",
            "libwinpthread-1.dll",
        ):
            self.assertEqual(
                (windows_dir / filename).read_bytes(),
                (legacy_native / "windows" / filename).read_bytes(),
            )


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


class FeatureCompatibilityTests(unittest.TestCase):
    def test_legacy_modules_reexport_core_runtime(self):
        scripts = PROJECT_ROOT / "iq_feature_extraction/scripts"
        backend_wrapper = _load_module(
            "legacy_feature_backend", scripts / "ctypes_backend.py"
        )
        map_wrapper = _load_module(
            "legacy_feature_map", scripts / "feature_map_utils.py"
        )
        cli_wrapper = _load_module(
            "legacy_feature_cli", scripts / "feature_extractor.py"
        )

        self.assertIs(backend_wrapper.IQFeatureCtypesBackend, IQFeatureCtypesBackend)
        self.assertIs(map_wrapper.load_feature_map, load_feature_map)
        self.assertIs(
            cli_wrapper.extract_features_from_dataset,
            extract_features_from_dataset,
        )

    def test_real_wifi_fixture_preserves_five_field_npz_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "features.npz"
            summary = extract_features_from_dataset(
                data_path=str(WIFI_FIXTURE),
                output_path=str(output_path),
                data_format="mat",
                x_key="iq",
                seq_len=4096,
                sample_rate=100_000_000,
                max_samples=5,
                progress_every=0,
            )
            with np.load(output_path, allow_pickle=False) as actual:
                actual_payload = {
                    field: actual[field].copy() for field in actual.files
                }
            with np.load(FEATURE_GOLDEN, allow_pickle=False) as expected:
                expected_payload = {
                    field: expected[field].copy() for field in expected.files
                }

        self.assertEqual(
            list(actual_payload),
            ["features", "sample_rate", "seq_len", "feature_count", "feature_names"],
        )
        self.assertEqual(summary["feature_shape"], (5, FEATURE_COUNT))
        np.testing.assert_allclose(
            actual_payload["features"],
            expected_payload["features"],
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            actual_payload["feature_names"], expected_payload["feature_names"]
        )


if __name__ == "__main__":
    unittest.main()
