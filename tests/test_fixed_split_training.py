import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _write_split(
    path: Path,
    *,
    split_name: str,
    group_offset: int,
    source_prefix: str | None = None,
) -> None:
    rng = np.random.default_rng(group_offset)
    samples_per_class = 4
    y = np.repeat(np.arange(2, dtype=np.int64), samples_per_class)
    group_id = group_offset + np.repeat(np.arange(4, dtype=np.int64), 2)
    source_prefix = split_name if source_prefix is None else source_prefix
    sample_source_id = np.asarray(
        [f"{source_prefix}_class_{label}" for label in y]
    )
    source_region_id = np.tile(np.repeat(np.arange(2), 2), 2).astype(np.int64)
    np.savez_compressed(
        path,
        X=rng.normal(size=(8, 2, 128)).astype(np.float32),
        y=y,
        dataset_id=np.asarray("fixed_fixture_v1"),
        split=np.asarray(split_name),
        seq_len=np.asarray(128, dtype=np.int64),
        label_map_json=np.asarray(json.dumps({"0": "A", "1": "B"})),
        group_id=group_id,
        burst_id=group_id,
        sample_source_id=sample_source_id,
        source_region_id=source_region_id,
        source_id=np.asarray(f"fixed_fixture_v1:{split_name}"),
    )


def _write_bundle(directory: Path, *, test_source_prefix: str | None = None) -> None:
    _write_split(directory / "train.npz", split_name="train", group_offset=0)
    _write_split(
        directory / "validation.npz",
        split_name="validation",
        group_offset=100,
    )
    _write_split(
        directory / "test.npz",
        split_name="test",
        group_offset=200,
        source_prefix=test_source_prefix,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch optional dependency is not installed")
class FixedSplitTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global load_fixed_split_bundle
        from signal_fusion.training.fixed_splits import load_fixed_split_bundle

    def test_loads_fixed_files_without_resplitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_bundle(root)
            bundle = load_fixed_split_bundle(
                root,
                class_num=2,
                batch_size=3,
                num_workers=0,
            )

        self.assertEqual(bundle.dataset_id, "fixed_fixture_v1")
        self.assertEqual(bundle.label_map, {"0": "A", "1": "B"})
        self.assertEqual(bundle.split_sizes, {"train": 8, "validation": 8, "test": 8})
        self.assertEqual(bundle.seq_len, 128)
        self.assertEqual(bundle.input_channels, 2)
        self.assertEqual(len(bundle.train_loader.dataset), 8)
        self.assertEqual(len(bundle.validation_loader.dataset), 8)
        self.assertEqual(len(bundle.test_loader.dataset), 8)

    def test_rejects_source_leakage_between_fixed_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_bundle(root, test_source_prefix="train")

            with self.assertRaisesRegex(ValueError, "source_id leakage"):
                load_fixed_split_bundle(
                    root,
                    class_num=2,
                    batch_size=4,
                    num_workers=0,
                )

    def test_rejects_class_num_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_bundle(root)

            with self.assertRaisesRegex(ValueError, "class_num=3"):
                load_fixed_split_bundle(
                    root,
                    class_num=3,
                    batch_size=4,
                    num_workers=0,
                )


if __name__ == "__main__":
    unittest.main()
