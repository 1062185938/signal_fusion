import ast
from argparse import Namespace
import importlib.util
import inspect
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_TRAINING_SCRIPT = (
    PROJECT_ROOT / "radioml-iq-modulation/scripts/train_and_export.py"
)
TRAINING_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("matplotlib", "onnx", "onnxruntime", "torch")
)


class TrainingWrapperStructureTests(unittest.TestCase):
    def test_legacy_script_is_only_a_core_compatibility_wrapper(self):
        tree = ast.parse(LEGACY_TRAINING_SCRIPT.read_text(encoding="utf-8"))
        classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        imports = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }

        self.assertEqual(classes, [])
        self.assertIn("signal_fusion.training", imports)
        self.assertIn("signal_fusion.training.losses", imports)

    def test_pyproject_exposes_training_extra_and_cli(self):
        contents = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('training = [', contents)
        self.assertIn(
            'signal-train = "signal_fusion.training.cli:main"', contents
        )

    def test_training_cli_exposes_fixed_split_directory(self):
        contents = (
            PROJECT_ROOT / "src/signal_fusion/training/cli.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--dataset_dir"', contents)
        self.assertIn("load_fixed_split_bundle", contents)


@unittest.skipUnless(
    TRAINING_DEPS_AVAILABLE,
    "offline training dependencies are not installed",
)
class TrainingRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

        global nn, ort, torch, DataLoader, TensorDataset
        global LabelSmoothingCrossEntropy, LogitNormLoss
        global build_training_loaders, eval_epoch, train_epoch, training_cli

        import onnxruntime as ort
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from signal_fusion.training import cli as training_cli
        from signal_fusion.training.losses import (
            LabelSmoothingCrossEntropy,
            LogitNormLoss,
        )
        from signal_fusion.training.splitting import build_training_loaders
        from signal_fusion.training.trainer import eval_epoch, train_epoch

    def setUp(self):
        self.previous_args = training_cli.args

    def tearDown(self):
        training_cli.args = self.previous_args

    def test_loss_values_match_pre_migration_baseline(self):
        logits = torch.tensor([[1.0, -0.5, 2.0], [-2.0, 0.25, 1.5]])
        targets = torch.tensor([2, 1])

        self.assertAlmostEqual(
            LogitNormLoss(t=0.2)(logits, targets).item(),
            1.3395042419433594,
        )
        self.assertAlmostEqual(
            LabelSmoothingCrossEntropy(epsilon=0.1)(logits, targets).item(),
            1.0233346223831177,
        )
        self.assertAlmostEqual(
            nn.CrossEntropyLoss()(logits, targets).item(),
            0.9483346939086914,
        )

    def test_random_split_indices_match_pre_migration_baseline(self):
        training_cli.args = Namespace(
            split_mode="random",
            max_samples=24,
            seed=44,
            batch_size=8,
            num_workers=0,
        )
        train, val, test, seq_len, channels = training_cli.load_data(
            str(
                PROJECT_ROOT
                / "data/processed/slices/test/sigmf_lora_dataset_128.npz"
            ),
            data_format="npz",
        )

        self.assertEqual(
            list(train.dataset.indices),
            [4, 3, 15, 2, 7, 21, 20, 5, 11, 6, 9, 13, 1, 23, 18, 19],
        )
        self.assertEqual(list(val.dataset.indices), [12, 14, 8])
        self.assertEqual(list(test.dataset.indices), [22, 16, 17, 0, 10])
        self.assertEqual(int(seq_len), 128)
        self.assertEqual(channels, 2)

    def test_group_split_never_mixes_bursts(self):
        X = np.arange(24 * 2 * 8, dtype=np.float32).reshape(24, 2, 8)
        y = np.arange(24, dtype=np.int64) % 2
        burst_id = np.repeat(np.arange(12), 2)
        loaders = build_training_loaders(
            X,
            y,
            {"burst_id": burst_id},
            split_mode="group",
            max_samples=None,
            seed=44,
            batch_size=4,
            num_workers=0,
        )

        split_bursts = []
        for loader in loaders:
            first_values = loader.dataset.tensors[0][:, 0, 0].numpy()
            sample_indices = (first_values / 16).astype(np.int64)
            split_bursts.append(set(burst_id[sample_indices].tolist()))
        self.assertTrue(split_bursts[0].isdisjoint(split_bursts[1]))
        self.assertTrue(split_bursts[0].isdisjoint(split_bursts[2]))
        self.assertTrue(split_bursts[1].isdisjoint(split_bursts[2]))

    def test_train_and_eval_epoch_keep_dual_output_model_contract(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, inputs):
                return self.linear(inputs), inputs

        inputs = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]
        )
        targets = torch.tensor([0, 1, 1, 0])
        loader = DataLoader(
            TensorDataset(inputs, targets), batch_size=2, shuffle=False
        )
        torch.manual_seed(44)
        model = TinyModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        train_result = train_epoch(
            model,
            loader,
            criterion,
            optimizer,
            scaler,
            torch.device("cpu"),
            False,
        )
        eval_result = eval_epoch(
            model, loader, criterion, torch.device("cpu"), False
        )

        np.testing.assert_allclose(train_result, (25.0, 0.9123342037200928))
        np.testing.assert_allclose(eval_result, (25.0, 0.9066800475120544))

    def test_full_training_and_onnx_outputs_match_baseline(self):
        with tempfile.TemporaryDirectory(prefix="signal-fusion-training-") as root:
            root_path = Path(root)
            rng = np.random.default_rng(17)
            X = rng.normal(size=(20, 2, 128)).astype(np.float32)
            y = np.array([0, 1] * 10, dtype=np.int64)
            data_path = root_path / "synthetic.npz"
            output_path = root_path / "out"
            np.savez(data_path, X=X, y=y)

            result = training_cli.train_and_export_model(
                data_path=str(data_path),
                data_format="npz",
                save_dir=str(output_path),
                onnx_filename="smoke.onnx",
                model_name="deepconvnet_1d",
                class_num=2,
                batch_size=4,
                lr_model=0.001,
                optimizer="adam",
                max_epoch=1,
                split_mode="random",
                patience=2,
                seed=44,
                num_workers=0,
                loss="ce",
                device="cpu",
                no_plot=True,
                no_amp=True,
            )

            self.assertAlmostEqual(result["test_loss"], 0.6925864219665527)
            self.assertAlmostEqual(result["test_acc"], 66.66666666666666)
            self.assertEqual(result["seq_len"], 128)
            self.assertEqual(result["input_channels"], 2)
            self.assertEqual(
                {path.name for path in output_path.iterdir()},
                {
                    "best_model.pth",
                    "smoke.onnx",
                    "train_acc.npy",
                    "train_loss.npy",
                    "training_curves.png",
                    "val_acc.npy",
                    "val_loss.npy",
                },
            )

            session = ort.InferenceSession(
                str(output_path / "smoke.onnx"),
                providers=["CPUExecutionProvider"],
            )
            logits, features = session.run(None, {"input": X[:1]})
            self.assertEqual(tuple(logits.shape), (1, 2))
            self.assertEqual(tuple(features.shape), (1, 300))
            self.assertEqual(
                [output.name for output in session.get_outputs()],
                ["output", "feature"],
            )

    def test_legacy_module_reexports_core_and_synchronizes_args(self):
        module_spec = importlib.util.spec_from_file_location(
            "legacy_train_and_export_for_test", LEGACY_TRAINING_SCRIPT
        )
        legacy = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(legacy)

        self.assertIs(legacy.LogitNormLoss, LogitNormLoss)
        self.assertIs(legacy.train_epoch, train_epoch)
        self.assertIs(legacy.parser, training_cli.parser)
        self.assertEqual(
            inspect.signature(legacy.train_and_export_model),
            inspect.signature(training_cli.train_and_export_model),
        )

        legacy.args = Namespace(
            split_mode="random",
            max_samples=24,
            seed=44,
            batch_size=8,
            num_workers=0,
        )
        legacy.load_data(
            str(
                PROJECT_ROOT
                / "data/processed/slices/test/sigmf_lora_dataset_128.npz"
            ),
            data_format="npz",
        )
        self.assertIs(training_cli.args, legacy.args)


if __name__ == "__main__":
    unittest.main()
