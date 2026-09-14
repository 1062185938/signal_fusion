"""Compatibility wrapper for :mod:`signal_fusion.training`.

The historical module-level ``args`` variable remains synchronized so callers
that configured this script programmatically continue to work during migration.
"""

from pathlib import Path
import sys


_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from signal_fusion.modeling import build_model  # noqa: E402
from signal_fusion.training import (  # noqa: E402
    EarlyStopping,
    LabelSmoothingCrossEntropy,
    LogitNormLoss,
    eval_epoch,
    export_onnx,
    train_epoch,
)
from signal_fusion.training.losses import (  # noqa: E402
    linear_combination,
    reduce_loss,
)
from signal_fusion.training import cli as _core_cli  # noqa: E402


parser = _core_cli.parser
args = None


def _push_args():
    _core_cli.args = args


def _pull_args():
    global args
    args = _core_cli.args


def load_data(
    filepath=None,
    data_format="mat",
    seq_len=None,
    label_path=None,
    x_key=None,
    y_key=None,
    dataset_dir=None,
):
    _push_args()
    try:
        return _core_cli.load_data(
            filepath=filepath,
            data_format=data_format,
            seq_len=seq_len,
            label_path=label_path,
            x_key=x_key,
            y_key=y_key,
            dataset_dir=dataset_dir,
        )
    finally:
        _pull_args()


def _run_training():
    _push_args()
    try:
        return _core_cli._run_training()
    finally:
        _pull_args()


def train_and_export_model(
    data_path="./data/processed/radioml2016_train.mat",
    dataset_dir=None,
    data_format="mat",
    label_path=None,
    x_key=None,
    y_key=None,
    save_dir="./outputs_logit_norm",
    onnx_filename="deep_iq_cnn.onnx",
    model_name="deepconvnet_1d",
    class_num=11,
    batch_size=128,
    lr_model=0.008,
    optimizer="sgd",
    max_epoch=30,
    max_samples=None,
    split_mode="random",
    patience=15,
    seed=44,
    num_workers=0,
    loss="logit_norm",
    temp=0.2,
    epsilon=0.1,
    device="auto",
    no_plot=True,
    no_amp=False,
    awgn_probability=0.0,
    awgn_snr_min=5.0,
    awgn_snr_max=20.0,
    frequency_shift_probability=0.0,
    frequency_shift_max_fraction=0.0,
    spectral_inversion_probability=0.0,
):
    try:
        return _core_cli.train_and_export_model(
            data_path=data_path,
            dataset_dir=dataset_dir,
            data_format=data_format,
            label_path=label_path,
            x_key=x_key,
            y_key=y_key,
            save_dir=save_dir,
            onnx_filename=onnx_filename,
            model_name=model_name,
            class_num=class_num,
            batch_size=batch_size,
            lr_model=lr_model,
            optimizer=optimizer,
            max_epoch=max_epoch,
            max_samples=max_samples,
            split_mode=split_mode,
            patience=patience,
            seed=seed,
            num_workers=num_workers,
            loss=loss,
            temp=temp,
            epsilon=epsilon,
            device=device,
            no_plot=no_plot,
            no_amp=no_amp,
            awgn_probability=awgn_probability,
            awgn_snr_min=awgn_snr_min,
            awgn_snr_max=awgn_snr_max,
            frequency_shift_probability=frequency_shift_probability,
            frequency_shift_max_fraction=frequency_shift_max_fraction,
            spectral_inversion_probability=spectral_inversion_probability,
        )
    finally:
        _pull_args()


def main(argv=None):
    try:
        return _core_cli.main(argv)
    finally:
        _pull_args()


if __name__ == "__main__":
    main()
