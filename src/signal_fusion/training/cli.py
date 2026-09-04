"""Command-line and Python entry points for offline IQ model training."""

from __future__ import annotations

import argparse

import numpy as np

from signal_fusion.io import load_signal_dataset
from signal_fusion.training.fixed_splits import load_fixed_split_bundle
from signal_fusion.training.splitting import build_training_loaders
from signal_fusion.training.trainer import run_training


def build_arg_parser():
    parser = argparse.ArgumentParser(description="IQ CNN Training and ONNX Export")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--data_path",
        type=str,
        default="./data/processed/radioml2016_train.mat",
        help="Path to one dataset that the trainer will split",
    )
    input_group.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help=(
            "Directory containing fixed train.npz, validation.npz, and test.npz; "
            "no additional split is performed"
        ),
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="mat",
        choices=["auto", "pkl", "mat", "npz", "npy"],
        help="Dataset file format",
    )
    parser.add_argument("--label_path", type=str, default=None)
    parser.add_argument("--x_key", type=str, default=None)
    parser.add_argument("--y_key", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./outputs_logit_norm")
    parser.add_argument("--onnx_filename", type=str, default="deep_iq_cnn.onnx")
    parser.add_argument("--model_name", type=str, default="deepconvnet_1d")
    parser.add_argument("--class_num", type=int, default=11)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_model", type=float, default=0.008)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--max_epoch", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--split_mode", choices=["random", "group"], default="random"
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--loss", choices=["ce", "logit_norm", "ls"], default="logit_norm"
    )
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument(
        "--awgn_probability",
        type=float,
        default=0.0,
        help="Per-window probability of dynamic AWGN during training only",
    )
    parser.add_argument(
        "--awgn_snr_min",
        type=float,
        default=5.0,
        help="Minimum uniformly sampled training AWGN SNR in dB",
    )
    parser.add_argument(
        "--awgn_snr_max",
        type=float,
        default=20.0,
        help="Maximum uniformly sampled training AWGN SNR in dB",
    )
    return parser


parser = build_arg_parser()
args = None


def load_data(
    filepath=None,
    data_format="mat",
    seq_len=None,
    label_path=None,
    x_key=None,
    y_key=None,
    dataset_dir=None,
):
    global args
    if args is None:
        args = parser.parse_args([])

    effective_dataset_dir = (
        dataset_dir
        if dataset_dir is not None
        else getattr(args, "dataset_dir", None)
    )
    if effective_dataset_dir is not None:
        if getattr(args, "max_samples", None) is not None:
            raise ValueError("max_samples is not supported with dataset_dir")
        print(f"Loading fixed splits from {effective_dataset_dir}...")
        bundle = load_fixed_split_bundle(
            effective_dataset_dir,
            class_num=args.class_num,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"Dataset ID: {bundle.dataset_id}")
        print(f"Label map: {bundle.label_map}")
        print(f"Fixed split samples: {bundle.split_sizes}")
        print(
            f"Loaded input seq_len: {bundle.seq_len}, "
            f"channels: {bundle.input_channels}"
        )
        return bundle.loader_tuple()

    if filepath is None:
        raise ValueError("filepath is required when dataset_dir is not provided")
    print(f"Loading data from {filepath}...")
    loaded = load_signal_dataset(
        path=filepath,
        data_format=data_format,
        seq_len=seq_len,
        label_path=label_path,
        x_key=x_key,
        y_key=y_key,
    )
    X = loaded["X"]
    Y = loaded["y"]
    meta = loaded["meta"]
    if Y is None:
        raise ValueError("Training requires labels, but loaded dataset returned y=None.")

    input_channels = int(X.shape[1])
    seq_len = meta["seq_len"]
    print(f"Loaded input seq_len: {seq_len}, channels: {X.shape[1]}")

    X = X.astype(np.float32, copy=False)
    Y = np.asarray(Y).squeeze().astype(np.int64)
    if Y.min() == 1:
        Y = Y - 1

    train_loader, val_loader, test_loader = build_training_loaders(
        X,
        Y,
        meta,
        split_mode=getattr(args, "split_mode", "random"),
        max_samples=args.max_samples,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader, test_loader, seq_len, input_channels


def _run_training():
    return run_training(args, load_data_fn=load_data)


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
):
    global args
    args = argparse.Namespace(
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
    )
    return _run_training()


def main(argv=None):
    global args
    args = parser.parse_args(argv)
    return _run_training()


if __name__ == "__main__":
    main()
