"""Command-line and Python entry points for offline IQ model training."""

from __future__ import annotations

import argparse

import numpy as np

from signal_fusion.io import load_signal_dataset
from signal_fusion.training.splitting import build_training_loaders
from signal_fusion.training.trainer import run_training


def build_arg_parser():
    parser = argparse.ArgumentParser(description="IQ CNN Training and ONNX Export")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/processed/radioml2016_train.mat",
        help="Path to the .mat dataset",
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
    return parser


parser = build_arg_parser()
args = None


def load_data(
    filepath,
    data_format="mat",
    seq_len=None,
    label_path=None,
    x_key=None,
    y_key=None,
):
    global args
    if args is None:
        args = parser.parse_args([])

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
):
    global args
    args = argparse.Namespace(
        data_path=data_path,
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
    )
    return _run_training()


def main(argv=None):
    global args
    args = parser.parse_args(argv)
    return _run_training()


if __name__ == "__main__":
    main()
