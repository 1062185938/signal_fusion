"""Offline training loop for modulation-recognition models."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from signal_fusion.modeling import build_model
from signal_fusion.training.augmentation import add_random_complex_awgn
from signal_fusion.training.exporter import export_onnx
from signal_fusion.training.losses import (
    LabelSmoothingCrossEntropy,
    LogitNormLoss,
)

try:
    from thop import profile
except ImportError:
    profile = None


class EarlyStopping:
    """Stop after ``patience`` epochs without lower validation loss."""

    def __init__(self, patience=20, verbose=False, path="best_model.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> "
                f"{val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp,
    *,
    awgn_probability=0.0,
    awgn_snr_min=5.0,
    awgn_snr_max=20.0,
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = add_random_complex_awgn(
            inputs,
            probability=awgn_probability,
            snr_min_db=awgn_snr_min,
            snr_max_db=awgn_snr_max,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += inputs.size(0)
    return correct / total * 100.0, total_loss / total


def eval_epoch(model, dataloader, criterion, device, use_amp):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = (
                inputs.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
            )
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total += inputs.size(0)
    return correct / total * 100.0, total_loss / total


def _resolve_device(options):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    if options.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, fallback to CPU.")
        return torch.device("cpu")
    if options.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_criterion(options):
    if options.loss == "ce":
        print("损失函数: 标准交叉熵 (CrossEntropyLoss)")
        return nn.CrossEntropyLoss()
    if options.loss == "logit_norm":
        print(f"损失函数: LogitNormLoss (temp={options.temp})")
        return LogitNormLoss(t=options.temp)
    if options.loss == "ls":
        print(f"损失函数: 标签平滑 (LabelSmoothing, epsilon={options.epsilon})")
        return LabelSmoothingCrossEntropy(epsilon=options.epsilon)
    raise ValueError(f"Unsupported loss: {options.loss}")


def _build_optimizer(options, model):
    if options.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=options.lr_model,
            momentum=0.9,
            weight_decay=1e-4,
        )
    if options.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=options.lr_model, weight_decay=1e-4
        )
    raise ValueError(f"Unsupported optimizer: {options.optimizer}")


def _profile_model(model, model_name, input_channels, seq_len, device):
    print("\n" + "=" * 40 + " Profiling Model " + "=" * 40)
    dummy_input = torch.randn(1, input_channels, seq_len).to(device)
    if profile is None:
        print("thop not installed, skip FLOPs profiling.")
        return
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"输入尺寸:[Batch, {input_channels}, {seq_len}]")
    print("FLOPs = {:.4f} G".format(flops / 1000**3))
    print("FLOPs = {:.4f} M".format(flops / 10**6))
    print("Params = {:.4f} M".format(params / 1000**2))
    print(
        "%s | Params: %.4fM | FLOPs: %.4fG\n"
        % (model_name, params / (1000**2), flops / (1000**3))
    )


def _save_training_curves(history, save_dir, no_plot):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    plt.plot(history["val_acc"], label="Validation Accuracy", linewidth=2)
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path)
    print(f"Training curves plotted and saved to {plot_path}")
    if not no_plot:
        plt.show()
    else:
        plt.close()


def run_training(options, load_data_fn=None):
    """Run training from an argparse-compatible options namespace."""

    if load_data_fn is None:
        from signal_fusion.training.cli import load_data as load_data_fn

    torch.manual_seed(options.seed)
    device = _resolve_device(options)
    print(f"Using device: {device}")
    os.makedirs(options.save_dir, exist_ok=True)

    dataset_dir = getattr(options, "dataset_dir", None)
    fixed_split_bundle = None
    if dataset_dir is not None:
        if getattr(options, "max_samples", None) is not None:
            raise ValueError("max_samples is not supported with dataset_dir")
        from signal_fusion.training.fixed_splits import load_fixed_split_bundle

        fixed_split_bundle = load_fixed_split_bundle(
            dataset_dir,
            class_num=options.class_num,
            batch_size=options.batch_size,
            num_workers=options.num_workers,
        )
        (
            train_loader,
            val_loader,
            test_loader,
            seq_len,
            input_channels,
        ) = fixed_split_bundle.loader_tuple()
        print(f"Dataset ID: {fixed_split_bundle.dataset_id}")
        print(f"Label map: {fixed_split_bundle.label_map}")
        print(f"Split mode: fixed ({fixed_split_bundle.split_sizes})")
        label_map_path = Path(options.save_dir) / "label_map.json"
        label_map_path.write_text(
            json.dumps(
                fixed_split_bundle.label_map,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        train_loader, val_loader, test_loader, seq_len, input_channels = load_data_fn(
            filepath=options.data_path,
            data_format=options.data_format,
            label_path=options.label_path,
            x_key=options.x_key,
            y_key=options.y_key,
        )
    model = build_model(
        model_name=options.model_name,
        class_num=options.class_num,
        input_channels=input_channels,
        seq_len=seq_len,
    ).to(device)
    criterion = _build_criterion(options)
    optimizer = _build_optimizer(options, model)
    print(f"Optimizer: {options.optimizer} (lr={options.lr_model})")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=25, eta_min=5e-6, last_epoch=-1
    )
    use_amp = device.type == "cuda" and not options.no_amp
    print(f"AMP enabled: {use_amp}")
    print(
        "Training AWGN: "
        f"probability={options.awgn_probability}, "
        f"SNR={options.awgn_snr_min:g}..{options.awgn_snr_max:g} dB"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_model_path = os.path.join(options.save_dir, "best_model.pth")
    early_stopping = EarlyStopping(
        patience=options.patience, verbose=True, path=best_model_path
    )
    _profile_model(model, options.model_name, input_channels, seq_len, device)

    history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
    print("=" * 40 + " Start Training " + "=" * 40)
    start_time = time.time()

    for epoch in range(options.max_epoch):
        train_acc, train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            awgn_probability=options.awgn_probability,
            awgn_snr_min=options.awgn_snr_min,
            awgn_snr_max=options.awgn_snr_max,
        )
        val_acc, val_loss = eval_epoch(
            model, val_loader, criterion, device, use_amp
        )
        scheduler.step()
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch[{epoch + 1:03d}/{options.max_epoch:03d}] "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        )
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(
                "\n!!! Early stopping triggered. Training stopped to prevent "
                "overfitting !!!"
            )
            break

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds.")
    for name in ("train_acc", "train_loss", "val_acc", "val_loss"):
        np.save(os.path.join(options.save_dir, f"{name}.npy"), history[name])

    try:
        best_state = torch.load(
            best_model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        best_state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_state)
    test_acc, test_loss = eval_epoch(
        model, test_loader, criterion, device, use_amp
    )
    print("\n" + "=" * 40 + " Testing (Best Model) " + "=" * 40)
    print(f"Best Model Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%\n")

    onnx_full_path = os.path.join(options.save_dir, options.onnx_filename)
    export_onnx(
        model, onnx_full_path, seq_len, device, input_channels=input_channels
    )
    _save_training_curves(history, options.save_dir, options.no_plot)

    result = {
        "best_model_path": best_model_path,
        "onnx_path": onnx_full_path,
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "seq_len": int(seq_len),
        "input_channels": int(input_channels),
        "class_num": int(options.class_num),
        "model_name": options.model_name,
        "optimizer": options.optimizer,
        "amp_enabled": bool(use_amp),
        "device": str(device),
        "elapsed_sec": float(elapsed_time),
        "epochs_completed": len(history["train_loss"]),
        "training_augmentation": {
            "type": (
                "dynamic_complex_awgn"
                if options.awgn_probability > 0.0
                else "none"
            ),
            "probability": float(options.awgn_probability),
            "snr_db_min": float(options.awgn_snr_min),
            "snr_db_max": float(options.awgn_snr_max),
            "seed": int(options.seed),
            "post_noise_remove_dc": True,
            "post_noise_rms_normalize": True,
        },
    }
    if fixed_split_bundle is not None:
        result.update(
            {
                "dataset_id": fixed_split_bundle.dataset_id,
                "dataset_dir": str(dataset_dir),
                "split_mode": "fixed",
                "split_sizes": dict(fixed_split_bundle.split_sizes),
                "label_map_path": str(Path(options.save_dir) / "label_map.json"),
            }
        )
        training_result_path = Path(options.save_dir) / "training_result.json"
        result["training_result_path"] = str(training_result_path)
        training_result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result


__all__ = ["EarlyStopping", "eval_epoch", "run_training", "train_epoch"]
