"""One-dimensional convolutional IQ classifier."""

from __future__ import annotations

import torch.nn as nn


class DeepConvNet1D(nn.Module):
    """Historical DeepConvNet architecture with logits and pooled features."""

    def __init__(
        self,
        class_num: int,
        input_channels: int = 2,
        seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.class_num = class_num
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 2 * 25, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(2 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(2 * 25, 4 * 25, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(4 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            nn.Conv1d(4 * 25, 6 * 25, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(6 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(6 * 25, 8 * 25, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(8 * 25),
            nn.ReLU(inplace=True),
            nn.Conv1d(8 * 25, 8 * 25, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(8 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(8 * 25, 12 * 25, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.Conv1d(12 * 25, 12 * 25, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(12 * 25, 12 * 25, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.Conv1d(12 * 25, 12 * 25, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.Conv1d(12 * 25, 12 * 25, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(12 * 25),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(12 * 25, class_num),
        )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        feature = x.view(x.size(0), -1)
        logits = self.fc1(feature)
        return logits, feature


DeepConvNet_1D = DeepConvNet1D


__all__ = ["DeepConvNet1D", "DeepConvNet_1D"]
