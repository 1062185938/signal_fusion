"""Two-layer LSTM classifier for channel-first IQ windows."""

from __future__ import annotations

import torch.nn as nn


class LSTMIQ(nn.Module):
    """Historical LSTM architecture with logits and hidden-state features."""

    def __init__(
        self,
        class_num: int,
        input_channels: int = 2,
        seq_len: int = 128,
        hidden_size: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.class_num = class_num
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.input_norm = nn.BatchNorm1d(input_channels)
        self.lstm1 = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_size, class_num)

    def forward(self, x):
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        sequence_output, _ = self.lstm1(x)
        sequence_output = self.dropout(sequence_output)
        _, (hidden, _) = self.lstm2(sequence_output)
        feature = hidden[-1]
        feature = self.dropout(feature)
        logits = self.fc(feature)
        return logits, feature


__all__ = ["LSTMIQ"]
