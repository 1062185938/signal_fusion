"""Loss functions used by modulation-recognition training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce_loss(loss, reduction="mean"):
    return (
        loss.mean()
        if reduction == "mean"
        else loss.sum()
        if reduction == "sum"
        else loss
    )


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


class LogitNormLoss(nn.Module):
    def __init__(self, t=1.0):
        super().__init__()
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction="mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


__all__ = [
    "LabelSmoothingCrossEntropy",
    "LogitNormLoss",
    "linear_combination",
    "reduce_loss",
]
