"""Small linear classifier for standardized 62-dimensional feature vectors."""

from __future__ import annotations

import torch.nn as nn


class LinearFeatureClassifier(nn.Module):
    def __init__(self, feature_count: int, class_count: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(feature_count, class_count)

    def forward(self, features):
        return self.classifier(features)


__all__ = ["LinearFeatureClassifier"]
