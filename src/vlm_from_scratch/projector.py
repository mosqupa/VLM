from __future__ import annotations

import torch
from torch import nn


class LlavaMlpProjector(nn.Module):
    """Small MLP that maps vision features into the language model hidden size."""

    def __init__(self, vision_hidden_size: int, text_hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.layers(image_features)
