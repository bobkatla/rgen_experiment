# rgen/models/y_encoder.py
from __future__ import annotations
import torch as t
import torch.nn as nn

class YEncoder(nn.Module):
    """
    3x32x32 -> d_y features. Lightweight, fast, differentiable.
    """
    def __init__(self, d_y: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),   # 16x16
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),  # 8x8
        )
        self.head = nn.Linear(128, d_y)

    def forward(self, x: t.Tensor) -> t.Tensor:
        h = self.conv(x)              # [B,128,8,8]
        h = h.mean(dim=(2, 3))        # GAP -> [B,128]
        y = self.head(h)              # [B,d_y]
        return y
