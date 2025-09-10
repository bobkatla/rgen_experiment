from __future__ import annotations
import torch
from torch import nn

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    def update(self, model: nn.Module):
        i = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            self.shadow[i].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
            i += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        i = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.shadow[i])
            i += 1
