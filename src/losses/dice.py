"""Dice loss theo từng mẫu — SPECS §4.2.1, [P] Eq. 13, [A] D1 (eps = 1e-6).

`L_b = 1 − (2·Σ P_b·G_b + ε) / (Σ P_b + Σ G_b + ε)`, trả `mean_b(L_b)`. Tính ở fp32.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["dice_loss", "DiceLoss"]


def dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = prob.float().flatten(1)
    g = target.float().flatten(1)
    inter = (p * g).sum(dim=1)
    denom = p.sum(dim=1) + g.sum(dim=1)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(prob, target, self.eps)
