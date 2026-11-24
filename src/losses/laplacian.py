"""Laplacian edge loss — SPECS §4.2.3, [P] Eq. 23–28 (Algorithm 3).

`K = [[0,1,0],[1,−4,1],[0,1,0]]` (zero padding, D3), loss = `mean|K * P − K * G|` ([A] A6: trung bình).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["LaplacianLoss", "LAPLACIAN_KERNEL"]

LAPLACIAN_KERNEL = ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))


class LaplacianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("kernel", torch.tensor(LAPLACIAN_KERNEL).view(1, 1, 3, 3), persistent=False)

    def response(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x.float(), self.kernel.float(), padding=1)

    def forward(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(self.response(prob) - self.response(target)))
