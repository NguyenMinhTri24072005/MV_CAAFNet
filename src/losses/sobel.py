"""Sobel edge loss — SPECS §4.2.2, [P] Eq. 16–21 (Algorithm 2).

`M(X) = sqrt((S_x * X)² + (S_y * X)² + ε)` (zero padding, D3), loss = `mean|M(P) − M(G)|` ([A] A6:
trung bình thay vì tổng). Dấu của kernel không ảnh hưởng vì chỉ dùng độ lớn (A7).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["SobelLoss", "SOBEL_X", "SOBEL_Y"]

SOBEL_X = ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
SOBEL_Y = ((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0))


class SobelLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("kx", torch.tensor(SOBEL_X).view(1, 1, 3, 3), persistent=False)
        self.register_buffer("ky", torch.tensor(SOBEL_Y).view(1, 1, 3, 3), persistent=False)

    def magnitude(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        gx = F.conv2d(x, self.kx.float(), padding=1)
        gy = F.conv2d(x, self.ky.float(), padding=1)
        return torch.sqrt(gx * gx + gy * gy + self.eps)

    def forward(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(self.magnitude(prob) - self.magnitude(target)))
