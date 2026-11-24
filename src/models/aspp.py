"""ASPP — SPECS §3.2. Mỗi nhánh theo đúng thứ tự Figure 1: `Conv → ReLU → BN` [P]."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ASPP"]


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling, dilation `rates` (mặc định 6/12/18), nhánh GAP tùy chọn (A12).

    Ở 8×8, nhánh dilation 12 và 18 chỉ chạm điểm tâm — vẫn giữ nguyên theo bài báo.
    """

    def __init__(self, in_ch: int, out_ch: int, rates: Sequence[int] = (6, 12, 18), gap: bool = False):
        super().__init__()
        self.rates = tuple(int(r) for r in rates)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
            )
            for r in self.rates
        )
        self.gap = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
            )
            if gap
            else None
        )
        n_branches = len(self.rates) + (1 if gap else 0)
        self.project = nn.Conv2d(n_branches * out_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        if self.gap is not None:
            outs.append(F.interpolate(self.gap(x), size=x.shape[-2:], mode="nearest"))
        return self.project(torch.cat(outs, dim=1))
