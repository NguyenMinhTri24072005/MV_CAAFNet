"""Adaptive Attention Controller (AAC) — SPECS §3.4, [P] Algorithm 1, Eq. 1–9.

Nhận `F0 = stem(x)` (C = 32, [F] A5), chạy một lần mỗi forward và trả `w = (w1, w2)` dùng chung
cho cả 3 khối AAF (w1 nhân PA, w2 nhân CA).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.models.attention import fp32_softmax_bmm

__all__ = ["AAC"]


class AAC(nn.Module):
    def __init__(self, in_ch: int = 32, pool_size: int = 8, qkv_bias: bool = False):
        super().__init__()
        if in_ch < 2:
            raise ValueError("AAC cần in_ch >= 2 (FC C → C/2)")
        self.in_ch = int(in_ch)
        self.pool_size = int(pool_size)
        self.w_q = nn.Linear(in_ch, in_ch, bias=qkv_bias)  # Eq. 3 không có bias [A] C3
        self.w_k = nn.Linear(in_ch, in_ch, bias=qkv_bias)
        self.w_v = nn.Linear(in_ch, in_ch, bias=qkv_bias)
        self.fc1 = nn.Linear(in_ch, in_ch // 2)             # Eq. 7
        self.fc2 = nn.Linear(in_ch // 2, 2)                  # Eq. 8

    def forward(self, f_in: torch.Tensor) -> torch.Tensor:
        f_ds = F.adaptive_avg_pool2d(f_in, (self.pool_size, self.pool_size))           # Eq. 1 (B, C, 8, 8)
        f_seq = f_ds.flatten(2).transpose(1, 2)                                          # Eq. 2 (B, 64, C)
        q, k, v = self.w_q(f_seq), self.w_k(f_seq), self.w_v(f_seq)                     # Eq. 3
        attn = fp32_softmax_bmm(q, k.transpose(1, 2), scale=1.0 / math.sqrt(self.in_ch))  # Eq. 4
        f_sa = torch.bmm(attn.to(v.dtype), v)                                            # Eq. 5 (B, 64, C)
        f_g = f_sa.mean(dim=1)                                                           # Eq. 6 (B, C)
        z = self.fc2(torch.relu(self.fc1(f_g)))                                          # Eq. 7–8 (B, 2)
        with torch.autocast(device_type=z.device.type, enabled=False):
            w = torch.softmax(z.float(), dim=-1)                                         # Eq. 9
        return w.to(f_in.dtype)
