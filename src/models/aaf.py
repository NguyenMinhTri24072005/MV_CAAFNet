"""Adaptive Attention Fusion (AAF) — SPECS §3.5, [P] Eq. 11–12, Table 3.

`fused = w1·PAM(x) + w2·CAM(x)` (quy ước cố định: **w1 nhân PA, w2 nhân CA**), rồi `Conv3×3`
(không BN/ReLU sau phép trộn, C2). Bốn chế độ:
- `pa`: chỉ PAM; `ca`: chỉ CAM (ablation Table 7)
- `static`: PAM + CAM với trọng số cố định `static_weights` (buffer, không học)
- `adaptive`: PAM + CAM với `w` từ AAC (dùng chung cho cả 3 tầng)
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.models.attention import CAM, PAM

__all__ = ["AAF", "AAF_MODES"]

AAF_MODES = ("pa", "ca", "static", "adaptive")


class AAF(nn.Module):
    def __init__(
        self,
        ch: int,
        mode: str,
        pam_reduction: int = 8,
        static_weights: Sequence[float] = (0.5, 0.5),
    ):
        super().__init__()
        if mode not in AAF_MODES:
            raise ValueError(f"aaf.mode phải thuộc {AAF_MODES}, nhận {mode!r}")
        self.mode = mode
        self.pam = PAM(ch, pam_reduction) if mode in ("pa", "static", "adaptive") else None
        self.cam = CAM(ch) if mode in ("ca", "static", "adaptive") else None
        if mode == "static":
            sw = torch.as_tensor([float(v) for v in static_weights], dtype=torch.float32)
            if sw.shape != (2,):
                raise ValueError(f"static_weights cần 2 giá trị, nhận {list(static_weights)}")
            self.register_buffer("static_weights", sw)
        self.out_conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "pa":
            fused = self.pam(x)
        elif self.mode == "ca":
            fused = self.cam(x)
        elif self.mode == "static":
            sw = self.static_weights.to(x.dtype)
            fused = sw[0] * self.pam(x) + sw[1] * self.cam(x)
        else:
            if w is None:
                raise ValueError("AAF mode 'adaptive' cần w (B, 2) từ AAC")
            pa, ca = self.pam(x), self.cam(x)
            w = w.to(pa.dtype)
            w1 = w[:, 0].view(-1, 1, 1, 1)
            w2 = w[:, 1].view(-1, 1, 1, 1)
            fused = w1 * pa + w2 * ca
        return self.out_conv(fused)
