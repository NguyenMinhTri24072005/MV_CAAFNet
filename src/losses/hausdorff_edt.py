"""Hausdorff-EDT loss — SPECS §4.2.4, [P] Eq. 33–38 (Algorithm 5), [A] A9 / D4 / D5.

Khoảng cách tính bằng pixel. `D_G` (EDT của G) được tính sẵn trong `.h5`; `D_P` tính trên CPU từ
`P` nhị phân (ngưỡng 0.5, detach) bằng `edt_from_mask`, `P_b` rỗng → hằng `empty_value` (64·√2).

- Chiều P → G luôn khả vi theo P:  `d_PG_b = Σ P·D_G / (Σ P + 1e-6)`.
- Chiều G → P theo `mode`:
  - `soft_pg_detach_gp` (mặc định): `d_GP_b = Σ G·D_P / (Σ G + 1e-6)` (hằng theo P) → loss = mean(d_PG + d_GP)
  - `bidir_soft` (S3): `d_GP_b = Σ G·(1−P)·D_P / (Σ G + 1e-6)` (phạt FN có gradient) → loss = mean(d_PG + d_GP)
  - `karimi_sq` (S3, Karimi & Salcudean 2020): loss = mean((P − G)²·(D_G + D_P)); `d_GP` chỉ để log.
Trả `(loss, {"d_pg", "d_gp", "empty_pred_frac"})`. Tính toàn bộ ở fp32.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.data.compute_edt import EMPTY_VALUE, edt_from_mask

__all__ = ["HAUSDORFF_MODES", "hausdorff_edt_loss", "HausdorffEDTLoss", "pred_edt"]

HAUSDORFF_MODES = ("soft_pg_detach_gp", "bidir_soft", "karimi_sq")
_EPS = 1e-6


def pred_edt(prob: torch.Tensor, empty_value: float = EMPTY_VALUE) -> tuple[torch.Tensor, float]:
    """`D_P` (detach, fp32, cùng device với `prob`) từ `prob > 0.5`, kèm tỉ lệ mẫu dự đoán rỗng."""
    p_bin = (prob.detach() > 0.5).cpu().numpy().astype(np.uint8)  # (B, 1, H, W)
    empty = p_bin.reshape(p_bin.shape[0], -1).sum(axis=1) == 0
    d_p = np.empty(p_bin.shape, dtype=np.float32)
    for b in range(p_bin.shape[0]):
        for c in range(p_bin.shape[1]):
            d_p[b, c] = edt_from_mask(p_bin[b, c], empty_value)
    return torch.from_numpy(d_p).to(device=prob.device), float(empty.mean()) if len(empty) else 0.0


def hausdorff_edt_loss(
    prob: torch.Tensor,
    target: torch.Tensor,
    dist_g: torch.Tensor,
    empty_value: float = EMPTY_VALUE,
    mode: str = "soft_pg_detach_gp",
) -> tuple[torch.Tensor, dict[str, float]]:
    if mode not in HAUSDORFF_MODES:
        raise ValueError(f"loss.hausdorff_mode phải thuộc {HAUSDORFF_MODES}, nhận {mode!r}")
    p = prob.float()
    g = target.float()
    d_g = dist_g.float()
    d_p, empty_frac = pred_edt(p, empty_value)
    dims = tuple(range(1, p.dim()))

    d_pg = (p * d_g).sum(dim=dims) / (p.sum(dim=dims) + _EPS)
    g_sum = g.sum(dim=dims) + _EPS
    if mode == "soft_pg_detach_gp":
        d_gp = (g * d_p).sum(dim=dims) / g_sum
        loss = (d_pg + d_gp).mean()
    elif mode == "bidir_soft":
        d_gp = (g * (1.0 - p) * d_p).sum(dim=dims) / g_sum
        loss = (d_pg + d_gp).mean()
    else:  # karimi_sq
        loss = torch.mean((p - g) ** 2 * (d_g + d_p))
        d_gp = (g * d_p).sum(dim=dims) / g_sum
    stats = {
        "d_pg": float(d_pg.detach().mean().item()),
        "d_gp": float(d_gp.detach().mean().item()),
        "empty_pred_frac": empty_frac,
    }
    return loss, stats


class HausdorffEDTLoss(nn.Module):
    def __init__(self, empty_value: float = EMPTY_VALUE, mode: str = "soft_pg_detach_gp"):
        super().__init__()
        if mode not in HAUSDORFF_MODES:
            raise ValueError(f"loss.hausdorff_mode phải thuộc {HAUSDORFF_MODES}, nhận {mode!r}")
        self.empty_value = float(empty_value)
        self.mode = mode

    def forward(self, prob: torch.Tensor, target: torch.Tensor, dist_g: torch.Tensor):
        return hausdorff_edt_loss(prob, target, dist_g, self.empty_value, self.mode)
