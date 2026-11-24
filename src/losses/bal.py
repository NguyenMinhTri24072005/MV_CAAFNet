"""Boundary-Aware Loss — SPECS §4.2.5, [P] Eq. 14: `total = L_Dice + λ·L_boundary`.

`loss.name` ∈ {dice, dice+sobel, dice+laplacian, dice+hausdorff}; với `dice` thì `L_boundary = 0`.
Mọi phép tính ở **fp32** (tắt autocast cục bộ; `prob = sigmoid(logits.float())` khi có logits),
kể cả khi bật AMP ở forward của mô hình.

    total, parts = criterion(model_out, {"mask": G, "dist": D_G})
    parts = {"dice", "boundary", "total", "d_pg", "d_gp", "empty_pred_frac"}  (3 khóa cuối NaN nếu không phải Hausdorff)
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn

from src.losses.dice import dice_loss
from src.losses.hausdorff_edt import HAUSDORFF_MODES, hausdorff_edt_loss
from src.losses.laplacian import LaplacianLoss
from src.losses.sobel import SobelLoss

__all__ = ["LOSS_NAMES", "BoundaryAwareLoss"]

LOSS_NAMES = ("dice", "dice+sobel", "dice+laplacian", "dice+hausdorff")


class BoundaryAwareLoss(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        lc = cfg["loss"]
        self.name = str(lc["name"])
        if self.name not in LOSS_NAMES:
            raise ValueError(f"loss.name phải thuộc {LOSS_NAMES}, nhận {self.name!r}")
        self.lam = float(lc["lambda"])
        self.dice_eps = float(lc["dice_eps"])
        self.hausdorff_mode = str(lc["hausdorff_mode"])
        if self.hausdorff_mode not in HAUSDORFF_MODES:
            raise ValueError(f"loss.hausdorff_mode phải thuộc {HAUSDORFF_MODES}, nhận {self.hausdorff_mode!r}")
        self.empty_value = float(lc["hausdorff_empty_value"])
        self.sobel = SobelLoss(float(lc["sobel_eps"])) if self.name == "dice+sobel" else None
        self.laplacian = LaplacianLoss() if self.name == "dice+laplacian" else None

    @staticmethod
    def _prob(out: Any) -> torch.Tensor:
        if isinstance(out, Mapping):
            if out.get("logits") is not None:
                return torch.sigmoid(out["logits"].float())
            return out["prob"].float()
        return out.float()

    def forward(self, out: Any, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        nan = float("nan")
        ref = batch["mask"]
        with torch.autocast(device_type=ref.device.type, enabled=False):
            prob = self._prob(out)
            mask = batch["mask"].float()
            l_dice = dice_loss(prob, mask, self.dice_eps)
            extra = {"d_pg": nan, "d_gp": nan, "empty_pred_frac": nan}
            if self.name == "dice":
                boundary = torch.zeros((), device=prob.device, dtype=torch.float32)
            elif self.name == "dice+sobel":
                boundary = self.sobel(prob, mask)
            elif self.name == "dice+laplacian":
                boundary = self.laplacian(prob, mask)
            else:
                boundary, extra = hausdorff_edt_loss(
                    prob, mask, batch["dist"].float(), self.empty_value, self.hausdorff_mode
                )
            total = l_dice + self.lam * boundary
        parts = {
            "dice": float(l_dice.detach().item()),
            "boundary": float(boundary.detach().item()),
            "total": float(total.detach().item()),
            **extra,
        }
        if not math.isfinite(parts["total"]):
            parts["nonfinite"] = 1.0
        return total, parts
