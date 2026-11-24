"""Hàm mất mát — SPECS §4.2, Phụ lục C (cần torch).

Ca dịch 1 pixel (tính tay): G = ô vuông 4×4 tại hàng 10–13, cột 10–13 trên ảnh 64×64; P = G dịch phải 1 cột.
- Dice: TP = 12, |P| = |G| = 16 → L = 1 − 24/32 = 0.25.
- Laplacian: P − G = +1 ở cột 14, −1 ở cột 10 (mỗi thanh 4 pixel, cách xa nhau). Với một thanh dọc 4 pixel,
  Σ|K * thanh| = (2+3+3+2) [chính thanh] + 8 [2 cột kề] + 2 [trên/dưới] = 20 → tổng 40 → mean = 40/4096.
- Hausdorff: cột 14 của P cách G 1 px → d_PG = 4/16; cột 10 của G cách P 1 px → d_GP = 4/16 → 0.5
  (cả `soft_pg_detach_gp` và `bidir_soft`); `karimi_sq` = 8 pixel lệch × (1 + 0) / 4096.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.compute_edt import edt_from_mask  # noqa: E402
from src.losses import (  # noqa: E402
    HAUSDORFF_MODES, LOSS_NAMES, BoundaryAwareLoss, LaplacianLoss, SobelLoss, dice_loss, hausdorff_edt_loss,
)
from src.losses.sobel import SOBEL_X, SOBEL_Y  # noqa: E402
from src.utils import load_config  # noqa: E402

EMPTY = 90.51


def square(row: int = 10, col: int = 10, size: int = 4) -> np.ndarray:
    m = np.zeros((64, 64), np.float32)
    m[row: row + size, col: col + size] = 1
    return m


def batch(*masks: np.ndarray) -> tuple:
    g = torch.from_numpy(np.stack(masks)[:, None])
    d = torch.from_numpy(np.stack([edt_from_mask(m, EMPTY) for m in masks])[:, None])
    return g, d


@pytest.fixture
def shifted():
    g, d = batch(square())
    p = torch.from_numpy(square(col=11)[None, None])
    return p, g, d


def test_identical_binary_gives_zero():
    g, d = batch(square(), square(30, 20, 7))
    assert dice_loss(g, g).item() < 1e-5
    assert SobelLoss()(g, g).item() == 0.0
    assert LaplacianLoss()(g, g).item() == 0.0
    for mode in HAUSDORFF_MODES:
        loss, stats = hausdorff_edt_loss(g, g, d, EMPTY, mode)
        assert loss.item() == pytest.approx(0.0, abs=1e-6), mode
        assert stats["empty_pred_frac"] == 0.0


def test_one_pixel_shift_by_hand(shifted):
    p, g, d = shifted
    assert dice_loss(p, g).item() == pytest.approx(0.25, abs=1e-6)
    assert LaplacianLoss()(p, g).item() == pytest.approx(40 / 4096, abs=1e-7)
    for mode in ("soft_pg_detach_gp", "bidir_soft"):
        loss, stats = hausdorff_edt_loss(p, g, d, EMPTY, mode)
        assert loss.item() == pytest.approx(0.5, abs=1e-5), mode
        assert stats["d_pg"] == pytest.approx(0.25, abs=1e-5) and stats["d_gp"] == pytest.approx(0.25, abs=1e-5)
    loss, _ = hausdorff_edt_loss(p, g, d, EMPTY, "karimi_sq")
    assert loss.item() == pytest.approx(8 / 4096, abs=1e-7)


def test_sobel_matches_numpy_reference(shifted):
    p, g, _ = shifted

    def mag(x):
        gx = ndimage.correlate(x, np.array(SOBEL_X), mode="constant")
        gy = ndimage.correlate(x, np.array(SOBEL_Y), mode="constant")
        return np.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    ref = np.mean(np.abs(mag(p[0, 0].numpy().astype(np.float64)) - mag(g[0, 0].numpy().astype(np.float64))))
    assert SobelLoss()(p, g).item() == pytest.approx(ref, rel=1e-5)
    assert ref > 0


@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_constant_prediction_is_finite(fill):
    g, d = batch(square(), np.zeros((64, 64), np.float32))
    p = torch.full_like(g, fill)
    assert math.isfinite(dice_loss(p, g).item())
    assert math.isfinite(SobelLoss()(p, g).item()) and math.isfinite(LaplacianLoss()(p, g).item())
    for mode in HAUSDORFF_MODES:
        loss, stats = hausdorff_edt_loss(p, g, d, EMPTY, mode)
        assert math.isfinite(loss.item()) and all(math.isfinite(v) for v in stats.values()), mode
        assert stats["empty_pred_frac"] == (1.0 if fill == 0.0 else 0.0)


def test_empty_prediction_uses_empty_value():
    g, d = batch(square())
    loss, stats = hausdorff_edt_loss(torch.zeros_like(g), g, d, EMPTY, "soft_pg_detach_gp")
    assert stats["d_pg"] == pytest.approx(0.0) and stats["d_gp"] == pytest.approx(EMPTY, rel=1e-6)
    assert loss.item() == pytest.approx(EMPTY, rel=1e-6)
    _, stats7 = hausdorff_edt_loss(torch.zeros_like(g), g, d, 7.0, "soft_pg_detach_gp")
    assert stats7["d_gp"] == pytest.approx(7.0, rel=1e-6)


def test_invalid_hausdorff_mode(shifted):
    p, g, d = shifted
    with pytest.raises(ValueError):
        hausdorff_edt_loss(p, g, d, EMPTY, "max")


@pytest.fixture
def cfg():
    return load_config(ROOT / "configs" / "base.yaml")


@pytest.mark.parametrize("name", LOSS_NAMES)
def test_bal_parts_and_total(cfg, name, shifted):
    p, g, d = shifted
    cfg["loss"]["name"] = name
    cfg["loss"]["lambda"] = 0.7
    crit = BoundaryAwareLoss(cfg)
    logits = torch.logit(p.clamp(1e-4, 1 - 1e-4))
    total, parts = crit({"logits": logits, "prob": torch.sigmoid(logits)}, {"mask": g, "dist": d})
    assert set(parts) >= {"dice", "boundary", "total", "d_pg", "d_gp", "empty_pred_frac"}
    assert total.dtype == torch.float32 and total.dim() == 0
    assert parts["total"] == pytest.approx(parts["dice"] + 0.7 * parts["boundary"], rel=1e-6)
    if name == "dice":
        assert parts["boundary"] == 0.0
    if name == "dice+hausdorff":
        assert all(math.isfinite(parts[k]) for k in ("d_pg", "d_gp", "empty_pred_frac"))
    else:
        assert all(math.isnan(parts[k]) for k in ("d_pg", "d_gp", "empty_pred_frac"))


def test_bal_fp32_under_autocast(cfg, shifted):
    p, g, d = shifted
    cfg["loss"]["name"] = "dice+sobel"
    crit = BoundaryAwareLoss(cfg)
    logits = torch.logit(p.clamp(1e-4, 1 - 1e-4)).to(torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        total, _ = crit({"logits": logits}, {"mask": g.to(torch.bfloat16), "dist": d})
    assert total.dtype == torch.float32


def test_bal_rejects_unknown_names(cfg):
    cfg["loss"]["name"] = "dice+focal"
    with pytest.raises(ValueError):
        BoundaryAwareLoss(cfg)
    cfg["loss"]["name"] = "dice+hausdorff"
    cfg["loss"]["hausdorff_mode"] = "max"
    with pytest.raises(ValueError):
        BoundaryAwareLoss(cfg)


def test_kernels_are_non_persistent_buffers():
    assert SobelLoss().state_dict() == {} and LaplacianLoss().state_dict() == {}
    assert {n for n, _ in SobelLoss().named_buffers()} == {"kx", "ky"}
