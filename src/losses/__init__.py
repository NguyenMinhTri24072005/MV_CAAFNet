"""Hàm mất mát — SPECS Phase 4 (§4.2)."""
from src.losses.bal import LOSS_NAMES, BoundaryAwareLoss
from src.losses.dice import DiceLoss, dice_loss
from src.losses.hausdorff_edt import HAUSDORFF_MODES, HausdorffEDTLoss, hausdorff_edt_loss, pred_edt
from src.losses.laplacian import LaplacianLoss
from src.losses.sobel import SobelLoss

__all__ = [
    "LOSS_NAMES", "BoundaryAwareLoss", "DiceLoss", "dice_loss", "HAUSDORFF_MODES", "HausdorffEDTLoss",
    "hausdorff_edt_loss", "pred_edt", "LaplacianLoss", "SobelLoss",
]
