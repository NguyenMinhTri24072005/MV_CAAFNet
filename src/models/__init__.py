"""Mô hình CAAF-ResUNet và baseline ResUNet++ — SPECS Phase 3."""
from __future__ import annotations

from torch import nn

from src.models.aac import AAC
from src.models.aaf import AAF, AAF_MODES
from src.models.aspp import ASPP
from src.models.attention import CAM, PAM
from src.models.blocks import AttentionModule, ResidualConvModule, SEBlock, StemBlock, UpSample
from src.models.caaf_resunet import ENC_ATTN_MODES, CAAFResUNet, ResUNetFamily, family_kwargs
from src.models.resunetpp import ResUNetPP

__all__ = [
    "AAC", "AAF", "AAF_MODES", "ASPP", "CAM", "PAM", "AttentionModule", "ResidualConvModule", "SEBlock",
    "StemBlock", "UpSample", "ENC_ATTN_MODES", "ResUNetFamily", "CAAFResUNet", "ResUNetPP", "family_kwargs",
    "MODEL_NAMES", "build_model", "count_parameters",
]

MODEL_NAMES = ("caaf_resunet", "resunetpp")


def build_model(cfg: dict) -> ResUNetFamily:
    """`model.name == "resunetpp"` → ResUNet++ (SE); `"caaf_resunet"` → `enc_attn = aaf.mode`."""
    name = cfg["model"]["name"]
    if name == "resunetpp":
        return ResUNetPP.from_config(cfg)
    if name == "caaf_resunet":
        return CAAFResUNet.from_config(cfg)
    raise ValueError(f"model.name phải thuộc {MODEL_NAMES}, nhận {name!r}")


def count_parameters(model: nn.Module) -> dict:
    """Số tham số: tổng, cần học, và theo từng module con cấp 1 (cho `reports/model_params.md`)."""
    by_module = {
        name: sum(p.numel() for p in child.parameters())
        for name, child in model.named_children()
    }
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "by_module": {k: v for k, v in by_module.items() if v},
    }
