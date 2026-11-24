"""Baseline ResUNet++ — SPECS §3.6, [A] C8.

Cùng encoder / bridge / decoder ConvTranspose với CAAF-ResUNet, khối chú ý encoder là SE (reduction 8),
không có AAC/AAF. Chọn qua `model.name: resunetpp` (`aaf.mode` bị bỏ qua).
"""
from __future__ import annotations

from src.models.caaf_resunet import ResUNetFamily, family_kwargs

__all__ = ["ResUNetPP"]


class ResUNetPP(ResUNetFamily):
    def __init__(self, **kwargs):
        kwargs.pop("enc_attn", None)
        super().__init__(enc_attn="se", **kwargs)

    @classmethod
    def from_config(cls, cfg: dict) -> "ResUNetPP":
        return cls(**family_kwargs(cfg))
