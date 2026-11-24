"""Mạng tổng thể họ ResUNet (CAAF-ResUNet và các biến thể ablation) — SPECS §3.6, [P] Figure 1.

`enc_attn` chọn khối chú ý sau stem / RCM1 / RCM2 của encoder:
`se` (ResUNet++ baseline), `pa`, `ca`, `static` (PCAM w = 0.5/0.5), `adaptive` (CAAF, w từ AAC).

Luồng forward (B, 1, 64, 64), channels = (32, 64, 128, 256):
    x1 = stem(x)            32×64×64   → a1 = enc_attn1(x1, w)
    x2 = rcm1(a1)  s=2      64×32×32   → a2 = enc_attn2(x2, w)
    x3 = rcm2(a2)  s=2     128×16×16   → a3 = enc_attn3(x3, w)
    x4 = rcm3(a3)  s=2     256× 8× 8
    b  = aspp_bridge(x4)   256× 8× 8
    decoder: att1(enc=x3, dec=b) → up1 → cat x3 → rcm_d1 → att2(enc=x2) → up2 → cat x2 → rcm_d2
             → att3(enc=x1) → up3 → cat x1 → rcm_d3   (skip lấy từ x1, x2, x3 TRƯỚC AAF, C7;
             decoder dùng AttentionModule, không dùng AAF, [F] A3)
    head: aspp_out → Conv1×1 → logits, prob = sigmoid(logits)
"""
from __future__ import annotations

from typing import Any, Literal, Sequence

import torch
from torch import nn

from src.models.aac import AAC
from src.models.aaf import AAF, AAF_MODES
from src.models.aspp import ASPP
from src.models.blocks import AttentionModule, ResidualConvModule, SEBlock, StemBlock, UpSample

__all__ = ["ENC_ATTN_MODES", "ResUNetFamily", "CAAFResUNet", "family_kwargs"]

ENC_ATTN_MODES = ("se", *AAF_MODES)
_CONST_W = {"pa": (1.0, 0.0), "ca": (0.0, 1.0)}


class ResUNetFamily(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        enc_attn: Literal["se", "pa", "ca", "static", "adaptive"] = "adaptive",
        *,
        upsample: str = "convtranspose",
        pool_size: int = 8,
        qkv_bias: bool = False,
        pam_reduction: int = 8,
        att_sigmoid: bool = False,
        aspp_rates: Sequence[int] = (6, 12, 18),
        aspp_gap: bool = False,
        se_reduction: int = 8,
        static_weights: Sequence[float] = (0.5, 0.5),
    ):
        super().__init__()
        if enc_attn not in ENC_ATTN_MODES:
            raise ValueError(f"enc_attn phải thuộc {ENC_ATTN_MODES}, nhận {enc_attn!r}")
        if upsample != "convtranspose":
            raise ValueError(f"model.upsample chỉ hỗ trợ 'convtranspose' [P] Fig. 1, nhận {upsample!r}")
        c0, c1, c2, c3 = (int(c) for c in channels)
        self.enc_attn = enc_attn
        self.channels = (c0, c1, c2, c3)
        self.static_weights = tuple(float(v) for v in static_weights)

        # Encoder
        self.stem = StemBlock(in_ch, c0)
        self.aac = AAC(c0, pool_size=pool_size, qkv_bias=qkv_bias) if enc_attn == "adaptive" else None
        self.enc_attn1 = self._make_attn(c0, enc_attn, pam_reduction, se_reduction, static_weights)
        self.rcm1 = ResidualConvModule(c0, c1, stride=2)
        self.enc_attn2 = self._make_attn(c1, enc_attn, pam_reduction, se_reduction, static_weights)
        self.rcm2 = ResidualConvModule(c1, c2, stride=2)
        self.enc_attn3 = self._make_attn(c2, enc_attn, pam_reduction, se_reduction, static_weights)
        self.rcm3 = ResidualConvModule(c2, c3, stride=2)
        # Bridge
        self.aspp_bridge = ASPP(c3, c3, rates=aspp_rates, gap=aspp_gap)
        # Decoder
        self.att1 = AttentionModule(c2, c3, att_sigmoid)
        self.up1 = UpSample(c3)
        self.rcm_d1 = ResidualConvModule(c3 + c2, c2, stride=1)
        self.att2 = AttentionModule(c1, c2, att_sigmoid)
        self.up2 = UpSample(c2)
        self.rcm_d2 = ResidualConvModule(c2 + c1, c1, stride=1)
        self.att3 = AttentionModule(c0, c1, att_sigmoid)
        self.up3 = UpSample(c1)
        self.rcm_d3 = ResidualConvModule(c1 + c0, c0, stride=1)
        # Head
        self.aspp_out = ASPP(c0, c0, rates=aspp_rates, gap=aspp_gap)
        self.head = nn.Conv2d(c0, 1, kernel_size=1)

    @staticmethod
    def _make_attn(ch: int, mode: str, pam_reduction: int, se_reduction: int, static_weights) -> nn.Module:
        if mode == "se":
            return SEBlock(ch, se_reduction)
        return AAF(ch, mode, pam_reduction=pam_reduction, static_weights=static_weights)

    def _fusion_weights(self, x1: torch.Tensor) -> torch.Tensor | None:
        if self.enc_attn == "adaptive":
            return self.aac(x1)
        if self.enc_attn == "se":
            return None
        const = self.static_weights if self.enc_attn == "static" else _CONST_W[self.enc_attn]
        return x1.new_tensor(const).expand(x1.shape[0], 2)

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> dict[str, Any]:
        x1 = self.stem(x)
        w = self._fusion_weights(x1)
        a1 = self.enc_attn1(x1, w)
        x2 = self.rcm1(a1)
        a2 = self.enc_attn2(x2, w)
        x3 = self.rcm2(a2)
        a3 = self.enc_attn3(x3, w)
        x4 = self.rcm3(a3)

        b = self.aspp_bridge(x4)

        d = self.att1(enc=x3, dec=b)
        d = self.rcm_d1(torch.cat([self.up1(d), x3], dim=1))
        d = self.att2(enc=x2, dec=d)
        d = self.rcm_d2(torch.cat([self.up2(d), x2], dim=1))
        d = self.att3(enc=x1, dec=d)
        d = self.rcm_d3(torch.cat([self.up3(d), x1], dim=1))

        logits = self.head(self.aspp_out(d))
        out: dict[str, Any] = {
            "prob": torch.sigmoid(logits),
            "logits": logits,
            "w": w,
            "feats": {"aaf1": a1, "aaf2": a2, "aaf3": a3},
        }
        if return_intermediates:
            out["intermediates"] = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "b": b, "a1": a1, "a2": a2, "a3": a3, "d": d}
        return out


def family_kwargs(cfg: dict) -> dict[str, Any]:
    """Ánh xạ khối `model` / `aaf` của config (configs/base.yaml) sang tham số `ResUNetFamily`."""
    m = cfg["model"]
    return {
        "in_ch": int(m["in_channels"]),
        "channels": tuple(m["channels"]),
        "upsample": m["upsample"],
        "pool_size": int(m["aac"]["pool_size"]),
        "qkv_bias": bool(m["aac"]["qkv_bias"]),
        "pam_reduction": int(m["attention"]["pam_reduction"]),
        "att_sigmoid": bool(m["attention"]["att_sigmoid"]),
        "aspp_rates": tuple(m["aspp"]["rates"]),
        "aspp_gap": bool(m["aspp"]["gap"]),
        "se_reduction": int(m["se_reduction"]),
        "static_weights": tuple(cfg["aaf"]["static_weights"]),
    }


class CAAFResUNet(ResUNetFamily):
    """CAAF-ResUNet: `enc_attn` ∈ {pa, ca, static, adaptive} lấy từ `aaf.mode`."""

    @classmethod
    def from_config(cls, cfg: dict) -> "CAAFResUNet":
        mode = cfg["aaf"]["mode"]
        if mode not in AAF_MODES:
            raise ValueError(f"aaf.mode phải thuộc {AAF_MODES}, nhận {mode!r}")
        return cls(enc_attn=mode, **family_kwargs(cfg))
