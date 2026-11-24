"""Khối cơ bản của CAAF-ResUNet / ResUNet++ — SPECS §3.1.

Quy ước [A]: `Conv2d` đứng ngay trước `BatchNorm2d` có `bias=False`, các `Conv2d` khác `bias=True`;
khởi tạo mặc định của PyTorch (C10); không dùng Dropout (C11).
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["StemBlock", "ResidualConvModule", "AttentionModule", "SEBlock", "UpSample"]


def conv3x3(in_ch: int, out_ch: int, stride: int = 1, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=bias)


class StemBlock(nn.Module):
    """`main = Conv3×3 → BN → ReLU → Conv3×3`, `skip = Conv3×3 → BN`, trả `main + skip` [P] Fig. 1."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            conv3x3(in_ch, out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            conv3x3(out_ch, out_ch),
        )
        self.skip = nn.Sequential(conv3x3(in_ch, out_ch, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class ResidualConvModule(nn.Module):
    """Khối residual pre-activation (RCM): `BN → ReLU → Conv(s) → BN → ReLU → Conv` + `Conv(s) → BN`."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            conv3x3(in_ch, out_ch, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            conv3x3(out_ch, out_ch),
        )
        self.skip = nn.Sequential(conv3x3(in_ch, out_ch, stride=stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class AttentionModule(nn.Module):
    """Attention gate kiểu ResUNet++ ở decoder [P] Fig. 1 §2.

    `e = MaxPool(Conv3×3(ReLU(BN(enc))))`, `d = Conv3×3(ReLU(BN(dec)))`,
    `a = Conv1×1(dec→1, p=0)(ReLU(BN(e + d)))` ([F] A4: p=0); tùy chọn sigmoid (mặc định tắt, C6/C13);
    trả `a * dec`. `enc` có độ phân giải gấp đôi `dec`.
    """

    def __init__(self, enc_ch: int, dec_ch: int, att_sigmoid: bool = False):
        super().__init__()
        self.enc = nn.Sequential(
            nn.BatchNorm2d(enc_ch),
            nn.ReLU(inplace=True),
            conv3x3(enc_ch, dec_ch),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dec = nn.Sequential(nn.BatchNorm2d(dec_ch), nn.ReLU(inplace=True), conv3x3(dec_ch, dec_ch))
        self.attn = nn.Sequential(
            nn.BatchNorm2d(dec_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_ch, 1, kernel_size=1, padding=0),
        )
        self.att_sigmoid = bool(att_sigmoid)

    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        a = self.attn(self.enc(enc) + self.dec(dec))
        if self.att_sigmoid:
            a = torch.sigmoid(a)
        return a * dec


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (reduction 8) cho baseline ResUNet++ [A] C8.

    `w` bị bỏ qua — chỉ để có cùng giao diện `forward(x, w)` với `AAF`.
    """

    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, ch // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(ch, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, ch), nn.Sigmoid())

    def forward(self, x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
        b, c = x.shape[:2]
        s = self.fc(self.pool(x).flatten(1))
        return x * s.view(b, c, 1, 1)


class UpSample(nn.Module):
    """`ConvTranspose2d(ch, ch, kernel_size=2, stride=2)` — giữ nguyên số kênh [A] C5/C12."""

    def __init__(self, ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
