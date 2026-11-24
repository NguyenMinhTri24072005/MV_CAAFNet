"""PAM & CAM theo DANet — SPECS §3.3, [A] C1.

Năng lượng attention và softmax luôn tính ở **fp32**: tắt autocast cục bộ (nếu không, `bmm` bị autocast
ép xuống fp16 và `X·Xᵀ` của CAM với N = 4096 có thể tràn số), rồi ép kết quả về kiểu của nhánh value.
`gamma` khởi tạo 0, nên lúc đầu mỗi khối là ánh xạ đồng nhất (residual).
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["PAM", "CAM", "fp32_softmax_bmm"]


def fp32_softmax_bmm(a: torch.Tensor, b: torch.Tensor, scale: float = 1.0, shift_max: bool = False) -> torch.Tensor:
    """`softmax(a @ b · scale)` (hoặc `softmax(max − a @ b)` khi `shift_max`) tính hoàn toàn ở fp32."""
    with torch.autocast(device_type=a.device.type, enabled=False):
        energy = torch.bmm(a.float(), b.float())
        if scale != 1.0:
            energy = energy * scale
        if shift_max:
            energy = energy.max(dim=-1, keepdim=True).values - energy
        return torch.softmax(energy, dim=-1)


class PAM(nn.Module):
    """Position Attention Module: attention giữa N = H·W vị trí; Q, K có `C // reduction` kênh."""

    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        inner = max(1, ch // reduction)
        self.query = nn.Conv2d(ch, inner, kernel_size=1)
        self.key = nn.Conv2d(ch, inner, kernel_size=1)
        self.value = nn.Conv2d(ch, ch, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Ma trận attention fp32 `(B, N, N)` — tách riêng để kiểm thử độ bão hòa."""
        b, _, h, w = x.shape
        n = h * w
        q = self.query(x).reshape(b, -1, n).transpose(1, 2)  # (B, N, C')
        k = self.key(x).reshape(b, -1, n)                    # (B, C', N)
        return fp32_softmax_bmm(q, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        v = self.value(x).reshape(b, c, h * w)                # (B, C, N)
        attn = self.attention(x).to(v.dtype)                  # (B, N, N)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(b, c, h, w)
        return self.gamma.to(out.dtype) * out + x


class CAM(nn.Module):
    """Channel Attention Module: attention giữa C kênh, dạng `softmax(max(E) − E)` của DANet."""

    def __init__(self, ch: int):
        super().__init__()
        self.channels = int(ch)
        self.gamma = nn.Parameter(torch.zeros(1))

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Ma trận attention fp32 `(B, C, C)`."""
        b, c, h, w = x.shape
        flat = x.reshape(b, c, h * w)
        return fp32_softmax_bmm(flat, flat.transpose(1, 2), shift_max=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.reshape(b, c, h * w)
        attn = self.attention(x).to(flat.dtype)
        out = torch.bmm(attn, flat).reshape(b, c, h, w)
        return self.gamma.to(out.dtype) * out + x
