"""AAC — SPECS §3.4, [P] Algorithm 1, Eq. 1–9, Phụ lục C (cần torch)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.aac import AAC  # noqa: E402


@pytest.fixture
def aac():
    torch.manual_seed(0)
    return AAC(in_ch=32, pool_size=8, qkv_bias=False)


@pytest.mark.parametrize("size", [64, 32, 16])
def test_output_is_softmax_pair(aac, size):
    w = aac(torch.randn(3, 32, size, size))
    assert w.shape == (3, 2)
    assert torch.allclose(w.sum(-1), torch.ones(3), atol=1e-6)
    assert ((w > 0) & (w < 1)).all()


def test_input_dependent(aac):
    torch.manual_seed(1)
    w = aac(torch.randn(2, 32, 64, 64) * torch.tensor([1.0, 3.0]).view(2, 1, 1, 1))
    assert not torch.allclose(w[0], w[1])


def test_matches_algorithm_1(aac):
    x = torch.randn(2, 32, 64, 64)
    f = F.adaptive_avg_pool2d(x, 8).flatten(2).transpose(1, 2)               # Eq. 1–2
    q, k, v = f @ aac.w_q.weight.T, f @ aac.w_k.weight.T, f @ aac.w_v.weight.T  # Eq. 3 (không bias)
    a = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(32), dim=-1)       # Eq. 4
    g = (a @ v).mean(dim=1)                                                  # Eq. 5–6
    z = aac.fc2(torch.relu(aac.fc1(g)))                                      # Eq. 7–8
    assert torch.allclose(aac(x), torch.softmax(z, dim=-1), atol=1e-6)       # Eq. 9


def test_pooling_to_8x8(aac):
    small = torch.randn(2, 32, 8, 8)
    big = F.interpolate(small, scale_factor=8, mode="nearest")  # adaptive_avg_pool(8) khôi phục đúng `small`
    assert torch.allclose(aac(small), aac(big), atol=1e-6)


def test_layer_layout(aac):
    for lin in (aac.w_q, aac.w_k, aac.w_v):
        assert lin.bias is None and lin.weight.shape == (32, 32)
    assert aac.fc1.weight.shape == (16, 32) and aac.fc2.weight.shape == (2, 16)
    assert AAC(32, qkv_bias=True).w_q.bias is not None
    with pytest.raises(ValueError):
        AAC(1)


def test_fp32_under_autocast(aac):
    x = torch.randn(2, 32, 64, 64)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        w = aac(x)
    assert w.dtype == torch.float32 and torch.isfinite(w).all()
    assert torch.allclose(w.sum(-1), torch.ones(2), atol=1e-5)
