"""4 chế độ AAF và baseline SE — SPECS §3.5–3.6, Table 3, Phụ lục C (cần torch)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import AAC, AAF, CAM, PAM, ResUNetFamily, SEBlock  # noqa: E402


def kinds(model: nn.Module) -> set[type]:
    return {type(m) for m in model.modules()}


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(2, 32, 16, 16)


def make_aaf(mode: str) -> AAF:
    torch.manual_seed(0)
    aaf = AAF(32, mode)
    for m in (aaf.pam, aaf.cam):
        if m is not None:
            m.gamma.data.fill_(0.7)  # khác 0 để PAM/CAM không còn là ánh xạ đồng nhất
    return aaf


def test_pa_only_pam(x):
    aaf = make_aaf("pa")
    assert aaf.cam is None and CAM not in kinds(aaf) and PAM in kinds(aaf)
    assert torch.allclose(aaf(x), aaf.out_conv(aaf.pam(x)))


def test_ca_only_cam(x):
    aaf = make_aaf("ca")
    assert aaf.pam is None and PAM not in kinds(aaf) and CAM in kinds(aaf)
    assert torch.allclose(aaf(x), aaf.out_conv(aaf.cam(x)))


def test_static_fixed_half_weights(x):
    aaf = make_aaf("static")
    assert torch.equal(aaf.static_weights, torch.tensor([0.5, 0.5]))
    assert "static_weights" in dict(aaf.named_buffers()) and "static_weights" not in dict(aaf.named_parameters())
    assert torch.allclose(aaf(x), aaf.out_conv(0.5 * aaf.pam(x) + 0.5 * aaf.cam(x)), atol=1e-6)


def test_adaptive_w1_multiplies_pa_w2_multiplies_ca(x):
    aaf = make_aaf("adaptive")
    pa, ca = aaf.pam(x), aaf.cam(x)
    assert torch.allclose(aaf(x, torch.tensor([[1.0, 0.0]] * 2)), aaf.out_conv(pa), atol=1e-6)
    assert torch.allclose(aaf(x, torch.tensor([[0.0, 1.0]] * 2)), aaf.out_conv(ca), atol=1e-6)
    w = torch.tensor([[0.26, 0.74], [0.9, 0.1]])
    expect = aaf.out_conv(w[:, :1, None, None] * pa + w[:, 1:, None, None] * ca)
    assert torch.allclose(aaf(x, w), expect, atol=1e-6)
    with pytest.raises(ValueError):
        aaf(x)


def test_output_conv_only(x):
    aaf = AAF(32, "adaptive")
    assert isinstance(aaf.out_conv, nn.Conv2d) and aaf.out_conv.kernel_size == (3, 3) and aaf.out_conv.padding == (1, 1)
    assert not any(isinstance(m, (nn.BatchNorm2d, nn.ReLU)) for m in aaf.modules())


def test_invalid_mode():
    with pytest.raises(ValueError):
        AAF(32, "none")


@pytest.mark.parametrize(
    "mode, has_aac, has_pam, has_cam, has_se",
    [
        ("se", False, False, False, True),
        ("pa", False, True, False, False),
        ("ca", False, False, True, False),
        ("static", False, True, True, False),
        ("adaptive", True, True, True, False),
    ],
)
def test_model_level_modules(mode, has_aac, has_pam, has_cam, has_se):
    model = ResUNetFamily(enc_attn=mode)
    k = kinds(model)
    assert (AAC in k, PAM in k, CAM in k, SEBlock in k) == (has_aac, has_pam, has_cam, has_se)
    assert (model.aac is not None) == has_aac


def test_static_model_reports_constant_w():
    model = ResUNetFamily(enc_attn="static")
    out = model(torch.rand(2, 1, 64, 64))
    assert torch.equal(out["w"], torch.tensor([[0.5, 0.5]] * 2))
