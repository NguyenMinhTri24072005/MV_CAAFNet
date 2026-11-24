"""Shape & cấu trúc mô hình — SPECS §3.1–3.6, Phụ lục C (cần torch; chạy trên Colab)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import (  # noqa: E402
    ASPP, ENC_ATTN_MODES, AttentionModule, ResUNetFamily, ResUNetPP, UpSample, build_model, count_parameters,
)
from src.models.aaf import AAF  # noqa: E402
from src.utils import load_config  # noqa: E402

B = 2
EXPECTED = {
    "x1": (B, 32, 64, 64), "a1": (B, 32, 64, 64),
    "x2": (B, 64, 32, 32), "a2": (B, 64, 32, 32),
    "x3": (B, 128, 16, 16), "a3": (B, 128, 16, 16),
    "x4": (B, 256, 8, 8), "b": (B, 256, 8, 8),
    "d": (B, 32, 64, 64),
}
CONFIGS = {
    "baseline_resunetpp": "se", "pa_resunet": "pa", "ca_resunet": "ca", "pcam_dice": "static",
    "pcam_sobel": "static", "pcam_laplacian": "static", "pcam_hausdorff": "static",
    "caaf_sobel": "adaptive", "caaf_laplacian": "adaptive", "caaf_hausdorff": "adaptive",
}


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(0)
    return torch.rand(B, 1, 64, 64)


@pytest.mark.parametrize("att_sigmoid", [False, True])
@pytest.mark.parametrize("enc_attn", ENC_ATTN_MODES)
def test_forward_shapes(enc_attn, att_sigmoid, x):
    torch.manual_seed(0)
    model = ResUNetFamily(enc_attn=enc_attn, att_sigmoid=att_sigmoid)
    out = model(x, return_intermediates=True)
    assert set(out) == {"prob", "logits", "w", "feats", "intermediates"}
    assert out["prob"].shape == out["logits"].shape == (B, 1, 64, 64)
    assert torch.isfinite(out["logits"]).all()
    assert (out["prob"] >= 0).all() and (out["prob"] <= 1).all()
    for name, shape in EXPECTED.items():
        assert tuple(out["intermediates"][name].shape) == shape, name
    for i in (1, 2, 3):
        assert out["feats"][f"aaf{i}"] is out["intermediates"][f"a{i}"]
    w = out["w"]
    if enc_attn == "se":
        assert w is None
    else:
        assert w.shape == (B, 2) and torch.allclose(w.sum(-1), torch.ones(B), atol=1e-6)
        expect = {"pa": [1.0, 0.0], "ca": [0.0, 1.0], "static": [0.5, 0.5]}.get(enc_attn)
        if expect is not None:
            assert torch.equal(w, torch.tensor([expect] * B))


def test_att1_takes_x3_and_bridge(x):
    model = ResUNetFamily(enc_attn="adaptive")
    seen = {}
    model.att1.register_forward_hook(lambda m, args, kwargs, out: seen.update(kwargs) or None, with_kwargs=True)
    out = model(x, return_intermediates=True)
    inter = out["intermediates"]
    assert seen["enc"] is inter["x3"] and seen["dec"] is inter["b"]
    assert tuple(seen["enc"].shape) == (B, 128, 16, 16) and tuple(seen["dec"].shape) == (B, 256, 8, 8)


def test_skips_taken_before_aaf(x):
    model = ResUNetFamily(enc_attn="adaptive")
    captured = {}
    for name in ("rcm_d1", "rcm_d2", "rcm_d3"):
        getattr(model, name).register_forward_pre_hook(lambda m, args, n=name: captured.update({n: args[0]}))
    inter = model(x, return_intermediates=True)["intermediates"]
    for name, skip, aaf, up_ch in (("rcm_d1", "x3", "a3", 256), ("rcm_d2", "x2", "a2", 128), ("rcm_d3", "x1", "a1", 64)):
        part = captured[name][:, up_ch:]
        assert torch.equal(part, inter[skip]), name
        assert not torch.equal(part, inter[aaf]), name


@pytest.mark.parametrize("stem, enc_attn", sorted(CONFIGS.items()))
def test_build_model_from_configs(stem, enc_attn, x):
    cfg = load_config(ROOT / "configs" / f"{stem}.yaml")
    model = build_model(cfg)
    assert model.enc_attn == enc_attn
    assert isinstance(model, ResUNetPP) == (enc_attn == "se")
    assert (model.aac is not None) == (enc_attn == "adaptive")
    assert model(x[:1].repeat(2, 1, 1, 1))["prob"].shape == (B, 1, 64, 64)


def test_build_model_rejects_unknown():
    cfg = load_config(ROOT / "configs" / "base.yaml")
    cfg["model"]["name"] = "unet"
    with pytest.raises(ValueError):
        build_model(cfg)
    cfg["model"]["name"] = "caaf_resunet"
    cfg["aaf"]["mode"] = "none"
    with pytest.raises(ValueError):
        build_model(cfg)


def test_aspp_branch_order_conv_relu_bn():
    aspp = ASPP(32, 32, rates=(6, 12, 18), gap=False)
    assert aspp.gap is None and len(aspp.branches) == 3
    for branch, r in zip(aspp.branches, (6, 12, 18)):
        assert [type(m) for m in branch] == [nn.Conv2d, nn.ReLU, nn.BatchNorm2d]
        assert branch[0].dilation == (r, r) and branch[0].padding == (r, r) and branch[0].bias is not None
    assert aspp.project.in_channels == 96
    with_gap = ASPP(32, 32, gap=True)
    assert with_gap.project.in_channels == 128
    assert with_gap(torch.rand(2, 32, 8, 8)).shape == (2, 32, 8, 8)


def test_attention_module_and_upsample_layout():
    att = AttentionModule(128, 256)
    last = att.attn[-1]
    assert isinstance(last, nn.Conv2d) and last.kernel_size == (1, 1) and last.padding == (0, 0) and last.out_channels == 1
    assert att.att_sigmoid is False
    up = UpSample(256).up
    assert isinstance(up, nn.ConvTranspose2d)
    assert (up.in_channels, up.out_channels, up.kernel_size, up.stride) == (256, 256, (2, 2), (2, 2))


@pytest.mark.parametrize("enc_attn", ENC_ATTN_MODES)
def test_no_dropout_and_bias_rules(enc_attn):
    model = ResUNetFamily(enc_attn=enc_attn)
    assert not any(isinstance(m, nn.modules.dropout._DropoutNd) for m in model.modules())
    for seq in (m for m in model.modules() if isinstance(m, nn.Sequential)):
        layers = list(seq)
        for a, b in zip(layers, layers[1:]):
            if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                assert a.bias is None, "Conv2d ngay trước BN phải bias=False"
    aafs = [m for m in model.modules() if isinstance(m, AAF)]
    assert len(aafs) == (0 if enc_attn == "se" else 3)


def test_count_parameters():
    model = ResUNetFamily(enc_attn="adaptive")
    info = count_parameters(model)
    assert info["total"] == info["trainable"] == sum(p.numel() for p in model.parameters())
    assert sum(info["by_module"].values()) == info["total"]
    assert {"stem", "aac", "enc_attn1", "aspp_bridge", "head"} <= set(info["by_module"])
    assert "aac" not in count_parameters(ResUNetFamily(enc_attn="static"))["by_module"]
