"""Dòng gradient của loss & mô hình — SPECS §3.3 (C14), §4.2, Phụ lục C (cần torch)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.compute_edt import edt_from_mask  # noqa: E402
from src.losses import LaplacianLoss, SobelLoss, dice_loss, hausdorff_edt_loss  # noqa: E402
from src.models import CAM, PAM, ResUNetFamily  # noqa: E402

EMPTY = 90.51


def target():
    g = np.zeros((64, 64), np.float32)
    g[20:36, 20:36] = 1
    return torch.from_numpy(g)[None, None], torch.from_numpy(edt_from_mask(g, EMPTY))[None, None]


def soft_prediction():
    """P ≈ 0.9 trên nửa trái của G (cột 20–27), 0.1 ở nơi khác → nửa phải của G là FN."""
    p = torch.full((1, 1, 64, 64), 0.1)
    p[..., 20:36, 20:28] = 0.9
    return p.requires_grad_(True)


@pytest.mark.parametrize("loss_fn", [SobelLoss(), LaplacianLoss()], ids=["sobel", "laplacian"])
def test_edge_losses_have_gradient(loss_fn):
    g, _ = target()
    p = soft_prediction()
    loss_fn(p, g).backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_d_pg_gradient_and_detached_d_gp():
    g, d = target()
    p = soft_prediction()
    loss, _ = hausdorff_edt_loss(p, g, d, EMPTY, "soft_pg_detach_gp")
    loss.backward()
    assert p.grad.abs().sum() > 0
    q = p.detach().clone().requires_grad_(True)  # gradient chỉ đến từ d_PG (d_GP là hằng theo P)
    ((q * d).sum() / (q.sum() + 1e-6)).backward()
    assert torch.allclose(p.grad, q.grad, atol=1e-7)


@pytest.mark.parametrize("mode", ["bidir_soft", "karimi_sq"])
def test_false_negative_direction_has_gradient(mode):
    g, d = target()
    fn = (g > 0) & (soft_prediction().detach() < 0.5)
    assert fn.sum() > 0
    p_soft = soft_prediction()
    hausdorff_edt_loss(p_soft, g, d, EMPTY, "soft_pg_detach_gp")[0].backward()
    p = soft_prediction()
    hausdorff_edt_loss(p, g, d, EMPTY, mode)[0].backward()
    assert (p.grad[fn] < 0).all(), "tăng P ở pixel FN phải làm giảm loss"
    if mode == "bidir_soft":
        assert (p.grad[fn] < p_soft.grad[fn]).all(), "bidir_soft phạt FN mạnh hơn soft_pg_detach_gp"


def test_aac_gets_gradient_from_dice():
    torch.manual_seed(0)
    model = ResUNetFamily(enc_attn="adaptive")
    for m in model.modules():
        if isinstance(m, (PAM, CAM)):
            m.gamma.data.fill_(0.1)  # trạng thái sau vài bước học
    g, _ = target()
    x = torch.rand(2, 1, 64, 64)
    dice_loss(model(x)["prob"], g.repeat(2, 1, 1, 1)).backward()
    grads = [p.grad for p in model.aac.parameters()]
    assert all(gr is not None and torch.isfinite(gr).all() for gr in grads)
    assert sum(gr.abs().sum() for gr in grads) > 0


def test_aac_gradient_is_zero_while_gammas_are_zero():
    """Lúc khởi tạo γ = 0 nên PAM(x) = CAM(x) = x → ∂L/∂w1 = ∂L/∂w2 → gradient qua softmax của AAC = 0.

    AAC chỉ bắt đầu học khi γ rời 0 (γ luôn nhận gradient ngay từ bước đầu). Ghi lại hành vi này để
    khi đọc log `train_mean_w1` các epoch đầu không nhầm là lỗi.
    """
    g, _ = target()
    x = torch.rand(2, 1, 64, 64)

    def grads(gamma: float) -> tuple[float, float]:
        torch.manual_seed(0)
        model = ResUNetFamily(enc_attn="adaptive")
        for m in model.modules():
            if isinstance(m, (PAM, CAM)):
                m.gamma.data.fill_(gamma)
        dice_loss(model(x)["prob"], g.repeat(2, 1, 1, 1)).backward()
        aac = sum(p.grad.abs().sum().item() for p in model.aac.parameters())
        gam = sum(m.gamma.grad.abs().sum().item() for m in model.modules() if isinstance(m, (PAM, CAM)))
        return aac, gam

    aac0, gamma0 = grads(0.0)
    aac1, _ = grads(0.1)
    assert gamma0 > 0 and aac1 > 0
    assert aac0 < 1e-3 * aac1


def test_pam_cam_gammas_receive_gradient():
    torch.manual_seed(0)
    model = ResUNetFamily(enc_attn="static")
    g, _ = target()
    dice_loss(model(torch.rand(2, 1, 64, 64))["prob"], g.repeat(2, 1, 1, 1)).backward()
    gammas = [m.gamma for m in model.modules() if isinstance(m, (PAM, CAM))]
    assert len(gammas) == 6
    assert all(gm.grad is not None and gm.grad.abs().item() > 0 for gm in gammas)


def test_cam_gradient_at_64x64_not_vanishing():
    """C14: CAM ở 64×64 (N = 4096) trên đặc trưng stem thật.

    Bắt buộc: gradient tới đầu vào và γ hữu hạn, khác 0. Chẩn đoán (cảnh báo, không làm fail):
    tỉ lệ gradient đi qua nhánh logits attention so với nhánh value, và xác suất lớn nhất trung bình
    của softmax — nếu ≈ 1 thì attention gần one-hot (bão hòa), cần quyết định ở OPEN_ISSUES C14.
    """
    torch.manual_seed(0)
    model = ResUNetFamily(enc_attn="adaptive")
    feats = model.stem(torch.rand(2, 1, 64, 64)).detach()
    cam = CAM(32)
    cam.gamma.data.fill_(1.0)
    upstream = torch.randn_like(feats)

    x_full = feats.clone().requires_grad_(True)
    (cam(x_full) * upstream).sum().backward()
    assert torch.isfinite(x_full.grad).all() and x_full.grad.abs().sum() > 0
    assert torch.isfinite(cam.gamma.grad).all() and cam.gamma.grad.abs().item() > 0

    x_val = feats.clone().requires_grad_(True)
    attn = cam.attention(x_val).detach()
    flat = x_val.reshape(2, 32, -1)
    out = cam.gamma.detach() * torch.bmm(attn, flat).reshape_as(x_val) + x_val
    (out * upstream).sum().backward()
    ratio = ((x_full.grad - x_val.grad).norm() / x_val.grad.norm()).item()
    max_prob = attn.max(dim=-1).values.mean().item()
    print(f"CAM 64×64: attention-path/value-path grad ratio = {ratio:.3e}, mean max softmax = {max_prob:.4f}")
    if ratio < 1e-3 or max_prob > 0.99:
        warnings.warn(
            f"C14: softmax CAM ở 64×64 gần bão hòa (ratio={ratio:.2e}, max_prob={max_prob:.3f}); "
            "gradient chỉ còn đi qua nhánh value — ghi vào OPEN_ISSUES C14",
            stacklevel=1,
        )


def test_pam_gradient_at_64x64():
    torch.manual_seed(0)
    pam = PAM(32)
    pam.gamma.data.fill_(1.0)
    x = torch.randn(1, 32, 64, 64, requires_grad=True)
    (pam(x) * torch.randn(1, 32, 64, 64)).sum().backward()
    assert torch.isfinite(x.grad).all() and pam.gamma.grad.abs().item() > 0
    assert pam.query.weight.grad.abs().sum() > 0
