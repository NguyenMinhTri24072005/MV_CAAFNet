"""Trực quan hóa — SPECS §9.3 (Figure 2, Figure 3, Table 11–13).

    python -m src.visualize --run-dirs <run_dir> [<run_dir> ...] --kind {loss,dotplot,diffmap,gradcam,weights} \
        --out-dir reports/figures [--data-dir /content/processed] [--ckpt last] [--split test] \
        [--patch-ids ID ...] [--layer aaf1] [--dataset luna16]

- `loss`    (Figure 2): `train_loss` theo epoch, mỗi loss một hình, mỗi run một đường (LUNA16, LIDC).
- `dotplot` (Figure 3): Dice (%) từng patch, sắp theo (patient_id, patch_id); < 80% tô đỏ.
- `diffmap` (Table 11–13): CT | GT | dự đoán | bản đồ sai khác (TP xanh lá, TN xanh dương, FP đỏ, FN vàng).
- `gradcam` (Table 11–13): Grad-CAM trên `enc_attn{1,2,3}` (`--layer aaf1/2/3`); **chỉ chế độ này cần torch**.
- `weights`: histogram `w1`, `w2` trên tập đánh giá.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import RESOLVED_CONFIG_NAME, load_config

KINDS = ("loss", "dotplot", "diffmap", "gradcam", "weights")
LAYERS = ("aaf1", "aaf2", "aaf3")
COLORS = {"tp": (0, 255, 0), "tn": (0, 0, 255), "fp": (255, 0, 0), "fn": (255, 255, 0)}  # [P]


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def run_label(run_dir: Path) -> str:
    return f"{run_dir.parent.name}/{run_dir.name}"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def eval_dir(run_dir: Path, split: str, ckpt: str) -> Path:
    return run_dir / f"eval_{split}_{ckpt}"


def run_config(run_dir: Path) -> dict:
    return load_config(run_dir / RESOLVED_CONFIG_NAME)


def diff_rgb(pred_bin: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Bản đồ sai khác uint8 (H, W, 3): TP xanh lá, TN xanh dương, FP đỏ, FN vàng."""
    p, g = np.asarray(pred_bin) > 0, np.asarray(gt) > 0
    out = np.empty((*p.shape, 3), dtype=np.uint8)
    out[p & g] = COLORS["tp"]
    out[~p & ~g] = COLORS["tn"]
    out[p & ~g] = COLORS["fp"]
    out[~p & g] = COLORS["fn"]
    return out


def load_split_images(data_dir: Path, dataset: str, split: str, patch_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Đọc `image`, `mask` (N, 64, 64) của các `patch_id` từ `<data_dir>/<dataset>/<split>.h5`."""
    import h5py

    with h5py.File(data_dir / dataset / f"{split}.h5", "r") as h5:
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in h5["patch_id"][...]]
        pos = {p: i for i, p in enumerate(ids)}
        missing = [p for p in patch_ids if p not in pos]
        if missing:
            raise KeyError(f"patch_id không có trong {split}.h5: {missing[:5]}")
        idx = [pos[p] for p in patch_ids]
        order = np.argsort(idx)
        sorted_idx = np.asarray(idx)[order]
        img = h5["image"][sorted_idx][:, 0]
        msk = h5["mask"][sorted_idx][:, 0]
    inv = np.argsort(order)
    return img[inv], msk[inv]


def default_patch_ids(run_dir: Path, split: str, ckpt: str, k: int = 4) -> list[str]:
    df = pd.read_csv(eval_dir(run_dir, split, ckpt) / "per_patch_metrics.csv", dtype={"patch_id": str})
    return df.sort_values(["dice", "patch_id"]).head(k)["patch_id"].tolist()


# ---------------------------------------------------------------------------
# Các loại hình
# ---------------------------------------------------------------------------
def plot_loss(run_dirs: list[Path], out_dir: Path) -> list[Path]:
    plt = _plt()
    groups: dict[str, list[Path]] = {}
    for rd in run_dirs:
        groups.setdefault(run_config(rd)["loss"]["name"], []).append(rd)
    paths = []
    for loss, rds in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(6, 4))
        for rd in rds:
            log = pd.read_csv(rd / "train_log.csv")
            ax.plot(log["epoch"], log["train_loss"], label=run_label(rd))
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
        ax.set_title(f"Figure 2 — {loss}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        path = out_dir / f"fig2_loss_{_slug(loss)}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_dotplot(run_dirs: list[Path], out_dir: Path, split: str, ckpt: str) -> list[Path]:
    plt = _plt()
    paths = []
    for rd in run_dirs:
        df = pd.read_csv(eval_dir(rd, split, ckpt) / "per_patch_metrics.csv", dtype={"patch_id": str, "patient_id": str})
        df = df.sort_values(["patient_id", "patch_id"]).reset_index(drop=True)
        dice = df["dice"].to_numpy() * 100
        low = dice < 80
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.scatter(np.flatnonzero(~low), dice[~low], s=4, c="tab:blue", label="Dice ≥ 80%")
        ax.scatter(np.flatnonzero(low), dice[low], s=4, c="tab:red", label="Dice < 80%")
        ax.axhline(80, color="gray", ls="--", lw=0.8)
        ax.set_ylim(-2, 102)
        ax.set_xlabel("patch (sắp theo patient_id, patch_id)")
        ax.set_ylabel("Dice (%)")
        ax.set_title(f"Figure 3 — {run_label(rd)} [{split}, {ckpt}] mean {dice.mean():.2f}%, < 80%: {int(low.sum())}/{len(dice)}")
        ax.legend(fontsize=8, loc="lower left")
        path = out_dir / f"fig3_dotplot_{_slug(run_label(rd))}_{split}_{ckpt}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_diffmap(run_dirs: list[Path], out_dir: Path, data_dir: Path, split: str, ckpt: str,
                 patch_ids: list[str] | None, dataset: str | None) -> list[Path]:
    plt = _plt()
    paths = []
    for rd in run_dirs:
        ds = dataset or run_config(rd)["dataset"]
        ids = list(patch_ids) if patch_ids else default_patch_ids(rd, split, ckpt)
        with np.load(eval_dir(rd, split, ckpt) / "preds.npz") as z:
            pos = {str(p): i for i, p in enumerate(z["patch_id"])}
            pred = np.stack([z["pred_bin"][pos[p]] for p in ids])
        img, gt = load_split_images(data_dir, ds, split, ids)
        fig, axes = plt.subplots(len(ids), 4, figsize=(8.4, 2.2 * len(ids)), squeeze=False)
        for r, pid in enumerate(ids):
            panels = [(img[r], "gray", "CT"), (gt[r], "gray", "GT"), (pred[r], "gray", "Dự đoán"),
                      (diff_rgb(pred[r], gt[r]), None, "TP lục · TN lam · FP đỏ · FN vàng")]
            for c, (data, cmap, title) in enumerate(panels):
                ax = axes[r, c]
                scale = {"cmap": cmap, "vmin": 0, "vmax": 1} if cmap else {}
                ax.imshow(data, interpolation="nearest", **scale)
                ax.set_title(title if r == 0 else "", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            axes[r, 0].set_ylabel(pid.replace("LIDC-IDRI-", ""), fontsize=7)
        fig.suptitle(f"{run_label(rd)} [{split}, {ckpt}]", fontsize=9)
        path = out_dir / f"diffmap_{_slug(run_label(rd))}_{split}_{ckpt}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_weights(run_dirs: list[Path], out_dir: Path, split: str, ckpt: str) -> list[Path]:
    plt = _plt()
    paths = []
    for rd in run_dirs:
        df = pd.read_csv(eval_dir(rd, split, ckpt) / "per_patch_metrics.csv")
        w1, w2 = df["w1"].to_numpy(dtype=float), df["w2"].to_numpy(dtype=float)
        if not np.isfinite(w1).any():
            print(f"bỏ qua weights cho {run_label(rd)}: mô hình không có trọng số w (baseline SE)")
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bins = np.linspace(0, 1, 41)
        ax.hist(w1[np.isfinite(w1)], bins=bins, alpha=0.7, label=f"w1 (PA) mean {np.nanmean(w1):.3f}")
        ax.hist(w2[np.isfinite(w2)], bins=bins, alpha=0.7, label=f"w2 (CA) mean {np.nanmean(w2):.3f}")
        ax.set_xlabel("trọng số")
        ax.set_ylabel("số patch")
        ax.set_title(f"Phân bố w1/w2 — {run_label(rd)} [{split}, {ckpt}]")
        ax.legend(fontsize=8)
        path = out_dir / f"weights_{_slug(run_label(rd))}_{split}_{ckpt}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def grad_cam(model, image: np.ndarray, layer: str, threshold: float = 0.5) -> tuple[np.ndarray, float]:
    """Grad-CAM (64×64, [0, 1]) trên `enc_attn{k}`; mục tiêu Σ(logits·pred_bin), fallback Σ(logits·prob)."""
    import torch
    import torch.nn.functional as F

    module = getattr(model, f"enc_attn{LAYERS.index(layer) + 1}")
    store: dict[str, Any] = {}
    h_fwd = module.register_forward_hook(lambda m, i, o: store.__setitem__("A", o))
    h_bwd = module.register_full_backward_hook(lambda m, gi, go: store.__setitem__("dA", go[0]))
    try:
        device = next(model.parameters()).device
        x = torch.as_tensor(image, dtype=torch.float32, device=device).view(1, 1, *image.shape[-2:]).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out = model(x)
        pred_bin = (out["prob"] >= threshold).float()
        target = (out["logits"] * pred_bin).sum() if pred_bin.sum() > 0 else (out["logits"] * out["prob"]).sum()
        target.backward()
        a, da = store["A"].detach(), store["dA"].detach()
        alpha = da.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((alpha * a).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else torch.zeros_like(cam)
        return cam.cpu().numpy(), float(pred_bin.sum().item())
    finally:
        h_fwd.remove()
        h_bwd.remove()


def plot_gradcam(run_dirs: list[Path], out_dir: Path, data_dir: Path, split: str, ckpt: str,
                 patch_ids: list[str] | None, layer: str, dataset: str | None) -> list[Path]:
    import torch  # chỉ chế độ gradcam cần torch

    from src.models import build_model
    from src.utils import load_checkpoint

    plt = _plt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = []
    for rd in run_dirs:
        cfg = run_config(rd)
        ds = dataset or cfg["dataset"]
        ids = list(patch_ids) if patch_ids else default_patch_ids(rd, split, ckpt)
        model = build_model(cfg).to(device)
        model.load_state_dict(load_checkpoint(rd / f"{ckpt}.pt", map_location="cpu")["model"])
        model.eval()
        img, gt = load_split_images(data_dir, ds, split, ids)
        fig, axes = plt.subplots(len(ids), 2, figsize=(4.4, 2.2 * len(ids)), squeeze=False)
        for r, pid in enumerate(ids):
            cam, n_pos = grad_cam(model, img[r], layer, float(cfg["eval"]["threshold"]))
            axes[r, 0].imshow(img[r], cmap="gray", vmin=0, vmax=1)
            axes[r, 0].contour(gt[r], levels=[0.5], colors="lime", linewidths=0.8)
            axes[r, 1].imshow(img[r], cmap="gray", vmin=0, vmax=1)
            axes[r, 1].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            axes[r, 0].set_ylabel(pid.replace("LIDC-IDRI-", ""), fontsize=7)
            if r == 0 or not n_pos:
                axes[r, 1].set_title(layer if n_pos else f"{layer} (pred rỗng → mục tiêu Σ logits·prob)", fontsize=7)
            for ax in axes[r]:
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0, 0].set_title("CT + GT", fontsize=8)
        fig.suptitle(f"Grad-CAM {run_label(rd)} [{split}, {ckpt}]", fontsize=9)
        path = out_dir / f"gradcam_{_slug(run_label(rd))}_{layer}_{split}_{ckpt}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Một hoặc nhiều thư mục run")
    parser.add_argument("--kind", choices=KINDS, required=True, help="Loại hình")
    parser.add_argument("--out-dir", default="reports/figures", help="Nơi ghi .png")
    parser.add_argument("--data-dir", default="data/processed", help="Thư mục <dataset>/<split>.h5 (diffmap, gradcam)")
    parser.add_argument("--ckpt", choices=("last", "best"), default="last")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--patch-ids", nargs="*", default=None, help="Mặc định: 4 patch Dice thấp nhất")
    parser.add_argument("--layer", choices=LAYERS, default="aaf1", help="Tầng Grad-CAM (enc_attn1/2/3)")
    parser.add_argument("--dataset", choices=("luna16", "lidc"), default=None, help="Mặc định theo config_resolved.yaml")
    return parser


def visualize(args: argparse.Namespace) -> list[Path]:
    run_dirs = [Path(p) for p in args.run_dirs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    if args.kind == "loss":
        return plot_loss(run_dirs, out_dir)
    if args.kind == "dotplot":
        return plot_dotplot(run_dirs, out_dir, args.split, args.ckpt)
    if args.kind == "diffmap":
        return plot_diffmap(run_dirs, out_dir, data_dir, args.split, args.ckpt, args.patch_ids, args.dataset)
    if args.kind == "weights":
        return plot_weights(run_dirs, out_dir, args.split, args.ckpt)
    return plot_gradcam(run_dirs, out_dir, data_dir, args.split, args.ckpt, args.patch_ids, args.layer, args.dataset)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in visualize(args):
        print(f"đã ghi {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
