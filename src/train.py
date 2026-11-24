"""Huấn luyện CAAF-ResUNet / ResUNet++ — SPECS §6.1.

20 tham số CLI (Phụ lục A): 5 chung (`--config`, `--dataset`, `--data-dir`, `--out-dir`, `--set`)
+ 4 điều khiển (`--resume`, `--run-name`, `--max-batches`, `--overfit-steps`) + 11 cờ tắt (§6.1.1).

    python -m src.train --config configs/caaf_hausdorff.yaml --dataset luna16 \
        --data-dir /content/processed --out-dir "$DRIVE_ROOT/experiments" [--resume] [--batch-size 8 ...]

Thư mục run: `<out-dir>/<dataset>/<run_name>/` gồm `config_resolved.yaml`, `overrides.json`,
`commit.txt`, `env.json`, `env_history.jsonl`, `train_log.csv`, `train.log`, `last.pt` (mỗi epoch,
atomic), `best.pt` (chỉ trọng số + metadata, khi `val_dice` tăng).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.dataset import make_loaders
from src.losses import BoundaryAwareLoss
from src.metrics import compute_patch_metrics
from src.models import build_model, count_parameters
from src.utils import (
    OVERRIDES_NAME,
    RESOLVED_CONFIG_NAME,
    add_common_args,
    add_train_shortcuts,
    get_logger,
    git_commit,
    load_checkpoint,
    load_config,
    prepare_run_config,
    save_checkpoint,
    save_json,
    save_yaml,
    seed_everything,
)

# Khóa được phép khác `config_resolved.yaml` khi --resume: kéo dài số epoch của một run đang dở.
RESUME_MUTABLE_KEYS = frozenset({"train.epochs"})

LOG_COLUMNS = [
    "epoch", "lr", "train_loss", "train_dice_loss", "train_boundary_loss", "val_loss", "val_dice",
    "train_mean_w1", "train_mean_w2", "d_pg", "d_gp", "empty_pred_frac", "seconds",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ last.pt của run cùng tên")
    parser.add_argument("--run-name", default=None, help="Ghi đè tên run tự sinh")
    parser.add_argument("--max-batches", type=int, default=None, help="Giới hạn số batch mỗi epoch (smoke test)")
    parser.add_argument("--overfit-steps", type=int, default=None,
                        help="Lặp N bước trên đúng 1 batch train để kiểm tra khả năng học (không tạo run_dir)")
    add_train_shortcuts(parser)
    return parser


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
class WeightedMean:
    """Trung bình có trọng số theo kích thước batch thực (drop_last=False); bỏ qua giá trị NaN/None."""

    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.weight: dict[str, float] = {}

    def add(self, weight: float, **values: float | None) -> None:
        for key, value in values.items():
            if value is None or not math.isfinite(float(value)):
                continue
            self.total[key] = self.total.get(key, 0.0) + float(value) * weight
            self.weight[key] = self.weight.get(key, 0.0) + weight

    def mean(self, key: str) -> float:
        w = self.weight.get(key, 0.0)
        return self.total[key] / w if w > 0 else float("nan")


def make_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # PyTorch < 2.3


def ckpt_tmp_dir() -> str | None:
    """Trên Linux/Colab ghi checkpoint ra /tmp (SSD) trước rồi mới chép sang Drive FUSE."""
    return "/tmp" if sys.platform.startswith("linux") else None


def env_info(device: torch.device) -> dict[str, Any]:
    info = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
        "git_commit": git_commit(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(device).total_memory / 1024 ** 3, 1)
    return info


def rng_state(loader) -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    gen = getattr(loader, "generator", None)
    if gen is not None:
        state["loader"] = gen.get_state()
    return state


def set_rng_state(state: dict[str, Any], loader) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    gen = getattr(loader, "generator", None)
    if gen is not None and "loader" in state:
        gen.set_state(state["loader"].cpu())


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def config_diff(a: dict, b: dict) -> list[str]:
    fa, fb = _flatten(a), _flatten(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k, "<thiếu>") != fb.get(k, "<thiếu>"))


def append_log(path: Path, row: dict[str, Any]) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


def truncate_log(path: Path, max_epoch: int) -> int:
    """Xóa mọi dòng `epoch > max_epoch` (ghi lại atomic); trả số dòng đã xóa."""
    if not path.exists():
        return 0
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    keep = [r for r in rows if r.get("epoch") not in (None, "") and int(float(r["epoch"])) <= max_epoch]
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in LOG_COLUMNS} for r in keep)
    tmp.replace(path)
    return len(rows) - len(keep)


def _append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _to_device(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {k: batch[k].to(device, non_blocking=True) for k in ("image", "mask", "dist")}


def _batch_dice(prob: torch.Tensor, mask: torch.Tensor, threshold: float) -> np.ndarray:
    return compute_patch_metrics(prob, mask, threshold)["dice"]


# ---------------------------------------------------------------------------
# Một epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp: bool, grad_clip, max_batches) -> WeightedMean:
    model.train()
    acc = WeightedMean()
    for i, raw in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = _to_device(raw, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            out = model(batch["image"])
        total, parts = criterion(out, batch)  # fp32 bên trong criterion
        scaler.scale(total).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        bs = int(batch["image"].shape[0])
        w = out["w"]
        w_mean = w.detach().float().mean(dim=0).tolist() if w is not None else [None, None]
        acc.add(bs, total=parts["total"], dice=parts["dice"], boundary=parts["boundary"], d_pg=parts["d_pg"],
                d_gp=parts["d_gp"], empty_pred_frac=parts["empty_pred_frac"], w1=w_mean[0], w2=w_mean[1])
    return acc


@torch.no_grad()
def validate(model, loader, criterion, device, amp: bool, threshold: float, max_batches) -> tuple[float, float]:
    model.eval()
    acc = WeightedMean()
    dices: list[np.ndarray] = []
    for i, raw in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = _to_device(raw, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            out = model(batch["image"])
        _, parts = criterion(out, batch)
        acc.add(int(batch["image"].shape[0]), total=parts["total"])
        dices.append(_batch_dice(out["prob"], batch["mask"], threshold))
    val_dice = float(np.concatenate(dices).mean()) if dices else float("nan")
    return acc.mean("total"), val_dice


# ---------------------------------------------------------------------------
# Overfit 1 batch (P6.4b)
# ---------------------------------------------------------------------------
def overfit(cfg: dict, loaders: dict, device: torch.device, steps: int, logger) -> int:
    batch = _to_device(next(iter(loaders["train"])), device)
    model = build_model(cfg).to(device)
    criterion = BoundaryAwareLoss(cfg).to(device)
    opt_cfg = cfg["train"]["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg["lr"]),
                                  weight_decay=float(opt_cfg["weight_decay"]), betas=tuple(opt_cfg["betas"]))
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = make_scaler(amp)
    threshold = float(cfg["eval"]["threshold"])
    logger.info("overfit: %d bước trên 1 batch (%d patch), loss=%s", steps, batch["image"].shape[0], cfg["loss"]["name"])
    dice = float("nan")
    for step in range(1, steps + 1):
        model.train()
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            out = model(batch["image"])
        total, parts = criterion(out, batch)
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 25 == 0 or step == steps:
            dice = float(_batch_dice(out["prob"].detach(), batch["mask"], threshold).mean())
            logger.info("overfit bước %4d | loss %.4f | dice_loss %.4f | Dice(batch) %.2f%%", step, parts["total"],
                        parts["dice"], 100 * dice)
    verdict = "ĐẠT" if dice > 0.95 else "CHƯA ĐẠT"
    logger.info("overfit kết thúc: Dice(batch) = %.2f%% → %s ngưỡng 95%%", 100 * dice, verdict)
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, cfg, run_name, overrides = prepare_run_config(args)
    seed_everything(int(cfg["seed"]), bool(cfg["train"]["deterministic"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = make_loaders(cfg, args.data_dir, cfg["dataset"])

    if args.overfit_steps:
        return overfit(cfg, loaders, device, int(args.overfit_steps), get_logger("train.overfit"))

    run_dir = Path(args.out_dir) / cfg["dataset"] / run_name
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    log_csv = run_dir / "train_log.csv"
    if last_path.exists() and not args.resume:
        raise SystemExit(f"{run_dir} đã có last.pt — thêm --resume để tiếp tục, hoặc đổi --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"train.{run_name}", run_dir / "train.log")

    model = build_model(cfg).to(device)
    criterion = BoundaryAwareLoss(cfg).to(device)
    opt_cfg, sch_cfg = cfg["train"]["optimizer"], cfg["train"]["scheduler"]
    if opt_cfg["name"] != "adamw" or sch_cfg["name"] != "steplr":
        raise SystemExit("Chỉ hỗ trợ optimizer adamw và scheduler steplr ([P] §2.6.3)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg["lr"]),
                                  weight_decay=float(opt_cfg["weight_decay"]), betas=tuple(opt_cfg["betas"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(sch_cfg["step_size"]), gamma=float(sch_cfg["gamma"]))
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = make_scaler(amp)
    epochs = int(cfg["train"]["epochs"])
    best_val = float("-inf")
    start_epoch = 1
    env = env_info(device)

    if args.resume and last_path.exists():
        saved = load_config(run_dir / RESOLVED_CONFIG_NAME)
        diff = config_diff(saved, cfg)
        blocked = [k for k in diff if k not in RESUME_MUTABLE_KEYS]
        if blocked:
            raise SystemExit(f"--resume: config hiện tại khác {RESOLVED_CONFIG_NAME} ở các khóa {blocked}")
        if diff:  # chỉ train.epochs đổi → cập nhật config_resolved.yaml và ghi lại lý do
            logger.info("--resume: đổi train.epochs %s → %s", saved["train"]["epochs"], epochs)
            save_yaml(cfg, run_dir / RESOLVED_CONFIG_NAME)
        ckpt = load_checkpoint(last_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        best_val = float(ckpt["best_val_dice"])
        set_rng_state(ckpt["rng"], loaders["train"])
        removed = truncate_log(log_csv, int(ckpt["epoch"]))
        start_epoch = int(ckpt["epoch"]) + 1
        _append_jsonl(run_dir / "env_history.jsonl", {**env, "event": "resume", "from_epoch": start_epoch,
                                                       "log_rows_removed": removed, "config_changed": diff})
        logger.info("resume từ epoch %d (đã bỏ %d dòng log thừa)", start_epoch, removed)
        if start_epoch > epochs:
            logger.info("run đã xong đủ %d epoch", epochs)
            return 0
    else:
        if args.resume:
            logger.info("--resume nhưng chưa có last.pt → bắt đầu run mới")
        save_yaml(cfg, run_dir / RESOLVED_CONFIG_NAME)
        save_json(overrides, run_dir / OVERRIDES_NAME)
        (run_dir / "commit.txt").write_text(env["git_commit"] + "\n", encoding="utf-8")
        save_json({**env, "params": count_parameters(model)}, run_dir / "env.json")
        _append_jsonl(run_dir / "env_history.jsonl", {**env, "event": "start", "from_epoch": 1})
        log_csv.unlink(missing_ok=True)

    params = count_parameters(model)["total"]
    logger.info("run %s | %s | %s | %d tham số | device %s | amp %s | train %d / val %d patch",
                run_name, cfg["model"]["name"], cfg["loss"]["name"], params, device, amp,
                len(loaders["train"].dataset), len(loaders["val"].dataset))
    threshold = float(cfg["eval"]["threshold"])
    grad_clip = cfg["train"]["grad_clip"]

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        loaders["train"].dataset.set_epoch(epoch)
        epoch_lr = optimizer.param_groups[0]["lr"]  # ghi TRƯỚC scheduler.step()
        acc = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, device, amp, grad_clip, args.max_batches)
        val_loss, val_dice = validate(model, loaders["val"], criterion, device, amp, threshold, args.max_batches)
        scheduler.step()
        row = {
            "epoch": epoch, "lr": epoch_lr, "train_loss": acc.mean("total"), "train_dice_loss": acc.mean("dice"),
            "train_boundary_loss": acc.mean("boundary"), "val_loss": val_loss, "val_dice": val_dice,
            "train_mean_w1": acc.mean("w1"), "train_mean_w2": acc.mean("w2"), "d_pg": acc.mean("d_pg"),
            "d_gp": acc.mean("d_gp"), "empty_pred_frac": acc.mean("empty_pred_frac"),
            "seconds": round(time.time() - t0, 2),
        }
        improved = math.isfinite(val_dice) and val_dice > best_val
        if improved:
            best_val = val_dice
        append_log(log_csv, row)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_dice": best_val,
            "rng": rng_state(loaders["train"]),
            "cfg": cfg,
        }
        save_checkpoint(last_path, state, tmp_dir=ckpt_tmp_dir())
        if improved:
            save_checkpoint(best_path, {"epoch": epoch, "model": model.state_dict(), "best_val_dice": best_val, "cfg": cfg},
                            tmp_dir=ckpt_tmp_dir())
        logger.info(
            "epoch %3d/%d | lr %.2e | train %.4f (dice %.4f, bnd %.4f) | val %.4f | val Dice %.2f%%%s | w1 %.3f | %.0fs",
            epoch, epochs, epoch_lr, row["train_loss"], row["train_dice_loss"], row["train_boundary_loss"], val_loss,
            100 * val_dice, " *" if improved else "", row["train_mean_w1"], row["seconds"],
        )
    logger.info("xong: best val Dice %.2f%% → %s", 100 * best_val, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
