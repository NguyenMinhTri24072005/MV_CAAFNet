"""Đánh giá một run trên tập test/val — SPECS §6.2.

    python -m src.evaluate --run-dir "$DRIVE_ROOT/experiments/luna16/caaf_hausdorff_s42" \
        --data-dir /content/processed [--dataset luna16] [--ckpt last|best] [--split test|val] \
        [--batch-size 32] [--threshold 0.5] [--save-preds | --no-save-preds]

`dataset`, `ckpt`, `threshold` mặc định đọc từ `config_resolved.yaml` của run. Kết quả ghi vào
`<run_dir>/eval_<split>_<ckpt>/`:
- `per_patch_metrics.csv` — thang [0, 1], một dòng/patch (sắp theo patch_id)
- `results.json` — mean ± std theo %, std `ddof=1`, kèm `mean_w1`, `mean_w2`
- `preds.npz` — `prob` float16 (N, 64, 64), `pred_bin` uint8 (N, 64, 64), `patch_id` (N,)

Phần ghi kết quả (`write_eval_outputs`) không cần torch; torch chỉ được import khi chạy mô hình.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.ct_utils import save_npz_atomic
from src.metrics import METRIC_NAMES, compute_patch_metrics, summarize_metrics
from src.utils import RESOLVED_CONFIG_NAME, load_config, save_json

PER_PATCH_COLUMNS = [
    "patch_id", "patient_id", "dice", "iou", "sensitivity", "miss_rate", "specificity",
    "tp", "fp", "fn", "tn", "gt_area", "pred_area", "w1", "w2",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="Thư mục run (chứa config_resolved.yaml, last.pt)")
    parser.add_argument("--data-dir", default="data/processed", help="Thư mục chứa <dataset>/<split>.h5")
    parser.add_argument("--dataset", choices=("luna16", "lidc"), default=None, help="Mặc định theo config_resolved.yaml")
    parser.add_argument("--ckpt", choices=("last", "best"), default=None, help="Mặc định eval.ckpt trong config")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--batch-size", type=int, default=None, help="Mặc định train.batch_size")
    parser.add_argument("--threshold", type=float, default=None, help="Mặc định eval.threshold (0.5)")
    parser.add_argument("--save-preds", action=argparse.BooleanOptionalAction, default=True, help="Ghi preds.npz")
    return parser


def eval_dir_name(split: str, ckpt: str) -> str:
    return f"eval_{split}_{ckpt}"


def per_patch_table(prob: np.ndarray, mask: np.ndarray, patch_id, patient_id, w: np.ndarray | None,
                    threshold: float) -> pd.DataFrame:
    """Bảng metric từng patch (thang [0, 1]); `w` (N, 2) hoặc None (baseline SE → NaN)."""
    m = compute_patch_metrics(prob, mask, threshold)
    n = len(m["dice"])
    w = np.full((n, 2), np.nan) if w is None else np.asarray(w, dtype=np.float64).reshape(n, 2)
    df = pd.DataFrame({"patch_id": list(patch_id), "patient_id": list(patient_id), **m, "w1": w[:, 0], "w2": w[:, 1]})
    return df[PER_PATCH_COLUMNS]


def write_eval_outputs(
    out_dir: str | Path,
    df: pd.DataFrame,
    meta: dict[str, Any],
    prob: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Ghi `per_patch_metrics.csv`, `results.json` (+ `preds.npz` nếu có `prob`); trả nội dung results.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(df["patch_id"].astype(str).to_numpy(), kind="stable")  # `prob` sắp theo cùng thứ tự với df
    df = df.iloc[order].reset_index(drop=True)
    df.to_csv(out_dir / "per_patch_metrics.csv", index=False, lineterminator="\n")
    w1, w2 = df["w1"].to_numpy(dtype=float), df["w2"].to_numpy(dtype=float)
    results = {
        **meta,
        "n": int(len(df)),
        "threshold": float(threshold),
        "metrics": summarize_metrics(df, METRIC_NAMES),
        "mean_w1": float(np.nanmean(w1)) if np.isfinite(w1).any() else None,
        "mean_w2": float(np.nanmean(w2)) if np.isfinite(w2).any() else None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_json(results, out_dir / "results.json")
    if prob is not None:
        p = np.asarray(prob, dtype=np.float32).reshape(len(df), *np.asarray(prob).shape[-2:])[order]
        save_npz_atomic(
            out_dir / "preds.npz",
            prob=p.astype(np.float16),
            pred_bin=(p >= threshold).astype(np.uint8),
            patch_id=np.asarray(df["patch_id"].astype(str).tolist()),
        )
    return results


def predict(model, loader, device, amp: bool) -> dict[str, Any]:
    """Chạy mô hình trên loader (không shuffle) → prob (N, 1, H, W), mask, patch_id, patient_id, w (hoặc None)."""
    import torch

    model.eval()
    probs, masks, ws, pids, patients = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                out = model(img)
            probs.append(out["prob"].float().cpu().numpy())
            masks.append(batch["mask"].numpy())
            ws.append(None if out["w"] is None else out["w"].float().cpu().numpy())
            pids += list(batch["patch_id"])
            patients += list(batch["patient_id"])
    empty = (0, 1, 64, 64)
    return {
        "prob": np.concatenate(probs) if probs else np.zeros(empty, np.float32),
        "mask": np.concatenate(masks) if masks else np.zeros(empty, np.float32),
        "w": None if not ws or ws[0] is None else np.concatenate(ws),
        "patch_id": pids,
        "patient_id": patients,
    }


def evaluate_run(
    run_dir: str | Path,
    data_dir: str | Path = "data/processed",
    dataset: str | None = None,
    ckpt: str | None = None,
    split: str = "test",
    batch_size: int | None = None,
    threshold: float | None = None,
    save_preds: bool = True,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import H5SegDataset
    from src.models import build_model
    from src.utils import load_checkpoint

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / RESOLVED_CONFIG_NAME)
    dataset = dataset or cfg["dataset"]
    ckpt = ckpt or cfg["eval"]["ckpt"]
    threshold = float(cfg["eval"]["threshold"] if threshold is None else threshold)
    batch_size = int(batch_size or cfg["train"]["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = load_checkpoint(run_dir / f"{ckpt}.pt", map_location="cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model"])
    ds = H5SegDataset(Path(data_dir) / dataset / f"{split}.h5", augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=int(cfg["data"].get("num_workers", 0)),
                        pin_memory=device.type == "cuda")
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    pred = predict(model, loader, device, amp)
    df = per_patch_table(pred["prob"], pred["mask"], pred["patch_id"], pred["patient_id"], pred["w"], threshold)
    meta = {
        "run": run_dir.name,
        "dataset": dataset,
        "split": split,
        "ckpt": ckpt,
        "epoch": int(state.get("epoch", -1)),
        "model": cfg["model"]["name"],
        "aaf_mode": cfg["aaf"]["mode"] if cfg["model"]["name"] == "caaf_resunet" else None,
        "loss": cfg["loss"]["name"],
        "seed": int(cfg["seed"]),
    }
    prob = pred["prob"][:, 0] if save_preds else None  # cùng thứ tự với df; write_eval_outputs tự sắp theo patch_id
    return write_eval_outputs(run_dir / eval_dir_name(split, ckpt), df, meta, prob, threshold)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    res = evaluate_run(args.run_dir, args.data_dir, args.dataset, args.ckpt, args.split, args.batch_size,
                       args.threshold, args.save_preds)
    m = res["metrics"]
    fmt = lambda k: f"{m[k]['mean']:.2f} ± {m[k]['std']:.2f}" if m[k]["mean"] is not None and m[k]["std"] is not None else "—"  # noqa: E731
    w = f" | w1 {res['mean_w1']:.3f} w2 {res['mean_w2']:.3f}" if res["mean_w1"] is not None else ""
    print(f"{res['dataset']}/{res['run']} [{res['split']}, {res['ckpt']}, epoch {res['epoch']}] n={res['n']} | "
          f"Dice {fmt('dice')} | IoU {fmt('iou')} | Sens {fmt('sensitivity')} | Miss {fmt('miss_rate')} | "
          f"Spec {fmt('specificity')}{w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
