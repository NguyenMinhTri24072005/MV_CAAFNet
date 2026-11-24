"""Độ đo theo từng patch — SPECS §4.3, [P] Eq. 39–43, [A] F1–F5.

Viết bằng numpy (nhận cả `np.ndarray` lẫn `torch.Tensor`), nên dùng được ở máy không có torch.

Quy ước biên (theo từng mẫu):
- Dice = 2TP / (2TP + FP + FN); IoU = TP / (TP + FP + FN). Cả P và G rỗng (TP+FP+FN = 0) → 1.0;
  chỉ P rỗng mà G không rỗng → 0.0 (F4, vẫn tính vào trung bình).
- Sensitivity = TP / (TP + FN), Miss Rate = FN / (TP + FN) = 1 − Sensitivity; TP + FN = 0 → NaN.
- Specificity = TN / (TN + FP); TN + FP = 0 → NaN. NaN bị bỏ qua khi tổng hợp (nanmean / nanstd).
Đơn vị: `compute_patch_metrics` trả thang [0, 1] (cho per_patch_metrics.csv); `summarize_metrics` trả `%`
(× 100) với `std` dùng `ddof=1` (F3) (cho results.json, báo cáo).
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

__all__ = ["METRIC_NAMES", "COUNT_NAMES", "confusion", "compute_patch_metrics", "summarize_metrics"]

METRIC_NAMES = ("dice", "iou", "sensitivity", "miss_rate", "specificity")
COUNT_NAMES = ("tp", "fp", "fn", "tn", "gt_area", "pred_area")


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy() if x.is_floating_point() else x.detach().cpu().numpy()
    return np.asarray(x)


def confusion(p_bin: Any, g: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(tp, fp, fn, tn)` int64 theo từng mẫu `(B,)`; `p_bin`, `g` shape `(B, ...)`, giá trị > 0 là dương."""
    p = _to_numpy(p_bin) > 0
    t = _to_numpy(g) > 0
    if p.shape != t.shape:
        raise ValueError(f"shape dự đoán {p.shape} khác shape nhãn {t.shape}")
    p = p.reshape(p.shape[0], -1)
    t = t.reshape(t.shape[0], -1)
    tp = (p & t).sum(axis=1).astype(np.int64)
    fp = (p & ~t).sum(axis=1).astype(np.int64)
    fn = (~p & t).sum(axis=1).astype(np.int64)
    tn = (~p & ~t).sum(axis=1).astype(np.int64)
    return tp, fp, fn, tn


def _ratio(num: np.ndarray, den: np.ndarray, when_zero: float) -> np.ndarray:
    num = num.astype(np.float64)
    den = den.astype(np.float64)
    out = np.full(num.shape, when_zero, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def compute_patch_metrics(prob: Any, mask: Any, threshold: float = 0.5) -> dict[str, np.ndarray]:
    """Metric từng patch, thang [0, 1]: dice, iou, sensitivity, miss_rate, specificity + tp/fp/fn/tn, diện tích."""
    p_bin = _to_numpy(prob) >= float(threshold)
    g = _to_numpy(mask) > 0.5
    tp, fp, fn, tn = confusion(p_bin, g)
    return {
        "dice": _ratio(2 * tp, 2 * tp + fp + fn, 1.0),
        "iou": _ratio(tp, tp + fp + fn, 1.0),
        "sensitivity": _ratio(tp, tp + fn, np.nan),
        "miss_rate": _ratio(fn, tp + fn, np.nan),
        "specificity": _ratio(tn, tn + fp, np.nan),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gt_area": tp + fn,
        "pred_area": tp + fp,
    }


def _column(data: Any, name: str) -> np.ndarray | None:
    if hasattr(data, "columns"):
        return data[name].to_numpy(dtype=np.float64) if name in data.columns else None
    if isinstance(data, Mapping):
        return np.asarray(data[name], dtype=np.float64) if name in data else None
    raise TypeError("summarize_metrics cần pandas.DataFrame hoặc dict tên → mảng")


def _clean(v: float) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)


def summarize_metrics(data: Any, metrics: Iterable[str] = METRIC_NAMES) -> dict[str, dict[str, float | None]]:
    """`{metric: {"mean", "std", "n"}}` theo **%** (× 100), `std` với `ddof=1`, bỏ qua NaN.

    `n` là số patch có giá trị hữu hạn; `std` = None khi n < 2, `mean` = None khi n = 0.
    """
    out: dict[str, dict[str, float | None]] = {}
    for name in metrics:
        col = _column(data, name)
        if col is None:
            continue
        vals = col[np.isfinite(col)] * 100.0
        n = int(vals.size)
        out[name] = {
            "mean": _clean(vals.mean()) if n else None,
            "std": _clean(vals.std(ddof=1)) if n > 1 else None,
            "n": n,
        }
    return out
