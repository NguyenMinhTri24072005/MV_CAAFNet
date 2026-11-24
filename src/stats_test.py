"""Paired t-test hai phía giữa hai mô hình trên cùng tập patch — SPECS §9.1, [P] Table 6.

    python -m src.stats_test --a <caaf>/eval_test_last/per_patch_metrics.csv \
        --b <pcam>/eval_test_last/per_patch_metrics.csv --metric dice --out reports/stats/caaf_vs_pcam.json

Inner-join theo `patch_id` và bắt buộc khớp 100% (cùng 1.200 patch). Đổi sang %, `d_i = a_i − b_i`
(t > 0 ⇔ A tốt hơn B, đúng chiều Table 6: A = CAAF, B = PCAM tĩnh). Kèm Wilcoxon signed-rank
(bọc try/except khi mọi d_i = 0).
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import save_json

__all__ = ["paired_test", "build_parser", "main"]


def _load(x: Any, metric: str) -> pd.DataFrame:
    df = x if isinstance(x, pd.DataFrame) else pd.read_csv(x, dtype={"patch_id": str})
    if "patch_id" not in df.columns or metric not in df.columns:
        raise KeyError(f"cần cột patch_id và {metric!r}")
    if df["patch_id"].duplicated().any():
        raise ValueError("patch_id trùng lặp trong một tệp per_patch_metrics")
    return df[["patch_id", metric]].assign(patch_id=lambda d: d["patch_id"].astype(str))


def _num(v: Any) -> float | None:
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def paired_test(a: Any, b: Any, metric: str = "dice") -> dict[str, Any]:
    """Kiểm định ghép cặp `a` (mô hình A) vs `b` (mô hình B); `a`, `b` là đường dẫn CSV hoặc DataFrame."""
    da, db = _load(a, metric), _load(b, metric)
    joined = da.merge(db, on="patch_id", how="inner", suffixes=("_a", "_b"))
    if not (len(da) == len(db) == len(joined)):
        raise ValueError(
            f"hai tệp không cùng tập patch: a={len(da)}, b={len(db)}, chung={len(joined)} (cần khớp 100%)"
        )
    a_pct = joined[f"{metric}_a"].to_numpy(dtype=np.float64) * 100.0
    b_pct = joined[f"{metric}_b"].to_numpy(dtype=np.float64) * 100.0
    ok = np.isfinite(a_pct) & np.isfinite(b_pct)
    a_pct, b_pct = a_pct[ok], b_pct[ok]
    d = a_pct - b_pct
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t, p = stats.ttest_rel(a_pct, b_pct, alternative="two-sided")
    w_stat = w_p = None
    if len(d) >= 2 and np.any(d != 0):  # scipy mới trả NaN (không raise) khi mọi d_i = 0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                w_stat, w_p = stats.wilcoxon(a_pct, b_pct)
        except ValueError:
            w_stat = w_p = None
    return {
        "metric": metric,
        "unit": "%",
        "n": int(len(joined)),
        "n_used": int(ok.sum()),
        "mean_a": _num(a_pct.mean()) if len(a_pct) else None,
        "std_a": _num(a_pct.std(ddof=1)) if len(a_pct) > 1 else None,
        "mean_b": _num(b_pct.mean()) if len(b_pct) else None,
        "std_b": _num(b_pct.std(ddof=1)) if len(b_pct) > 1 else None,
        "mean_diff": _num(d.mean()) if len(d) else None,
        "std_diff": _num(d.std(ddof=1)) if len(d) > 1 else None,
        "t": _num(t),
        "p": _num(p),
        "wilcoxon_stat": _num(w_stat),
        "wilcoxon_p": _num(w_p),
        "direction": "d_i = a_i - b_i; t > 0 nghĩa là A tốt hơn B",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="per_patch_metrics.csv của mô hình A (CAAF)")
    parser.add_argument("--b", required=True, help="per_patch_metrics.csv của mô hình B (PCAM tĩnh)")
    parser.add_argument("--metric", default="dice", help="Cột metric (dice, iou, sensitivity, ...)")
    parser.add_argument("--out", required=True, help="Tệp JSON kết quả")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    res = paired_test(args.a, args.b, args.metric)
    res.update(a=Path(args.a).as_posix(), b=Path(args.b).as_posix())
    save_json(res, args.out)
    print(f"{args.metric}: A {res['mean_a']:.2f} vs B {res['mean_b']:.2f} (%) | Δ {res['mean_diff']:+.2f} | "
          f"t = {res['t']} | p = {res['p']} | n = {res['n']} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
