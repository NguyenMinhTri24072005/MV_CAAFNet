"""Chia train/val/test — SPECS §5.4.

- `patient` (chuẩn, mọi bảng chính): hoán vị bệnh nhân tất định, lấy đủ đúng `test_patches` (1.200,
  [P] Table 4) patch cho test; bệnh nhân ranh giới chỉ lấy đúng số patch còn thiếu và **bỏ phần dư**
  (`dropped`) để 0% rò rỉ bệnh nhân. Phần còn lại chia val (10% số bệnh nhân) và train.
- `slice` (chỉ cho thử nghiệm độ nhạy S1, Phase 8b): hoán vị patch, có rò rỉ bệnh nhân (ghi rõ trong JSON).

    python -m src.data.make_splits --index data/interim/luna16/patches.csv --out data/splits/luna16.json \
        --config configs/data.yaml [--mode patient|slice]
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.ct_utils import save_json_atomic
from src.utils import load_config

SPLIT_NAMES = ("train", "val", "test")
MODES = ("patient", "slice")


def _n_val(n: int, val_ratio: float) -> int:
    return max(1, int(round(val_ratio * n))) if n > 0 else 0


def split_patient(df: pd.DataFrame, seed: int, test_patches: int, val_ratio: float) -> dict[str, Any]:
    """Chia theo bệnh nhân (§5.4 bước 1–5)."""
    by_patient = {str(pid): sorted(g["patch_id"].astype(str)) for pid, g in df.groupby("patient_id")}
    all_patients = sorted(by_patient)
    rng = np.random.default_rng(seed)
    perm = [str(p) for p in rng.permutation(np.array(all_patients, dtype=object))]

    test_ids: list[str] = []
    test_patients: list[str] = []
    dropped: list[str] = []
    boundary = None
    idx = 0
    while idx < len(perm) and len(test_ids) < test_patches:
        pid = perm[idx]
        idx += 1
        patches = by_patient[pid]
        need = test_patches - len(test_ids)
        if len(patches) <= need:
            test_ids.extend(patches)
        else:
            chosen = sorted(int(i) for i in rng.choice(len(patches), size=need, replace=False))
            chosen_set = set(chosen)
            test_ids.extend(patches[i] for i in chosen)
            dropped.extend(patches[i] for i in range(len(patches)) if i not in chosen_set)
            boundary = pid
        test_patients.append(pid)

    rem = perm[idx:]
    n_val = _n_val(len(rem), val_ratio)
    val_patients, train_patients = rem[:n_val], rem[n_val:]
    return {
        "train": sorted(p for pid in train_patients for p in by_patient[pid]),
        "val": sorted(p for pid in val_patients for p in by_patient[pid]),
        "test": sorted(test_ids),
        "dropped": sorted(dropped),
        "patients": {
            "train": sorted(train_patients),
            "val": sorted(val_patients),
            "test": sorted(test_patients),
        },
        "boundary_patient": boundary,
    }


def split_slice(df: pd.DataFrame, seed: int, test_patches: int, val_ratio: float) -> dict[str, Any]:
    """Chia theo patch (S1): có thể rò rỉ bệnh nhân giữa các tập."""
    all_ids = sorted(df["patch_id"].astype(str))
    rng = np.random.default_rng(seed)
    perm = [str(p) for p in rng.permutation(np.array(all_ids, dtype=object))]
    rem = perm[test_patches:]
    n_val = _n_val(len(rem), val_ratio)
    out = {
        "train": sorted(rem[n_val:]),
        "val": sorted(rem[:n_val]),
        "test": sorted(perm[:test_patches]),
        "dropped": [],
        "boundary_patient": None,
    }
    pid_of = dict(zip(df["patch_id"].astype(str), df["patient_id"].astype(str)))
    out["patients"] = {name: sorted({pid_of[p] for p in out[name]}) for name in SPLIT_NAMES}
    return out


def verify_splits(result: dict, df: pd.DataFrame, test_patches: int, mode: str) -> None:
    """Kiểm chứng bắt buộc (§5.4 bước 6); dùng `raise` thay vì `assert` để không bị tắt bởi `python -O`."""
    def check(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    check(len(result["test"]) == test_patches, f"test có {len(result['test'])} patch, cần {test_patches}")
    pats = {k: set(v) for k, v in result["patients"].items()}
    if mode == "patient":
        check(pats["train"].isdisjoint(pats["val"]), "train và val chung bệnh nhân")
        check(pats["train"].isdisjoint(pats["test"]), "train và test chung bệnh nhân")
        check(pats["val"].isdisjoint(pats["test"]), "val và test chung bệnh nhân")
    lists = [result[k] for k in (*SPLIT_NAMES, "dropped")]
    total = sum(len(x) for x in lists)
    check(total == len(df), f"tổng {total} patch khác số patch đầu vào {len(df)}")
    check(len(set().union(*map(set, lists))) == total, "có patch_id xuất hiện ở nhiều danh sách")
    check(set().union(*map(set, lists)) == set(df["patch_id"].astype(str)), "danh sách patch_id không khớp đầu vào")


def _num(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def split_stats(df: pd.DataFrame, result: dict) -> dict[str, Any]:
    idx = df.set_index(df["patch_id"].astype(str))
    stats = {}
    for name in (*SPLIT_NAMES, "dropped"):
        sub = idx.loc[result[name]] if result[name] else idx.iloc[0:0]
        stats[name] = {
            "n_patients": int(sub["patient_id"].nunique()),
            "n_scans": int(sub["series_uid"].nunique()) if "series_uid" in sub else None,
            "n_patches": int(len(sub)),
            "diameter_mean": _num(sub["diameter_mm"].mean()) if "diameter_mm" in sub and len(sub) else None,
            "diameter_std": _num(sub["diameter_mm"].std()) if "diameter_mm" in sub and len(sub) > 1 else None,
            "area_mean": _num(sub["area_union_px"].mean()) if "area_union_px" in sub and len(sub) else None,
            "area_std": _num(sub["area_union_px"].std()) if "area_union_px" in sub and len(sub) > 1 else None,
            "touches_border_frac": _num(sub["touches_border"].astype(bool).mean())
            if "touches_border" in sub and len(sub) else None,
        }
    pats = {k: set(v) for k, v in result["patients"].items()}
    stats["patient_overlap"] = {
        "train_val": len(pats["train"] & pats["val"]),
        "train_test": len(pats["train"] & pats["test"]),
        "val_test": len(pats["val"] & pats["test"]),
    }
    return stats


def make_splits(
    df: pd.DataFrame,
    mode: str = "patient",
    seed: int = 2024,
    test_patches: int = 1200,
    val_ratio: float = 0.1,
) -> dict[str, Any]:
    """Chia tập theo `mode`, kiểm chứng và kèm thống kê phân bố."""
    if mode not in MODES:
        raise ValueError(f"mode phải là {MODES}, nhận {mode!r}")
    if df["patch_id"].duplicated().any():
        raise ValueError("patches.csv có patch_id trùng lặp")
    if len(df) < test_patches:
        raise ValueError(f"chỉ có {len(df)} patch, không đủ {test_patches} patch cho test")
    df = df.copy()
    df["patch_id"] = df["patch_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    fn = split_patient if mode == "patient" else split_slice
    result = fn(df, int(seed), int(test_patches), float(val_ratio))
    verify_splits(result, df, int(test_patches), mode)
    return {
        "mode": mode,
        "seed": int(seed),
        "test_patches": int(test_patches),
        "val_ratio": float(val_ratio),
        "n_total": int(len(df)),
        **result,
        "stats": split_stats(df, result),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="patches.csv của bước tiền xử lý")
    parser.add_argument("--out", required=True, help="Tệp JSON kết quả, ví dụ data/splits/luna16.json")
    parser.add_argument("--config", default="configs/data.yaml", help="Config dữ liệu (khối split)")
    parser.add_argument("--mode", choices=MODES, default=None, help="Ghi đè split.mode")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)["split"]
    mode = args.mode or cfg.get("mode", "patient")
    df = pd.read_csv(args.index, dtype={"patch_id": str, "patient_id": str, "series_uid": str})
    result = make_splits(df, mode, cfg["seed"], cfg["test_patches"], cfg["val_ratio"])
    payload = {
        "index": Path(args.index).as_posix(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **result,
    }
    save_json_atomic(args.out, payload)
    st = result["stats"]
    print(
        f"[{mode}] train {st['train']['n_patches']} patch/{st['train']['n_patients']} BN, "
        f"val {st['val']['n_patches']}/{st['val']['n_patients']}, test {st['test']['n_patches']}/"
        f"{st['test']['n_patients']}, dropped {st['dropped']['n_patches']} → {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
