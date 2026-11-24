"""Đóng gói train/val/test thành HDF5 + manifest SHA-256 — SPECS §5.5.

Mỗi `<split>.h5` gồm (sắp xếp theo `patch_id` tăng dần để ghép cặp paired t-test giữa mọi mô hình):
- `image`  float32 (N, 1, 64, 64) ∈ [0, 1] = `window_hu(HU, lo, hi)`        — nén gzip mức 4
- `mask`   uint8   (N, 1, 64, 64) ∈ {0, 1}                                  — nén gzip mức 4
- `dist`   float32 (N, 1, 64, 64) = `edt_from_mask(mask)` (pixel, 0 trong mask) — nén gzip mức 4
- `patch_id`, `patient_id` (utf-8), `diameter_mm` float32, `area_px` int32 (= area_union_px),
  `touches_border` uint8; thuộc tính gốc `dataset`, `split`, `data_config_json`, `split_mode`,
  `split_seed`, `created`, `git_commit`, ...

Luôn ghi ra `--out-dir` (SSD `/content/...` trên Colab) trước, chép kèm `patches.csv` và `<ds>.json`,
ghi `manifest.json`; nếu có `--copy-to-drive-dir` thì chép một lần sang Drive và kiểm lại SHA-256.

    python -m src.data.pack_h5 --interim-dir data/interim/luna16 --splits data/splits/luna16.json \
        --out-dir /content/processed_tmp/luna16 --config configs/data.yaml \
        [--copy-to-drive-dir "$DRIVE_ROOT/data/processed/luna16"]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.compute_edt import EMPTY_VALUE, edt_from_mask
from src.data.ct_utils import save_json_atomic, window_hu
from src.utils import git_commit, load_config

SPLITS = ("train", "val", "test")
MANIFEST = "manifest.json"


def sha256_file(path: str | Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_interim(interim_dir: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    interim_dir = Path(interim_dir)
    hu = np.load(interim_dir / "patches_hu.npy")
    masks = np.load(interim_dir / "masks.npy")
    meta = pd.read_csv(interim_dir / "patches.csv", dtype={"patch_id": str, "patient_id": str, "series_uid": str})
    if not (len(hu) == len(masks) == len(meta)):
        raise ValueError(f"patches_hu ({len(hu)}), masks ({len(masks)}), patches.csv ({len(meta)}) lệch số dòng")
    if meta["patch_id"].duplicated().any():
        raise ValueError("patches.csv có patch_id trùng lặp")
    return hu, masks, meta


def _create(h5, name: str, data: np.ndarray, compress: bool) -> None:
    if compress and len(data):
        chunks = (min(len(data), 128), *data.shape[1:])
        h5.create_dataset(name, data=data, compression="gzip", compression_opts=4, chunks=chunks)
    else:
        h5.create_dataset(name, data=data)


def write_split_h5(
    path: str | Path,
    hu: np.ndarray,
    masks: np.ndarray,
    meta: pd.DataFrame,
    attrs: dict[str, Any],
    hu_window: tuple[float, float],
    empty_value: float = EMPTY_VALUE,
) -> int:
    """Ghi một split ra `path` (atomic qua `<path>.tmp`); trả số patch."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(meta)
    lo, hi = hu_window
    image = window_hu(hu, lo, hi)[:, None].astype(np.float32)
    mask = (masks > 0).astype(np.uint8)[:, None]
    dist = np.empty(mask.shape, dtype=np.float32)
    for i in range(n):
        dist[i, 0] = edt_from_mask(mask[i, 0], empty_value)
    str_dt = h5py.string_dtype("utf-8")
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        with h5py.File(tmp, "w") as h5:
            _create(h5, "image", image, True)
            _create(h5, "mask", mask, True)
            _create(h5, "dist", dist, True)
            h5.create_dataset("patch_id", data=np.array(meta["patch_id"].astype(str).tolist(), dtype=object), dtype=str_dt)
            h5.create_dataset("patient_id", data=np.array(meta["patient_id"].astype(str).tolist(), dtype=object), dtype=str_dt)
            h5.create_dataset("diameter_mm", data=meta["diameter_mm"].to_numpy(dtype=np.float32))
            h5.create_dataset("area_px", data=meta["area_union_px"].to_numpy(dtype=np.int32))
            h5.create_dataset("touches_border", data=meta["touches_border"].astype(bool).to_numpy(dtype=np.uint8))
            for key, value in {**attrs, "n": n, "hu_window": list(hu_window), "dist_empty_value": empty_value}.items():
                h5.attrs[key] = value if not isinstance(value, (dict, list)) else json.dumps(value, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return n


def _copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f"{dst.name}.tmp")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def verify_manifest(directory: str | Path) -> dict[str, str]:
    """Đối chiếu SHA-256 mọi tệp trong `manifest.json`; trả `{tên: lý_do}` cho tệp sai/thiếu (rỗng = OK)."""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    bad = {}
    for name, info in manifest["files"].items():
        p = directory / name
        if not p.is_file():
            bad[name] = "thiếu tệp"
        elif p.stat().st_size != info["bytes"]:
            bad[name] = f"kích thước {p.stat().st_size} != {info['bytes']}"
        elif sha256_file(p) != info["sha256"]:
            bad[name] = "SHA-256 không khớp"
    return bad


def copy_to_drive(src_dir: str | Path, dst_dir: str | Path) -> list[str]:
    """Chép các tệp trong manifest + `manifest.json` (chép cuối) sang `dst_dir`, rồi kiểm SHA-256 ở đích."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    manifest = json.loads((src_dir / MANIFEST).read_text(encoding="utf-8"))
    dst_dir.mkdir(parents=True, exist_ok=True)
    names = list(manifest["files"]) + [MANIFEST]
    for name in names:
        _copy_atomic(src_dir / name, dst_dir / name)
    bad = verify_manifest(dst_dir)
    if sha256_file(dst_dir / MANIFEST) != sha256_file(src_dir / MANIFEST):
        bad[MANIFEST] = "SHA-256 không khớp"
    if bad:
        raise RuntimeError(f"Sao chép sang {dst_dir} lỗi: {bad}")
    return names


def pack(
    interim_dir: str | Path,
    splits_path: str | Path,
    out_dir: str | Path,
    data_cfg: dict,
    copy_to: str | Path | None = None,
) -> dict[str, Any]:
    """Đóng gói đủ 3 split + bản sao `patches.csv`, `<ds>.json` và `manifest.json`."""
    interim_dir, splits_path, out_dir = Path(interim_dir), Path(splits_path), Path(out_dir)
    dataset = splits_path.stem
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    hu, masks, meta = load_interim(interim_dir)
    row_of = {pid: i for i, pid in enumerate(meta["patch_id"].astype(str))}
    hu_window = tuple(float(v) for v in data_cfg["common"]["hu_window"])
    created = time.strftime("%Y-%m-%dT%H:%M:%S")
    commit = git_commit()
    out_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict] = {}
    for split in SPLITS:
        ids = sorted(str(p) for p in splits[split])
        missing = [p for p in ids if p not in row_of]
        if missing:
            raise KeyError(f"{split}: {len(missing)} patch_id không có trong patches.csv, ví dụ {missing[:3]}")
        rows = np.array([row_of[p] for p in ids], dtype=np.int64)
        attrs = {
            "dataset": dataset,
            "split": split,
            "data_config_json": json.dumps(data_cfg, ensure_ascii=False),
            "split_mode": splits.get("mode", ""),
            "split_seed": int(splits.get("seed", -1)),
            "created": created,
            "git_commit": commit,
        }
        name = f"{split}.h5"
        n = write_split_h5(out_dir / name, hu[rows], masks[rows], meta.iloc[rows], attrs, hu_window)
        files[name] = {"n": n}
    _copy_atomic(interim_dir / "patches.csv", out_dir / "patches.csv")
    _copy_atomic(splits_path, out_dir / f"{dataset}.json")
    files["patches.csv"] = {"n": int(len(meta))}
    files[f"{dataset}.json"] = {"n": None}
    for name, info in files.items():
        info["sha256"] = sha256_file(out_dir / name)
        info["bytes"] = (out_dir / name).stat().st_size
    manifest = {
        "dataset": dataset,
        "created": created,
        "git_commit": commit,
        "split_mode": splits.get("mode", ""),
        "split_seed": splits.get("seed"),
        "hu_window": list(hu_window),
        "data_config": data_cfg,
        "files": files,
    }
    save_json_atomic(out_dir / MANIFEST, manifest)
    if copy_to is not None:
        copy_to_drive(out_dir, copy_to)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interim-dir", required=True, help="Thư mục chứa patches_hu.npy, masks.npy, patches.csv")
    parser.add_argument("--splits", required=True, help="Tệp <ds>.json của make_splits")
    parser.add_argument("--out-dir", required=True, help="Nơi ghi .h5 (SSD cục bộ trên Colab)")
    parser.add_argument("--config", default="configs/data.yaml", help="Config dữ liệu (hu_window)")
    parser.add_argument("--copy-to-drive-dir", default=None, help="Chép một lần sang thư mục Drive và kiểm SHA-256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = pack(args.interim_dir, args.splits, args.out_dir, load_config(args.config), args.copy_to_drive_dir)
    counts = {k: v["n"] for k, v in manifest["files"].items() if k.endswith(".h5")}
    dest = f" → đã chép và kiểm SHA-256 tại {args.copy_to_drive_dir}" if args.copy_to_drive_dir else ""
    print(f"{manifest['dataset']}: {counts} → {args.out_dir}{dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
