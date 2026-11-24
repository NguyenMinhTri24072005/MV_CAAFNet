"""Thư viện tiền xử lý CT dùng chung — SPECS §5.1.

Chuyển thể từ `SOURCES/common/data_pipeline/medical_prep.py` (chép và tinh chỉnh, không import chéo):
giữ ý tưởng cắt đáy HU, cửa sổ HU (numpy + torch), resample mặt phẳng trục, ghi tệp atomic và chạy
song song có resume; bỏ OpenCV (resample bằng `scipy.ndimage.zoom`), bỏ tách phổi, bỏ lượng tử hóa,
bỏ `configs.config`. Thêm `crop_patch` (cắt quanh tâm nốt), đường kính nốt và hình thái patch.

Quy ước: volume luôn có trục `(Z, Y, X)`, spacing `(sz, sy, sx)` tính bằng mm.
Không hàm nào sửa mảng đầu vào tại chỗ.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy import ndimage

__all__ = [
    "HU_FLOOR",
    "ERROR_COLUMNS",
    "StageError",
    "floor_hu",
    "window_hu",
    "window_hu_torch",
    "resample_inplane",
    "crop_patch",
    "compute_nodule_diameters",
    "compute_patch_morphology",
    "save_npy_atomic",
    "save_npz_atomic",
    "save_text_atomic",
    "save_json_atomic",
    "save_csv_atomic",
    "run_parallel",
]

HU_FLOOR = -1000.0
ERROR_COLUMNS = ["series_uid", "patient_id", "stage", "error"]


class StageError(RuntimeError):
    """Lỗi của một scan kèm tên bước (`read_ct`, `seg_map`, ...) để ghi vào errors.csv."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


# ---------------------------------------------------------------------------
# HU
# ---------------------------------------------------------------------------
def floor_hu(vol: np.ndarray, floor: float = HU_FLOOR) -> np.ndarray:
    """Cắt đáy HU: `max(vol, floor)` → bản sao float32 mới (vùng ngoài khung quét -3024/-2048 → -1000)."""
    return np.maximum(np.asarray(vol), floor).astype(np.float32, copy=True)


def _check_window(lo: float, hi: float) -> tuple[float, float]:
    lo, hi = float(lo), float(hi)
    if not hi > lo:
        raise ValueError(f"cửa sổ HU cần hi > lo, nhận [{lo}, {hi}]")
    return lo, hi


def window_hu(x: np.ndarray, lo: float = -1000.0, hi: float = 400.0) -> np.ndarray:
    """Kẹp HU vào `[lo, hi]` rồi đưa tuyến tính về `[0, 1]` (float32); không khí (-1000) → 0."""
    lo, hi = _check_window(lo, hi)
    clipped = np.clip(np.asarray(x, dtype=np.float32), lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32, copy=False)


def window_hu_torch(x, lo: float = -1000.0, hi: float = 400.0, out_dtype=None):
    """Cùng công thức `window_hu` cho tensor (chạy được trên GPU). Import torch lười."""
    import torch

    lo, hi = _check_window(lo, hi)
    y = (torch.clamp(x.to(torch.float32), lo, hi) - lo) / (hi - lo)
    return y if out_dtype is None else y.to(out_dtype)


# ---------------------------------------------------------------------------
# Hình học
# ---------------------------------------------------------------------------
def resample_inplane(
    vol_zyx: np.ndarray,
    spacing_yx: tuple[float, float],
    target_mm: float,
    order: int = 1,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Đưa mặt phẳng trục về `target_mm` mm/pixel, giữ nguyên trục Z.

    Ảnh dùng `order=1`, mask dùng `order=0` (giữ đúng tập giá trị, ví dụ `{0, 1}`).
    Trả `(vol_out, (sy_new, sx_new))` với spacing mới tính theo số pixel thực tế sau khi làm tròn.
    """
    vol = np.asarray(vol_zyx)
    if vol.ndim != 3:
        raise ValueError(f"cần volume 3D (Z, Y, X), nhận shape {vol.shape}")
    sy, sx = (float(s) for s in spacing_yx)
    target = float(target_mm)
    if target <= 0:
        raise ValueError("target_mm phải > 0")
    zoom = (1.0, sy / target, sx / target)
    if math.isclose(zoom[1], 1.0) and math.isclose(zoom[2], 1.0):
        return vol.copy(), (sy, sx)
    out = ndimage.zoom(vol, zoom, order=int(order), mode="nearest", grid_mode=False)
    new_sy = sy * vol.shape[1] / out.shape[1]
    new_sx = sx * vol.shape[2] / out.shape[2]
    return out, (new_sy, new_sx)


def crop_patch(
    img2d_yx: np.ndarray, cy: int, cx: int, size: int = 64, pad_value: float = HU_FLOOR
) -> np.ndarray:
    """Cắt ô `size × size` có tâm `(cy, cx)`; phần nằm ngoài ảnh được pad bằng `pad_value`.

    Góc trên-trái nguồn là `(cy - size // 2, cx - size // 2)`, nên pixel `(cy, cx)` nằm ở `(size//2, size//2)`.
    """
    img = np.asarray(img2d_yx)
    if img.ndim != 2:
        raise ValueError(f"cần ảnh 2D, nhận shape {img.shape}")
    h, w = img.shape
    out = np.full((size, size), pad_value, dtype=img.dtype)
    y0, x0 = int(cy) - size // 2, int(cx) - size // 2
    sy0, sy1 = max(y0, 0), min(y0 + size, h)
    sx0, sx1 = max(x0, 0), min(x0 + size, w)
    if sy0 < sy1 and sx0 < sx1:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def compute_nodule_diameters(
    mask_3d: np.ndarray, spacing_zyx: tuple[float, float, float]
) -> tuple[float, float]:
    """Đường kính tương đương (mm): khối cầu cùng thể tích và hình tròn cùng diện tích lát lớn nhất."""
    m = np.asarray(mask_3d) > 0
    if m.ndim != 3:
        raise ValueError(f"cần mask 3D, nhận shape {m.shape}")
    sz, sy, sx = (float(s) for s in spacing_zyx)
    n_vox = int(m.sum())
    if n_vox == 0:
        return 0.0, 0.0
    volume_mm3 = n_vox * sz * sy * sx
    diam_3d = (6.0 * volume_mm3 / math.pi) ** (1.0 / 3.0)
    max_area_mm2 = float(m.sum(axis=(1, 2)).max()) * sy * sx
    diam_2d = 2.0 * math.sqrt(max_area_mm2 / math.pi)
    return float(diam_3d), float(diam_2d)


def _solidity(mask: np.ndarray) -> float:
    if int(mask.sum()) < 4:
        return float("nan")
    try:
        from skimage.measure import regionprops

        props = regionprops(mask.astype(np.uint8))
        if not props:
            return float("nan")
        return float(props[0].area / props[0].area_convex)
    except Exception:  # skimage thiếu hoặc Qhull lỗi với hình suy biến (một hàng pixel)
        return float("nan")


def compute_patch_morphology(patch_hu: np.ndarray, patch_mask: np.ndarray) -> dict[str, Any]:
    """Thông tin hình thái của một patch: chạm biên, thống kê HU trong mask và solidity."""
    hu = np.asarray(patch_hu, dtype=np.float64)
    m = np.asarray(patch_mask) > 0
    if hu.shape != m.shape or m.ndim != 2:
        raise ValueError(f"patch_hu {hu.shape} và patch_mask {m.shape} phải cùng shape 2D")
    touches = bool(m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())
    if m.any():
        vals = hu[m]
        mean_hu, min_hu, std_hu = float(vals.mean()), float(vals.min()), float(vals.std())
    else:
        mean_hu = min_hu = std_hu = float("nan")
    return {
        "touches_border": touches,
        "mean_hu": mean_hu,
        "min_hu": min_hu,
        "std_hu": std_hu,
        "solidity": _solidity(m),
    }


# ---------------------------------------------------------------------------
# Ghi tệp atomic
# ---------------------------------------------------------------------------
def _atomic_write(path: str | Path, writer: Callable[[io.BufferedWriter], None]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        with open(tmp, "wb") as fh:
            writer(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def save_npy_atomic(path: str | Path, arr: np.ndarray) -> Path:
    """`np.save` qua tệp tạm `<path>.tmp` rồi `os.replace` (truyền file handle để numpy không đổi đuôi)."""
    return _atomic_write(path, lambda fh: np.save(fh, np.asarray(arr), allow_pickle=False))


def save_npz_atomic(path: str | Path, compress: bool = True, **arrays: np.ndarray) -> Path:
    """`np.savez[_compressed]` atomic; mặc định nén vì patch int16 nén rất tốt."""
    saver = np.savez_compressed if compress else np.savez
    return _atomic_write(path, lambda fh: saver(fh, **{k: np.asarray(v) for k, v in arrays.items()}))


def save_text_atomic(path: str | Path, text: str) -> Path:
    return _atomic_write(path, lambda fh: fh.write(text.encode("utf-8")))


def save_json_atomic(path: str | Path, obj: Any) -> Path:
    return save_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default) + "\n")


def _json_default(obj: Any):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"không ghi JSON được kiểu {type(obj).__name__}")


def save_csv_atomic(path: str | Path, rows: Iterable[dict] | Any, columns: list[str] | None = None) -> Path:
    """Ghi CSV UTF-8 atomic từ `pandas.DataFrame` hoặc danh sách dict (luôn có dòng tiêu đề)."""
    if hasattr(rows, "to_csv"):
        text = rows.to_csv(index=False, lineterminator="\n")
        return save_text_atomic(path, text)
    rows = list(rows)
    if columns is None:
        columns = []
        for row in rows:
            columns.extend(k for k in row if k not in columns)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return save_text_atomic(path, buf.getvalue())


# ---------------------------------------------------------------------------
# Chạy song song có resume
# ---------------------------------------------------------------------------
def _call_job(fn: Callable[[dict], Any], job: dict) -> tuple:
    """Bọc lời gọi trong tiến trình con: không để ngoại lệ lan ra (có thể không pickle được)."""
    try:
        return ("ok", fn(job))
    except Exception as exc:  # noqa: BLE001 — mọi lỗi của một scan đều ghi lại, không dừng cả lượt
        stage = getattr(exc, "stage", "process")
        last = traceback.extract_tb(exc.__traceback__)[-1:] if exc.__traceback__ else []
        where = f" [{Path(last[0].filename).name}:{last[0].lineno}]" if last else ""
        return ("error", str(stage), f"{type(exc).__name__}: {exc}{where}"[:2000])


def run_parallel(
    fn: Callable[[dict], Any],
    jobs: list[dict],
    workers: int,
    errors_csv: str | Path,
    desc: str = "scan",
) -> dict[str, Any]:
    """Chạy `fn(job)` cho từng scan, song song bằng `ProcessPoolExecutor` (tuần tự khi `workers <= 1`).

    Mỗi `job` là dict có `series_uid`, `patient_id` và `done_marker` (đường dẫn tệp `_DONE`).
    - Scan đã có `_DONE` (kể cả scan 0 nốt hợp lệ) bị bỏ qua ngay → resume khi Colab ngắt phiên.
    - `fn` tự ghi kết quả và `_DONE` khi thành công; ngoại lệ được bắt lại, ghi vào `errors_csv`
      với các cột `series_uid, patient_id, stage, error`, không làm sập cả lượt chạy.
    - `errors_csv` được ghi lại mỗi lượt (chỉ gồm lỗi của lượt này); scan lỗi không có `_DONE`
      nên lượt sau sẽ chạy lại.
    """
    pending = [j for j in jobs if not Path(j["done_marker"]).exists()]
    n_skipped = len(jobs) - len(pending)
    errors: list[dict] = []
    results: list[Any] = []

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(it, **_):  # type: ignore[no-redef]
            return it

    def handle(job: dict, outcome: tuple) -> None:
        if outcome[0] == "ok":
            results.append(outcome[1])
        else:
            errors.append(
                {
                    "series_uid": job.get("series_uid", ""),
                    "patient_id": job.get("patient_id", ""),
                    "stage": outcome[1],
                    "error": outcome[2],
                }
            )

    if workers <= 1 or len(pending) <= 1:
        for job in tqdm(pending, desc=desc):
            handle(job, _call_job(fn, job))
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(pending))) as pool:
            futures = {pool.submit(_call_job, fn, job): job for job in pending}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                job = futures[fut]
                try:
                    outcome = fut.result()
                except Exception as exc:  # tiến trình con chết (hết RAM, ...)
                    outcome = ("error", "worker", f"{type(exc).__name__}: {exc}")
                handle(job, outcome)

    errors.sort(key=lambda r: (r["patient_id"], r["series_uid"]))
    save_csv_atomic(errors_csv, errors, columns=ERROR_COLUMNS)
    return {
        "n_jobs": len(jobs),
        "n_skipped": n_skipped,
        "n_done": len(results),
        "n_failed": len(errors),
        "results": results,
        "errors": errors,
    }
