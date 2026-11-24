"""Bản đồ khoảng cách Euclid D_G cho Hausdorff-EDT loss — SPECS §4.1, [P] Eq. 35.

`D_G[i, j]` = khoảng cách (đơn vị pixel) từ pixel (i, j) tới pixel gần nhất thuộc mask G;
bằng 0 bên trong G. Mask rỗng trả về hằng `empty_value` (64·√2 ≈ 90.51, đường chéo patch 64×64).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

EMPTY_VALUE = 90.51  # khớp `loss.hausdorff_empty_value` trong configs/base.yaml

__all__ = ["EMPTY_VALUE", "edt_from_mask", "edt_batch"]


def edt_from_mask(mask: np.ndarray, empty_value: float = EMPTY_VALUE) -> np.ndarray:
    """EDT của nền tới mask (float32, cùng shape); mask rỗng → toàn `empty_value`. Không sửa đầu vào."""
    m = np.asarray(mask) > 0
    if not m.any():
        return np.full(m.shape, empty_value, dtype=np.float32)
    return ndimage.distance_transform_edt(~m).astype(np.float32)


def edt_batch(masks: np.ndarray, empty_value: float = EMPTY_VALUE) -> np.ndarray:
    """Áp `edt_from_mask` cho từng ảnh 2D theo hai trục cuối, ví dụ shape `(N, 1, H, W)`."""
    arr = np.asarray(masks)
    if arr.ndim < 2:
        raise ValueError(f"cần ít nhất 2 chiều, nhận shape {arr.shape}")
    flat = arr.reshape(-1, *arr.shape[-2:])
    out = np.empty(flat.shape, dtype=np.float32)
    for i, m in enumerate(flat):
        out[i] = edt_from_mask(m, empty_value)
    return out.reshape(arr.shape)
