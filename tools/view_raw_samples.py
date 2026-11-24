#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script xem trực quan (có thanh trượt lát z mặc định) cho dữ liệu gốc LIDC-IDRI và LUNA16.

Mặc định đọc từ:
  - LIDC-IDRI: D:/GG_1/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/MV_CAAFNet/data/raw/LIDC-IDRI
  - LUNA16   : D:/GG_1/COMPUTER SCIENCE/NAM4_HK1_(2026-2027)/CT208E/MV_CAAFNet/data/raw/LUNA16

Tính năng khi mở cửa sổ tương tác (MẶC ĐỊNH khi không truyền --save-dir):
  - Thanh trượt lát `z` đi qua TOÀN BỘ lát cắt (0 .. Z-1) của volume CT + đánh dấu dải lát có nốt.
  - Hỗ trợ cuộn con lăn chuột và phím mũi tên (Trái/Phải hoặc Lên/Xuống) để đổi lát mượt mà.
  - Hiển thị mask/votes trên CẢ ảnh toàn phổi 512×512 lẫn patch 64×64, kèm nút Bật/Tắt Mask & Nhảy tới nốt.
  - Hỗ trợ xem đúng 1 mẫu bất kỳ của LUNA16 hoặc LIDC-IDRI qua `--index`, `--patient`, `--series-uid` (gõ ngắn).

Ví dụ chạy (từ gốc repo MV_CAAFNet):
  # 1) Xem đúng 1 mẫu đầu tiên của LUNA16 (có thanh trượt toàn bộ lát z + chồng mask DICOM-SEG)
  python tools/view_raw_samples.py --dataset luna16

  # 2) Xem đúng 1 mẫu thứ 10 của LUNA16 (không cần gõ seriesuid dài)
  python tools/view_raw_samples.py --dataset luna16 --index 10

  # 3) Xem 1 mẫu LUNA16 theo vài số cuối của seriesuid (nhanh, thêm --no-seg nếu không cần ghép mask LIDC)
  python tools/view_raw_samples.py --dataset luna16 --series-uid 603192

  # 4) Xem đúng 1 mẫu LIDC-IDRI (hỗ trợ gõ ngắn 0001 hoặc LIDC-IDRI-0001)
  python tools/view_raw_samples.py --dataset lidc --patient 0001

  # 5) Xem 1 mẫu ngẫu nhiên của LUNA16 hoặc LIDC
  python tools/view_raw_samples.py --dataset luna16 --random

  # 6) Xuất PNG tĩnh (không mở cửa sổ) cho 3 mẫu
  python tools/view_raw_samples.py --dataset both --n 3 --save-dir reports/raw_preview
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy import ndimage

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIDC_DIR = REPO_ROOT / "data" / "raw" / "LIDC-IDRI"
DEFAULT_LUNA_DIR = REPO_ROOT / "data" / "raw" / "LUNA16"
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "index"

NODULE_RE = re.compile(r"Nodule\s+(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Tiện ích chuẩn hóa & crop 64x64
# ---------------------------------------------------------------------------
def window_hu(img: np.ndarray, lo: float = -1000.0, hi: float = 400.0) -> np.ndarray:
    clipped = np.clip(img.astype(np.float32), lo, hi)
    return (clipped - lo) / (hi - lo)


def crop_patch(img2d: np.ndarray, cy: int, cx: int, size: int = 64, pad_val: float = -1000.0) -> np.ndarray:
    H, W = img2d.shape
    out = np.full((size, size), pad_val, dtype=img2d.dtype)
    y0 = cy - size // 2
    x0 = cx - size // 2
    sy0, sy1 = max(0, y0), min(H, y0 + size)
    sx0, sx1 = max(0, x0), min(W, x0 + size)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img2d[sy0:sy1, sx0:sx1]
    return out


# ---------------------------------------------------------------------------
# Đọc LIDC-IDRI (NCI IDC DICOM-SEG)
# ---------------------------------------------------------------------------
def load_lidc_ct(ct_dir: Path) -> tuple[np.ndarray, dict[str, int], tuple[float, float, float], str]:
    dcm_files = sorted(ct_dir.glob("*.dcm"))
    if len(dcm_files) < 2:
        raise ValueError(f"Thư mục CT {ct_dir} chỉ có {len(dcm_files)} tệp .dcm")
    slices = [pydicom.dcmread(str(f)) for f in dcm_files]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    zs = np.array([float(s.ImagePositionPatient[2]) for s in slices], dtype=np.float64)
    dz = float(np.median(np.abs(np.diff(zs))))
    py, px = map(float, slices[0].PixelSpacing)
    rows, cols = int(slices[0].Rows), int(slices[0].Columns)
    vol = np.empty((len(slices), rows, cols), dtype=np.float32)
    for i, s in enumerate(slices):
        slope = float(getattr(s, "RescaleSlope", 1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        vol[i] = s.pixel_array.astype(np.float32) * slope + intercept
    sop2idx = {str(s.SOPInstanceUID): i for i, s in enumerate(slices)}
    series_uid = str(slices[0].SeriesInstanceUID)
    return vol, sop2idx, (dz, py, px), series_uid


def parse_nodule_key(ds: pydicom.Dataset) -> str:
    label = ""
    if hasattr(ds, "SegmentSequence") and len(ds.SegmentSequence) > 0:
        label = str(getattr(ds.SegmentSequence[0], "SegmentLabel", ""))
    if not label:
        label = str(getattr(ds, "SeriesDescription", ""))
    m = NODULE_RE.search(label)
    return f"Nodule {int(m.group(1))}" if m else (label or "Unknown")


def load_lidc_nodules(
    patient_dir: Path,
    series_uid: str,
    sop2idx: dict[str, int],
    shape: tuple[int, int, int],
    clevel: float = 0.5,
) -> list[dict]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for seg_dcm in sorted(patient_dir.glob("SEG*/*.dcm")):
        hdr = pydicom.dcmread(str(seg_dcm), stop_before_pixels=True)
        ref_uid = (
            str(hdr.ReferencedSeriesSequence[0].SeriesInstanceUID)
            if hasattr(hdr, "ReferencedSeriesSequence") and len(hdr.ReferencedSeriesSequence) > 0
            else ""
        )
        if ref_uid != series_uid:
            continue
        groups[parse_nodule_key(hdr)].append(seg_dcm)

    nodules = []
    for nkey in sorted(groups.keys()):
        seg_files = groups[nkey]
        votes = np.zeros(shape, dtype=np.uint8)
        for sf in seg_files:
            ds = pydicom.dcmread(str(sf))
            arr = ds.pixel_array
            if arr.ndim == 2:
                arr = arr[None]
            m = np.zeros(shape, dtype=bool)
            for i, pf in enumerate(ds.PerFrameFunctionalGroupsSequence):
                sop = str(pf.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)
                k = sop2idx.get(sop)
                if k is not None:
                    m[k] |= arr[i] > 0
            votes += m.astype(np.uint8)
        n_readers = len(seg_files)
        thresh = max(1, math.ceil(clevel * n_readers))
        cmask = (votes >= thresh).astype(np.uint8)
        if cmask.sum() == 0:
            continue
        nodules.append({"nodule_key": nkey, "n_readers": n_readers, "votes": votes, "cmask": cmask})
    return nodules


# ---------------------------------------------------------------------------
# Tìm nhanh SEG của LIDC khớp với 1 SeriesInstanceUID của LUNA16
# ---------------------------------------------------------------------------
def find_seg_files_for_uid(lidc_dir: Path, target_uid: str) -> tuple[Optional[str], list[Path]]:
    """Dùng data/index nếu có (tức thì), nếu chưa có thì quét header CT trong LIDC-IDRI."""
    seg_csv = DEFAULT_INDEX_DIR / "seg_index.csv"
    if seg_csv.is_file():
        try:
            df_seg = pd.read_csv(seg_csv)
            sub = df_seg[df_seg["ref_series_uid"] == target_uid]
            if len(sub) > 0:
                pid = str(sub.iloc[0]["patient_id"])
                paths = [lidc_dir / str(rel) for rel in sub["seg_rel_path"].tolist()]
                paths = [p for p in paths if p.is_file()]
                if paths:
                    return pid, paths
        except Exception:
            pass

    if not lidc_dir.is_dir():
        return None, []
    print("  (Đang tìm DICOM-SEG khớp trong LIDC-IDRI... dùng --no-seg để bỏ qua bước này)")
    for pdir in sorted(lidc_dir.glob("LIDC-IDRI-*")):
        if not pdir.is_dir():
            continue
        for ct_dir in sorted(pdir.glob("CT*")):
            dcms = sorted(ct_dir.glob("*.dcm"))
            if not dcms:
                continue
            hdr = pydicom.dcmread(str(dcms[0]), stop_before_pixels=True)
            if str(getattr(hdr, "SeriesInstanceUID", "")) == target_uid:
                matched_segs = []
                for sf in sorted(pdir.glob("SEG*/*.dcm")):
                    sh = pydicom.dcmread(str(sf), stop_before_pixels=True)
                    ref = (
                        str(sh.ReferencedSeriesSequence[0].SeriesInstanceUID)
                        if hasattr(sh, "ReferencedSeriesSequence") and len(sh.ReferencedSeriesSequence) > 0
                        else ""
                    )
                    if ref == target_uid:
                        matched_segs.append(sf)
                return pdir.name, matched_segs
    return None, []


def map_segs_to_mhd(
    seg_files: list[Path],
    sitk_img: sitk.Image,
    shape_zyx: tuple[int, int, int],
    clevel: float = 0.5,
) -> list[dict]:
    """Ánh xạ DICOM-SEG lên lưới voxel .mhd qua tọa độ vật lý ImagePositionPatient."""
    Z, H, W = shape_zyx
    direction = np.array(sitk_img.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.array(sitk_img.GetSpacing(), dtype=np.float64)
    origin = np.array(sitk_img.GetOrigin(), dtype=np.float64)
    inv_mat = np.linalg.inv(direction @ np.diag(spacing))

    groups: dict[str, list[Path]] = defaultdict(list)
    for sf in seg_files:
        hdr = pydicom.dcmread(str(sf), stop_before_pixels=True)
        groups[parse_nodule_key(hdr)].append(sf)

    nodules = []
    for nkey in sorted(groups.keys()):
        files = groups[nkey]
        votes = np.zeros(shape_zyx, dtype=np.uint8)
        for sf in files:
            ds = pydicom.dcmread(str(sf))
            arr = ds.pixel_array
            if arr.ndim == 2:
                arr = arr[None]
            sfg = ds.SharedFunctionalGroupsSequence[0]
            iop = [float(v) for v in sfg.PlaneOrientationSequence[0].ImageOrientationPatient]
            ps_row, ps_col = map(float, sfg.PixelMeasuresSequence[0].PixelSpacing)
            row_dir = np.array(iop[0:3], dtype=np.float64)
            col_dir = np.array(iop[3:6], dtype=np.float64)
            m = np.zeros(shape_zyx, dtype=bool)
            for i, pf in enumerate(ds.PerFrameFunctionalGroupsSequence):
                if not np.any(arr[i]):
                    continue
                ipp = np.array([float(v) for v in pf.PlanePositionSequence[0].ImagePositionPatient], dtype=np.float64)
                ix0, iy0, iz0 = sitk_img.TransformPhysicalPointToContinuousIndex(tuple(ipp))
                k = int(round(iz0))
                if (
                    arr.shape[1:] == (H, W)
                    and np.allclose(direction, np.eye(3), atol=1e-3)
                    and np.allclose(iop, [1, 0, 0, 0, 1, 0], atol=1e-3)
                    and abs(ix0) < 0.25
                    and abs(iy0) < 0.25
                    and 0 <= k < Z
                ):
                    m[k] |= arr[i] > 0
                else:
                    rs, cs = np.nonzero(arr[i] > 0)
                    pts = (
                        ipp[None, :]
                        + (cs[:, None] * ps_col) * row_dir[None, :]
                        + (rs[:, None] * ps_row) * col_dir[None, :]
                    )
                    idxs = (pts - origin[None, :]) @ inv_mat.T
                    xs = np.rint(idxs[:, 0]).astype(int)
                    ys = np.rint(idxs[:, 1]).astype(int)
                    zs = np.rint(idxs[:, 2]).astype(int)
                    valid = (0 <= xs) & (xs < W) & (0 <= ys) & (ys < H) & (0 <= zs) & (zs < Z)
                    m[zs[valid], ys[valid], xs[valid]] = True
            votes += m.astype(np.uint8)
        n_readers = len(files)
        thresh = max(1, math.ceil(clevel * n_readers))
        cmask = (votes >= thresh).astype(np.uint8)
        if cmask.sum() > 0:
            nodules.append({"nodule_key": nkey, "n_readers": n_readers, "votes": votes, "cmask": cmask})
    return nodules


# ---------------------------------------------------------------------------
# Vẽ tĩnh (khi xuất PNG bằng --save-dir hoặc --static)
# ---------------------------------------------------------------------------
def draw_sample_figure(
    plt,
    patches_mod,
    vol: np.ndarray,
    z: int,
    cy: int,
    cx: int,
    title: str,
    votes: Optional[np.ndarray] = None,
    cmask: Optional[np.ndarray] = None,
    circle_radius_px: Optional[float] = None,
    hu_window: tuple[float, float] = (-1000.0, 400.0),
    patch_size: int = 64,
    save_path: Optional[Path] = None,
):
    sl_img = window_hu(vol[z], *hu_window)
    p_img = window_hu(crop_patch(vol[z], cy, cx, patch_size, hu_window[0]), *hu_window)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    # Ô 1: Toàn lát 512x512 + mask + khung 64x64
    axes[0].imshow(sl_img, cmap="gray", vmin=0, vmax=1)
    if cmask is not None and cmask[z].max() > 0:
        axes[0].contour(cmask[z], levels=[0.5], colors=["red"], linewidths=1.0)
    axes[0].add_patch(
        patches_mod.Rectangle((cx - patch_size // 2, cy - patch_size // 2), patch_size, patch_size,
                              linewidth=1.5, edgecolor="lime", facecolor="none")
    )
    if circle_radius_px is not None:
        axes[0].add_patch(patches_mod.Circle((cx, cy), circle_radius_px, lw=1.2, ec="cyan", fc="none"))
    axes[0].set_title(f"Toàn lát CT (z={z}/{vol.shape[0]-1}) | khung {patch_size}×{patch_size}", fontsize=9.5)

    # Ô 2: Patch 64x64 ảnh CT
    axes[1].imshow(p_img, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Patch {patch_size}×{patch_size} quanh tâm (cy={cy}, cx={cx})", fontsize=9.5)

    # Ô 3: Patch 64x64 chồng nhãn
    axes[2].imshow(p_img, cmap="gray", vmin=0, vmax=1)
    line1_parts = []
    if votes is not None:
        p_votes = crop_patch(votes[z], cy, cx, patch_size, 0)
        if p_votes.max() > 0:
            axes[2].imshow(np.ma.masked_equal(p_votes, 0), cmap="autumn_r", vmin=1, vmax=4, alpha=0.55)
            line1_parts.append(f"Votes (max={int(p_votes.max())})")
    if cmask is not None:
        p_mask = crop_patch(cmask[z], cy, cx, patch_size, 0)
        if p_mask.max() > 0:
            axes[2].contour(p_mask, levels=[0.5], colors=["lime"], linewidths=1.5)
            line1_parts.append("Xanh lá: consensus ≥50%")
    title_ax2 = " | ".join(line1_parts) if line1_parts else "Overlay nhãn"
    if circle_radius_px is not None:
        axes[2].add_patch(
            patches_mod.Circle((patch_size // 2, patch_size // 2), circle_radius_px,
                               lw=1.5, ec="cyan", ls="--", fc="none")
        )
        title_ax2 = f"{title_ax2}\nXanh lam: bán kính LUNA16"
    axes[2].set_title(title_ax2, fontsize=9.5)

    for ax in axes:
        ax.axis("off")
    plt.tight_layout(rect=(0, 0, 1, 0.93))

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Đã lưu ảnh: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Trình xem tương tác có thanh trượt z (MẶC ĐỊNH)
# ---------------------------------------------------------------------------
def view_interactive_3d(
    plt,
    patches_mod,
    vol: np.ndarray,
    z_init: int,
    cy: int,
    cx: int,
    title: str,
    votes: Optional[np.ndarray] = None,
    cmask: Optional[np.ndarray] = None,
    union_mask: Optional[np.ndarray] = None,
    sphere_info: Optional[tuple[float, float, float, float]] = None,  # (z_center, radius_mm, sz_mm, sxy_mm)
    hu_window: tuple[float, float] = (-1000.0, 400.0),
    patch_size: int = 64,
):
    """Cửa sổ tương tác: thanh trượt z + cuộn chuột + phím mũi tên + bật/tắt mask + nhảy tới nốt."""
    from matplotlib.widgets import Button, Slider

    Z = vol.shape[0]
    mask_ref = union_mask if union_mask is not None else cmask
    pos_slices = np.where(mask_ref.reshape(Z, -1).sum(1) > 0)[0] if mask_ref is not None else np.array([], dtype=int)
    if len(pos_slices) > 0:
        z_range_txt = f"Lát có mask: [{int(pos_slices[0])}..{int(pos_slices[-1])}] ({len(pos_slices)} lát)"
    elif sphere_info is not None:
        zc, r_mm, sz_mm, _ = sphere_info
        dz_sl = int(math.ceil(r_mm / max(sz_mm, 1e-3)))
        z_range_txt = f"Vùng cầu LUNA16: [{max(0, int(zc)-dz_sl)}..{min(Z-1, int(zc)+dz_sl)}]"
    else:
        z_range_txt = "Không có mask"

    state = {"z": int(np.clip(z_init, 0, Z - 1)), "show_mask": True}

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.6))
    plt.subplots_adjust(left=0.03, right=0.98, top=0.86, bottom=0.19, wspace=0.08)
    fig.suptitle(
        f"{title}\n{z_range_txt}  |  Cuộn chuột hoặc phím ←/→ để đổi lát",
        fontsize=10.5,
        fontweight="bold",
    )

    def sphere_radius_at_z(z_idx: int) -> Optional[float]:
        if sphere_info is None:
            return None
        zc, r_mm, sz_mm, sxy_mm = sphere_info
        dz_mm = abs(z_idx - zc) * sz_mm
        if dz_mm >= r_mm:
            return None
        r_slice_mm = math.sqrt(max(0.0, r_mm * r_mm - dz_mm * dz_mm))
        return r_slice_mm / max(sxy_mm, 1e-3)

    def render(z_idx: int):
        z_idx = int(np.clip(z_idx, 0, Z - 1))
        state["z"] = z_idx
        for ax in axes:
            ax.clear()
            ax.axis("off")

        sl = window_hu(vol[z_idx], *hu_window)
        p = window_hu(crop_patch(vol[z_idx], cy, cx, patch_size, hu_window[0]), *hu_window)
        r_px = sphere_radius_at_z(z_idx)

        # Ô 1: Toàn lát CT 512x512 + chồng toàn bộ mask trên lát đó + khung 64x64
        axes[0].imshow(sl, cmap="gray", vmin=0, vmax=1)
        if state["show_mask"]:
            if votes is not None and votes[z_idx].max() > 0:
                axes[0].imshow(np.ma.masked_equal(votes[z_idx], 0), cmap="autumn_r", vmin=1, vmax=4, alpha=0.55)
            if mask_ref is not None and mask_ref[z_idx].max() > 0:
                axes[0].contour(mask_ref[z_idx], levels=[0.5], colors=["lime"], linewidths=1.2)
            if r_px is not None:
                axes[0].add_patch(patches_mod.Circle((cx, cy), r_px, lw=1.2, ec="cyan", ls="--", fc="none"))
        axes[0].add_patch(
            patches_mod.Rectangle((cx - patch_size // 2, cy - patch_size // 2), patch_size, patch_size,
                                  lw=1.3, ec="yellow", fc="none")
        )
        full_area = int(mask_ref[z_idx].sum()) if mask_ref is not None else 0
        axes[0].set_title(f"Toàn lát CT z={z_idx}/{Z-1} | mask toàn lát: {full_area} px", fontsize=9.5)

        # Ô 2: Patch 64x64 ảnh CT thuần
        axes[1].imshow(p, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Patch {patch_size}×{patch_size} tại (cy={cy}, cx={cx})", fontsize=9.5)

        # Ô 3: Patch 64x64 chồng votes + consensus + bán kính LUNA16 theo lát z
        axes[2].imshow(p, cmap="gray", vmin=0, vmax=1)
        info_bits = []
        if state["show_mask"]:
            if votes is not None:
                pv = crop_patch(votes[z_idx], cy, cx, patch_size, 0)
                if pv.max() > 0:
                    axes[2].imshow(np.ma.masked_equal(pv, 0), cmap="autumn_r", vmin=1, vmax=4, alpha=0.55)
                    info_bits.append(f"votes max={int(pv.max())}")
            if cmask is not None:
                pm = crop_patch(cmask[z_idx], cy, cx, patch_size, 0)
                if pm.max() > 0:
                    axes[2].contour(pm, levels=[0.5], colors=["lime"], linewidths=1.6)
                info_bits.append(f"area={int(pm.sum())} px")
            if r_px is not None:
                axes[2].add_patch(
                    patches_mod.Circle((patch_size // 2, patch_size // 2), r_px,
                                       lw=1.5, ec="cyan", ls="--", fc="none")
                )
                info_bits.append(f"LUNA r={r_px:.1f} px")
        else:
            info_bits.append("Mask đang TẮT")
        axes[2].set_title(" | ".join(info_bits) if info_bits else "Overlay", fontsize=9.5)
        fig.canvas.draw_idle()

    # Thanh trượt z (tô nền xanh nhạt ở đúng đoạn lát chứa mask để dễ kéo tới nốt!)
    ax_slider = fig.add_axes([0.12, 0.08, 0.62, 0.045])
    if len(pos_slices) > 0:
        ax_slider.axvspan(int(pos_slices[0]), int(pos_slices[-1]), color="limegreen", alpha=0.35)
    slider = Slider(ax_slider, "Lát z", 0, Z - 1, valinit=state["z"], valstep=1)

    def on_slider(val):
        render(int(val))

    slider.on_changed(on_slider)

    # Nút "Bật/Tắt Mask" và "Về lát nốt"
    ax_btn_mask = fig.add_axes([0.76, 0.075, 0.10, 0.055])
    btn_mask = Button(ax_btn_mask, "Mask: BẬT")

    def toggle_mask(_event):
        state["show_mask"] = not state["show_mask"]
        btn_mask.label.set_text("Mask: BẬT" if state["show_mask"] else "Mask: TẮT")
        render(state["z"])

    btn_mask.on_clicked(toggle_mask)

    ax_btn_jump = fig.add_axes([0.87, 0.075, 0.11, 0.055])
    btn_jump = Button(ax_btn_jump, f"Về lát z={z_init}")
    btn_jump.on_clicked(lambda _event: slider.set_val(z_init))

    # Hỗ trợ cuộn chuột & phím mũi tên
    def on_scroll(event):
        step = 1 if event.button == "up" else -1
        new_z = int(np.clip(state["z"] + step, 0, Z - 1))
        if new_z != state["z"]:
            slider.set_val(new_z)

    def on_key(event):
        if event.key in ("right", "up"):
            slider.set_val(min(Z - 1, state["z"] + 1))
        elif event.key in ("left", "down"):
            slider.set_val(max(0, state["z"] - 1))
        elif event.key == "m":
            toggle_mask(None)
        elif event.key == "home":
            slider.set_val(z_init)

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)

    render(state["z"])
    plt.show()


# ---------------------------------------------------------------------------
# Chạy xem LIDC-IDRI & LUNA16
# ---------------------------------------------------------------------------
def inspect_lidc(args, plt, patches_mod):
    lidc_dir = Path(args.lidc_dir)
    if not lidc_dir.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục LIDC-IDRI: {lidc_dir}")

    all_p = [p for p in sorted(lidc_dir.glob("LIDC-IDRI-*")) if p.is_dir() and any(p.glob("SEG*/*.dcm"))]
    if args.patient:
        q = args.patient.strip()
        if q.isdigit():
            q = f"LIDC-IDRI-{int(q):04d}"
        candidates = [p for p in all_p if q.lower() in p.name.lower()]
        if not candidates:
            direct = lidc_dir / q
            if direct.is_dir():
                candidates = [direct]
            else:
                raise FileNotFoundError(f"Không tìm thấy bệnh nhân khớp '{args.patient}' trong {lidc_dir}")
        candidates = candidates[:1]
    elif args.index is not None:
        if not (0 <= args.index < len(all_p)):
            raise IndexError(f"--index {args.index} ngoài phạm vi [0, {len(all_p)-1}]")
        candidates = [all_p[args.index]]
    elif args.random:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(all_p), size=min(args.n, len(all_p)), replace=False)
        candidates = [all_p[i] for i in sorted(idx)]
    else:
        candidates = all_p[: args.n]

    print(f"\n=== XEM MẪU LIDC-IDRI ({len(candidates)}/{len(all_p)} bệnh nhân có SEG từ {lidc_dir}) ===")
    for pdir in candidates:
        ct_dirs = sorted(d for d in pdir.glob("CT*") if d.is_dir())
        if not ct_dirs:
            print(f"[Bỏ qua] {pdir.name}: không có thư mục CT_*")
            continue
        vol, sop2idx, (sz, sy, sx), series_uid = load_lidc_ct(ct_dirs[0])
        nodules = load_lidc_nodules(pdir, series_uid, sop2idx, vol.shape, clevel=args.clevel)
        print(
            f"[LIDC] {pdir.name} | CT={ct_dirs[0].name} | shape={vol.shape} | "
            f"spacing=({sz:.2f}, {sy:.3f}, {sx:.3f}) mm | n_nodules={len(nodules)}"
        )
        if not nodules:
            continue
        nod_idx = min(args.nodule_idx, len(nodules) - 1)
        nod = nodules[nod_idx]
        cmask = nod["cmask"]
        union_mask = np.zeros_like(cmask)
        union_votes = np.zeros_like(nod["votes"])
        for nd in nodules:
            union_mask |= nd["cmask"]
            union_votes = np.maximum(union_votes, nd["votes"])

        cz, cy, cx = ndimage.center_of_mass(cmask)
        cy_i, cx_i = int(round(cy)), int(round(cx))
        z_best = int(cmask.reshape(cmask.shape[0], -1).sum(1).argmax())
        vol_mm3 = float(cmask.sum()) * sz * sy * sx
        diam_3d = (6.0 * vol_mm3 / math.pi) ** (1.0 / 3.0)
        area_px = int(cmask[z_best].sum())
        print(
            f"       {nod['nodule_key']} (#{nod_idx+1}/{len(nodules)}) | readers={nod['n_readers']} | "
            f"z_best={z_best} | center=({cy_i},{cx_i}) | diam_3d={diam_3d:.2f} mm | area={area_px} px"
        )
        title = (
            f"LIDC-IDRI: {pdir.name} ({nod['nodule_key']}, {nod['n_readers']} bác sĩ, "
            f"d≈{diam_3d:.1f} mm, spacing={sy:.2f} mm)"
        )
        if args.save_dir is None and not args.static:
            view_interactive_3d(
                plt, patches_mod, vol, z_best, cy_i, cx_i, title,
                votes=union_votes, cmask=cmask, union_mask=union_mask,
            )
        else:
            save_path = (Path(args.save_dir) / f"lidc_{pdir.name}.png") if args.save_dir else None
            draw_sample_figure(
                plt, patches_mod, vol, z_best, cy_i, cx_i, title,
                votes=nod["votes"], cmask=cmask, save_path=save_path,
            )


def inspect_luna16(args, plt, patches_mod):
    luna_dir = Path(args.luna_dir)
    ann_csv = luna_dir / "annotations.csv"
    if not ann_csv.is_file():
        raise FileNotFoundError(f"Không thấy annotations.csv tại: {ann_csv}")
    df = pd.read_csv(ann_csv)

    mhd_map = {p.stem: p for p in luna_dir.glob("subset*/*.mhd")}
    df = df[df["seriesuid"].isin(mhd_map)].copy()
    df["subset"] = df["seriesuid"].map(lambda u: mhd_map[u].parent.name)
    if args.subset:
        df = df[df["subset"] == args.subset]
    df = df.reset_index(drop=True)

    if args.series_uid:
        q = args.series_uid.strip()
        rows = df[df["seriesuid"].str.contains(q, regex=False)]
        if len(rows) == 0:
            raise FileNotFoundError(f"Không tìm thấy seriesuid chứa '{q}' trong annotations.csv")
        rows = rows.iloc[[min(args.nodule_idx, len(rows) - 1)]]
    elif args.index is not None:
        if not (0 <= args.index < len(df)):
            raise IndexError(f"--index {args.index} ngoài phạm vi [0, {len(df)-1}]")
        rows = df.iloc[[args.index]]
    elif args.random:
        rows = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    else:
        rows = df.head(args.n)

    print(f"\n=== XEM MẪU LUNA16 ({len(rows)}/{len(df)} nốt từ {luna_dir}) ===")
    for row_idx, row in rows.iterrows():
        uid = str(row["seriesuid"])
        mhd_path = mhd_map[uid]
        sitk_img = sitk.ReadImage(str(mhd_path))
        vol = sitk.GetArrayFromImage(sitk_img).astype(np.float32)  # (Z, Y, X)
        sx, sy, sz = sitk_img.GetSpacing()
        wx, wy, wz = float(row["coordX"]), float(row["coordY"]), float(row["coordZ"])
        diam_mm = float(row["diameter_mm"])
        ix, iy, iz = sitk_img.TransformPhysicalPointToContinuousIndex((wx, wy, wz))
        cx_i, cy_i, z_i = int(round(ix)), int(round(iy)), int(round(iz))
        sxy = (sx + sy) / 2.0
        radius_px = (diam_mm / 2.0) / sxy

        votes, cmask, union_mask, extra_info = None, None, None, "chỉ vòng tròn annotations.csv"
        if args.with_seg and Path(args.lidc_dir).is_dir():
            pid_found, seg_files = find_seg_files_for_uid(Path(args.lidc_dir), uid)
            if seg_files:
                nods = map_segs_to_mhd(seg_files, sitk_img, vol.shape, clevel=args.clevel)
                union_mask = np.zeros(vol.shape, dtype=np.uint8)
                union_votes = np.zeros(vol.shape, dtype=np.uint8)
                best_nod, best_dist = None, 1e9
                for nod in nods:
                    union_mask |= nod["cmask"]
                    union_votes = np.maximum(union_votes, nod["votes"])
                    ncz, ncy, ncx = ndimage.center_of_mass(nod["cmask"])
                    nwx, nwy, nwz = sitk_img.TransformContinuousIndexToPhysicalPoint((float(ncx), float(ncy), float(ncz)))
                    d = math.sqrt((nwx - wx) ** 2 + (nwy - wy) ** 2 + (nwz - wz) ** 2)
                    if d < best_dist:
                        best_dist, best_nod = d, nod
                if best_nod is not None:
                    votes, cmask = union_votes, best_nod["cmask"]
                    extra_info = f"khớp {pid_found}/{best_nod['nodule_key']} (lệch tâm={best_dist:.2f} mm)"

        print(
            f"[LUNA16 #{row_idx}] {mhd_path.parent.name}/{uid[-18:]} | shape={vol.shape} | "
            f"spacing=({sz:.2f},{sy:.3f},{sx:.3f}) | voxel=(z={z_i},y={cy_i},x={cx_i}) | "
            f"d={diam_mm:.2f} mm | {extra_info}"
        )
        title = f"LUNA16 #{row_idx} ({mhd_path.parent.name}): ...{uid[-14:]} | d={diam_mm:.1f} mm | {extra_info}"
        if args.save_dir is None and not args.static:
            view_interactive_3d(
                plt, patches_mod, vol, z_i, cy_i, cx_i, title,
                votes=votes, cmask=cmask, union_mask=union_mask,
                sphere_info=(iz, diam_mm / 2.0, sz, sxy),
            )
        else:
            save_path = (Path(args.save_dir) / f"luna16_{uid[-14:]}_z{z_i}.png") if args.save_dir else None
            draw_sample_figure(
                plt, patches_mod, vol, z_i, cy_i, cx_i, title,
                votes=votes, cmask=cmask, circle_radius_px=radius_px, save_path=save_path,
            )


def main():
    parser = argparse.ArgumentParser(description="Xem trực quan mẫu dữ liệu gốc LIDC-IDRI và LUNA16 (có thanh trượt z)")
    parser.add_argument("--dataset", choices=["lidc", "luna16", "both"], default="both",
                        help="Chọn tập muốn xem: lidc | luna16 | both (mặc định: both)")
    parser.add_argument("--lidc-dir", default=str(DEFAULT_LIDC_DIR))
    parser.add_argument("--luna-dir", default=str(DEFAULT_LUNA_DIR))
    parser.add_argument("--n", type=int, default=1, help="Số mẫu hiển thị cho mỗi tập (mặc định: 1)")
    parser.add_argument("--index", type=int, default=None, help="Xem đúng 1 mẫu ở vị trí thứ K (0, 1, 2, ...)")
    parser.add_argument("--patient", default=None, help="Xem 1 bệnh nhân LIDC cụ thể (ví dụ: 0001 hoặc LIDC-IDRI-0001)")
    parser.add_argument("--series-uid", default=None, help="Xem 1 mẫu LUNA16 theo seriesuid (có thể gõ vài số cuối, vd: 603192)")
    parser.add_argument("--subset", default=None, help="Lọc theo subset của LUNA16 (ví dụ: subset0 .. subset9)")
    parser.add_argument("--nodule-idx", type=int, default=0, help="Chọn nốt thứ mấy trong cùng 1 scan nếu có nhiều nốt (mặc định 0)")
    parser.add_argument("--random", action="store_true", help="Chọn ngẫu nhiên")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clevel", type=float, default=0.5, help="Ngưỡng đồng thuận bác sĩ (mặc định 0.5)")
    parser.add_argument("--no-seg", dest="with_seg", action="store_false",
                        help="Không ghép DICOM-SEG khi xem LUNA16 (mở tức thì chỉ với vòng tròn annotations.csv)")
    parser.add_argument("--static", action="store_true", help="Tắt thanh trượt z, chỉ vẽ lát trung tâm tĩnh")
    parser.add_argument("--save-dir", default=None, help="Lưu PNG tĩnh ra thư mục thay vì mở cửa sổ tương tác")
    args = parser.parse_args()

    import matplotlib
    if args.save_dir is not None:
        matplotlib.use("Agg")
    import matplotlib.patches as patches_mod
    import matplotlib.pyplot as plt

    if args.dataset in ("lidc", "both"):
        inspect_lidc(args, plt, patches_mod)
    if args.dataset in ("luna16", "both"):
        inspect_luna16(args, plt, patches_mod)


if __name__ == "__main__":
    main()
