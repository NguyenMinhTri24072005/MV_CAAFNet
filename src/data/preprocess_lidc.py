"""Tiền xử lý LIDC-IDRI (NCI IDC DICOM-SEG) → patch 64×64 — SPECS §5.2.

Đọc `CT_*/*.dcm` + các `SEG_*/*.dcm` trỏ vào scan đó (`ReferencedSeriesSequence`), gom cụm theo
nhãn `Nodule <K>` (đã có trong `seg_index.csv` của Phase 1), tạo mask đồng thuận
`votes >= max(1, ceil(clevel · R))`, lọc theo số bác sĩ và đường kính, rồi cắt patch quanh trọng tâm
3D của từng nốt trên mọi lát có mask. Không dùng `pylidc`.

Đầu ra:
    <out_dir>/by_scan/<series_uid>/{patches.npz, meta.csv, summary.json, _DONE}   (resume theo _DONE)
    <out_dir>/{patches_hu.npy, masks.npy, patches.csv, errors.csv, preprocess_summary.json}

    python -m src.data.preprocess_lidc --raw-dir data/raw/LIDC-IDRI --index-dir data/index \
        --out-dir data/interim/lidc --config configs/data.yaml --workers 4 [--limit 3]

Mô-đun này cũng chứa phần dùng chung với `preprocess_luna16.py`: đọc SEG, gom cụm, đồng thuận,
cắt patch (bước 3–5) và gộp kết quả.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import ndimage

from src.data.ct_utils import (
    StageError,
    compute_nodule_diameters,
    compute_patch_morphology,
    crop_patch,
    floor_hu,
    resample_inplane,
    run_parallel,
    save_csv_atomic,
    save_json_atomic,
    save_npy_atomic,
    save_npz_atomic,
    save_text_atomic,
)
from src.utils import load_config

DONE = "_DONE"
PATCH_COLUMNS = [
    "patch_id", "patient_id", "series_uid", "scan_idx", "nodule_idx", "z", "cx", "cy", "n_readers",
    "diameter_mm", "diameter_2d_mm", "area_nodule_px", "area_union_px", "n_other_nodules", "touches_border",
    "spacing_x", "spacing_y", "spacing_z", "mean_hu", "min_hu", "std_hu", "solidity",
]
ID_DTYPES = {"patch_id": str, "patient_id": str, "series_uid": str, "nodule_key": str}


# ---------------------------------------------------------------------------
# Đọc CT DICOM
# ---------------------------------------------------------------------------
@dataclass
class CTVolume:
    vol_zyx: np.ndarray                     # float32 HU (Z, Y, X)
    spacing_zyx: tuple[float, float, float]
    sop2idx: dict[str, int]
    z_pos: np.ndarray                       # (Z,) tọa độ z thế giới của từng lát
    ipp_xy: np.ndarray                      # (Z, 2) góc (0, 0) của từng lát
    iop: np.ndarray                         # (6,)


def read_ct_series(ct_dir: str | Path) -> CTVolume:
    """Đọc mọi lát của một CT series, sắp theo `ImagePositionPatient[2]` tăng dần, đổi sang HU."""
    import pydicom

    ct_dir = Path(ct_dir)
    files = sorted(p for p in ct_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm")
    slices = []
    for f in files:
        ds = pydicom.dcmread(f)
        if "PixelData" in ds:
            slices.append(ds)
    if not slices:
        raise StageError("read_ct", f"không có lát CT nào có pixel trong {ct_dir.name}")
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except (AttributeError, TypeError) as exc:
        raise StageError("read_ct", f"thiếu ImagePositionPatient: {exc}") from exc
    shapes = {(int(s.Rows), int(s.Columns)) for s in slices}
    if len(shapes) != 1:
        raise StageError("read_ct", f"các lát có kích thước khác nhau: {sorted(shapes)}")
    z = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    dz = np.diff(z)
    if len(dz) and (dz <= 1e-3).any():
        raise StageError("read_ct", f"có {int((dz <= 1e-3).sum())} lát trùng tọa độ z")
    h, w = shapes.pop()
    vol = np.empty((len(slices), h, w), dtype=np.float32)
    for i, s in enumerate(slices):
        slope = float(getattr(s, "RescaleSlope", 1) or 1)
        intercept = float(getattr(s, "RescaleIntercept", 0) or 0)
        vol[i] = s.pixel_array.astype(np.float32) * slope + intercept
    ps = [float(v) for v in slices[0].PixelSpacing]  # [khoảng cách hàng (y), khoảng cách cột (x)]
    sz = float(np.median(dz)) if len(dz) else float(getattr(slices[0], "SliceThickness", 1.0) or 1.0)
    return CTVolume(
        vol_zyx=vol,
        spacing_zyx=(sz, ps[0], ps[1]),
        sop2idx={str(s.SOPInstanceUID): i for i, s in enumerate(slices)},
        z_pos=z,
        ipp_xy=np.array([[float(s.ImagePositionPatient[0]), float(s.ImagePositionPatient[1])] for s in slices]),
        iop=np.array([float(v) for v in slices[0].ImageOrientationPatient]),
    )


# ---------------------------------------------------------------------------
# Đọc DICOM-SEG
# ---------------------------------------------------------------------------
@dataclass
class SegFrames:
    arr: np.ndarray            # bool (F, H, W)
    sops: list[str | None]     # ReferencedSOPInstanceUID của từng frame
    ipps: np.ndarray           # (F, 3), NaN nếu thiếu
    pixel_spacing: tuple[float, float]  # (hàng, cột) = (y, x)
    iop: np.ndarray            # (6,)


def _functional_item(ds, frame_idx: int, seq_name: str):
    """Lấy `<seq_name>[0]` trong per-frame group của frame, nếu không có thì ở shared group."""
    pfg = ds.get("PerFrameFunctionalGroupsSequence")
    if pfg is not None and frame_idx < len(pfg) and seq_name in pfg[frame_idx]:
        return pfg[frame_idx].get(seq_name)[0]
    sfg = ds.get("SharedFunctionalGroupsSequence")
    if sfg and seq_name in sfg[0]:
        return sfg[0].get(seq_name)[0]
    return None


def read_seg_frames(path: str | Path) -> SegFrames:
    """Đọc một tệp DICOM-SEG (1 bác sĩ × 1 nốt) thành các frame nhị phân kèm tham chiếu lát."""
    import pydicom

    try:
        ds = pydicom.dcmread(path)
        n_frames = int(ds.get("NumberOfFrames", 1) or 1)
        arr = np.asarray(ds.pixel_array).reshape(n_frames, int(ds.Rows), int(ds.Columns)) > 0
    except Exception as exc:  # noqa: BLE001
        raise StageError("read_seg", f"{Path(path).name}: {type(exc).__name__}: {exc}") from exc
    pfg = ds.get("PerFrameFunctionalGroupsSequence") or []
    sops: list[str | None] = []
    ipps = np.full((n_frames, 3), np.nan)
    for i in range(n_frames):
        sop = None
        if i < len(pfg):
            try:
                sop = str(pfg[i].DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)
            except (AttributeError, IndexError, KeyError):
                sop = None
        sops.append(sop)
        pos = _functional_item(ds, i, "PlanePositionSequence")
        if pos is not None and "ImagePositionPatient" in pos:
            ipps[i] = [float(v) for v in pos.ImagePositionPatient]
    measures = _functional_item(ds, 0, "PixelMeasuresSequence")
    orient = _functional_item(ds, 0, "PlaneOrientationSequence")
    ps = tuple(float(v) for v in measures.PixelSpacing) if measures is not None else (float("nan"),) * 2
    iop = np.array([float(v) for v in orient.ImageOrientationPatient]) if orient is not None else np.full(6, np.nan)
    return SegFrames(arr=arr, sops=sops, ipps=ipps, pixel_spacing=ps, iop=iop)


def lidc_reader_slices(seg: SegFrames, ct: CTVolume) -> dict[int, np.ndarray]:
    """Đặt các frame của một SEG lên lưới CT: `{chỉ_số_lát: mask 2D bool}`.

    Ánh xạ bằng `ReferencedSOPInstanceUID`; nếu thiếu thì theo z gần nhất (sai số < 0.35·dz).
    Kiểm tra lưới SEG trùng lưới CT (kích thước, PixelSpacing, IOP, góc (0, 0) của lát).
    """
    _, h, w = ct.vol_zyx.shape
    if seg.arr.shape[1:] != (h, w):
        raise StageError("seg_map", f"SEG {seg.arr.shape[1:]} khác lưới CT {(h, w)}")
    ps_ct = ct.spacing_zyx[1:]
    if not np.allclose(seg.pixel_spacing, ps_ct, rtol=1e-3, atol=1e-4):
        raise StageError("seg_geometry", f"PixelSpacing SEG {seg.pixel_spacing} khác CT {ps_ct}")
    if not np.allclose(seg.iop, ct.iop, atol=1e-3):
        raise StageError("seg_geometry", f"IOP SEG {seg.iop.tolist()} khác CT {ct.iop.tolist()}")
    dz = ct.spacing_zyx[0]
    tol_xy = 0.5 * min(ps_ct)
    out: dict[int, np.ndarray] = {}
    for i, frame in enumerate(seg.arr):
        if not frame.any():
            continue
        k = ct.sop2idx.get(seg.sops[i]) if seg.sops[i] else None
        if k is None:
            if np.isnan(seg.ipps[i, 2]):
                raise StageError("seg_map", f"frame {i} không có SOP tham chiếu lẫn tọa độ z")
            dist = np.abs(ct.z_pos - seg.ipps[i, 2])
            k = int(dist.argmin())
            if dist[k] > 0.35 * dz:
                raise StageError("seg_map", f"frame {i} (z={seg.ipps[i, 2]:.2f}) không khớp lát CT nào")
        if not np.isnan(seg.ipps[i, 0]) and not np.allclose(seg.ipps[i, :2], ct.ipp_xy[k], atol=tol_xy):
            raise StageError("seg_geometry", f"frame {i}: góc lát SEG {seg.ipps[i, :2]} khác CT {ct.ipp_xy[k]}")
        out[k] = out[k] | frame if k in out else frame.copy()
    return out


# ---------------------------------------------------------------------------
# Gom cụm & đồng thuận
# ---------------------------------------------------------------------------
def group_clusters(seg_rows: Iterable[dict]) -> list[dict]:
    """Gom các tệp SEG theo `nodule_key`; thứ tự tất định (số nốt tăng dần, nhãn dị biệt cuối) → `nodule_idx`."""
    groups: dict[str, list[dict]] = {}
    for row in seg_rows:
        groups.setdefault(str(row["nodule_key"]), []).append(row)

    def order(item):
        key, rows = item
        num = int(rows[0].get("nodule_num", -1))
        return (num < 0, num, key)

    return [
        {"nodule_idx": idx, "nodule_key": key, "rows": sorted(rows, key=lambda r: str(r["seg_rel_path"]))}
        for idx, (key, rows) in enumerate(sorted(groups.items(), key=order))
    ]


def consensus_threshold(n_readers: int, clevel: float) -> int:
    """Số phiếu tối thiểu `max(1, ceil(clevel · R))` (trừ 1e-9 để tránh sai số dấu phẩy động)."""
    return max(1, math.ceil(float(clevel) * int(n_readers) - 1e-9))


def votes_from_slices(readers: list[dict[int, np.ndarray]], shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """Cộng phiếu của các bác sĩ (mỗi bác sĩ tối đa 1 phiếu/voxel) → uint8 (Z, Y, X)."""
    votes = np.zeros(shape_zyx, dtype=np.uint8)
    for slices in readers:
        for k, m in slices.items():
            votes[k] += m.astype(np.uint8)
    return votes


# ---------------------------------------------------------------------------
# Bước 3–5: cắt patch (dùng chung LIDC & LUNA16)
# ---------------------------------------------------------------------------
def _jitter(patch_id: str, j: int) -> tuple[int, int]:
    seed = int.from_bytes(hashlib.sha256(patch_id.encode("utf-8")).digest()[:8], "little")
    dy, dx = np.random.default_rng(seed).integers(-j, j + 1, size=2)
    return int(dy), int(dx)


def extract_scan_patches(
    vol_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    nodules: list[dict],
    *,
    patient_id: str,
    series_uid: str,
    scan_idx: int,
    common: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Cắt đáy HU, resample (nếu bật), rồi cắt patch theo từng lát cho từng nốt đạt chuẩn.

    Mỗi phần tử `nodules` cần: `nodule_idx`, `n_readers`, `diameter_mm`, `diameter_2d_mm`,
    `cmask` (uint8 3D trên lưới gốc), tùy chọn `extra` (dict cột bổ sung cho patches.csv).
    Trả `(patches_hu int16 (N,S,S), masks uint8 (N,S,S), rows)`.
    """
    size = int(common["patch_size"])
    sz, sy, sx = (float(s) for s in spacing_zyx)
    vol = floor_hu(vol_zyx, float(common.get("hu_floor", -1000)))
    cmasks = [np.asarray(n["cmask"]) for n in nodules]
    target = common.get("resample_mm")
    if target is not None:
        order = int(common.get("resample_order", 1))
        vol, (ny, nx) = resample_inplane(vol, (sy, sx), float(target), order=order)
        cmasks = [resample_inplane(m, (sy, sx), float(target), order=0)[0] for m in cmasks]
        sy, sx = ny, nx
    binary = [m > 0 for m in cmasks]
    union = np.zeros(vol.shape, dtype=bool)
    for m in binary:
        union |= m
    union_u8 = union.astype(np.uint8)
    min_area = max(1, int(common.get("min_mask_area_px", 1)))
    include_other = bool(common.get("include_other_nodules", True))
    pad_hu = float(common.get("pad_value_hu", -1000))
    jitter = int(common.get("center_jitter_px", 0) or 0)

    patches, masks, rows = [], [], []
    for k, (nod, m) in enumerate(zip(nodules, binary)):
        if not m.any():
            continue
        _, cy, cx = ndimage.center_of_mass(m)
        cy0, cx0 = int(round(cy)), int(round(cx))
        areas = m.sum(axis=(1, 2))
        for z in np.nonzero(areas >= min_area)[0]:
            z = int(z)
            patch_id = f"{patient_id}_s{int(scan_idx):02d}_n{int(nod['nodule_idx']):02d}_z{z:03d}"
            cy_c, cx_c = cy0, cx0
            if jitter > 0:
                dy, dx = _jitter(patch_id, jitter)
                cy_c, cx_c = cy0 + dy, cx0 + dx
            patch_hu = crop_patch(vol[z], cy_c, cx_c, size, pad_value=pad_hu)
            src = union_u8[z] if include_other else m[z].astype(np.uint8)
            patch_mask = crop_patch(src, cy_c, cx_c, size, pad_value=0).astype(np.uint8)
            n_other = sum(
                1
                for j, other in enumerate(binary)
                if j != k and other[z].any() and crop_patch(other[z], cy_c, cx_c, size, pad_value=False).any()
            )
            morph = compute_patch_morphology(patch_hu, patch_mask)
            rows.append(
                {
                    "patch_id": patch_id,
                    "patient_id": patient_id,
                    "series_uid": series_uid,
                    "scan_idx": int(scan_idx),
                    "nodule_idx": int(nod["nodule_idx"]),
                    "z": z,
                    "cx": cx_c,
                    "cy": cy_c,
                    "n_readers": int(nod["n_readers"]),
                    "diameter_mm": float(nod["diameter_mm"]),
                    "diameter_2d_mm": float(nod["diameter_2d_mm"]),
                    "area_nodule_px": int(areas[z]),
                    "area_union_px": int(patch_mask.sum()),
                    "n_other_nodules": int(n_other),
                    "touches_border": bool(morph["touches_border"]),
                    "spacing_x": sx,
                    "spacing_y": sy,
                    "spacing_z": sz,
                    "mean_hu": morph["mean_hu"],
                    "min_hu": morph["min_hu"],
                    "std_hu": morph["std_hu"],
                    "solidity": morph["solidity"],
                    **nod.get("extra", {}),
                }
            )
            patches.append(np.clip(np.rint(patch_hu), -32768, 32767).astype(np.int16))
            masks.append(patch_mask)
    if not rows:
        empty_hu = np.zeros((0, size, size), np.int16)
        return empty_hu, np.zeros((0, size, size), np.uint8), []
    return np.stack(patches), np.stack(masks), rows


def _columns_for(rows: list[dict]) -> list[str]:
    cols = list(PATCH_COLUMNS)
    for row in rows:
        cols.extend(c for c in row if c not in cols)
    return cols


def save_scan_outputs(
    scan_dir: str | Path,
    patches: np.ndarray,
    masks: np.ndarray,
    rows: list[dict],
    summary: dict,
    reports: dict[str, list[dict]] | None = None,
) -> None:
    """Ghi kết quả một scan (atomic) rồi mới tạo `_DONE` rỗng."""
    scan_dir = Path(scan_dir)
    scan_dir.mkdir(parents=True, exist_ok=True)
    (scan_dir / DONE).unlink(missing_ok=True)
    if rows:
        save_npz_atomic(scan_dir / "patches.npz", patches_hu=patches, masks=masks)
        save_csv_atomic(scan_dir / "meta.csv", rows, columns=_columns_for(rows))
    else:
        (scan_dir / "patches.npz").unlink(missing_ok=True)
        (scan_dir / "meta.csv").unlink(missing_ok=True)
    for name, report_rows in (reports or {}).items():
        save_csv_atomic(scan_dir / name, report_rows)
    save_json_atomic(scan_dir / "summary.json", summary)
    save_text_atomic(scan_dir / DONE, "")


def merge_by_scan(
    out_dir: str | Path,
    patch_size: int = 64,
    reports: dict[str, str] | None = None,
    report_columns: dict[str, list[str]] | None = None,
) -> dict:
    """Gộp mọi `by_scan/*` đã có `_DONE`, sắp xếp tất định theo `patch_id`, assert không trùng.

    `reports` = `{tên_tệp_trong_by_scan: tên_tệp_gộp}`, ví dụ `{"match.csv": "match_report.csv"}`.
    """
    out_dir = Path(out_dir)
    by_scan = out_dir / "by_scan"
    dirs = sorted(d for d in by_scan.iterdir() if (d / DONE).exists()) if by_scan.is_dir() else []
    metas, hus, mks = [], [], []
    report_parts: dict[str, list[pd.DataFrame]] = {name: [] for name in (reports or {})}
    for d in dirs:
        if (d / "meta.csv").exists():
            meta = pd.read_csv(d / "meta.csv", dtype=ID_DTYPES)
            with np.load(d / "patches.npz") as npz:
                hu, mk = npz["patches_hu"], npz["masks"]
            if not (len(meta) == len(hu) == len(mk)):
                raise RuntimeError(f"{d.name}: meta.csv ({len(meta)}) và patches.npz ({len(hu)}) lệch số dòng")
            metas.append(meta)
            hus.append(hu)
            mks.append(mk)
        for name in report_parts:
            if (d / name).exists() and (d / name).stat().st_size > 0:
                part = pd.read_csv(d / name, dtype={"series_uid": str})
                if len(part):
                    report_parts[name].append(part)
    if metas:
        meta_all = pd.concat(metas, ignore_index=True)
        hu_all, mk_all = np.concatenate(hus), np.concatenate(mks)
        order = np.argsort(meta_all["patch_id"].to_numpy(dtype=str), kind="stable")
        meta_all = meta_all.iloc[order].reset_index(drop=True)
        hu_all, mk_all = hu_all[order], mk_all[order]
    else:
        meta_all = pd.DataFrame(columns=PATCH_COLUMNS)
        hu_all = np.zeros((0, patch_size, patch_size), np.int16)
        mk_all = np.zeros((0, patch_size, patch_size), np.uint8)
    dup = meta_all["patch_id"][meta_all["patch_id"].duplicated()]
    if len(dup):
        raise AssertionError(f"patch_id trùng lặp: {dup.tolist()[:10]}")
    save_npy_atomic(out_dir / "patches_hu.npy", hu_all)
    save_npy_atomic(out_dir / "masks.npy", mk_all)
    save_csv_atomic(out_dir / "patches.csv", meta_all)
    for name, out_name in (reports or {}).items():
        parts = report_parts[name]
        cols = (report_columns or {}).get(name)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols or [])
        save_csv_atomic(out_dir / out_name, df)
    return {
        "n_scans_done": len(dirs),
        "n_scans_with_patches": len(metas),
        "n_patches": int(len(meta_all)),
        "n_patients": int(meta_all["patient_id"].nunique()) if len(meta_all) else 0,
    }


# ---------------------------------------------------------------------------
# Một scan LIDC
# ---------------------------------------------------------------------------
def process_lidc_scan(job: dict) -> dict:
    """Xử lý trọn một CT scan LIDC theo §5.2 (bước 0–5); trả tóm tắt nhỏ (không chứa mảng)."""
    t0 = time.time()
    scan_dir = Path(job["scan_dir"])
    lidc_cfg, common = job["lidc_cfg"], job["common_cfg"]
    raw = Path(job["raw_dir"])
    seg_rows = job["seg_rows"]
    summary: dict[str, Any] = {
        "series_uid": job["series_uid"],
        "patient_id": job["patient_id"],
        "scan_idx": int(job["scan_idx"]),
        "n_seg_files": len(seg_rows),
        "clusters": [],
        "n_nodules_kept": 0,
        "n_patches": 0,
    }
    size = int(common["patch_size"])
    empty = (np.zeros((0, size, size), np.int16), np.zeros((0, size, size), np.uint8), [])
    clusters = group_clusters(seg_rows)
    min_readers = int(lidc_cfg.get("min_readers", 1))
    if not any(len(c["rows"]) >= min_readers for c in clusters):  # không cần đọc CT
        summary["clusters"] = [
            {"nodule_idx": c["nodule_idx"], "nodule_key": c["nodule_key"], "n_readers": len(c["rows"]),
             "status": "few_readers"} for c in clusters
        ]
        summary["seconds"] = round(time.time() - t0, 2)
        save_scan_outputs(scan_dir, *empty, summary)
        return summary

    ct = read_ct_series(raw / job["ct_rel_dir"])
    nodules = []
    for cl in clusters:
        n_readers = len(cl["rows"])
        info: dict[str, Any] = {"nodule_idx": cl["nodule_idx"], "nodule_key": cl["nodule_key"], "n_readers": n_readers}
        summary["clusters"].append(info)
        if n_readers < min_readers:
            info["status"] = "few_readers"
            continue
        readers = [lidc_reader_slices(read_seg_frames(raw / r["seg_rel_path"]), ct) for r in cl["rows"]]
        votes = votes_from_slices(readers, ct.vol_zyx.shape)
        thresh = consensus_threshold(n_readers, float(lidc_cfg.get("clevel", 0.5)))
        cmask = (votes >= thresh).astype(np.uint8)
        info["consensus_thresh"] = thresh
        info["n_voxels"] = int(cmask.sum())
        if not cmask.any():
            info["status"] = "empty_consensus"
            continue
        d3, d2 = compute_nodule_diameters(cmask, ct.spacing_zyx)
        info.update(diameter_3d_mm=round(d3, 3), diameter_2d_mm=round(d2, 3))
        if d3 < float(lidc_cfg.get("min_diameter_mm", 3.0)):
            info["status"] = "small"
            continue
        info["status"] = "kept"
        nodules.append(
            {
                "nodule_idx": cl["nodule_idx"],
                "n_readers": n_readers,
                "diameter_mm": d3,
                "diameter_2d_mm": d2,
                "cmask": cmask,
                "extra": {"nodule_key": cl["nodule_key"]},
            }
        )
    patches, masks, rows = extract_scan_patches(
        ct.vol_zyx, ct.spacing_zyx, nodules,
        patient_id=job["patient_id"], series_uid=job["series_uid"], scan_idx=int(job["scan_idx"]), common=common,
    ) if nodules else empty
    summary.update(
        n_nodules_kept=len(nodules),
        n_patches=len(rows),
        shape_zyx=list(ct.vol_zyx.shape),
        spacing_zyx=[round(float(s), 6) for s in ct.spacing_zyx],
        seconds=round(time.time() - t0, 2),
    )
    save_scan_outputs(scan_dir, patches, masks, rows, summary)
    return summary


# ---------------------------------------------------------------------------
# Hỗ trợ khảo sát Phase 1 (chỉ đọc SEG, không đọc CT)
# ---------------------------------------------------------------------------
def cluster_slice_areas(raw_dir: str | Path, seg_rel_paths: list[str], clevel: float = 0.5) -> list[int]:
    """Diện tích (pixel) mask đồng thuận trên từng lát của một cụm nốt, chỉ dùng các tệp SEG.

    Lát được nhận diện bằng `ReferencedSOPInstanceUID` (hoặc z làm tròn 0.01 mm nếu thiếu).
    """
    readers = []
    for rel in seg_rel_paths:
        seg = read_seg_frames(Path(raw_dir) / rel)
        slices: dict[str, np.ndarray] = {}
        for i, frame in enumerate(seg.arr):
            if not frame.any():
                continue
            key = seg.sops[i] or f"z{seg.ipps[i, 2]:.2f}"
            slices[key] = slices[key] | frame if key in slices else frame
        readers.append(slices)
    thresh = consensus_threshold(len(readers), clevel)
    keys = sorted({k for r in readers for k in r})
    areas = []
    for key in keys:
        votes = sum(r[key].astype(np.uint8) for r in readers if key in r)
        areas.append(int((votes >= thresh).sum()))
    return areas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_jobs(index_dir: str | Path, raw_dir: str | Path, out_dir: str | Path, cfg: dict, limit: int | None) -> list[dict]:
    index_dir, out_dir = Path(index_dir), Path(out_dir)
    ct = pd.read_csv(index_dir / "ct_scans.csv", dtype={"patient_id": str, "series_uid": str, "ct_rel_dir": str})
    seg = pd.read_csv(index_dir / "seg_index.csv", dtype={"patient_id": str, "ref_series_uid": str,
                                                          "seg_rel_path": str, "nodule_key": str})
    seg_by_scan = {
        uid: grp[["seg_rel_path", "nodule_key", "nodule_num", "label_anomaly"]].to_dict("records")
        for uid, grp in seg.groupby("ref_series_uid")
    }
    ct = ct.sort_values(["patient_id", "scan_idx"]).reset_index(drop=True)
    if limit:
        ct = ct.head(int(limit))
    jobs = []
    for row in ct.itertuples(index=False):
        scan_dir = out_dir / "by_scan" / row.series_uid
        jobs.append(
            {
                "series_uid": row.series_uid,
                "patient_id": row.patient_id,
                "scan_idx": int(row.scan_idx),
                "ct_rel_dir": row.ct_rel_dir,
                "seg_rows": seg_by_scan.get(row.series_uid, []),
                "raw_dir": str(raw_dir),
                "scan_dir": str(scan_dir),
                "done_marker": str(scan_dir / DONE),
                "common_cfg": dict(cfg["common"]),
                "lidc_cfg": dict(cfg["lidc"]),
            }
        )
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=None, help="Thư mục LIDC-IDRI (mặc định paths.lidc_raw)")
    parser.add_argument("--index-dir", default=None, help="Thư mục chỉ mục Phase 1 (mặc định paths.index_dir)")
    parser.add_argument("--out-dir", required=True, help="Thư mục interim, ví dụ data/interim/lidc")
    parser.add_argument("--config", default="configs/data.yaml", help="Config dữ liệu")
    parser.add_argument("--workers", type=int, default=None, help="Số tiến trình (mặc định common.workers)")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N scan đầu (debug)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    raw_dir = Path(args.raw_dir or cfg["paths"]["lidc_raw"])
    index_dir = Path(args.index_dir or cfg["paths"]["index_dir"])
    out_dir = Path(args.out_dir)
    workers = int(args.workers if args.workers is not None else cfg["common"].get("workers", 1))
    jobs = build_jobs(index_dir, raw_dir, out_dir, cfg, args.limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run = run_parallel(process_lidc_scan, jobs, workers, out_dir / "errors.csv", desc="LIDC")
    merged = merge_by_scan(out_dir, int(cfg["common"]["patch_size"]))
    save_json_atomic(
        out_dir / "preprocess_summary.json",
        {
            "dataset": "lidc",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(time.time() - t0, 1),
            "raw_dir": raw_dir.as_posix(),
            "index_dir": index_dir.as_posix(),
            "limit": args.limit,
            "data_config": cfg,
            "n_jobs": run["n_jobs"],
            "n_skipped_done": run["n_skipped"],
            "n_processed": run["n_done"],
            "n_failed": run["n_failed"],
            **merged,
        },
    )
    print(
        f"LIDC: {run['n_done']} scan mới, {run['n_skipped']} đã có _DONE, {run['n_failed']} lỗi "
        f"(xem errors.csv) → {merged['n_patches']} patch / {merged['n_patients']} bệnh nhân"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
