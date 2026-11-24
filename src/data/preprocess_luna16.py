"""Tiền xử lý LUNA16 → patch 64×64 — SPECS §5.3.

Ảnh CT đọc từ `.mhd/.raw` bằng SimpleITK; nhãn phân đoạn đọc trực tiếp từ các tệp `SEG_*/*.dcm`
tương ứng trong LIDC-IDRI (cùng `SeriesInstanceUID`), **không mở CT DICOM của LIDC**. Mỗi frame SEG
được đặt lên lưới `.mhd` qua tọa độ thế giới (`TransformPhysicalPointToContinuousIndex`, hoặc
`world_to_voxel_batch` cho từng pixel khi Direction/IOP khác ma trận đơn vị). Các cụm nốt đồng
thuận được khớp 1-1 với `annotations.csv` (khoảng cách tâm ≤ `diameter_mm / 2`); chỉ cụm đã khớp
được cắt patch, với `diameter_mm` lấy từ `annotations.csv`.

Đầu ra: <out_dir>/{patches_hu.npy, masks.npy, patches.csv, match_report.csv, errors.csv}
        + by_scan/<series_uid>/{patches.npz, meta.csv, match.csv, summary.json, _DONE}

    python -m src.data.preprocess_luna16 --raw-dir data/raw/LUNA16 --lidc-dir data/raw/LIDC-IDRI \
        --index-dir data/index --out-dir data/interim/luna16 --config configs/data.yaml --workers 4

`--limit N` xử lý N scan dương tính đầu tiên (scan âm tính chỉ là đánh dấu `_DONE`, được bỏ qua).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

from src.data.build_index import load_annotations
from src.data.ct_utils import StageError, compute_nodule_diameters, run_parallel, save_json_atomic
from src.data.preprocess_lidc import (
    DONE,
    SegFrames,
    consensus_threshold,
    extract_scan_patches,
    group_clusters,
    merge_by_scan,
    read_seg_frames,
    save_scan_outputs,
    votes_from_slices,
)
from src.utils import load_config

MATCH_COLUMNS = [
    "series_uid", "ann_idx", "coordX", "coordY", "coordZ", "diameter_mm", "matched_nodule_key",
    "matched_nodule_idx", "dist_mm", "n_candidates_in_radius", "status",
]
IDENTITY_IOP = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Hình học .mhd
# ---------------------------------------------------------------------------
class MhdGeometry:
    """Hình học ảnh (origin, spacing, direction, size) với cùng API biến đổi tọa độ như `sitk.Image`.

    Dùng khi chỉ cần hình học mà không muốn đọc `.raw` (ví dụ khảo sát Phase 1 trên 601 scan).
    """

    def __init__(self, origin, spacing, direction, size):
        self._origin = tuple(float(v) for v in origin)
        self._spacing = tuple(float(v) for v in spacing)
        self._direction = tuple(float(v) for v in direction)
        self._size = tuple(int(v) for v in size)

    @classmethod
    def from_file(cls, path: str | Path) -> "MhdGeometry":
        import SimpleITK as sitk

        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        return cls(reader.GetOrigin(), reader.GetSpacing(), reader.GetDirection(), reader.GetSize())

    @classmethod
    def from_image(cls, img) -> "MhdGeometry":
        return cls(img.GetOrigin(), img.GetSpacing(), img.GetDirection(), img.GetSize())

    def GetOrigin(self):  # noqa: N802 — giữ tên giống SimpleITK
        return self._origin

    def GetSpacing(self):  # noqa: N802
        return self._spacing

    def GetDirection(self):  # noqa: N802
        return self._direction

    def GetSize(self):  # noqa: N802
        return self._size

    def TransformPhysicalPointToContinuousIndex(self, point):  # noqa: N802
        return tuple(float(v) for v in world_to_voxel_batch(self, np.asarray(point, float)[None])[0])

    def TransformContinuousIndexToPhysicalPoint(self, index):  # noqa: N802
        return tuple(float(v) for v in voxel_to_world_batch(self, np.asarray(index, float)[None])[0])


def _affine(img) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(img.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(img.GetSpacing(), dtype=np.float64)
    direction = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    return origin, direction @ np.diag(spacing)


def world_to_voxel_batch(img, pts: np.ndarray) -> np.ndarray:
    """Tọa độ thế giới (M, 3) mm → chỉ số liên tục (M, 3) theo thứ tự (x, y, z) của ảnh.

    `idx = inv(Direction · diag(spacing)) · (pts − origin)`, giống `TransformPhysicalPointToContinuousIndex`.
    """
    origin, a = _affine(img)
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    return np.linalg.solve(a, (pts - origin).T).T


def voxel_to_world_batch(img, idx: np.ndarray) -> np.ndarray:
    """Ngược lại của `world_to_voxel_batch`: chỉ số (x, y, z) → tọa độ thế giới (mm)."""
    origin, a = _affine(img)
    idx = np.atleast_2d(np.asarray(idx, dtype=np.float64))
    return origin + (a @ idx.T).T


# ---------------------------------------------------------------------------
# SEG → lưới .mhd
# ---------------------------------------------------------------------------
def luna_reader_slices(seg: SegFrames, img, shape_zyx: tuple[int, int, int]) -> tuple[dict[int, np.ndarray], int]:
    """Đặt các frame của một SEG lên lưới `.mhd` → (`{z: mask 2D bool}`, số frame đi đường tổng quát)."""
    z_dim, h, w = shape_zyx
    direction = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    sx, sy, _ = img.GetSpacing()
    grid_ok = (
        seg.arr.shape[1:] == (h, w)
        and np.allclose(direction, np.eye(3), atol=1e-3)
        and np.allclose(seg.iop, IDENTITY_IOP, atol=1e-3)
        and np.allclose(seg.pixel_spacing, (sy, sx), rtol=1e-3, atol=1e-4)
    )
    row_dir, col_dir = seg.iop[0:3], seg.iop[3:6]
    ps_row, ps_col = seg.pixel_spacing
    out: dict[int, np.ndarray] = {}
    n_general = 0
    for i, frame in enumerate(seg.arr):
        if not frame.any():
            continue
        ipp = seg.ipps[i]
        if np.isnan(ipp).any():
            raise StageError("seg_map", f"frame {i} thiếu ImagePositionPatient")
        ix0, iy0, iz0 = img.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in ipp))
        k = int(round(iz0))
        if abs(iz0 - k) >= 0.35 or not 0 <= k < z_dim:
            raise StageError("seg_map", f"frame {i}: z liên tục {iz0:.3f} không trùng lát .mhd (Z={z_dim})")
        if grid_ok and abs(ix0) < 0.25 and abs(iy0) < 0.25:
            out[k] = out[k] | frame if k in out else frame.copy()
            continue
        n_general += 1
        rs, cs = np.nonzero(frame)
        pts = ipp[None, :] + (cs[:, None] * ps_col) * row_dir[None, :] + (rs[:, None] * ps_row) * col_dir[None, :]
        idx = world_to_voxel_batch(img, pts)
        xs, ys, zs = (np.rint(idx[:, j]).astype(int) for j in range(3))
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & (zs >= 0) & (zs < z_dim)
        for z in np.unique(zs[valid]):
            sel = valid & (zs == z)
            plane = out.setdefault(int(z), np.zeros((h, w), dtype=bool))
            plane[ys[sel], xs[sel]] = True
    return out, n_general


def build_luna_clusters(
    img, seg_rows: list[dict], lidc_dir: str | Path, clevel: float, shape_zyx: tuple[int, int, int]
) -> list[dict]:
    """Dựng mask đồng thuận 3D cho từng cụm `nodule_key` trên lưới `.mhd`."""
    clusters = []
    for cl in group_clusters(seg_rows):
        readers, n_general = [], 0
        for r in cl["rows"]:
            slices, ng = luna_reader_slices(read_seg_frames(Path(lidc_dir) / r["seg_rel_path"]), img, shape_zyx)
            readers.append(slices)
            n_general += ng
        n_readers = len(readers)
        thresh = consensus_threshold(n_readers, clevel)
        cmask = (votes_from_slices(readers, shape_zyx) >= thresh).astype(np.uint8)
        clusters.append(
            {
                "nodule_idx": cl["nodule_idx"],
                "nodule_key": cl["nodule_key"],
                "n_readers": n_readers,
                "consensus_thresh": thresh,
                "cmask": cmask,
                "n_general_frames": n_general,
            }
        )
    return clusters


def match_clusters(clusters: list[dict], img, ann_rows: list[dict], series_uid: str) -> tuple[dict[int, tuple], list[dict]]:
    """Khớp 1-1 tham lam theo khoảng cách tăng dần, điều kiện `dist ≤ diameter_mm / 2`.

    Trả `({ann_idx: (vị_trí_cụm, dist_mm)}, các dòng match_report)`.
    """
    centers = []
    for pos, cl in enumerate(clusters):
        if cl["cmask"].any():
            cz, cy, cx = ndimage.center_of_mass(cl["cmask"] > 0)
            centers.append((pos, np.asarray(img.TransformContinuousIndexToPhysicalPoint((float(cx), float(cy), float(cz))))))
    edges, n_cand = [], {}
    for a, ann in enumerate(ann_rows):
        p = np.array([ann["coordX"], ann["coordY"], ann["coordZ"]], dtype=np.float64)
        radius = float(ann["diameter_mm"]) / 2.0
        n_cand[a] = 0
        for pos, c in centers:
            d = float(np.linalg.norm(c - p))
            if d <= radius:
                n_cand[a] += 1
                edges.append((d, a, pos))
    edges.sort()
    assign: dict[int, tuple] = {}
    used: set[int] = set()
    for d, a, pos in edges:
        if a not in assign and pos not in used:
            assign[a] = (pos, d)
            used.add(pos)
    report = []
    for a, ann in enumerate(ann_rows):
        hit = assign.get(a)
        status = "unmatched" if hit is None else ("matched" if n_cand[a] == 1 else "multi_cand")
        report.append(
            {
                "series_uid": series_uid,
                "ann_idx": a,
                "coordX": ann["coordX"],
                "coordY": ann["coordY"],
                "coordZ": ann["coordZ"],
                "diameter_mm": ann["diameter_mm"],
                "matched_nodule_key": clusters[hit[0]]["nodule_key"] if hit else "",
                "matched_nodule_idx": clusters[hit[0]]["nodule_idx"] if hit else -1,
                "dist_mm": round(hit[1], 4) if hit else float("nan"),
                "n_candidates_in_radius": n_cand[a],
                "status": status,
            }
        )
    return assign, report


# ---------------------------------------------------------------------------
# Một scan LUNA16
# ---------------------------------------------------------------------------
def process_luna16_scan(job: dict) -> dict:
    """Xử lý trọn một scan LUNA16 theo §5.3 (bước 0–4)."""
    import SimpleITK as sitk

    t0 = time.time()
    scan_dir = Path(job["scan_dir"])
    common = job["common_cfg"]
    size = int(common["patch_size"])
    empty = (np.zeros((0, size, size), np.int16), np.zeros((0, size, size), np.uint8), [])
    ann_rows, seg_rows = job["ann_rows"], job["seg_rows"]
    summary: dict[str, Any] = {
        "series_uid": job["series_uid"],
        "patient_id": job["patient_id"],
        "scan_idx": int(job["scan_idx"]),
        "n_annotations": len(ann_rows),
        "n_seg_files": len(seg_rows),
        "n_matched": 0,
        "n_patches": 0,
        "clusters": [],
    }
    if not ann_rows:  # scan âm tính
        save_scan_outputs(scan_dir, *empty, summary)
        return summary
    if not job["patient_id"] or int(job["scan_idx"]) < 0:
        raise StageError("lookup", "series_uid không có trong ct_scans.csv của LIDC-IDRI")

    img = sitk.ReadImage(str(Path(job["raw_dir"]) / job["mhd_rel_path"]))
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    sx, sy, sz = img.GetSpacing()
    spacing_zyx = (float(sz), float(sy), float(sx))
    clusters = build_luna_clusters(img, seg_rows, job["lidc_dir"], float(job["luna16_cfg"].get("clevel", 0.5)), vol.shape)
    assign, report = match_clusters(clusters, img, ann_rows, job["series_uid"])

    nodules = []
    for a, (pos, dist) in sorted(assign.items()):
        cl = clusters[pos]
        d3, d2 = compute_nodule_diameters(cl["cmask"], spacing_zyx)
        nodules.append(
            {
                "nodule_idx": cl["nodule_idx"],
                "n_readers": cl["n_readers"],
                "diameter_mm": float(ann_rows[a]["diameter_mm"]),
                "diameter_2d_mm": d2,
                "cmask": cl["cmask"],
                "extra": {
                    "nodule_key": cl["nodule_key"],
                    "ann_idx": a,
                    "match_dist_mm": round(dist, 4),
                    "diameter_3d_mm": round(d3, 4),
                },
            }
        )
    patches, masks, rows = extract_scan_patches(
        vol, spacing_zyx, nodules,
        patient_id=job["patient_id"], series_uid=job["series_uid"], scan_idx=int(job["scan_idx"]), common=common,
    ) if nodules else empty
    summary.update(
        n_matched=len(nodules),
        n_patches=len(rows),
        shape_zyx=list(vol.shape),
        spacing_zyx=[round(s, 6) for s in spacing_zyx],
        direction_is_identity=bool(np.allclose(np.reshape(img.GetDirection(), (3, 3)), np.eye(3), atol=1e-3)),
        clusters=[
            {k: cl[k] for k in ("nodule_idx", "nodule_key", "n_readers", "consensus_thresh", "n_general_frames")}
            | {"n_voxels": int(cl["cmask"].sum())}
            for cl in clusters
        ],
        seconds=round(time.time() - t0, 2),
    )
    save_scan_outputs(scan_dir, patches, masks, rows, summary, reports={"match.csv": report})
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_jobs(
    index_dir: str | Path,
    raw_dir: str | Path,
    lidc_dir: str | Path,
    out_dir: str | Path,
    cfg: dict,
    limit: int | None,
) -> list[dict]:
    index_dir, out_dir = Path(index_dir), Path(out_dir)
    scans = pd.read_csv(index_dir / "luna16_scans.csv", dtype={"series_uid": str, "patient_id": str, "mhd_rel_path": str})
    scans["patient_id"] = scans["patient_id"].fillna("")
    seg = pd.read_csv(index_dir / "seg_index.csv", dtype={"ref_series_uid": str, "seg_rel_path": str, "nodule_key": str})
    seg_by_scan = {
        uid: grp[["seg_rel_path", "nodule_key", "nodule_num", "label_anomaly"]].to_dict("records")
        for uid, grp in seg.groupby("ref_series_uid")
    }
    ann = load_annotations(Path(raw_dir), cfg["luna16"].get("annotations_csv", "annotations.csv"))
    ann_by_scan = {
        uid: grp[["coordX", "coordY", "coordZ", "diameter_mm"]].astype(float).to_dict("records")
        for uid, grp in ann.groupby("seriesuid", sort=False)
    }
    scans = scans.sort_values("series_uid").reset_index(drop=True)
    if limit:
        positive = scans["series_uid"].map(lambda u: u in ann_by_scan)
        scans = scans[positive].head(int(limit))
    jobs = []
    for row in scans.itertuples(index=False):
        scan_dir = out_dir / "by_scan" / row.series_uid
        jobs.append(
            {
                "series_uid": row.series_uid,
                "patient_id": row.patient_id,
                "scan_idx": int(row.scan_idx),
                "mhd_rel_path": row.mhd_rel_path,
                "ann_rows": ann_by_scan.get(row.series_uid, []),
                "seg_rows": seg_by_scan.get(row.series_uid, []),
                "raw_dir": str(raw_dir),
                "lidc_dir": str(lidc_dir),
                "scan_dir": str(scan_dir),
                "done_marker": str(scan_dir / DONE),
                "common_cfg": dict(cfg["common"]),
                "luna16_cfg": dict(cfg["luna16"]),
            }
        )
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=None, help="Thư mục LUNA16 (mặc định paths.luna16_raw)")
    parser.add_argument("--lidc-dir", default=None, help="Thư mục LIDC-IDRI chứa SEG_* (mặc định paths.lidc_raw)")
    parser.add_argument("--index-dir", default=None, help="Thư mục chỉ mục Phase 1 (mặc định paths.index_dir)")
    parser.add_argument("--out-dir", required=True, help="Thư mục interim, ví dụ data/interim/luna16")
    parser.add_argument("--config", default="configs/data.yaml", help="Config dữ liệu")
    parser.add_argument("--workers", type=int, default=None, help="Số tiến trình (mặc định common.workers)")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N scan dương tính đầu (debug)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    raw_dir = Path(args.raw_dir or cfg["paths"]["luna16_raw"])
    lidc_dir = Path(args.lidc_dir or cfg["paths"]["lidc_raw"])
    index_dir = Path(args.index_dir or cfg["paths"]["index_dir"])
    out_dir = Path(args.out_dir)
    workers = int(args.workers if args.workers is not None else cfg["common"].get("workers", 1))
    jobs = build_jobs(index_dir, raw_dir, lidc_dir, out_dir, cfg, args.limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run = run_parallel(process_luna16_scan, jobs, workers, out_dir / "errors.csv", desc="LUNA16")
    merged = merge_by_scan(
        out_dir,
        int(cfg["common"]["patch_size"]),
        reports={"match.csv": "match_report.csv"},
        report_columns={"match.csv": MATCH_COLUMNS},
    )
    match = pd.read_csv(out_dir / "match_report.csv")
    status_counts = match["status"].value_counts().to_dict() if len(match) else {}
    save_json_atomic(
        out_dir / "preprocess_summary.json",
        {
            "dataset": "luna16",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(time.time() - t0, 1),
            "raw_dir": raw_dir.as_posix(),
            "lidc_dir": lidc_dir.as_posix(),
            "index_dir": index_dir.as_posix(),
            "limit": args.limit,
            "data_config": cfg,
            "n_jobs": run["n_jobs"],
            "n_skipped_done": run["n_skipped"],
            "n_processed": run["n_done"],
            "n_failed": run["n_failed"],
            "match_status": {str(k): int(v) for k, v in status_counts.items()},
            **merged,
        },
    )
    print(
        f"LUNA16: {run['n_done']} scan mới, {run['n_skipped']} đã có _DONE, {run['n_failed']} lỗi; "
        f"khớp annotations: {status_counts} → {merged['n_patches']} patch / {merged['n_patients']} bệnh nhân"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
