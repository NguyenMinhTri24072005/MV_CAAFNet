"""Lập chỉ mục dữ liệu thô một lần duy nhất — SPECS §1.2.

Chỉ đọc header DICOM (`stop_before_pixels=True`) và tệp text `.mhd`, không đọc pixel:
- LIDC-IDRI (NCI IDC DICOM-SEG): 1 tệp `.dcm` đầu tiên của mỗi `CT_*` + mọi tệp `SEG_*/*.dcm`.
- LUNA16: mọi `subset*/*.mhd` (vài trăm byte), không mở `.raw`.

Đầu ra trong `--out-dir`: `ct_scans.csv`, `seg_index.csv`, `luna16_scans.csv`, `index_summary.json`.
Mọi đường dẫn trong CSV là tương đối (dạng POSIX) so với thư mục gốc của bộ dữ liệu, để dùng
được cả ở local lẫn trên Colab.

    python -m src.data.build_index --lidc-dir data/raw/LIDC-IDRI --luna16-dir data/raw/LUNA16 \
        --out-dir data/index --workers 4
"""
from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data.ct_utils import save_csv_atomic, save_json_atomic

CT_COLUMNS = [
    "patient_id", "series_uid", "scan_idx", "ct_rel_dir", "n_dcm_files", "rows", "cols",
    "pixel_spacing_y", "pixel_spacing_x", "slice_thickness", "iop", "iop_is_identity",
    "n_seg_files", "n_nodule_keys", "in_luna16",
]
SEG_COLUMNS = [
    "patient_id", "ref_series_uid", "seg_rel_path", "seg_series_uid", "series_desc", "segment_label",
    "nodule_key", "nodule_num", "annotation_id", "label_anomaly", "n_segments", "n_frames", "rows",
    "cols", "pixel_spacing_y", "pixel_spacing_x", "iop", "iop_is_identity", "z_min", "z_max",
    "ref_in_patient_ct",
]
LUNA_COLUMNS = [
    "series_uid", "subset", "mhd_rel_path", "dim_x", "dim_y", "dim_z", "spacing_x", "spacing_y",
    "spacing_z", "offset_x", "offset_y", "offset_z", "transform_matrix", "direction_is_identity",
    "element_type", "n_annotations", "n_seg_files", "in_lidc", "patient_id", "scan_idx",
]
ANNOTATIONS_CSV = "annotations.csv"
IDENTITY_IOP = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

_NODULE_RE = re.compile(r"Nodule\s+(\d+)", re.IGNORECASE)
_ANNOT_RE = re.compile(r"Annotation\s+(.+?)\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Nhãn nốt
# ---------------------------------------------------------------------------
def parse_nodule_label(segment_label: str | None, series_desc: str | None, fallback_id: str) -> dict:
    """Tách `nodule_key` từ nhãn dạng `"Nodule <K> - Annotation <ID>"`.

    Lấy lần khớp ĐẦU TIÊN của `Nodule <K>` (có ca `"Nodule 1 - Annotation Nodule 001"`).
    Nhãn dị biệt (không khớp regex) → `label_anomaly=True`, `nodule_key` là phần trước
    `" - Annotation"` (nếu có), nếu không thì `"anomaly:<fallback_id>"` (một cụm riêng cho tệp đó).
    """
    for text in (segment_label, series_desc):
        if text:
            m = _NODULE_RE.search(str(text))
            if m:
                num = int(m.group(1))
                ann = _ANNOT_RE.search(str(text))
                return {
                    "nodule_key": f"Nodule {num}",
                    "nodule_num": num,
                    "annotation_id": ann.group(1) if ann else "",
                    "label_anomaly": False,
                }
    text = str(segment_label or series_desc or "").strip()
    text = re.sub(r"^Segmentation of\s+", "", text, flags=re.IGNORECASE)
    head = re.split(r"\s+-\s+Annotation", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    ann = _ANNOT_RE.search(text)
    return {
        "nodule_key": head if head else f"anomaly:{fallback_id}",
        "nodule_num": -1,
        "annotation_id": ann.group(1) if ann else "",
        "label_anomaly": True,
    }


def _iop_tuple(value: Iterable | None) -> tuple[float, ...]:
    return tuple(float(v) for v in value) if value is not None else ()


def _iop_str(iop: tuple[float, ...]) -> str:
    return " ".join(f"{v:g}" for v in iop)


def _is_identity_iop(iop: tuple[float, ...]) -> bool:
    return len(iop) == 6 and bool(np.allclose(iop, IDENTITY_IOP, atol=1e-3))


# ---------------------------------------------------------------------------
# LIDC-IDRI
# ---------------------------------------------------------------------------
def _dcm_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".dcm")


def _seg_header_row(path: Path, lidc_dir: Path, patient_id: str) -> dict:
    import pydicom

    ds = pydicom.dcmread(path, stop_before_pixels=True)
    segs = list(ds.get("SegmentSequence", []))
    segment_label = str(segs[0].get("SegmentLabel", "")) if segs else ""
    series_desc = str(ds.get("SeriesDescription", ""))
    seg_uid = str(ds.get("SeriesInstanceUID", ""))
    refs = list(ds.get("ReferencedSeriesSequence", []))
    ref_uid = str(refs[0].SeriesInstanceUID) if refs else ""
    sfg = ds.SharedFunctionalGroupsSequence[0] if "SharedFunctionalGroupsSequence" in ds else None
    ps = (float("nan"), float("nan"))
    iop: tuple[float, ...] = ()
    if sfg is not None:
        if "PixelMeasuresSequence" in sfg:
            ps = tuple(float(v) for v in sfg.PixelMeasuresSequence[0].PixelSpacing)  # (row, col) = (sy, sx)
        if "PlaneOrientationSequence" in sfg:
            iop = _iop_tuple(sfg.PlaneOrientationSequence[0].ImageOrientationPatient)
    zs = [
        float(pf.PlanePositionSequence[0].ImagePositionPatient[2])
        for pf in ds.get("PerFrameFunctionalGroupsSequence", [])
        if "PlanePositionSequence" in pf
    ]
    label = parse_nodule_label(segment_label, series_desc, seg_uid or path.parent.name)
    return {
        "patient_id": str(ds.get("PatientID", patient_id)) or patient_id,
        "ref_series_uid": ref_uid,
        "seg_rel_path": path.relative_to(lidc_dir).as_posix(),
        "seg_series_uid": seg_uid,
        "series_desc": series_desc,
        "segment_label": segment_label,
        **label,
        "n_segments": len(segs),
        "n_frames": int(ds.get("NumberOfFrames", len(zs) or 1)),
        "rows": int(ds.get("Rows", 0)),
        "cols": int(ds.get("Columns", 0)),
        "pixel_spacing_y": ps[0],
        "pixel_spacing_x": ps[1],
        "iop": _iop_str(iop),
        "iop_is_identity": _is_identity_iop(iop),
        "z_min": min(zs) if zs else float("nan"),
        "z_max": max(zs) if zs else float("nan"),
    }


def _ct_header_row(ct_dir: Path, lidc_dir: Path, patient_id: str) -> dict:
    import pydicom

    files = _dcm_files(ct_dir)
    if not files:
        raise ValueError("thư mục CT không có tệp .dcm")
    ds = pydicom.dcmread(files[0], stop_before_pixels=True)
    ps = tuple(float(v) for v in ds.get("PixelSpacing", (float("nan"), float("nan"))))
    iop = _iop_tuple(ds.get("ImageOrientationPatient"))
    return {
        "patient_id": str(ds.get("PatientID", patient_id)) or patient_id,
        "series_uid": str(ds.SeriesInstanceUID),
        "ct_rel_dir": ct_dir.relative_to(lidc_dir).as_posix(),
        "n_dcm_files": len(files),
        "rows": int(ds.get("Rows", 0)),
        "cols": int(ds.get("Columns", 0)),
        "pixel_spacing_y": ps[0],
        "pixel_spacing_x": ps[1],
        "slice_thickness": float(ds.get("SliceThickness", float("nan")) or float("nan")),
        "iop": _iop_str(iop),
        "iop_is_identity": _is_identity_iop(iop),
    }


def index_patient(p_dir: Path, lidc_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Đọc header CT (1 tệp/scan) và mọi SEG của một bệnh nhân → (ct_rows, seg_rows, errors)."""
    patient_id = p_dir.name
    ct_rows, seg_rows, errors = [], [], []
    for sub in sorted(d for d in p_dir.iterdir() if d.is_dir()):
        if sub.name.upper().startswith("CT_"):
            try:
                ct_rows.append(_ct_header_row(sub, lidc_dir, patient_id))
            except Exception as exc:  # noqa: BLE001
                errors.append({"path": sub.relative_to(lidc_dir).as_posix(), "error": f"{type(exc).__name__}: {exc}"})
        elif sub.name.upper().startswith("SEG_"):
            for f in _dcm_files(sub):
                try:
                    seg_rows.append(_seg_header_row(f, lidc_dir, patient_id))
                except Exception as exc:  # noqa: BLE001
                    errors.append({"path": f.relative_to(lidc_dir).as_posix(), "error": f"{type(exc).__name__}: {exc}"})
    for idx, row in enumerate(sorted(ct_rows, key=lambda r: r["series_uid"])):
        row["scan_idx"] = idx
    ct_uids = {r["series_uid"] for r in ct_rows}
    for row in seg_rows:
        row["ref_in_patient_ct"] = row["ref_series_uid"] in ct_uids
    return ct_rows, seg_rows, errors


# ---------------------------------------------------------------------------
# LUNA16
# ---------------------------------------------------------------------------
def parse_mhd_header(path: Path) -> dict[str, str]:
    """Đọc phần header text của tệp MetaImage `.mhd` (dạng `Key = Value`)."""
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "=" in line:
                key, value = line.split("=", 1)
                out[key.strip()] = value.strip()
    return out


def _mhd_row(path: Path, luna16_dir: Path) -> dict:
    hdr = parse_mhd_header(path)
    dims = [int(v) for v in hdr.get("DimSize", "0 0 0").split()]
    spacing = [float(v) for v in hdr.get("ElementSpacing", hdr.get("ElementSize", "nan nan nan")).split()]
    offset = [float(v) for v in hdr.get("Offset", hdr.get("Origin", hdr.get("Position", "0 0 0"))).split()]
    matrix = [float(v) for v in hdr.get("TransformMatrix", "1 0 0 0 1 0 0 0 1").split()]
    return {
        "series_uid": path.stem,
        "subset": path.parent.name,
        "mhd_rel_path": path.relative_to(luna16_dir).as_posix(),
        "dim_x": dims[0], "dim_y": dims[1], "dim_z": dims[2],
        "spacing_x": spacing[0], "spacing_y": spacing[1], "spacing_z": spacing[2],
        "offset_x": offset[0], "offset_y": offset[1], "offset_z": offset[2],
        "transform_matrix": " ".join(f"{v:g}" for v in matrix),
        "direction_is_identity": bool(np.allclose(matrix, np.eye(3).ravel(), atol=1e-3)),
        "element_type": hdr.get("ElementType", ""),
    }


def load_annotations(luna16_dir: Path, name: str = ANNOTATIONS_CSV) -> pd.DataFrame:
    path = Path(luna16_dir) / name
    if not path.is_file():
        return pd.DataFrame(columns=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])
    return pd.read_csv(path, dtype={"seriesuid": str})


# ---------------------------------------------------------------------------
# Toàn bộ
# ---------------------------------------------------------------------------
def build_index(
    lidc_dir: str | Path,
    luna16_dir: str | Path | None,
    out_dir: str | Path,
    workers: int = 4,
    patients: Iterable[str] | None = None,
    luna16_series: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Lập chỉ mục và ghi 4 tệp vào `out_dir`; trả về nội dung `index_summary.json`.

    `patients` / `luna16_series` (chỉ dùng qua API, ví dụ khi dry-run) giới hạn phạm vi quét.
    """
    t0 = time.time()
    lidc_dir = Path(lidc_dir)
    out_dir = Path(out_dir)
    if not lidc_dir.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục LIDC-IDRI: {lidc_dir}")
    p_dirs = sorted(d for d in lidc_dir.iterdir() if d.is_dir() and d.name.upper().startswith("LIDC-IDRI-"))
    if patients is not None:
        wanted = set(patients)
        p_dirs = [d for d in p_dirs if d.name in wanted]

    ct_rows: list[dict] = []
    seg_rows: list[dict] = []
    errors: list[dict] = []
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(it, **_):  # type: ignore[no-redef]
            return it
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for c, s, e in tqdm(pool.map(lambda d: index_patient(d, lidc_dir), p_dirs), total=len(p_dirs), desc="LIDC"):
            ct_rows += c
            seg_rows += s
            errors += e

    seg_df = pd.DataFrame(seg_rows, columns=SEG_COLUMNS)
    seg_df = seg_df.sort_values(["patient_id", "ref_series_uid", "nodule_num", "nodule_key", "seg_rel_path"],
                                kind="mergesort").reset_index(drop=True)
    ct_df = pd.DataFrame(ct_rows)
    if ct_df.empty:
        ct_df = pd.DataFrame(columns=CT_COLUMNS)
    seg_counts = seg_df.groupby("ref_series_uid").size() if len(seg_df) else pd.Series(dtype=int)
    key_counts = seg_df.groupby("ref_series_uid")["nodule_key"].nunique() if len(seg_df) else pd.Series(dtype=int)
    ct_df["n_seg_files"] = ct_df["series_uid"].map(seg_counts).fillna(0).astype(int)
    ct_df["n_nodule_keys"] = ct_df["series_uid"].map(key_counts).fillna(0).astype(int)

    # LUNA16
    ann = pd.DataFrame(columns=["seriesuid"])
    luna_rows: list[dict] = []
    if luna16_dir is not None and Path(luna16_dir).is_dir():
        luna16_dir = Path(luna16_dir)
        mhds = sorted(luna16_dir.glob("subset*/*.mhd"))
        if luna16_series is not None:
            wanted_l = set(luna16_series)
            mhds = [p for p in mhds if p.stem in wanted_l]
        for p in mhds:
            try:
                luna_rows.append(_mhd_row(p, luna16_dir))
            except Exception as exc:  # noqa: BLE001
                errors.append({"path": p.relative_to(luna16_dir).as_posix(), "error": f"{type(exc).__name__}: {exc}"})
        ann = load_annotations(luna16_dir)
    luna_df = pd.DataFrame(luna_rows, columns=[c for c in LUNA_COLUMNS if c not in
                                               ("n_annotations", "n_seg_files", "in_lidc", "patient_id", "scan_idx")])
    ann_counts = ann.groupby("seriesuid").size() if len(ann) else pd.Series(dtype=int)
    ct_lookup = ct_df.set_index("series_uid")[["patient_id", "scan_idx"]] if len(ct_df) else None
    luna_df["n_annotations"] = luna_df["series_uid"].map(ann_counts).fillna(0).astype(int)
    luna_df["n_seg_files"] = luna_df["series_uid"].map(seg_counts).fillna(0).astype(int)
    luna_df["in_lidc"] = luna_df["series_uid"].isin(ct_df["series_uid"]) if len(ct_df) else False
    if ct_lookup is not None:
        luna_df["patient_id"] = luna_df["series_uid"].map(ct_lookup["patient_id"]).fillna("")
        luna_df["scan_idx"] = luna_df["series_uid"].map(ct_lookup["scan_idx"]).fillna(-1).astype(int)
    else:
        luna_df["patient_id"] = ""
        luna_df["scan_idx"] = -1
    luna_df = luna_df[LUNA_COLUMNS].sort_values("series_uid").reset_index(drop=True)
    ct_df["in_luna16"] = ct_df["series_uid"].isin(luna_df["series_uid"])
    ct_df = ct_df[CT_COLUMNS].sort_values(["patient_id", "scan_idx"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv_atomic(out_dir / "ct_scans.csv", ct_df)
    save_csv_atomic(out_dir / "seg_index.csv", seg_df)
    save_csv_atomic(out_dir / "luna16_scans.csv", luna_df)

    readers = seg_df.groupby(["ref_series_uid", "nodule_key"]).size() if len(seg_df) else pd.Series(dtype=int)
    scans_per_patient = ct_df.groupby("patient_id").size() if len(ct_df) else pd.Series(dtype=int)
    seg_patients = set(seg_df["patient_id"]) if len(seg_df) else set()
    ann_series = set(ann["seriesuid"]) if len(ann) else set()
    luna_series = set(luna_df["series_uid"])
    positive = luna_df[luna_df["n_annotations"] > 0]
    summary = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
        "lidc_dir": lidc_dir.as_posix(),
        "luna16_dir": Path(luna16_dir).as_posix() if luna16_dir is not None else None,
        "lidc": {
            "n_patients": len(p_dirs),
            "n_ct_scans": int(len(ct_df)),
            "patients_multi_scan": sorted(scans_per_patient[scans_per_patient > 1].index.tolist()),
            "n_patients_without_seg": int(sum(1 for d in p_dirs if d.name not in seg_patients)),
            "n_seg_files": int(len(seg_df)),
            "n_nodule_clusters": int(len(readers)),
            "readers_per_cluster": {str(k): int(v) for k, v in sorted(Counter(readers.tolist()).items())},
            "n_label_anomaly": int(seg_df["label_anomaly"].sum()) if len(seg_df) else 0,
            "n_seg_ref_not_in_patient_ct": int((~seg_df["ref_in_patient_ct"].astype(bool)).sum()) if len(seg_df) else 0,
            "n_seg_multi_segment": int((seg_df["n_segments"] > 1).sum()) if len(seg_df) else 0,
            "n_seg_iop_not_identity": int((~seg_df["iop_is_identity"].astype(bool)).sum()) if len(seg_df) else 0,
            "n_ct_iop_not_identity": int((~ct_df["iop_is_identity"].astype(bool)).sum()) if len(ct_df) else 0,
        },
        "luna16": {
            "n_scans": int(len(luna_df)),
            "n_scans_with_annotations": int(len(positive)),
            "n_annotations": int(len(ann)),
            "n_annotation_series": len(ann_series),
            "n_annotation_series_without_mhd": len(ann_series - luna_series),
            "n_positive_scans_without_seg": int((positive["n_seg_files"] == 0).sum()),
            "n_scans_not_in_lidc": int((~luna_df["in_lidc"].astype(bool)).sum()),
            "n_direction_not_identity": int((~luna_df["direction_is_identity"].astype(bool)).sum()),
        },
        "n_errors": len(errors),
        "errors": errors,
    }
    save_json_atomic(out_dir / "index_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lidc-dir", default="data/raw/LIDC-IDRI", help="Thư mục LIDC-IDRI (IDC DICOM-SEG)")
    parser.add_argument("--luna16-dir", default="data/raw/LUNA16", help="Thư mục LUNA16 (subset*/*.mhd + annotations.csv)")
    parser.add_argument("--out-dir", default="data/index", help="Nơi ghi ct_scans.csv, seg_index.csv, ...")
    parser.add_argument("--workers", type=int, default=4, help="Số luồng đọc header (2-4 trên Drive FUSE)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_index(args.lidc_dir, args.luna16_dir, args.out_dir, args.workers)
    lidc, luna = summary["lidc"], summary["luna16"]
    print(
        f"LIDC: {lidc['n_patients']} bệnh nhân, {lidc['n_ct_scans']} scan, {lidc['n_seg_files']} SEG, "
        f"{lidc['n_nodule_clusters']} cụm nốt, {lidc['n_label_anomaly']} nhãn dị biệt"
    )
    print(
        f"LUNA16: {luna['n_scans']} scan, {luna['n_scans_with_annotations']} dương tính, "
        f"{luna['n_annotations']} nốt, {luna['n_positive_scans_without_seg']} scan dương tính thiếu SEG"
    )
    print(f"Lỗi đọc: {summary['n_errors']} — xem {Path(args.out_dir) / 'index_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
