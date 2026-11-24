"""Kiểm thử pipeline dữ liệu — SPECS §1.2, §5.2–5.6, Phụ lục C.

Dữ liệu thô được sinh tổng hợp (CT DICOM, DICOM-SEG, `.mhd`) nên chạy ở local không cần dữ liệu thật.
Phần build_index / preprocess / make_splits / pack_h5 / D4 (numpy) không cần torch; phần
`H5SegDataset` / `make_loaders` dùng `pytest.importorskip("torch")` và chạy khi lên Colab.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import build_index, make_splits, pack_h5, preprocess_lidc, preprocess_luna16  # noqa: E402
from src.data.compute_edt import edt_from_mask  # noqa: E402
from src.data.ct_utils import crop_patch, floor_hu, window_hu  # noqa: E402
from src.data.dataset import d4_transform  # noqa: E402
from src.utils import load_config  # noqa: E402

DATA_CFG = ROOT / "configs" / "data.yaml"
CT_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
SEG_STORAGE = "1.2.840.10008.5.1.4.1.1.66.4"

# Hình học chung của các scan tổng hợp
Z, H, W = 12, 48, 48
PS, DZ = 0.7, 2.5
ORIGIN = (-20.0, -30.0, -50.0)
UID_A1, UID_A2 = "1.2.826.0.1.3680043.1.1", "1.2.826.0.1.3680043.1.2"  # A1 < A2 theo thứ tự chuỗi
UID_B, UID_C = "1.2.826.0.1.3680043.2.1", "1.2.826.0.1.3680043.3.1"


# ---------------------------------------------------------------------------
# Sinh DICOM / MHD tổng hợp
# ---------------------------------------------------------------------------
def _save_dicom(ds, path: Path, sop_class: str, sop_uid: str) -> None:
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = meta
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds.save_as(path, enforce_file_format=True)
    except TypeError:  # pydicom < 3
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(path, write_like_original=False)


def write_ct(folder: Path, patient_id: str, series_uid: str, vol_hu: np.ndarray, z_offsets=None) -> list[str]:
    """Ghi một CT series; tên tệp đặt NGƯỢC thứ tự z để kiểm tra bước sắp xếp."""
    from pydicom.dataset import Dataset
    from pydicom.uid import generate_uid

    n = len(vol_hu)
    zs = [ORIGIN[2] + k * DZ for k in range(n)] if z_offsets is None else z_offsets
    sops = [generate_uid() for _ in range(n)]
    for k in range(n):
        ds = Dataset()
        ds.PatientID = patient_id
        ds.Modality = "CT"
        ds.StudyInstanceUID = "1.2.826.0.1.3680043.9"
        ds.SeriesInstanceUID = series_uid
        ds.Rows, ds.Columns = vol_hu.shape[1:]
        ds.PixelSpacing = [PS, PS]
        ds.SliceThickness = DZ
        ds.ImagePositionPatient = [ORIGIN[0], ORIGIN[1], zs[k]]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
        ds.PixelData = (vol_hu[k].astype(np.int32) + 1024).astype(np.int16).tobytes()
        _save_dicom(ds, folder / f"{n - k:08d}.dcm", CT_STORAGE, sops[k])
    return sops


def write_seg(path: Path, patient_id: str, ref_series: str, label: str, frames: list[tuple]) -> None:
    """Một tệp DICOM-SEG 8-bit: `frames` = [(sop_tham_chiếu, ipp, mask2d), ...]."""
    from pydicom.dataset import Dataset
    from pydicom.uid import generate_uid

    ds = Dataset()
    ds.PatientID = patient_id
    ds.Modality = "SEG"
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDescription = f"Segmentation of {label}"
    seg = Dataset()
    seg.SegmentNumber = 1
    seg.SegmentLabel = label
    ds.SegmentSequence = [seg]
    ref = Dataset()
    ref.SeriesInstanceUID = ref_series
    ds.ReferencedSeriesSequence = [ref]
    pm, po, shared = Dataset(), Dataset(), Dataset()
    pm.PixelSpacing = [PS, PS]
    pm.SliceThickness = DZ
    po.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    shared.PixelMeasuresSequence = [pm]
    shared.PlaneOrientationSequence = [po]
    ds.SharedFunctionalGroupsSequence = [shared]
    per = []
    for sop, ipp, _ in frames:
        src, der, pos, item = Dataset(), Dataset(), Dataset(), Dataset()
        src.ReferencedSOPInstanceUID = sop
        der.SourceImageSequence = [src]
        pos.ImagePositionPatient = list(ipp)
        item.DerivationImageSequence = [der]
        item.PlanePositionSequence = [pos]
        per.append(item)
    ds.PerFrameFunctionalGroupsSequence = per
    ds.NumberOfFrames = len(frames)
    ds.Rows, ds.Columns = frames[0][2].shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = np.stack([f[2] for f in frames]).astype(np.uint8).tobytes()
    _save_dicom(ds, path, SEG_STORAGE, generate_uid())


def disk(cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).astype(np.uint8)


def reader_volume(spec: list[tuple]) -> np.ndarray:
    """spec = [(z, cy, cx, r), ...] → mask 3D (Z, H, W) của một bác sĩ."""
    vol = np.zeros((Z, H, W), np.uint8)
    for z, cy, cx, r in spec:
        vol[z] |= disk(cy, cx, r)
    return vol


def seg_frames(mask3d: np.ndarray, sops: list[str]) -> list[tuple]:
    return [
        (sops[z], (ORIGIN[0], ORIGIN[1], ORIGIN[2] + z * DZ), mask3d[z])
        for z in range(Z)
        if mask3d[z].any()
    ]


def make_volume(seed: int, nodule_masks: list[np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = rng.integers(-900, -700, (Z, H, W)).astype(np.int16)
    vol[:, :, :3] = -3000  # vùng ngoài khung quét → floor_hu đưa về -1000
    for m in nodule_masks:
        vol[m > 0] = rng.integers(0, 80)
    return vol


def write_mhd(path: Path, vol_zyx: np.ndarray, flip_y: bool = False) -> None:
    import SimpleITK as sitk

    arr = vol_zyx[:, ::-1, :] if flip_y else vol_zyx
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr).astype(np.int16))
    img.SetSpacing((PS, PS, DZ))
    if flip_y:
        img.SetDirection((1, 0, 0, 0, -1, 0, 0, 0, 1))
        img.SetOrigin((ORIGIN[0], ORIGIN[1] + PS * (H - 1), ORIGIN[2]))
    else:
        img.SetOrigin(ORIGIN)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


def world_center(mask3d: np.ndarray) -> tuple[float, float, float]:
    from scipy import ndimage

    cz, cy, cx = ndimage.center_of_mass(mask3d > 0)
    return ORIGIN[0] + cx * PS, ORIGIN[1] + cy * PS, ORIGIN[2] + cz * DZ


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    """Cây dữ liệu thô tổng hợp:

    - LIDC-IDRI-0001: 2 scan (A1, A2). A1: Nodule 1 (3 bác sĩ) + Nodule 3 (1 bác sĩ, < 3 mm) +
      nhãn dị biệt. A2: Nodule 1 + Nodule 2 nằm cạnh (lọt vào patch của Nodule 1).
    - LIDC-IDRI-0002: 1 scan B không có SEG.
    - LIDC-IDRI-0003: 1 scan C có lát trùng z (phải ghi errors.csv) + 1 SEG.
    - LUNA16: subset0/A1.mhd (Direction = I), subset1/A2.mhd (lật trục y), subset2/B.mhd (âm tính).
    """
    root = tmp_path_factory.mktemp("raw")
    lidc, luna = root / "LIDC-IDRI", root / "LUNA16"
    pa = lidc / "LIDC-IDRI-0001"

    n1_readers = [
        reader_volume([(z, 20, 24, 5) for z in range(4, 8)]),
        reader_volume([(z, 20, 25, 5) for z in range(4, 8)]),
        reader_volume([(z, 20, 24, 4) for z in range(5, 8)]),
    ]
    n3 = reader_volume([(9, 8, 40, 1)])
    anomaly = reader_volume([(2, 36, 10, 3)])
    vol_a1 = make_volume(1, n1_readers + [n3, anomaly])
    sops_a1 = write_ct(pa / "CT_a1", "LIDC-IDRI-0001", UID_A1, vol_a1)
    for i, m in enumerate(n1_readers):
        write_seg(pa / f"SEG_a1n1r{i}" / "00000001.dcm", "LIDC-IDRI-0001", UID_A1,
                  f"Nodule 1 - Annotation R{i}", seg_frames(m, sops_a1))
    write_seg(pa / "SEG_a1n3" / "00000001.dcm", "LIDC-IDRI-0001", UID_A1, "Nodule 3 - Annotation Nodule 007",
              seg_frames(n3, sops_a1))
    write_seg(pa / "SEG_a1x" / "00000001.dcm", "LIDC-IDRI-0001", UID_A1, "Weird lesion - Annotation X1",
              seg_frames(anomaly, sops_a1))

    a2_n1 = [reader_volume([(z, 22, 20, 5) for z in range(3, 7)]) for _ in range(2)]
    a2_n2 = [reader_volume([(z, 22, 34, 3) for z in range(4, 6)]) for _ in range(2)]
    vol_a2 = make_volume(2, a2_n1 + a2_n2)
    sops_a2 = write_ct(pa / "CT_a2", "LIDC-IDRI-0001", UID_A2, vol_a2)
    for i, m in enumerate(a2_n1):
        write_seg(pa / f"SEG_a2n1r{i}" / "00000001.dcm", "LIDC-IDRI-0001", UID_A2,
                  f"Nodule 1 - Annotation Q{i}", seg_frames(m, sops_a2))
    for i, m in enumerate(a2_n2):
        write_seg(pa / f"SEG_a2n2r{i}" / "00000001.dcm", "LIDC-IDRI-0001", UID_A2,
                  f"Nodule 2 - Annotation Q{i}", seg_frames(m, sops_a2))

    vol_b = make_volume(3, [])
    write_ct(lidc / "LIDC-IDRI-0002" / "CT_b", "LIDC-IDRI-0002", UID_B, vol_b)

    c_mask = reader_volume([(z, 24, 24, 4) for z in range(4, 7)])
    zs_dup = [ORIGIN[2] + k * DZ for k in range(Z)]
    zs_dup[5] = zs_dup[4]
    sops_c = write_ct(lidc / "LIDC-IDRI-0003" / "CT_c", "LIDC-IDRI-0003", UID_C, make_volume(4, [c_mask]), zs_dup)
    write_seg(lidc / "LIDC-IDRI-0003" / "SEG_c1" / "00000001.dcm", "LIDC-IDRI-0003", UID_C,
              "Nodule 1 - Annotation Z", seg_frames(c_mask, sops_c))

    write_mhd(luna / "subset0" / f"{UID_A1}.mhd", vol_a1)
    write_mhd(luna / "subset1" / f"{UID_A2}.mhd", vol_a2, flip_y=True)
    write_mhd(luna / "subset2" / f"{UID_B}.mhd", vol_b)
    votes_a1 = sum(n1_readers)
    cons_a1 = (votes_a1 >= 2).astype(np.uint8)
    cons_a2 = (sum(a2_n1) >= 1).astype(np.uint8)
    ann = pd.DataFrame(
        [
            (UID_A1, *world_center(cons_a1), 6.0),
            (UID_A1, 5.0, 5.0, -20.0, 4.0),  # không có cụm nào gần → unmatched
            (UID_A2, *world_center(cons_a2), 6.0),
        ],
        columns=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"],
    )
    ann.to_csv(luna / "annotations.csv", index=False)
    return {
        "root": root, "lidc": lidc, "luna": luna,
        "vol_a1": vol_a1, "vol_a2": vol_a2, "cons_a1": cons_a1, "cons_a2": cons_a2,
        "a2_n2": (sum(a2_n2) >= 1).astype(np.uint8),
    }


@pytest.fixture(scope="module")
def index_dir(raw, tmp_path_factory):
    out = tmp_path_factory.mktemp("index")
    build_index.main(["--lidc-dir", str(raw["lidc"]), "--luna16-dir", str(raw["luna"]),
                      "--out-dir", str(out), "--workers", "2"])
    return out


@pytest.fixture(scope="module")
def lidc_out(raw, index_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("interim") / "lidc"
    assert preprocess_lidc.main(["--raw-dir", str(raw["lidc"]), "--index-dir", str(index_dir), "--out-dir", str(out),
                                 "--config", str(DATA_CFG), "--workers", "1"]) == 0
    return out


@pytest.fixture(scope="module")
def luna_out(raw, index_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("interim") / "luna16"
    assert preprocess_luna16.main(["--raw-dir", str(raw["luna"]), "--lidc-dir", str(raw["lidc"]),
                                   "--index-dir", str(index_dir), "--out-dir", str(out),
                                   "--config", str(DATA_CFG), "--workers", "1"]) == 0
    return out


def read_patches(out: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta = pd.read_csv(out / "patches.csv", dtype={"patch_id": str, "series_uid": str})
    return meta, np.load(out / "patches_hu.npy"), np.load(out / "masks.npy")


# ---------------------------------------------------------------------------
# CLI (Phụ lục A)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "module, flags",
    [
        (build_index, ["--lidc-dir", "--luna16-dir", "--out-dir", "--workers"]),
        (preprocess_lidc, ["--raw-dir", "--index-dir", "--out-dir", "--config", "--workers", "--limit"]),
        (preprocess_luna16, ["--raw-dir", "--lidc-dir", "--index-dir", "--out-dir", "--config", "--workers", "--limit"]),
        (make_splits, ["--index", "--out", "--config", "--mode"]),
        (pack_h5, ["--interim-dir", "--splits", "--out-dir", "--config", "--copy-to-drive-dir"]),
    ],
)
def test_cli_matches_appendix_a(module, flags):
    import argparse

    actions = [a for a in module.build_parser()._actions if not isinstance(a, argparse._HelpAction)]
    assert [s for a in actions for s in a.option_strings] == flags


# ---------------------------------------------------------------------------
# build_index (§1.2)
# ---------------------------------------------------------------------------
class TestBuildIndex:
    @pytest.mark.parametrize(
        "label, key, num, ann, anomaly",
        [
            ("Nodule 1 - Annotation IL057_127364", "Nodule 1", 1, "IL057_127364", False),
            ("Nodule 12 - Annotation Nodule 001", "Nodule 12", 12, "Nodule 001", False),
            ("Weird lesion - Annotation X1", "Weird lesion", -1, "X1", True),
            ("", "anomaly:uid9", -1, "", True),
        ],
    )
    def test_parse_nodule_label(self, label, key, num, ann, anomaly):
        out = build_index.parse_nodule_label(label, None, "uid9")
        assert (out["nodule_key"], out["nodule_num"], out["annotation_id"], out["label_anomaly"]) == (key, num, ann, anomaly)

    def test_falls_back_to_series_description(self):
        out = build_index.parse_nodule_label(None, "Segmentation of Nodule 4 - Annotation 2", "x")
        assert out["nodule_key"] == "Nodule 4" and not out["label_anomaly"]

    def test_index_files(self, index_dir):
        ct = pd.read_csv(index_dir / "ct_scans.csv", dtype={"series_uid": str})
        assert list(ct.columns) == build_index.CT_COLUMNS
        assert len(ct) == 4
        a = ct[ct["patient_id"] == "LIDC-IDRI-0001"].set_index("series_uid")
        assert a.loc[UID_A1, "scan_idx"] == 0 and a.loc[UID_A2, "scan_idx"] == 1
        assert a.loc[UID_A1, "n_dcm_files"] == Z and a.loc[UID_A1, "ct_rel_dir"] == "LIDC-IDRI-0001/CT_a1"
        assert a.loc[UID_A1, "n_seg_files"] == 5 and a.loc[UID_A1, "n_nodule_keys"] == 3
        seg = pd.read_csv(index_dir / "seg_index.csv", dtype={"ref_series_uid": str})
        assert list(seg.columns) == build_index.SEG_COLUMNS
        assert len(seg) == 10 and seg["ref_in_patient_ct"].all()
        weird = seg[seg["label_anomaly"]]
        assert weird["nodule_key"].tolist() == ["Weird lesion"]
        assert seg[seg["nodule_key"] == "Nodule 3"]["annotation_id"].tolist() == ["Nodule 007"]
        luna = pd.read_csv(index_dir / "luna16_scans.csv", dtype={"series_uid": str}).set_index("series_uid")
        assert list(luna.reset_index().columns) == build_index.LUNA_COLUMNS
        assert luna.loc[UID_A1, "n_annotations"] == 2 and luna.loc[UID_B, "n_annotations"] == 0
        assert luna.loc[UID_A2, "patient_id"] == "LIDC-IDRI-0001" and luna.loc[UID_A2, "scan_idx"] == 1
        assert bool(luna.loc[UID_A1, "direction_is_identity"]) and not bool(luna.loc[UID_A2, "direction_is_identity"])
        assert luna.loc[UID_A1, "mhd_rel_path"] == f"subset0/{UID_A1}.mhd"

    def test_summary(self, index_dir):
        s = json.loads((index_dir / "index_summary.json").read_text(encoding="utf-8"))
        lidc, luna = s["lidc"], s["luna16"]
        assert (lidc["n_patients"], lidc["n_ct_scans"], lidc["n_seg_files"]) == (3, 4, 10)
        assert lidc["patients_multi_scan"] == ["LIDC-IDRI-0001"]
        assert lidc["n_patients_without_seg"] == 1 and lidc["n_label_anomaly"] == 1
        assert lidc["readers_per_cluster"] == {"1": 3, "2": 2, "3": 1}
        assert (luna["n_scans"], luna["n_scans_with_annotations"], luna["n_annotations"]) == (3, 2, 3)
        assert luna["n_direction_not_identity"] == 1 and s["n_errors"] == 0

    def test_header_only(self, raw, monkeypatch, tmp_path):
        import pydicom

        calls = []
        real = pydicom.dcmread

        def spy(path, *a, **k):
            calls.append(k.get("stop_before_pixels", False))
            return real(path, *a, **k)

        monkeypatch.setattr(pydicom, "dcmread", spy)
        build_index.build_index(raw["lidc"], None, tmp_path, workers=1)
        assert calls and all(calls)
        assert len(calls) == 4 + 10  # 1 tệp/scan CT + mọi tệp SEG


# ---------------------------------------------------------------------------
# preprocess_lidc (§5.2)
# ---------------------------------------------------------------------------
class TestPreprocessLidc:
    def test_outputs_sorted_unique_and_two_scans(self, lidc_out):
        meta, hu, mk = read_patches(lidc_out)
        assert list(meta.columns[: len(preprocess_lidc.PATCH_COLUMNS)]) == preprocess_lidc.PATCH_COLUMNS
        assert meta["patch_id"].is_unique and meta["patch_id"].tolist() == sorted(meta["patch_id"])
        assert hu.shape == (len(meta), 64, 64) and hu.dtype == np.int16 and mk.dtype == np.uint8
        scans = {p.split("_")[1] for p in meta["patch_id"]}
        assert scans == {"s00", "s01"}  # cùng bệnh nhân 0001, hai scan → patch_id vẫn duy nhất
        assert set(meta["patient_id"]) == {"LIDC-IDRI-0001"}

    def test_consensus_mask_and_hu(self, raw, lidc_out):
        meta, hu, mk = read_patches(lidc_out)
        a1_all = meta[meta["series_uid"] == UID_A1]
        assert set(a1_all["nodule_idx"]) == {0, 2}  # Nodule 3 (< 3 mm) bị loại; nhãn dị biệt (~4 mm) được giữ
        a1 = a1_all[a1_all["nodule_idx"] == 0]
        cons = raw["cons_a1"]
        assert sorted(a1["z"]) == [z for z in range(Z) if cons[z].any()]
        assert (a1["n_readers"] == 3).all()
        vol = floor_hu(raw["vol_a1"])
        for i, row in a1.iterrows():
            exp_mask = crop_patch(cons[row.z], row.cy, row.cx, 64, 0)
            assert np.array_equal(mk[i], exp_mask)
            assert np.array_equal(hu[i], np.rint(crop_patch(vol[row.z], row.cy, row.cx, 64, -1000)).astype(np.int16))
            assert row.area_nodule_px == cons[row.z].sum() == row.area_union_px
        assert (hu >= -1000).all()

    def test_cluster_statuses(self, lidc_out):
        s = json.loads((lidc_out / "by_scan" / UID_A1 / "summary.json").read_text(encoding="utf-8"))
        status = {c["nodule_key"]: c["status"] for c in s["clusters"]}
        assert status == {"Nodule 1": "kept", "Nodule 3": "small", "Weird lesion": "kept"}
        assert [c["nodule_key"] for c in s["clusters"]] == ["Nodule 1", "Nodule 3", "Weird lesion"]
        n1 = next(c for c in s["clusters"] if c["nodule_key"] == "Nodule 1")
        assert n1["consensus_thresh"] == 2

    def test_include_other_nodules(self, raw, lidc_out):
        meta, _, mk = read_patches(lidc_out)
        a2n1 = meta[(meta["series_uid"] == UID_A2) & (meta["nodule_idx"] == 0)].set_index("z")
        for z in (4, 5):
            row = a2n1.loc[z]
            assert row.n_other_nodules == 1
            assert row.area_union_px == raw["cons_a2"][z].sum() + raw["a2_n2"][z].sum()
        assert a2n1.loc[3].n_other_nodules == 0

    def test_scan_without_seg_and_error_scan(self, lidc_out):
        assert (lidc_out / "by_scan" / UID_B / "_DONE").exists()
        assert not (lidc_out / "by_scan" / UID_B / "meta.csv").exists()
        assert not (lidc_out / "by_scan" / UID_C / "_DONE").exists()
        err = pd.read_csv(lidc_out / "errors.csv", dtype=str)
        assert list(err.columns) == ["series_uid", "patient_id", "stage", "error"]
        assert err[["series_uid", "patient_id", "stage"]].values.tolist() == [[UID_C, "LIDC-IDRI-0003", "read_ct"]]

    def test_resume_skips_done(self, raw, index_dir, lidc_out):
        stamp = (lidc_out / "by_scan" / UID_A1 / "meta.csv").stat().st_mtime_ns
        assert preprocess_lidc.main(["--raw-dir", str(raw["lidc"]), "--index-dir", str(index_dir),
                                     "--out-dir", str(lidc_out), "--config", str(DATA_CFG), "--workers", "1"]) == 0
        summary = json.loads((lidc_out / "preprocess_summary.json").read_text(encoding="utf-8"))
        assert summary["n_skipped_done"] == 3 and summary["n_failed"] == 1
        assert (lidc_out / "by_scan" / UID_A1 / "meta.csv").stat().st_mtime_ns == stamp

    def test_limit(self, raw, index_dir, tmp_path):
        out = tmp_path / "lidc"
        assert preprocess_lidc.main(["--raw-dir", str(raw["lidc"]), "--index-dir", str(index_dir),
                                     "--out-dir", str(out), "--config", str(DATA_CFG), "--limit", "1"]) == 0
        assert [d.name for d in (out / "by_scan").iterdir()] == [UID_A1]

    def test_consensus_threshold(self):
        assert [preprocess_lidc.consensus_threshold(r, 0.5) for r in (1, 2, 3, 4)] == [1, 1, 2, 2]
        assert preprocess_lidc.consensus_threshold(30, 0.1) == 3


# ---------------------------------------------------------------------------
# preprocess_luna16 (§5.3)
# ---------------------------------------------------------------------------
class TestPreprocessLuna16:
    def test_world_to_voxel_matches_sitk(self, tmp_path):
        import SimpleITK as sitk

        img = sitk.Image(10, 12, 7, sitk.sitkInt16)
        img.SetSpacing((0.7, 0.8, 2.5))
        img.SetOrigin((-10.0, 5.0, -100.0))
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        img.SetDirection(tuple(rot.ravel()))
        sitk.WriteImage(img, str(tmp_path / "r.mhd"))
        pts = np.random.default_rng(0).uniform(-50, 50, (20, 3))
        ref = np.array([img.TransformPhysicalPointToContinuousIndex(tuple(p)) for p in pts])
        assert np.allclose(preprocess_luna16.world_to_voxel_batch(img, pts), ref, atol=1e-9)
        geom = preprocess_luna16.MhdGeometry.from_file(tmp_path / "r.mhd")
        assert np.allclose(preprocess_luna16.world_to_voxel_batch(geom, pts), ref, atol=1e-6)
        assert np.allclose(geom.TransformPhysicalPointToContinuousIndex(tuple(pts[0])), ref[0], atol=1e-6)
        back = preprocess_luna16.voxel_to_world_batch(img, ref)
        assert np.allclose(back, pts, atol=1e-9)

    def test_match_report(self, luna_out):
        rep = pd.read_csv(luna_out / "match_report.csv", dtype={"series_uid": str})
        assert list(rep.columns) == preprocess_luna16.MATCH_COLUMNS
        got = {(r.series_uid, r.ann_idx): r.status for r in rep.itertuples()}
        assert got == {(UID_A1, 0): "matched", (UID_A1, 1): "unmatched", (UID_A2, 0): "matched"}
        assert rep[rep["status"] == "matched"]["dist_mm"].max() < 0.5

    def test_identity_grid_equals_lidc(self, lidc_out, luna_out):
        lm, lhu, lmk = read_patches(lidc_out)
        um, uhu, umk = read_patches(luna_out)
        a1_l = lm[(lm["series_uid"] == UID_A1) & (lm["nodule_idx"] == 0)].set_index("patch_id")
        a1_u = um[um["series_uid"] == UID_A1].set_index("patch_id")
        assert list(a1_u.index) == list(a1_l.index)
        li = {p: i for i, p in enumerate(lm["patch_id"])}
        for i, p in enumerate(um["patch_id"]):
            if p in a1_l.index:
                assert np.array_equal(uhu[i], lhu[li[p]]) and np.array_equal(umk[i], lmk[li[p]])
        assert (a1_u["diameter_mm"] == 6.0).all()
        assert (a1_u["ann_idx"] == 0).all()

    def test_flipped_direction_general_path(self, raw, luna_out):
        s = json.loads((luna_out / "by_scan" / UID_A2 / "summary.json").read_text(encoding="utf-8"))
        assert s["direction_is_identity"] is False
        assert all(c["n_general_frames"] > 0 for c in s["clusters"])
        um, _, umk = read_patches(luna_out)
        a2 = um[um["series_uid"] == UID_A2].set_index("z")
        cons = raw["cons_a2"]
        assert sorted(a2.index) == [z for z in range(Z) if cons[z].any()]
        for z, row in a2.iterrows():
            assert row.area_nodule_px == cons[z].sum()
            assert row.area_union_px == cons[z].sum()  # Nodule 2 không có trong annotations → không gộp

    def test_negative_scan_done(self, luna_out):
        assert (luna_out / "by_scan" / UID_B / "_DONE").exists()
        assert not (luna_out / "by_scan" / UID_B / "meta.csv").exists()
        assert pd.read_csv(luna_out / "errors.csv").empty

    def test_limit_counts_positive_scans(self, raw, index_dir):
        jobs = preprocess_luna16.build_jobs(index_dir, raw["luna"], raw["lidc"], Path("x"), load_config(DATA_CFG), 1)
        assert [j["series_uid"] for j in jobs] == [UID_A1]


# ---------------------------------------------------------------------------
# make_splits (§5.4)
# ---------------------------------------------------------------------------
def synthetic_patches(n_patients: int = 150, seed: int = 0, two_scans: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        pid = f"LIDC-IDRI-{p + 1:04d}"
        n_scans = 2 if (two_scans and p % 10 == 0) else 1
        for s in range(n_scans):
            for z in range(int(rng.integers(5, 40))):
                rows.append({
                    "patch_id": f"{pid}_s{s:02d}_n00_z{z:03d}", "patient_id": pid, "series_uid": f"{p}.{s}",
                    "diameter_mm": float(rng.uniform(3, 30)), "area_union_px": int(rng.integers(5, 900)),
                    "touches_border": bool(rng.random() < 0.05),
                })
    return pd.DataFrame(rows)


class TestMakeSplits:
    def test_patient_mode_exact_test_and_no_leak(self):
        df = synthetic_patches()
        out = make_splits.make_splits(df, "patient", 2024, 1200, 0.1)
        assert len(out["test"]) == 1200
        pid = dict(zip(df["patch_id"], df["patient_id"]))
        groups = {k: {pid[p] for p in out[k]} for k in ("train", "val", "test")}
        assert groups["train"].isdisjoint(groups["val"])
        assert groups["train"].isdisjoint(groups["test"])
        assert groups["val"].isdisjoint(groups["test"])
        assert groups == {k: set(out["patients"][k]) for k in groups}
        all_ids = out["train"] + out["val"] + out["test"] + out["dropped"]
        assert len(all_ids) == len(set(all_ids)) == len(df)
        assert out["stats"]["patient_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}
        for k in ("train", "val", "test", "dropped"):
            assert out[k] == sorted(out[k])

    def test_boundary_patient_dropped(self):
        df = synthetic_patches()
        out = make_splits.make_splits(df, "patient", 2024, 1200, 0.1)
        b = out["boundary_patient"]
        assert b is not None and out["dropped"]
        assert {p.split("_s")[0] for p in out["dropped"]} == {b}
        assert b in out["patients"]["test"]
        assert b not in out["patients"]["train"] + out["patients"]["val"]
        n_b = int((df["patient_id"] == b).sum())
        assert len(out["dropped"]) + sum(p.startswith(b + "_") for p in out["test"]) == n_b

    def test_val_ratio_by_patient(self):
        df = synthetic_patches()
        out = make_splits.make_splits(df, "patient", 2024, 1200, 0.1)
        n_rem = len(out["patients"]["train"]) + len(out["patients"]["val"])
        assert len(out["patients"]["val"]) == max(1, round(0.1 * n_rem))

    def test_deterministic_and_seed_sensitive(self):
        df = synthetic_patches()
        a = make_splits.make_splits(df, "patient", 2024, 1200, 0.1)
        b = make_splits.make_splits(df.sample(frac=1.0, random_state=3), "patient", 2024, 1200, 0.1)
        c = make_splits.make_splits(df, "patient", 7, 1200, 0.1)
        assert a["test"] == b["test"] and a["train"] == b["train"]
        assert a["test"] != c["test"]

    def test_two_scans_patch_ids_unique(self):
        df = synthetic_patches(two_scans=True)
        p = df[df["patient_id"] == "LIDC-IDRI-0001"]["patch_id"]
        assert {x.split("_")[1] for x in p} == {"s00", "s01"} and p.is_unique
        out = make_splits.make_splits(df, "patient", 2024, 1200, 0.1)
        split_of = {pid: k for k in ("train", "val", "test") for pid in out["patients"][k]}
        assert "LIDC-IDRI-0001" in split_of  # cả hai scan của bệnh nhân luôn ở cùng một tập

    def test_slice_mode(self):
        df = synthetic_patches()
        out = make_splits.make_splits(df, "slice", 2024, 1200, 0.1)
        assert len(out["test"]) == 1200 and out["dropped"] == []
        assert len(out["val"]) == max(1, round(0.1 * (len(df) - 1200)))
        assert len(set(out["train"] + out["val"] + out["test"])) == len(df)
        assert out["stats"]["patient_overlap"]["train_test"] > 0  # S1 cố ý rò rỉ bệnh nhân

    def test_errors(self):
        with pytest.raises(ValueError):
            make_splits.make_splits(synthetic_patches(10), "patient", 2024, 1200, 0.1)
        df = synthetic_patches()
        with pytest.raises(ValueError):
            make_splits.make_splits(pd.concat([df, df.iloc[:1]]), "patient", 2024, 1200, 0.1)
        with pytest.raises(ValueError):
            make_splits.make_splits(df, "random", 2024, 1200, 0.1)

    def test_verify_detects_leak(self):
        df = synthetic_patches()
        out = make_splits.make_splits(df, "patient", 2024, 1200, 0.1)
        bad = {**out, "patients": {**out["patients"], "val": out["patients"]["val"] + out["patients"]["train"][:1]}}
        with pytest.raises(AssertionError):
            make_splits.verify_splits(bad, df, 1200, "patient")

    def test_cli_writes_json(self, tmp_path):
        df = synthetic_patches()
        df.to_csv(tmp_path / "patches.csv", index=False)
        out_json = tmp_path / "splits" / "lidc.json"
        assert make_splits.main(["--index", str(tmp_path / "patches.csv"), "--out", str(out_json),
                                 "--config", str(DATA_CFG)]) == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["mode"] == "patient" and payload["seed"] == 2024 and len(payload["test"]) == 1200
        assert set(payload["stats"]) >= {"train", "val", "test", "dropped", "patient_overlap"}
        assert make_splits.main(["--index", str(tmp_path / "patches.csv"), "--out", str(out_json),
                                 "--config", str(DATA_CFG), "--mode", "slice"]) == 0
        assert json.loads(out_json.read_text(encoding="utf-8"))["mode"] == "slice"


# ---------------------------------------------------------------------------
# pack_h5 (§5.5)
# ---------------------------------------------------------------------------
@pytest.fixture
def interim(tmp_path):
    """Interim tổng hợp 40 patch, 8 bệnh nhân, có patch mask rỗng."""
    rng = np.random.default_rng(5)
    n = 40
    hu = rng.integers(-1000, 1200, (n, 64, 64)).astype(np.int16)
    masks = np.zeros((n, 64, 64), np.uint8)
    for i in range(n - 1):
        cy, cx = rng.integers(10, 54, 2)
        masks[i, cy - 4: cy + 4, cx - 3: cx + 5] = 1
    rows = [{"patch_id": f"LIDC-IDRI-{i // 5 + 1:04d}_s00_n00_z{i:03d}", "patient_id": f"LIDC-IDRI-{i // 5 + 1:04d}",
             "series_uid": str(i // 5), "diameter_mm": float(i), "area_union_px": int(masks[i].sum()),
             "touches_border": i % 7 == 0} for i in range(n)]
    order = rng.permutation(n)  # patches.csv không cần theo thứ tự
    d = tmp_path / "interim" / "luna16"
    d.mkdir(parents=True)
    np.save(d / "patches_hu.npy", hu[order])
    np.save(d / "masks.npy", masks[order])
    pd.DataFrame([rows[i] for i in order]).to_csv(d / "patches.csv", index=False)
    return d


def write_splits(path: Path, interim: Path, test_patches: int = 10, empty_val: bool = False) -> dict:
    df = pd.read_csv(interim / "patches.csv", dtype={"patch_id": str, "patient_id": str})
    out = make_splits.make_splits(df, "patient", 2024, test_patches, 0.1)
    if empty_val:
        out["train"] = sorted(out["train"] + out["val"])
        out["val"] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


class TestPackH5:
    def test_pack_contents_and_manifest(self, interim, tmp_path):
        import h5py

        splits = write_splits(tmp_path / "splits" / "luna16.json", interim)
        out_dir = tmp_path / "processed_tmp" / "luna16"
        cfg = load_config(DATA_CFG)
        manifest = pack_h5.pack(interim, tmp_path / "splits" / "luna16.json", out_dir, cfg)
        names = {"train.h5", "val.h5", "test.h5", "patches.csv", "luna16.json"}
        assert set(manifest["files"]) == names
        assert sorted(p.name for p in out_dir.iterdir()) == sorted(names | {"manifest.json"})
        for name, info in manifest["files"].items():
            assert info["sha256"] == hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
            assert info["bytes"] == (out_dir / name).stat().st_size
        assert pack_h5.verify_manifest(out_dir) == {}

        meta = pd.read_csv(interim / "patches.csv", dtype={"patch_id": str}).set_index("patch_id")
        hu_all, mk_all = np.load(interim / "patches_hu.npy"), np.load(interim / "masks.npy")
        row = {p: i for i, p in enumerate(meta.index)}
        lo, hi = cfg["common"]["hu_window"]
        for split in ("train", "val", "test"):
            with h5py.File(out_dir / f"{split}.h5", "r") as h5:
                ids = [x.decode() if isinstance(x, bytes) else x for x in h5["patch_id"][...]]
                assert ids == sorted(splits[split]) and manifest["files"][f"{split}.h5"]["n"] == len(ids)
                img, msk, dist = h5["image"][...], h5["mask"][...], h5["dist"][...]
                assert img.dtype == np.float32 and msk.dtype == np.uint8 and dist.dtype == np.float32
                assert img.shape == msk.shape == dist.shape == (len(ids), 1, 64, 64)
                for name in ("image", "mask", "dist"):
                    assert (h5[name].compression, h5[name].compression_opts) == ("gzip", 4)
                assert h5.attrs["dataset"] == "luna16" and h5.attrs["split"] == split
                assert h5.attrs["split_mode"] == "patient" and int(h5.attrs["split_seed"]) == 2024
                assert set(h5.attrs) >= {"data_config_json", "created", "git_commit"}
                for j, p in enumerate(ids):
                    i = row[p]
                    assert np.allclose(img[j, 0], window_hu(hu_all[i], lo, hi))
                    assert np.array_equal(msk[j, 0], mk_all[i])
                    assert np.array_equal(dist[j, 0], edt_from_mask(mk_all[i]))
                    assert h5["area_px"][j] == meta.loc[p, "area_union_px"]
                    assert h5["diameter_mm"][j] == pytest.approx(meta.loc[p, "diameter_mm"])
                    assert h5["touches_border"][j] == int(meta.loc[p, "touches_border"])
                    assert h5["patient_id"][j].decode() == meta.loc[p, "patient_id"]
                assert ((img >= 0) & (img <= 1)).all() and (dist[msk > 0] == 0).all()

    def test_copy_to_drive_and_verify(self, interim, tmp_path):
        write_splits(tmp_path / "splits" / "luna16.json", interim)
        drive = tmp_path / "drive" / "data" / "processed" / "luna16"
        assert pack_h5.main(["--interim-dir", str(interim), "--splits", str(tmp_path / "splits" / "luna16.json"),
                             "--out-dir", str(tmp_path / "local"), "--config", str(DATA_CFG),
                             "--copy-to-drive-dir", str(drive)]) == 0
        assert sorted(p.name for p in drive.iterdir()) == sorted(
            ["train.h5", "val.h5", "test.h5", "patches.csv", "luna16.json", "manifest.json"])
        assert pack_h5.verify_manifest(drive) == {}
        assert (drive / "manifest.json").read_bytes() == (tmp_path / "local" / "manifest.json").read_bytes()
        with open(drive / "test.h5", "r+b") as fh:  # làm hỏng 1 byte
            fh.seek(-1, 2)
            last = fh.read(1)
            fh.seek(-1, 2)
            fh.write(bytes([last[0] ^ 0xFF]))
        assert pack_h5.verify_manifest(drive) == {"test.h5": "SHA-256 không khớp"}
        (drive / "val.h5").unlink()
        assert pack_h5.verify_manifest(drive)["val.h5"] == "thiếu tệp"

    def test_empty_split(self, interim, tmp_path):
        import h5py

        write_splits(tmp_path / "s" / "lidc.json", interim, empty_val=True)
        manifest = pack_h5.pack(interim, tmp_path / "s" / "lidc.json", tmp_path / "out", load_config(DATA_CFG))
        assert manifest["files"]["val.h5"]["n"] == 0
        with h5py.File(tmp_path / "out" / "val.h5", "r") as h5:
            assert h5["image"].shape == (0, 1, 64, 64)

    def test_unknown_patch_id(self, interim, tmp_path):
        out = write_splits(tmp_path / "s" / "luna16.json", interim)
        out["test"] = out["test"] + ["không-tồn-tại"]
        (tmp_path / "s" / "luna16.json").write_text(json.dumps(out), encoding="utf-8")
        with pytest.raises(KeyError):
            pack_h5.pack(interim, tmp_path / "s" / "luna16.json", tmp_path / "out", load_config(DATA_CFG))


# ---------------------------------------------------------------------------
# D4 (numpy) & H5SegDataset (torch)
# ---------------------------------------------------------------------------
def asymmetric_mask() -> np.ndarray:
    m = np.zeros((1, 64, 64), np.uint8)
    m[0, 10:30, 12:18] = 1
    m[0, 25:30, 12:40] = 1
    return m


class TestD4:
    def test_preserves_edt(self):
        m = asymmetric_mask()
        d = edt_from_mask(m[0])[None]
        outs = []
        for op in range(8):
            tm, td = d4_transform(m, op), d4_transform(d, op)
            assert tm.shape == m.shape and tm.flags["C_CONTIGUOUS"]
            assert np.array_equal(td[0], edt_from_mask(tm[0]))
            outs.append(tm.tobytes())
        assert len(set(outs)) == 8  # 8 phép phân biệt

    def test_identity_and_bad_op(self):
        m = asymmetric_mask()
        assert np.array_equal(d4_transform(m, 0), m)
        with pytest.raises(ValueError):
            d4_transform(m, 8)


@pytest.fixture
def packed(interim, tmp_path):
    write_splits(tmp_path / "splits" / "luna16.json", interim)
    root = tmp_path / "processed"
    pack_h5.pack(interim, tmp_path / "splits" / "luna16.json", root / "luna16", load_config(DATA_CFG))
    return root


class TestH5SegDataset:
    def test_items_dtype_shape(self, packed):
        torch = pytest.importorskip("torch")
        from src.data.dataset import H5SegDataset

        ds = H5SegDataset(packed / "luna16" / "train.h5")
        item = ds[0]
        for key in ("image", "mask", "dist"):
            assert isinstance(item[key], torch.Tensor)
            assert item[key].dtype == torch.float32 and tuple(item[key].shape) == (1, 64, 64)
        assert isinstance(item["patch_id"], str) and isinstance(item["patient_id"], str)
        assert ds.patch_id == sorted(ds.patch_id)

    def test_augment_keeps_dist_equal_edt(self, packed):
        pytest.importorskip("torch")
        from src.data.dataset import H5SegDataset

        ds = H5SegDataset(packed / "luna16" / "train.h5", augment=True, seed=3)
        changed = 0
        for i in range(len(ds)):
            item = ds[i]
            m, d = item["mask"].numpy(), item["dist"].numpy()
            assert np.array_equal(d[0], edt_from_mask(m[0]))
            changed += int(not np.array_equal(m, ds.mask[i]))
        assert changed > 0
        ops0 = [ds.op_for(i) for i in range(len(ds))]
        assert ops0 == [ds.op_for(i) for i in range(len(ds))]
        ds.set_epoch(1)
        assert ops0 != [ds.op_for(i) for i in range(len(ds))]
        plain = H5SegDataset(packed / "luna16" / "train.h5", augment=False)
        assert all(plain.op_for(i) == 0 for i in range(len(plain)))

    def test_make_loaders(self, packed):
        torch = pytest.importorskip("torch")
        from src.data.dataset import make_loaders

        cfg = load_config(ROOT / "configs" / "base.yaml")
        cfg["train"]["batch_size"] = 4
        cfg["data"]["num_workers"] = 0
        loaders = make_loaders(cfg, packed, "luna16")
        assert set(loaders) == {"train", "val", "test"}
        batch = next(iter(loaders["test"]))
        assert batch["image"].shape[1:] == (1, 64, 64) and batch["image"].dtype == torch.float32
        assert loaders["train"].drop_last is False
