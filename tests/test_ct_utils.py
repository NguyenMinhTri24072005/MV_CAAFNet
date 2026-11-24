"""Kiểm thử `src/data/ct_utils.py` — SPECS §5.1, Phụ lục C (không cần torch, trừ 1 test so sánh torch)."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ct_utils  # noqa: E402
from src.data.ct_utils import (  # noqa: E402
    ERROR_COLUMNS,
    StageError,
    compute_nodule_diameters,
    compute_patch_morphology,
    crop_patch,
    floor_hu,
    resample_inplane,
    run_parallel,
    save_csv_atomic,
    save_npy_atomic,
    save_npz_atomic,
    window_hu,
)


# ---------------------------------------------------------------------------
# Hàm cấp module cho run_parallel (pickle được khi chạy ProcessPoolExecutor)
# ---------------------------------------------------------------------------
def touch_job(job: dict) -> dict:
    if job.get("fail") == "stage":
        raise StageError("read_ct", "lát trùng z")
    if job.get("fail") == "generic":
        raise ValueError("hỏng")
    Path(job["done_marker"]).write_text("", encoding="utf-8")
    return {"series_uid": job["series_uid"]}


def make_jobs(tmp_path: Path, n: int, fails: dict[int, str] | None = None) -> list[dict]:
    jobs = []
    for i in range(n):
        d = tmp_path / "by_scan" / f"s{i}"
        d.mkdir(parents=True, exist_ok=True)
        job = {"series_uid": f"s{i}", "patient_id": f"P{i:02d}", "done_marker": str(d / "_DONE")}
        if fails and i in fails:
            job["fail"] = fails[i]
        jobs.append(job)
    return jobs


def read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


# ---------------------------------------------------------------------------
# floor_hu, window_hu
# ---------------------------------------------------------------------------
class TestHU:
    def test_floor_hu_does_not_mutate(self):
        vol = np.array([[-3024, -2048, -1000, 50, 3000]], dtype=np.int16)
        before = vol.copy()
        out = floor_hu(vol)
        assert np.array_equal(vol, before)
        assert out.dtype == np.float32
        assert out.tolist() == [[-1000, -1000, -1000, 50, 3000]]

    def test_floor_hu_returns_new_array_for_float32(self):
        vol = np.full((2, 3, 3), -2000.0, np.float32)
        out = floor_hu(vol, -1000)
        assert not np.shares_memory(out, vol)
        assert (vol == -2000).all() and (out == -1000).all()

    def test_window_hu_maps_air_to_zero(self):
        x = np.array([-3024, -1000, -300, 400, 2000], dtype=np.int16)
        before = x.copy()
        y = window_hu(x, -1000, 400)
        assert np.array_equal(x, before)
        assert y.dtype == np.float32
        assert y[0] == 0.0 and y[1] == 0.0 and y[3] == 1.0 and y[4] == 1.0
        assert y[2] == pytest.approx(0.5)
        assert ((y >= 0) & (y <= 1)).all()

    def test_window_hu_alternative_window(self):
        y = window_hu(np.array([-1000.0, -425.0, 150.0]), -1000, 150)
        assert y.tolist() == pytest.approx([0.0, 0.5, 1.0])

    def test_window_hu_rejects_bad_window(self):
        with pytest.raises(ValueError):
            window_hu(np.zeros(3), 400, -1000)

    def test_window_hu_torch_matches_numpy(self):
        torch = pytest.importorskip("torch")
        x = np.random.default_rng(0).integers(-3000, 3000, (4, 64, 64)).astype(np.int16)
        t = ct_utils.window_hu_torch(torch.from_numpy(x), -1000, 400).numpy()
        assert np.allclose(t, window_hu(x, -1000, 400), atol=1e-6)


# ---------------------------------------------------------------------------
# crop_patch
# ---------------------------------------------------------------------------
class TestCropPatch:
    def setup_method(self):
        self.img = np.arange(100 * 120, dtype=np.int32).reshape(100, 120)

    def test_interior_crop_keeps_center(self):
        p = crop_patch(self.img, 50, 60, 64, pad_value=-1000)
        assert p.shape == (64, 64) and p.dtype == self.img.dtype
        assert p[32, 32] == self.img[50, 60]
        assert np.array_equal(p, self.img[18:82, 28:92])

    def test_corner_padding_image_and_mask(self):
        p = crop_patch(self.img, 3, 5, 64, pad_value=-1000)
        assert p[32, 32] == self.img[3, 5]
        assert (p[:29, :] == -1000).all() and (p[:, :27] == -1000).all()
        assert np.array_equal(p[29:, 27:], self.img[0:35, 0:37])
        m = np.ones((100, 120), np.uint8)
        pm = crop_patch(m, 99, 119, 64, pad_value=0)
        assert pm.dtype == np.uint8
        assert pm[32, 32] == 1 and pm.sum() == 33 * 33 and (pm[33:, :] == 0).all()

    def test_fully_outside_is_all_pad(self):
        p = crop_patch(self.img, -200, -200, 16, pad_value=-1000)
        assert (p == -1000).all()

    def test_no_mutation_and_odd_size(self):
        before = self.img.copy()
        p = crop_patch(self.img, 10, 10, 7, pad_value=0)
        p[...] = -1
        assert np.array_equal(self.img, before)
        assert crop_patch(self.img, 10, 10, 7)[3, 3] == self.img[10, 10]

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            crop_patch(np.zeros((2, 3, 3)), 1, 1, 4)


# ---------------------------------------------------------------------------
# resample_inplane
# ---------------------------------------------------------------------------
class TestResample:
    def test_mask_stays_binary_and_shape(self):
        mask = np.zeros((5, 100, 80), np.uint8)
        mask[2, 40:60, 30:50] = 1
        out, (sy, sx) = resample_inplane(mask, (0.7, 0.6), 1.0, order=0)
        assert out.dtype == np.uint8 and out.shape == (5, 70, 48)
        assert set(np.unique(out)) <= {0, 1} and out.sum() > 0
        assert sy == pytest.approx(0.7 * 100 / 70) and sx == pytest.approx(0.6 * 80 / 48)
        assert out[[0, 1, 3, 4]].sum() == 0  # trục Z giữ nguyên

    def test_image_linear_preserves_constant(self):
        vol = np.full((3, 40, 40), -500.0, np.float32)
        out, _ = resample_inplane(vol, (0.5, 0.5), 1.0, order=1)
        assert out.shape == (3, 20, 20)
        assert np.allclose(out, -500.0)

    def test_identity_returns_copy(self):
        vol = np.random.default_rng(0).random((2, 8, 8)).astype(np.float32)
        out, sp = resample_inplane(vol, (1.0, 1.0), 1.0)
        assert np.array_equal(out, vol) and not np.shares_memory(out, vol) and sp == (1.0, 1.0)

    def test_rejects_non_3d(self):
        with pytest.raises(ValueError):
            resample_inplane(np.zeros((8, 8)), (1.0, 1.0), 1.0)


# ---------------------------------------------------------------------------
# Đường kính & hình thái
# ---------------------------------------------------------------------------
def ball(shape, center_mm, radius_mm, spacing_zyx):
    zz, yy, xx = np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, spacing_zyx)], indexing="ij")
    cz, cy, cx = center_mm
    return (((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= radius_mm ** 2).astype(np.uint8)


class TestDiameters:
    @pytest.mark.parametrize("spacing", [(0.5, 0.5, 0.5), (1.25, 0.6, 0.6), (2.5, 0.7, 0.7)])
    def test_sphere_matches_theory(self, spacing):
        radius = 6.0
        shape = tuple(int(40 / s) for s in spacing)
        m = ball(shape, (20.0, 20.0, 20.0), radius, spacing)
        d3, d2 = compute_nodule_diameters(m, spacing)
        assert d3 == pytest.approx(2 * radius, rel=0.04)
        assert d2 == pytest.approx(2 * radius, rel=0.04)

    def test_known_box(self):
        m = np.zeros((4, 10, 10), np.uint8)
        m[1:3, 2:6, 2:6] = 1  # 2 lát × 16 pixel
        d3, d2 = compute_nodule_diameters(m, (2.0, 0.5, 0.5))
        vol = 32 * 2.0 * 0.25
        assert d3 == pytest.approx((6 * vol / math.pi) ** (1 / 3))
        assert d2 == pytest.approx(2 * math.sqrt(16 * 0.25 / math.pi))

    def test_empty_mask(self):
        assert compute_nodule_diameters(np.zeros((3, 4, 4)), (1, 1, 1)) == (0.0, 0.0)


class TestMorphology:
    def test_stats_inside_mask(self):
        hu = np.random.default_rng(1).normal(0, 100, (64, 64))
        m = np.zeros((64, 64), np.uint8)
        m[20:30, 20:30] = 1
        out = compute_patch_morphology(hu, m)
        vals = hu[m > 0]
        assert out["touches_border"] is False
        assert out["mean_hu"] == pytest.approx(vals.mean())
        assert out["min_hu"] == pytest.approx(vals.min())
        assert out["std_hu"] == pytest.approx(vals.std())
        assert out["solidity"] == pytest.approx(1.0)

    @pytest.mark.parametrize("edge", ["top", "bottom", "left", "right"])
    def test_touches_border(self, edge):
        m = np.zeros((64, 64), np.uint8)
        sl = {"top": (0, 30), "bottom": (63, 30), "left": (30, 0), "right": (30, 63)}[edge]
        m[sl] = 1
        assert compute_patch_morphology(np.zeros((64, 64)), m)["touches_border"] is True

    def test_solidity_l_shape_and_tiny(self):
        m = np.zeros((64, 64), np.uint8)
        m[10:30, 10:15] = 1
        m[25:30, 10:30] = 1
        s = compute_patch_morphology(np.zeros((64, 64)), m)["solidity"]
        assert 0.3 < s < 0.9
        tiny = np.zeros((64, 64), np.uint8)
        tiny[5, 5:8] = 1
        assert math.isnan(compute_patch_morphology(np.zeros((64, 64)), tiny)["solidity"])

    def test_empty_mask_gives_nan(self):
        out = compute_patch_morphology(np.zeros((64, 64)), np.zeros((64, 64), np.uint8))
        assert out["touches_border"] is False
        assert all(math.isnan(out[k]) for k in ("mean_hu", "min_hu", "std_hu", "solidity"))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            compute_patch_morphology(np.zeros((64, 64)), np.zeros((32, 32)))


# ---------------------------------------------------------------------------
# Ghi tệp atomic
# ---------------------------------------------------------------------------
class TestAtomicSave:
    def test_npy_roundtrip_no_tmp(self, tmp_path):
        arr = np.arange(12, dtype=np.int16).reshape(3, 4)
        p = save_npy_atomic(tmp_path / "sub" / "a.npy", arr)
        assert np.array_equal(np.load(p), arr)
        assert sorted(x.name for x in p.parent.iterdir()) == ["a.npy"]

    def test_npz_keeps_exact_name(self, tmp_path):
        a, b = np.ones((2, 2), np.uint8), np.zeros(3, np.int16)
        p = save_npz_atomic(tmp_path / "patches.npz", patches_hu=b, masks=a)
        with np.load(p) as z:
            assert np.array_equal(z["masks"], a) and np.array_equal(z["patches_hu"], b)
        assert sorted(x.name for x in tmp_path.iterdir()) == ["patches.npz"]

    def test_failure_keeps_old_file(self, tmp_path, monkeypatch):
        p = save_npy_atomic(tmp_path / "a.npy", np.array([1, 2, 3]))

        def boom(*_a, **_k):
            raise RuntimeError("đĩa đầy")

        monkeypatch.setattr(ct_utils.np, "save", boom)
        with pytest.raises(RuntimeError):
            save_npy_atomic(p, np.array([9]))
        monkeypatch.undo()
        assert np.load(p).tolist() == [1, 2, 3]
        assert sorted(x.name for x in tmp_path.iterdir()) == ["a.npy"]

    def test_csv_header_only_when_empty(self, tmp_path):
        p = save_csv_atomic(tmp_path / "errors.csv", [], columns=ERROR_COLUMNS)
        assert p.read_text(encoding="utf-8").strip() == ",".join(ERROR_COLUMNS)


# ---------------------------------------------------------------------------
# run_parallel
# ---------------------------------------------------------------------------
class TestRunParallel:
    def test_skips_done_and_records_errors(self, tmp_path):
        jobs = make_jobs(tmp_path, 5, fails={1: "stage", 3: "generic"})
        Path(jobs[4]["done_marker"]).write_text("", encoding="utf-8")  # đã xong từ lượt trước
        errors_csv = tmp_path / "errors.csv"
        out = run_parallel(touch_job, jobs, 1, errors_csv)
        assert (out["n_jobs"], out["n_skipped"], out["n_done"], out["n_failed"]) == (5, 1, 2, 2)
        cols, rows = read_csv_rows(errors_csv)
        assert cols == ["series_uid", "patient_id", "stage", "error"]
        by_uid = {r["series_uid"]: r for r in rows}
        assert by_uid["s1"]["stage"] == "read_ct" and "lát trùng z" in by_uid["s1"]["error"]
        assert by_uid["s3"]["stage"] == "process" and by_uid["s3"]["patient_id"] == "P03"
        assert not Path(jobs[1]["done_marker"]).exists()

    def test_resume_only_runs_unfinished(self, tmp_path):
        jobs = make_jobs(tmp_path, 4, fails={2: "stage"})
        run_parallel(touch_job, jobs, 1, tmp_path / "errors.csv")
        jobs[2].pop("fail")
        out = run_parallel(touch_job, jobs, 1, tmp_path / "errors.csv")
        assert (out["n_skipped"], out["n_done"], out["n_failed"]) == (3, 1, 0)
        assert read_csv_rows(tmp_path / "errors.csv")[1] == []

    def test_process_pool(self, tmp_path):
        jobs = make_jobs(tmp_path, 4, fails={0: "stage"})
        out = run_parallel(touch_job, jobs, 2, tmp_path / "errors.csv")
        assert (out["n_done"], out["n_failed"]) == (3, 1)
        assert sorted(r["series_uid"] for r in out["results"]) == ["s1", "s2", "s3"]
        assert all(Path(j["done_marker"]).exists() for j in jobs[1:])
