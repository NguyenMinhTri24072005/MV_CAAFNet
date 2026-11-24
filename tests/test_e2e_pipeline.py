"""Kiểm thử tích hợp đầu-cuối: make_splits → pack_h5 → train → resume → evaluate → stats_test → report → visualize.

Dữ liệu tổng hợp nhỏ trong `tmp_path` (12 bệnh nhân × 4 patch 64×64), không đụng `data/raw/`.
Mọi bước gọi đúng `main(argv)` của script như notebook gọi `python -m src.<module>`. Phần train /
evaluate / gradcam cần torch (chạy CPU: `--epochs 2 --batch-size 4 --max-batches 2`).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import make_splits, pack_h5  # noqa: E402
from src.data.compute_edt import edt_from_mask  # noqa: E402
from src.data.preprocess_lidc import PATCH_COLUMNS  # noqa: E402
from src.utils import load_config, save_yaml  # noqa: E402

N_PATIENTS, PER_PATIENT, TEST_PATCHES = 12, 4, 4
RUNS = {"caaf": "caaf_hausdorff", "pcam": "pcam_hausdorff"}
TRAIN_COLUMNS = [
    "epoch", "lr", "train_loss", "train_dice_loss", "train_boundary_loss", "val_loss", "val_dice",
    "train_mean_w1", "train_mean_w2", "d_pg", "d_gp", "empty_pred_frac", "seconds",
]


# ---------------------------------------------------------------------------
# Chuỗi dữ liệu (không cần torch)
# ---------------------------------------------------------------------------
def write_interim(d: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = N_PATIENTS * PER_PATIENT
    yy, xx = np.mgrid[:64, :64]
    hu = rng.normal(-800, 40, (n, 64, 64))
    masks = np.zeros((n, 64, 64), np.uint8)
    rows = []
    for i in range(n):
        p, z = divmod(i, PER_PATIENT)
        r = int(rng.integers(4, 11))
        cy, cx = 32 + rng.integers(-5, 6, 2)
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        masks[i][disk] = 1
        hu[i][disk] = rng.normal(40, 25, int(disk.sum()))
        pid = f"LIDC-IDRI-{p + 1:04d}"
        rows.append({
            "patch_id": f"{pid}_s00_n00_z{z:03d}", "patient_id": pid, "series_uid": f"1.2.826.0.{p}", "scan_idx": 0,
            "nodule_idx": 0, "z": z, "cx": int(cx), "cy": int(cy), "n_readers": 4, "diameter_mm": 2 * r * 0.7,
            "diameter_2d_mm": 2 * r * 0.7, "area_nodule_px": int(disk.sum()), "area_union_px": int(disk.sum()),
            "n_other_nodules": 0, "touches_border": False, "spacing_x": 0.7, "spacing_y": 0.7, "spacing_z": 2.5,
            "mean_hu": float(hu[i][disk].mean()), "min_hu": float(hu[i][disk].min()), "std_hu": float(hu[i][disk].std()),
            "solidity": 1.0,
        })
    d.mkdir(parents=True)
    np.save(d / "patches_hu.npy", np.clip(np.rint(hu), -1000, 3000).astype(np.int16))
    np.save(d / "masks.npy", masks)
    pd.DataFrame(rows, columns=PATCH_COLUMNS).to_csv(d / "patches.csv", index=False)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    data_cfg = load_config(ROOT / "configs" / "data.yaml")
    data_cfg["split"]["test_patches"] = TEST_PATCHES
    cfg_path = save_yaml(data_cfg, root / "data_small.yaml")
    interim = root / "interim" / "luna16"
    write_interim(interim)
    splits = root / "splits" / "luna16.json"
    assert make_splits.main(["--index", str(interim / "patches.csv"), "--out", str(splits), "--config", str(cfg_path),
                             "--mode", "patient"]) == 0
    processed = root / "processed"
    drive = root / "drive" / "data" / "processed" / "luna16"
    assert pack_h5.main(["--interim-dir", str(interim), "--splits", str(splits), "--out-dir", str(processed / "luna16"),
                         "--config", str(cfg_path), "--copy-to-drive-dir", str(drive)]) == 0
    return {"root": root, "interim": interim, "splits": splits, "processed": processed, "drive": drive,
            "exp": root / "experiments", "reports": root / "reports"}


class TestDataChain:
    def test_make_splits_patient_no_leak(self, pipeline):
        s = json.loads(pipeline["splits"].read_text(encoding="utf-8"))
        assert s["mode"] == "patient" and len(s["test"]) == TEST_PATCHES
        pid = {k: {p.split("_s")[0] for p in s[k]} for k in ("train", "val", "test")}
        assert pid["train"].isdisjoint(pid["val"]) and pid["train"].isdisjoint(pid["test"]) and pid["val"].isdisjoint(pid["test"])
        assert s["stats"]["patient_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}
        total = sum(len(s[k]) for k in ("train", "val", "test", "dropped"))
        assert total == N_PATIENTS * PER_PATIENT and len(s["val"]) > 0

    def test_pack_and_copy_to_drive(self, pipeline):
        import h5py

        for d in (pipeline["processed"] / "luna16", pipeline["drive"]):
            names = sorted(p.name for p in d.iterdir())
            assert names == sorted(["train.h5", "val.h5", "test.h5", "patches.csv", "luna16.json", "manifest.json"])
            assert pack_h5.verify_manifest(d) == {}
        s = json.loads(pipeline["splits"].read_text(encoding="utf-8"))
        for split in ("train", "val", "test"):
            with h5py.File(pipeline["drive"] / f"{split}.h5", "r") as h5:
                assert h5["image"].shape == (len(s[split]), 1, 64, 64)
                m, dist = h5["mask"][...], h5["dist"][...]
                assert all(np.array_equal(dist[i, 0], edt_from_mask(m[i, 0])) for i in range(len(m)))


# ---------------------------------------------------------------------------
# Chuỗi huấn luyện (torch, CPU)
# ---------------------------------------------------------------------------
def train_argv(p: dict, key: str, *extra: str, epochs: int = 2) -> list[str]:
    cfg = RUNS[key]
    return ["--config", str(ROOT / "configs" / f"{cfg}.yaml"), "--dataset", "luna16", "--data-dir", str(p["processed"]),
            "--out-dir", str(p["exp"]), "--run-name", f"{cfg}_s42", "--epochs", str(epochs), "--batch-size", "4",
            "--max-batches", "2", "--set", "data.num_workers=0", *extra]


def run_dir(p: dict, key: str) -> Path:
    return p["exp"] / "luna16" / f"{RUNS[key]}_s42"


def read_log(rd: Path) -> list[dict]:
    with open(rd / "train_log.csv", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def trained(pipeline):
    pytest.importorskip("torch")
    from src import train

    for key in RUNS:
        assert train.main(train_argv(pipeline, key)) == 0
    return pipeline


class TestTrainChain:
    def test_run_outputs(self, trained):
        import torch

        for key in RUNS:
            rd = run_dir(trained, key)
            for name in ("config_resolved.yaml", "overrides.json", "commit.txt", "env.json", "env_history.jsonl",
                         "train_log.csv", "train.log", "last.pt", "best.pt"):
                assert (rd / name).is_file(), f"{key}: thiếu {name}"
            cfg = load_config(rd / "config_resolved.yaml")
            assert cfg["dataset"] == "luna16" and cfg["train"]["epochs"] == 2 and cfg["train"]["batch_size"] == 4
            assert cfg["data"]["num_workers"] == 0
            ov = json.loads((rd / "overrides.json").read_text(encoding="utf-8"))
            assert ov["run_name"] == f"{RUNS[key]}_s42" and ov["dataset"] == "luna16"
            log = read_log(rd)
            assert list(log[0]) == TRAIN_COLUMNS
            assert [int(r["epoch"]) for r in log] == [1, 2]
            assert float(log[0]["lr"]) == pytest.approx(1e-4)
            assert all(np.isfinite(float(r["train_loss"])) and np.isfinite(float(r["val_dice"])) for r in log)
            assert np.isfinite(float(log[0]["d_pg"]))  # loss Hausdorff có log d_pg
            last = torch.load(rd / "last.pt", map_location="cpu", weights_only=False)
            assert {"epoch", "model", "optimizer", "scheduler", "scaler", "best_val_dice", "rng", "cfg"} <= set(last)
            assert last["epoch"] == 2 and {"python", "numpy", "torch", "loader"} <= set(last["rng"])
            best = torch.load(rd / "best.pt", map_location="cpu", weights_only=False)
            assert set(best) == {"epoch", "model", "best_val_dice", "cfg"}
            assert not {"optimizer", "scaler", "rng", "scheduler"} & set(best)
        w_caaf = float(read_log(run_dir(trained, "caaf"))[0]["train_mean_w1"])
        w_pcam = float(read_log(run_dir(trained, "pcam"))[0]["train_mean_w1"])
        assert 0 < w_caaf < 1 and w_pcam == pytest.approx(0.5)

    def test_resume_truncates_and_continues(self, trained):
        from src import train

        rd = run_dir(trained, "caaf")
        with open(rd / "train_log.csv", "a", encoding="utf-8", newline="") as fh:
            fh.write("3,0.5,9,9,9,9,0,0,0,0,0,0,0\n")  # dòng rác giả lập phiên bị ngắt
        assert train.main(train_argv(trained, "caaf", "--resume", epochs=3)) == 0
        log = read_log(rd)
        assert [int(r["epoch"]) for r in log] == [1, 2, 3]
        assert float(log[2]["train_loss"]) != 9.0
        events = [json.loads(line)["event"] for line in (rd / "env_history.jsonl").read_text(encoding="utf-8").splitlines()]
        assert events == ["start", "resume"]
        assert load_config(rd / "config_resolved.yaml")["train"]["epochs"] == 3
        import torch

        assert torch.load(rd / "last.pt", map_location="cpu", weights_only=False)["epoch"] == 3
        assert train.main(train_argv(trained, "caaf", "--resume", epochs=3)) == 0  # đã đủ epoch → thoát sớm
        assert [int(r["epoch"]) for r in read_log(rd)] == [1, 2, 3]

    def test_resume_rejects_other_config_changes(self, trained):
        from src import train

        with pytest.raises(SystemExit):
            train.main(train_argv(trained, "pcam", "--resume", "--lr", "0.001"))

    def test_refuses_to_overwrite_without_resume(self, trained):
        from src import train

        rd = run_dir(trained, "pcam")
        before = (rd / "last.pt").stat().st_mtime_ns
        with pytest.raises(SystemExit):
            train.main(train_argv(trained, "pcam"))
        assert (rd / "last.pt").stat().st_mtime_ns == before

    def test_overfit_does_not_touch_run_dir(self, trained):
        from src import train

        rd = run_dir(trained, "pcam")
        snapshot = {p.name: p.stat().st_mtime_ns for p in rd.iterdir()}
        assert train.main(train_argv(trained, "pcam", "--overfit-steps", "4")) == 0
        assert {p.name: p.stat().st_mtime_ns for p in rd.iterdir()} == snapshot


# ---------------------------------------------------------------------------
# Đánh giá, thống kê, báo cáo, hình
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def evaluated(trained):
    from src import evaluate

    for key in RUNS:
        for ck in ("last", "best"):
            assert evaluate.main(["--run-dir", str(run_dir(trained, key)), "--data-dir", str(trained["processed"]),
                                  "--ckpt", ck]) == 0
    return trained


class TestEvaluateReportVisualize:
    def test_evaluate_outputs(self, evaluated):
        for key in RUNS:
            for ck in ("last", "best"):
                ed = run_dir(evaluated, key) / f"eval_test_{ck}"
                df = pd.read_csv(ed / "per_patch_metrics.csv")
                assert len(df) == TEST_PATCHES
                for m in ("dice", "iou", "sensitivity", "miss_rate", "specificity"):
                    v = df[m].dropna()
                    assert ((v >= 0) & (v <= 1)).all(), m
                res = json.loads((ed / "results.json").read_text(encoding="utf-8"))
                assert res["dataset"] == "luna16" and res["ckpt"] == ck and res["split"] == "test" and res["n"] == TEST_PATCHES
                for m, s in res["metrics"].items():
                    if s["mean"] is not None:
                        assert 0 <= s["mean"] <= 100, m
                assert res["mean_w1"] is not None
                with np.load(ed / "preds.npz") as z:
                    assert z["prob"].dtype == np.float16 and z["pred_bin"].dtype == np.uint8
                    assert z["prob"].shape == (TEST_PATCHES, 64, 64)
                    assert z["patch_id"].tolist() == df["patch_id"].tolist()

    def test_stats_test(self, evaluated):
        from src import stats_test

        out = evaluated["reports"] / "stats" / "caaf_vs_pcam.json"
        a = run_dir(evaluated, "caaf") / "eval_test_last" / "per_patch_metrics.csv"
        b = run_dir(evaluated, "pcam") / "eval_test_last" / "per_patch_metrics.csv"
        assert stats_test.main(["--a", str(a), "--b", str(b), "--metric", "dice", "--out", str(out)]) == 0
        res = json.loads(out.read_text(encoding="utf-8"))
        assert res["n"] == TEST_PATCHES and res["unit"] == "%" and res["metric"] == "dice"

    def test_report_with_partial_runs(self, evaluated):
        from src import report

        out = evaluated["reports"]
        assert report.main(["--exp-root", str(evaluated["exp"]), "--out-dir", str(out), "--ckpt", "last", "--seed", "42",
                            "--include-sensitivity"]) == 0
        for name in ("compare_table5.md", "compare_table6.md", "compare_table7.md", "seed_variance.md", "sensitivity.md"):
            assert (out / name).stat().st_size > 0, name
        t6 = (out / "compare_table6.md").read_text(encoding="utf-8")
        assert "`caaf_hausdorff` vs `pcam_hausdorff`" in t6 and "15.91" in t6
        assert "caaf_hausdorff_s42" in (out / "sensitivity.md").read_text(encoding="utf-8")  # S2 best vs last

    @pytest.mark.parametrize(
        "kind, extra",
        [("loss", []), ("dotplot", []), ("diffmap", []), ("weights", []),
         ("gradcam", ["--layer", "aaf1"]), ("gradcam", ["--layer", "aaf2"]), ("gradcam", ["--layer", "aaf3"])],
    )
    def test_visualize_all_kinds(self, evaluated, kind, extra):
        from src import visualize

        out = evaluated["reports"] / "figures"
        before = set(out.glob("*.png")) if out.exists() else set()
        runs = [str(run_dir(evaluated, k)) for k in RUNS]
        assert visualize.main(["--run-dirs", *runs, "--kind", kind, "--out-dir", str(out),
                               "--data-dir", str(evaluated["processed"]), *extra]) == 0
        new = set(out.glob("*.png")) - before
        expected = 1 if kind == "loss" else 2  # loss: cả 2 run cùng loss dice+hausdorff → 1 hình
        assert len(new) == expected, sorted(p.name for p in new)
        for p in new:
            assert p.stat().st_size > 0 and p.read_bytes()[:4] == b"\x89PNG"
        if kind == "gradcam":
            assert all(extra[1] in p.name for p in new)
