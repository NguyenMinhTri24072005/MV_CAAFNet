"""Kiểm thử evaluate (phần ghi kết quả), stats_test, report, visualize — SPECS §6.2, §9.1–9.3.

Không cần torch: thư mục experiment được giả lập bằng `evaluate.write_eval_outputs`, còn visualize
chạy các chế độ `loss`, `dotplot`, `diffmap`, `weights`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import evaluate, report, stats_test, visualize  # noqa: E402
from src.utils import load_config, save_yaml  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n"
N = 40


def n_actions(parser: argparse.ArgumentParser) -> int:
    return len([a for a in parser._actions if not isinstance(a, argparse._HelpAction)])


def per_patch_df(dice_mean: float, seed: int, n: int = N, w=(0.3, 0.7)) -> pd.DataFrame:
    """Bảng per_patch_metrics giả lập (thang [0, 1]) trên cùng tập patch_id cho mọi run."""
    rng = np.random.default_rng(seed)
    dice = np.clip(dice_mean + rng.normal(0, 0.05, n), 0, 1)
    sens = np.clip(dice + 0.01, 0, 1)
    w1 = np.full(n, np.nan) if w is None else np.clip(w[0] + rng.normal(0, 0.02, n), 0, 1)
    return pd.DataFrame({
        "patch_id": [f"LIDC-IDRI-{i // 4 + 1:04d}_s00_n00_z{i:03d}" for i in range(n)],
        "patient_id": [f"LIDC-IDRI-{i // 4 + 1:04d}" for i in range(n)],
        "dice": dice, "iou": dice / (2 - dice), "sensitivity": sens, "miss_rate": 1 - sens,
        "specificity": np.full(n, 0.995), "tp": 100, "fp": 5, "fn": 5, "tn": 3986, "gt_area": 105, "pred_area": 105,
        "w1": w1, "w2": 1 - w1,
    })[evaluate.PER_PATCH_COLUMNS]


def make_run(exp: Path, ds: str, run: str, dice_mean: float, seed: int = 0, ckpts=("last",), w=(0.3, 0.7),
             loss: str = "dice+hausdorff") -> Path:
    rd = exp / ds / run
    rd.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "base.yaml")
    cfg["dataset"] = ds
    cfg["loss"]["name"] = loss
    save_yaml(cfg, rd / "config_resolved.yaml")
    pd.DataFrame({"epoch": range(1, 6), "train_loss": np.linspace(0.8, 0.2, 5) + 0.01 * seed}).to_csv(
        rd / "train_log.csv", index=False)
    for k, ck in enumerate(ckpts):
        df = per_patch_df(dice_mean + 0.005 * k, seed, w=w)
        evaluate.write_eval_outputs(rd / f"eval_test_{ck}", df,
                                    {"run": run, "dataset": ds, "split": "test", "ckpt": ck, "epoch": 100})
    return rd


# ---------------------------------------------------------------------------
# evaluate — phần ghi kết quả (không cần torch)
# ---------------------------------------------------------------------------
class TestEvaluateOutputs:
    def test_per_patch_results_and_preds(self, tmp_path):
        mask = np.zeros((4, 1, 64, 64), np.float32)
        mask[:, 0, 20:30, 20:30] = 1
        mask[3] = 0
        prob = np.zeros_like(mask)
        prob[0, 0, 20:30, 20:30] = 0.9            # khớp hoàn toàn
        prob[1, 0, 20:30, 22:32] = 0.8            # lệch 2 cột
        prob[2, 0] = 0.1                           # dự đoán rỗng
        ids = ["p3", "p1", "p0", "p2"]             # không theo thứ tự → phải được sắp lại
        df = evaluate.per_patch_table(prob, mask, ids, ["A"] * 4, np.array([[0.2, 0.8]] * 4), 0.5)
        assert list(df.columns) == evaluate.PER_PATCH_COLUMNS
        res = evaluate.write_eval_outputs(tmp_path / "eval_test_last", df, {"run": "r", "ckpt": "last"},
                                          prob[:, 0], 0.5)
        csv = pd.read_csv(tmp_path / "eval_test_last" / "per_patch_metrics.csv")
        assert csv["patch_id"].tolist() == ["p0", "p1", "p2", "p3"]
        assert ((csv["dice"] >= 0) & (csv["dice"] <= 1)).all()          # thang [0, 1]
        dice = dict(zip(df["patch_id"], df["dice"]))
        assert dice["p3"] == 1.0 and dice["p1"] == pytest.approx(0.8) and dice["p0"] == 0.0 and dice["p2"] == 1.0
        assert res["metrics"]["dice"]["mean"] == pytest.approx(100 * np.mean([1.0, 0.8, 0.0, 1.0]))
        assert res["metrics"]["dice"]["std"] == pytest.approx(100 * np.std([1.0, 0.8, 0.0, 1.0], ddof=1))
        assert res["n"] == 4 and res["mean_w1"] == pytest.approx(0.2) and res["threshold"] == 0.5
        assert json.loads((tmp_path / "eval_test_last" / "results.json").read_text(encoding="utf-8")) == res
        with np.load(tmp_path / "eval_test_last" / "preds.npz") as z:
            assert z["prob"].dtype == np.float16 and z["pred_bin"].dtype == np.uint8
            assert z["prob"].shape == z["pred_bin"].shape == (4, 64, 64)
            assert z["patch_id"].tolist() == ["p0", "p1", "p2", "p3"]  # cùng thứ tự với per_patch_metrics.csv
            order = np.argsort(ids)
            assert np.array_equal(z["pred_bin"], (prob[order, 0] >= 0.5).astype(np.uint8))
            assert np.allclose(z["prob"], prob[order, 0], atol=1e-3)

    def test_se_model_has_no_weights(self, tmp_path):
        df = evaluate.per_patch_table(np.zeros((2, 1, 8, 8)), np.zeros((2, 1, 8, 8)), ["a", "b"], ["A", "A"], None, 0.5)
        res = evaluate.write_eval_outputs(tmp_path, df, {})
        assert df["w1"].isna().all() and res["mean_w1"] is None and res["mean_w2"] is None
        assert not (tmp_path / "preds.npz").exists()

    def test_cli_has_8_args(self):
        assert n_actions(evaluate.build_parser()) == 8
        args = evaluate.build_parser().parse_args(["--run-dir", "x"])
        assert (args.split, args.save_preds, args.ckpt, args.threshold, args.dataset) == ("test", True, None, None, None)
        assert evaluate.build_parser().parse_args(["--run-dir", "x", "--no-save-preds"]).save_preds is False


# ---------------------------------------------------------------------------
# stats_test (§9.1)
# ---------------------------------------------------------------------------
class TestStatsTest:
    def test_direction_percent_and_matches_scipy(self):
        b = per_patch_df(0.85, 1)
        a = b.assign(dice=np.clip(b["dice"] + 0.03 + np.random.default_rng(2).normal(0, 0.01, N), 0, 1))
        res = stats_test.paired_test(a, b)
        t_ref, p_ref = stats.ttest_rel(a["dice"] * 100, b["dice"] * 100)
        assert res["t"] > 0 and res["t"] == pytest.approx(t_ref) and res["p"] == pytest.approx(p_ref)
        assert res["mean_a"] == pytest.approx(100 * a["dice"].mean())
        assert res["mean_diff"] == pytest.approx(100 * (a["dice"] - b["dice"]).mean())
        assert res["std_a"] == pytest.approx(100 * a["dice"].std(ddof=1))
        assert res["n"] == N and res["unit"] == "%" and res["wilcoxon_p"] is not None
        swapped = stats_test.paired_test(b, a)
        assert swapped["t"] == pytest.approx(-res["t"]) and swapped["mean_diff"] == pytest.approx(-res["mean_diff"])

    def test_inner_join_must_match_100_percent(self):
        b = per_patch_df(0.85, 1)
        with pytest.raises(ValueError):
            stats_test.paired_test(b.iloc[1:], b)
        other = b.assign(patch_id=b["patch_id"].str.replace("z000", "z999"))
        with pytest.raises(ValueError):
            stats_test.paired_test(other, b)
        with pytest.raises(ValueError):
            stats_test.paired_test(pd.concat([b, b.iloc[:1]]), b)

    def test_identical_inputs_do_not_crash(self):
        b = per_patch_df(0.85, 1)
        res = stats_test.paired_test(b, b.sample(frac=1.0, random_state=0))  # thứ tự khác vẫn ghép đúng cặp
        assert res["mean_diff"] == pytest.approx(0.0) and res["wilcoxon_stat"] is None and res["t"] is None

    def test_cli(self, tmp_path):
        a, b = per_patch_df(0.9, 3), per_patch_df(0.85, 3)
        a.to_csv(tmp_path / "a.csv", index=False)
        b.to_csv(tmp_path / "b.csv", index=False)
        assert n_actions(stats_test.build_parser()) == 4
        assert stats_test.main(["--a", str(tmp_path / "a.csv"), "--b", str(tmp_path / "b.csv"), "--metric", "dice",
                                "--out", str(tmp_path / "s" / "out.json")]) == 0
        out = json.loads((tmp_path / "s" / "out.json").read_text(encoding="utf-8"))
        assert out["t"] > 0 and out["metric"] == "dice"


# ---------------------------------------------------------------------------
# report (§9.2)
# ---------------------------------------------------------------------------
@pytest.fixture
def exp(tmp_path):
    e = tmp_path / "experiments"
    make_run(e, "luna16", "caaf_sobel_s42", 0.88, 1, loss="dice+sobel")
    make_run(e, "luna16", "pcam_sobel_s42", 0.86, 1, loss="dice+sobel")
    make_run(e, "luna16", "caaf_hausdorff_s42", 0.90, 1, ckpts=("last", "best"))
    make_run(e, "luna16", "caaf_hausdorff_s7", 0.89, 2)
    make_run(e, "luna16", "pcam_hausdorff_s42", 0.87, 1)
    make_run(e, "luna16", "baseline_resunetpp_s42", 0.83, 1, w=None, loss="dice")
    make_run(e, "luna16", "caaf_hausdorff_s42__loss-dice+sobel", 0.555, 1)
    make_run(e, "lidc", "caaf_hausdorff_s42", 0.85, 4)
    return e


def results_of(exp: Path, ds: str, run: str, ckpt: str = "last") -> dict:
    return json.loads((exp / ds / run / f"eval_test_{ckpt}" / "results.json").read_text(encoding="utf-8"))


def ms(res: dict, k: str = "dice") -> str:
    m = res["metrics"][k]
    return f"{m['mean']:.2f} ± {m['std']:.2f}"


class TestReport:
    def test_parse_run(self):
        assert report.parse_run("pcam_sobel_s42") == ("pcam_sobel", 42, "")
        assert report.parse_run("caaf_sobel_s7__set-ab12cd_bs-8") == ("caaf_sobel", 7, "set-ab12cd_bs-8")
        assert report.parse_run("smoke_x") is None

    def test_generates_all_tables(self, exp, tmp_path):
        out = tmp_path / "reports"
        paths = report.generate(exp, out, "last", 42, include_sensitivity=True)
        assert set(paths) == {"compare_table5.md", "compare_table6.md", "compare_table7.md", "seed_variance.md",
                              "sensitivity.md"}
        for p in paths.values():
            text = p.read_text(encoding="utf-8")
            assert "Bài báo" in text or "bài báo" in text
            assert "Δ" in text

    def test_table5_standard_runs_only(self, exp, tmp_path):
        report.generate(exp, tmp_path, "last", 42)
        t5 = (tmp_path / "compare_table5.md").read_text(encoding="utf-8")
        ours = results_of(exp, "luna16", "caaf_hausdorff_s42")
        assert "90.88 ± 6.16" in t5 and ms(ours) in t5
        assert f"{ours['metrics']['dice']['mean'] - 90.88:+.2f}" in t5
        assert ms(results_of(exp, "lidc", "caaf_hausdorff_s42")) in t5 and "85.92 ± 16.70" in t5
        assert ms(results_of(exp, "luna16", "caaf_hausdorff_s42__loss-dice+sobel")) not in t5  # bỏ hậu tố __*
        assert "0.26" in t5 and f"{ours['mean_w1']:.3f}" in t5

    def test_table6_calls_stats_test(self, exp, tmp_path):
        report.generate(exp, tmp_path, "last", 42)
        t6 = (tmp_path / "compare_table6.md").read_text(encoding="utf-8")
        st = json.loads((tmp_path / "stats" / "luna16_caaf_hausdorff_s42__vs__pcam_hausdorff_s42.json").read_text("utf-8"))
        ref = stats_test.paired_test(exp / "luna16/caaf_hausdorff_s42/eval_test_last/per_patch_metrics.csv",
                                     exp / "luna16/pcam_hausdorff_s42/eval_test_last/per_patch_metrics.csv")
        assert st["t"] == pytest.approx(ref["t"]) and st["t"] > 0
        assert f"{st['t']:.2f}" in t6 and "15.91" in t6 and "5.98" in t6 and "6.18e-52" in t6
        assert ms(results_of(exp, "luna16", "pcam_sobel_s42")) in t6 and "87.46 ± 7.32" in t6
        lap = next(line for line in t6.splitlines() if "pcam_laplacian" in line)
        assert "84.14 ± 16.90" in lap and "| — |" in lap  # chưa có run → "—"

    def test_table7(self, exp, tmp_path):
        report.generate(exp, tmp_path, "last", 42)
        t7 = (tmp_path / "compare_table7.md").read_text(encoding="utf-8")
        base = next(line for line in t7.splitlines() if "baseline_resunetpp" in line)
        res = results_of(exp, "luna16", "baseline_resunetpp_s42")
        assert ms(res) in base and "83.89" in base and f"{res['metrics']['dice']['mean'] - 83.89:+.2f}" in base
        assert "85.96" in t7 and "90.88" in t7

    def test_seed_variance(self, exp, tmp_path):
        report.generate(exp, tmp_path, "last", 42)
        sv = (tmp_path / "seed_variance.md").read_text(encoding="utf-8")
        d42 = results_of(exp, "luna16", "caaf_hausdorff_s42")["metrics"]["dice"]["mean"]
        d7 = results_of(exp, "luna16", "caaf_hausdorff_s7")["metrics"]["dice"]["mean"]
        assert "| 7 |" in sv and "| 42 |" in sv
        assert f"{np.mean([d42, d7]):.2f} ± {np.std([d42, d7], ddof=1):.2f}" in sv
        assert "loss-dice+sobel" not in sv and "90.88" in sv

    def test_sensitivity(self, exp, tmp_path):
        assert "sensitivity.md" not in report.generate(exp, tmp_path / "a", "last", 42)
        report.generate(exp, tmp_path / "b", "last", 42, include_sensitivity=True)
        sens = (tmp_path / "b" / "sensitivity.md").read_text(encoding="utf-8")
        assert "caaf_hausdorff_s42__loss-dice+sobel" in sens and "`loss-dice+sobel`" in sens
        last = results_of(exp, "luna16", "caaf_hausdorff_s42")["metrics"]["dice"]["mean"]
        best = results_of(exp, "luna16", "caaf_hausdorff_s42", "best")["metrics"]["dice"]["mean"]
        s2 = sens.split("S2")[1]
        assert f"{last:.2f}" in s2 and f"{best:.2f}" in s2 and f"{best - last:+.2f}" in s2

    def test_cli(self, exp, tmp_path):
        assert n_actions(report.build_parser()) == 5
        assert report.main(["--exp-root", str(exp), "--out-dir", str(tmp_path / "r"), "--ckpt", "last", "--seed", "7",
                            "--include-sensitivity"]) == 0
        t5 = (tmp_path / "r" / "compare_table5.md").read_text(encoding="utf-8")
        assert ms(results_of(exp, "luna16", "caaf_hausdorff_s7")) in t5  # --seed 7 chọn run _s7


# ---------------------------------------------------------------------------
# visualize (§9.3) — chế độ không cần torch
# ---------------------------------------------------------------------------
@pytest.fixture
def viz(tmp_path):
    import h5py

    exp, data = tmp_path / "experiments", tmp_path / "processed"
    ids = [f"LIDC-IDRI-{i // 2 + 1:04d}_s00_n00_z{i:03d}" for i in range(6)]
    mask = np.zeros((6, 1, 64, 64), np.float32)
    for i in range(6):
        mask[i, 0, 20:30 + i, 20:30] = 1
    image = np.random.default_rng(0).random((6, 1, 64, 64)).astype(np.float32)
    (data / "luna16").mkdir(parents=True)
    with h5py.File(data / "luna16" / "test.h5", "w") as h5:
        h5["image"], h5["mask"] = image, mask.astype(np.uint8)
        h5.create_dataset("patch_id", data=np.array(ids, dtype=object), dtype=h5py.string_dtype("utf-8"))
    runs = []
    for k, (run, loss, w) in enumerate([("caaf_hausdorff_s42", "dice+hausdorff", np.array([[0.3, 0.7]] * 6)),
                                        ("baseline_resunetpp_s42", "dice", None)]):
        rd = make_run(exp, "luna16", run, 0.9, k, loss=loss)
        prob = np.roll(mask, k + 1, axis=-1) * 0.9
        df = evaluate.per_patch_table(prob, mask, ids, [p[:14] for p in ids], w, 0.5)
        evaluate.write_eval_outputs(rd / "eval_test_last", df, {"run": run}, prob[:, 0], 0.5)
        runs.append(rd)
    return runs, data, ids, tmp_path / "figs"


def viz_main(runs, kind, out, *extra):
    return visualize.main(["--run-dirs", *map(str, runs), "--kind", kind, "--out-dir", str(out), *extra])


class TestVisualize:
    def test_diff_rgb_colors(self):
        pred = np.array([[1, 1], [0, 0]])
        gt = np.array([[1, 0], [1, 0]])
        rgb = visualize.diff_rgb(pred, gt)
        assert rgb.dtype == np.uint8
        assert tuple(rgb[0, 0]) == (0, 255, 0)      # TP xanh lá
        assert tuple(rgb[0, 1]) == (255, 0, 0)      # FP đỏ
        assert tuple(rgb[1, 0]) == (255, 255, 0)    # FN vàng
        assert tuple(rgb[1, 1]) == (0, 0, 255)      # TN xanh dương

    def test_loss_dotplot_diffmap_weights(self, viz):
        runs, data, ids, out = viz
        assert viz_main(runs, "loss", out) == 0
        assert viz_main(runs, "dotplot", out) == 0
        assert viz_main(runs, "diffmap", out, "--data-dir", str(data), "--patch-ids", ids[0], ids[3]) == 0
        assert viz_main(runs, "weights", out) == 0
        names = sorted(p.name for p in out.glob("*.png"))
        assert names == sorted([
            "fig2_loss_dice.png", "fig2_loss_dice_hausdorff.png",
            "fig3_dotplot_luna16_caaf_hausdorff_s42_test_last.png", "fig3_dotplot_luna16_baseline_resunetpp_s42_test_last.png",
            "diffmap_luna16_caaf_hausdorff_s42_test_last.png", "diffmap_luna16_baseline_resunetpp_s42_test_last.png",
            "weights_luna16_caaf_hausdorff_s42_test_last.png",   # baseline SE không có w → bỏ qua
        ])
        for p in out.glob("*.png"):
            assert p.read_bytes()[:8] == PNG and p.stat().st_size > 1000

    def test_diffmap_default_patches(self, viz):
        runs, data, _, out = viz
        assert viz_main(runs[:1], "diffmap", out, "--data-dir", str(data)) == 0
        assert len(visualize.default_patch_ids(runs[0], "test", "last")) == 4

    def test_unknown_patch_id(self, viz):
        runs, data, _, out = viz
        with pytest.raises(KeyError):
            viz_main(runs[:1], "diffmap", out, "--data-dir", str(data), "--patch-ids", "khong-co")

    def test_cli_has_9_args(self):
        assert n_actions(visualize.build_parser()) == 9
        assert visualize.KINDS == ("loss", "dotplot", "diffmap", "gradcam", "weights")

    def test_import_does_not_pull_torch(self):
        code = "import sys, src.visualize, src.report, src.stats_test, src.evaluate; print('torch' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "False"

    def test_gradcam_imports_torch_lazily(self, viz, monkeypatch):
        """Chỉ `--kind gradcam` cần torch: mô phỏng máy chưa cài torch → gradcam báo ImportError,
        còn các chế độ khác vẫn chạy (Grad-CAM thật được kiểm ở tests/test_e2e_pipeline.py)."""
        runs, data, ids, out = viz
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ImportError):
            viz_main(runs[:1], "gradcam", out, "--data-dir", str(data), "--patch-ids", ids[0])
        assert viz_main(runs[:1], "weights", out) == 0
