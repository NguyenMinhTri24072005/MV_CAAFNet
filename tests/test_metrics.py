"""Độ đo — SPECS §4.3, [A] F1–F5, Phụ lục C. Thuần numpy nên chạy được ở local không cần torch."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import METRIC_NAMES, compute_patch_metrics, confusion, summarize_metrics  # noqa: E402


def m(*rows):
    return np.array(rows, dtype=np.float32)


def test_confusion_by_hand():
    p = m([1, 1, 0, 0], [1, 0, 0, 0])[None]
    g = m([1, 0, 1, 0], [0, 0, 0, 0])[None]
    tp, fp, fn, tn = confusion(p, g)
    assert (tp[0], fp[0], fn[0], tn[0]) == (1, 2, 1, 4)


def test_metrics_by_hand():
    prob = m([0.9, 0.6, 0.2, 0.1], [0.5, 0.4, 0.0, 0.0])[None, None]
    mask = m([1, 0, 1, 0], [1, 0, 0, 0])[None, None]
    r = compute_patch_metrics(prob, mask, 0.5)  # 0.5 tính là dương (>=)
    assert (r["tp"][0], r["fp"][0], r["fn"][0], r["tn"][0]) == (2, 1, 1, 4)
    assert r["dice"][0] == pytest.approx(4 / 6)
    assert r["iou"][0] == pytest.approx(2 / 4)
    assert r["sensitivity"][0] == pytest.approx(2 / 3)
    assert r["miss_rate"][0] == pytest.approx(1 / 3)
    assert r["specificity"][0] == pytest.approx(4 / 5)
    assert r["gt_area"][0] == 3 and r["pred_area"][0] == 3


def test_empty_prediction_with_nonempty_gt():
    mask = np.zeros((1, 1, 8, 8), np.float32)
    mask[0, 0, 2:4, 2:4] = 1
    r = compute_patch_metrics(np.zeros_like(mask), mask)
    assert r["dice"][0] == 0.0 and r["iou"][0] == 0.0
    assert r["sensitivity"][0] == 0.0 and r["miss_rate"][0] == 1.0
    assert r["specificity"][0] == 1.0


def test_both_empty():
    z = np.zeros((1, 1, 8, 8), np.float32)
    r = compute_patch_metrics(z, z)
    assert r["dice"][0] == 1.0 and r["iou"][0] == 1.0
    assert math.isnan(r["sensitivity"][0]) and math.isnan(r["miss_rate"][0])
    assert r["specificity"][0] == 1.0


def test_full_patch_specificity_nan():
    o = np.ones((1, 1, 4, 4), np.float32)
    r = compute_patch_metrics(o, o)
    assert r["dice"][0] == 1.0 and math.isnan(r["specificity"][0])


def test_miss_rate_is_one_minus_sensitivity():
    rng = np.random.default_rng(0)
    prob = rng.random((50, 1, 16, 16))
    mask = (rng.random((50, 1, 16, 16)) < 0.2).astype(np.float32)
    r = compute_patch_metrics(prob, mask)
    ok = np.isfinite(r["sensitivity"])
    assert np.allclose(r["miss_rate"][ok], 1 - r["sensitivity"][ok])
    for name in METRIC_NAMES:
        vals = r[name][np.isfinite(r[name])]
        assert ((vals >= 0) & (vals <= 1)).all(), "thang [0, 1]"


def test_shape_mismatch():
    with pytest.raises(ValueError):
        confusion(np.zeros((2, 4)), np.zeros((2, 5)))


def test_summary_in_percent_ddof1():
    data = {"dice": np.array([1.0, 0.5, 0.0]), "sensitivity": np.array([1.0, np.nan, 0.5])}
    s = summarize_metrics(data)
    assert set(s) == {"dice", "sensitivity"}
    assert s["dice"]["mean"] == pytest.approx(50.0) and s["dice"]["std"] == pytest.approx(50.0)  # ddof=1
    assert s["dice"]["n"] == 3
    assert s["sensitivity"] == {"mean": pytest.approx(75.0), "std": pytest.approx(np.std([100, 50], ddof=1)), "n": 2}


def test_summary_edge_counts_and_dataframe():
    df = pd.DataFrame({"dice": [0.8], "iou": [np.nan], "miss_rate": [0.25]})
    s = summarize_metrics(df)
    assert s["dice"] == {"mean": pytest.approx(80.0), "std": None, "n": 1}
    assert s["iou"] == {"mean": None, "std": None, "n": 0}
    assert s["miss_rate"]["mean"] == pytest.approx(25.0)
    with pytest.raises(TypeError):
        summarize_metrics([0.1, 0.2])


def test_patch_metrics_roundtrip_to_summary():
    rng = np.random.default_rng(3)
    prob = rng.random((20, 1, 16, 16))
    mask = (rng.random((20, 1, 16, 16)) < 0.3).astype(np.float32)
    r = compute_patch_metrics(prob, mask)
    s = summarize_metrics(r)
    assert s["dice"]["mean"] == pytest.approx(100 * r["dice"].mean())
    assert s["dice"]["std"] == pytest.approx(100 * r["dice"].std(ddof=1))


def test_accepts_torch_tensors():
    torch = pytest.importorskip("torch")
    prob = torch.rand(4, 1, 16, 16)
    mask = (torch.rand(4, 1, 16, 16) < 0.3).float()
    r_t = compute_patch_metrics(prob, mask)
    r_n = compute_patch_metrics(prob.numpy(), mask.numpy())
    assert np.allclose(r_t["dice"], r_n["dice"])
