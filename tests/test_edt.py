"""Kiểm thử `src/data/compute_edt.py` — SPECS §4.1, Phụ lục C (không cần torch)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.compute_edt import EMPTY_VALUE, edt_batch, edt_from_mask  # noqa: E402
from src.utils import load_config  # noqa: E402


def random_mask(seed: int, shape=(64, 64), p=0.05) -> np.ndarray:
    return (np.random.default_rng(seed).random(shape) < p).astype(np.uint8)


def test_zero_inside_positive_outside():
    m = np.zeros((64, 64), np.uint8)
    m[20:30, 25:40] = 1
    d = edt_from_mask(m)
    assert d.dtype == np.float32 and d.shape == m.shape
    assert (d[m == 1] == 0).all()
    assert (d[m == 0] > 0).all()


@pytest.mark.parametrize("seed", range(5))
def test_matches_scipy(seed):
    m = random_mask(seed)
    ref = ndimage.distance_transform_edt(1 - (m > 0)).astype(np.float32)
    assert np.array_equal(edt_from_mask(m), ref)


def test_known_distance_in_pixels():
    m = np.zeros((32, 32), np.uint8)
    m[10, 10] = 1
    d = edt_from_mask(m)
    assert d[13, 14] == pytest.approx(5.0)
    assert d[10, 0] == pytest.approx(10.0)


def test_empty_mask_returns_empty_value():
    d = edt_from_mask(np.zeros((64, 64), np.uint8))
    assert d.dtype == np.float32
    assert np.all(d == np.float32(90.51))
    assert np.all(edt_from_mask(np.zeros((8, 8)), empty_value=7.0) == 7.0)


def test_empty_value_matches_base_config():
    cfg = load_config(ROOT / "configs" / "base.yaml")
    assert EMPTY_VALUE == cfg["loss"]["hausdorff_empty_value"]
    assert EMPTY_VALUE == pytest.approx(64 * 2 ** 0.5, abs=0.01)


def test_input_types_and_no_mutation():
    m = random_mask(7)
    before = m.copy()
    ref = edt_from_mask(m)
    assert np.array_equal(m, before)
    assert np.array_equal(edt_from_mask(m.astype(bool)), ref)
    assert np.array_equal(edt_from_mask(m.astype(np.float32) * 3), ref)


def test_batch_matches_per_sample():
    masks = np.stack([random_mask(s) for s in range(4)])[:, None]
    masks[2] = 0
    out = edt_batch(masks)
    assert out.shape == masks.shape and out.dtype == np.float32
    for i in range(4):
        assert np.array_equal(out[i, 0], edt_from_mask(masks[i, 0]))
    with pytest.raises(ValueError):
        edt_batch(np.zeros(5))
