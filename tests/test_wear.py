"""
Tests for wear-stage assignment.

    python3 -m pytest tests/test_wear.py -v
"""

import numpy as np
import pytest

from src.data.labels import build_labels
from src.data.splits import make_standard_split
from src.data.synthetic import make_synthetic
from src.data.windowing import make_windows
from src.features.wear import (StageMap, _binary_segment,
                               fit_changepoint_stages, fit_quantile_stages,
                               fit_stages, stage_table, window_age)


def bathtub(n=60000, seed=0, early=2000, late=14000):
    rng = np.random.default_rng(seed)
    age = rng.uniform(0, 20000, n)
    rate = np.where(age < early, 0.06,
                    np.where(age < late, 0.01,
                             0.01 + (age - late) / (20000 - late) * 0.05))
    y = (rng.uniform(size=n) < rate).astype(int)
    return age, y


# ---------------------------------------------------------------
# Quantile stages
# ---------------------------------------------------------------

def test_quantile_stages_are_equal_frequency():
    age, _ = bathtub()
    smap = fit_quantile_stages(age, n_stages=5)
    st = smap.assign(age)
    counts = np.bincount(st, minlength=5)
    assert smap.n_stages == 5
    assert counts.max() - counts.min() < 0.02 * len(age)


def test_quantile_edges_are_monotone():
    age, _ = bathtub()
    smap = fit_quantile_stages(age, n_stages=4)
    assert np.all(np.diff(smap.edges) > 0)


def test_quantile_collapses_duplicate_edges():
    """A degenerate age column must not produce empty stages."""
    age = np.array([5.0] * 100 + [10.0] * 100)
    smap = fit_quantile_stages(age, n_stages=5)
    assert smap.n_stages <= 3
    st = smap.assign(age)
    assert set(np.unique(st)) <= set(range(smap.n_stages))


# ---------------------------------------------------------------
# Change-point stages
# ---------------------------------------------------------------

def test_changepoint_recovers_bathtub_boundaries():
    age, y = bathtub()
    smap = fit_changepoint_stages(age, y, n_stages=3)
    assert smap.n_stages == 3
    lo, hi = smap.edges
    assert 1000 < lo < 3500, lo
    assert 12000 < hi < 17500, hi


def test_changepoint_is_fit_on_train_only_by_construction():
    """
    The map is a fitted object applied elsewhere; test data cannot
    influence its edges after fit.
    """
    age, y = bathtub(seed=1)
    smap = fit_changepoint_stages(age, y, n_stages=3)
    edges_before = smap.edges.copy()
    other_age, _ = bathtub(seed=99, early=8000, late=9000)
    smap.assign(other_age)
    np.testing.assert_array_equal(smap.edges, edges_before)


def test_binary_segment_finds_step_change():
    sig = np.concatenate([np.zeros(30), np.ones(30)])
    assert _binary_segment(sig, n_splits=1) == [30]


def test_binary_segment_respects_min_len():
    sig = np.concatenate([np.zeros(3), np.ones(3)])
    assert _binary_segment(sig, n_splits=1, min_len=5) == []


def test_changepoint_flat_curve_still_returns_requested_stages():
    rng = np.random.default_rng(2)
    age = rng.uniform(0, 1000, 20000)
    y = (rng.uniform(size=age.size) < 0.02).astype(int)
    smap = fit_changepoint_stages(age, y, n_stages=3)
    assert smap.n_stages == 3


# ---------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------

def test_assign_is_right_inclusive_of_edges():
    smap = StageMap(edges=np.array([10.0, 20.0]), strategy="quantile",
                    n_stages=3, fit_on=0)
    np.testing.assert_array_equal(
        smap.assign(np.array([5, 10, 15, 20, 25])), [0, 1, 1, 2, 2])


def test_labels_match_stage_count():
    smap = StageMap(edges=np.array([10.0, 20.0]), strategy="quantile",
                    n_stages=3, fit_on=0)
    assert len(smap.labels()) == 3


# ---------------------------------------------------------------
# Integration with the pipeline
# ---------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_windows():
    df, _ = make_synthetic(n_drives=200, seed=5)
    labelled, _ = build_labels(df)
    split = make_standard_split(labelled)
    return make_windows(split.apply(labelled, "train"), stride=15)


def test_window_age_reads_end_of_window(raw_windows):
    age = window_age(raw_windows)
    j = raw_windows.features.index("r_9")
    np.testing.assert_allclose(age, raw_windows.X[:, -1, j])


def test_window_age_requires_r9():
    from src.data.windowing import WindowSet
    ws = WindowSet(X=np.zeros((3, 2, 1), dtype=np.float32),
                   y=np.zeros(3, dtype=np.int8), drive_idx=np.arange(3),
                   end_ds=np.zeros(3, dtype=object),
                   drives=[("M", i) for i in range(3)],
                   vendors=np.array(["A"] * 3, dtype=object),
                   features=["n_5"], window_len=2, stride=1, positive_stride=1)
    with pytest.raises(ValueError, match="r_9"):
        window_age(ws)


@pytest.mark.parametrize("strategy", ["quantile", "changepoint"])
def test_fit_stages_on_real_windows(raw_windows, strategy):
    smap = fit_stages(raw_windows, strategy=strategy, n_stages=3)
    st = smap.assign(window_age(raw_windows))
    assert st.min() >= 0 and st.max() < smap.n_stages
    tbl = stage_table(st, raw_windows.y, smap)
    assert tbl.height == smap.n_stages
    assert int(tbl["n"].sum()) == len(raw_windows)


def test_unknown_strategy_raises(raw_windows):
    with pytest.raises(ValueError):
        fit_stages(raw_windows, strategy="nope")
