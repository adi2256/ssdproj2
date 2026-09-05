"""
Tests for trajectory windowing.

Windowing is the fiddliest logic in the pipeline: off-by-one at the
window boundary, label alignment to the wrong timestep, and silent date
gaps are all invisible in aggregate metrics. These tests pin the
behaviour against synthetic drives with known ground truth.

    python3 -m pytest tests/test_windowing.py -v
"""

from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from src.config import CFG, DRIVE_KEY
from src.data.labels import LABEL_COL, build_labels
from src.data.splits import make_lomo_split, make_standard_split
from src.data.synthetic import make_synthetic
from src.data.windowing import (Normalizer, WindowSet, flatten,
                                make_windows)
from src.schema import COMMON_COLS


@pytest.fixture(scope="module")
def labelled():
    df, _ = make_synthetic()
    out, _ = build_labels(df)
    return out


@pytest.fixture(scope="module")
def ws(labelled):
    return make_windows(labelled, stride=7)


# ---------------------------------------------------------------
# Shape and defaults
# ---------------------------------------------------------------

def test_shape_is_windows_by_time_by_features(ws):
    n, w, f = ws.X.shape
    assert w == CFG.window_len
    assert f == len(ws.features)
    assert ws.y.shape == (n,)
    assert ws.drive_idx.shape == (n,)
    assert ws.end_ds.shape == (n,)
    assert ws.vendors.shape == (n,)


def test_defaults_to_common_sixteen(ws):
    """
    Locked feature policy: only the 16 columns populated by every vendor
    are usable in every LOMO fold.
    """
    assert ws.features == list(COMMON_COLS)
    assert len(ws.features) == 16


def test_dtype_is_float32(ws):
    assert ws.X.dtype == np.float32
    assert ws.y.dtype == np.int8


def test_unknown_feature_raises(labelled):
    with pytest.raises(ValueError, match="not in frame"):
        make_windows(labelled, features=["nope_1"])


# ---------------------------------------------------------------
# Label alignment — the highest-risk logic
# ---------------------------------------------------------------

def test_label_comes_from_the_last_row_of_the_window(labelled):
    """
    A window ending at T is 'what we know as of T', so its label is the
    label of row T. Taking it from the first row or an aggregate
    silently shifts the prediction horizon.
    """
    ws = make_windows(labelled, stride=11)
    lookup = {
        (m, d, ds): lab
        for m, d, ds, lab in labelled.select(
            DRIVE_KEY + ["ds", LABEL_COL]
        ).iter_rows()
    }
    for i in range(0, len(ws), 37):
        model, disk_id = ws.drives[ws.drive_idx[i]]
        assert lookup[(model, disk_id, ws.end_ds[i])] == ws.y[i]


def test_no_window_emitted_for_undefined_label(labelled):
    """
    Reindexing can introduce days with no observed row. Those have no
    label and must never terminate a window.
    """
    ws = make_windows(labelled, stride=3)
    assert set(np.unique(ws.y)) <= {0, 1}


def test_positive_windows_end_inside_the_horizon(labelled):
    """Every positive window must end within `horizon` days of failure."""
    df, truth = make_synthetic()
    lab, _ = build_labels(df)
    ws = make_windows(lab, stride=5)

    fail_at = {(m, d): t for m, d, t in truth["failed"]}
    h = truth["horizon_days"]
    checked = 0
    for i in np.flatnonzero(ws.y == 1)[:200]:
        key = ws.drives[ws.drive_idx[i]]
        assert key in fail_at
        gap = (fail_at[key] - ws.end_ds[i]).days
        assert 0 < gap <= h
        checked += 1
    assert checked > 0


def test_negative_windows_are_outside_the_horizon(labelled):
    df, truth = make_synthetic()
    lab, _ = build_labels(df)
    ws = make_windows(lab, stride=5)
    fail_at = {(m, d): t for m, d, t in truth["failed"]}
    h = truth["horizon_days"]

    for i in np.flatnonzero(ws.y == 0)[:200]:
        key = ws.drives[ws.drive_idx[i]]
        if key in fail_at:
            assert (fail_at[key] - ws.end_ds[i]).days > h


# ---------------------------------------------------------------
# Window span — continuous calendar days
# ---------------------------------------------------------------

def test_window_spans_exactly_window_len_calendar_days(labelled):
    """
    Windows are built over a reindexed daily range, so a window always
    covers window_len calendar days. Windowing over positional index
    would let a window silently span months.
    """
    ws = make_windows(labelled, stride=13)
    starts = {}
    for m, d, ds in labelled.select(DRIVE_KEY + ["ds"]).iter_rows():
        starts.setdefault((m, d), []).append(ds)

    for i in range(0, len(ws), 29):
        key = ws.drives[ws.drive_idx[i]]
        first = min(starts[key])
        offset = (ws.end_ds[i] - first).days
        assert offset >= CFG.window_len - 1


# ---------------------------------------------------------------
# Stride and prevalence
# ---------------------------------------------------------------

def test_larger_stride_yields_fewer_windows(labelled):
    counts = [len(make_windows(labelled, stride=s)) for s in (1, 5, 20)]
    assert counts[0] > counts[1] > counts[2]


def test_default_does_not_oversample_positives(labelled):
    """
    positive_stride defaults to stride. The paper's central measurement
    is that failure-class coverage collapses under low prevalence, so
    inflating prevalence at the windowing stage would mask the effect
    being reported.
    """
    natural = make_windows(labelled, stride=7)
    explicit = make_windows(labelled, stride=7, positive_stride=7)
    assert len(natural) == len(explicit)
    assert natural.positive_rate == explicit.positive_rate


def test_oversampling_inflates_prevalence(labelled):
    """
    Documents the artefact so nobody enables it by accident and reads
    the resulting prevalence as real.
    """
    natural = make_windows(labelled, stride=7)
    over = make_windows(labelled, stride=7, positive_stride=1)
    assert over.positive_rate > 3 * natural.positive_rate
    assert len(over) > len(natural)


def test_prevalence_roughly_stable_across_stride(labelled):
    """Subsampling should not shift the class balance much."""
    a = make_windows(labelled, stride=1).positive_rate
    b = make_windows(labelled, stride=7).positive_rate
    assert abs(a - b) < 0.02


# ---------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------

def test_no_nan_in_output(ws):
    """Models cannot consume NaN; filling happens inside the windower."""
    assert np.isfinite(ws.X).all()


def test_min_valid_frac_rejects_sparse_windows(labelled):
    strict = make_windows(labelled, stride=7, min_valid_frac=0.999)
    loose = make_windows(labelled, stride=7, min_valid_frac=0.0)
    assert len(strict) <= len(loose)


def test_all_nan_feature_survives_as_zero(labelled):
    """A column absent for a vendor must not produce NaN or crash."""
    df = labelled.with_columns(
        pl.lit(float("nan")).alias("n_5")
    )
    out = make_windows(df, stride=11, min_valid_frac=0.0)
    assert np.isfinite(out.X).all()
    col = out.features.index("n_5")
    assert (out.X[:, :, col] == 0).all()


# ---------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------

def test_drive_idx_maps_into_drives(ws):
    assert ws.drive_idx.max() < len(ws.drives)
    assert ws.drive_idx.min() >= 0


def test_vendor_matches_drive_model(ws):
    for i in range(0, len(ws), 41):
        model, _ = ws.drives[ws.drive_idx[i]]
        assert ws.vendors[i] == model[1]


def test_every_listed_drive_emitted_a_window(ws):
    """Drives that produced nothing are removed from `drives`."""
    assert set(np.unique(ws.drive_idx)) == set(range(len(ws.drives)))


def test_drive_keys_helper(ws):
    keys = ws.drive_keys()
    assert len(keys) == len(ws)
    assert tuple(keys[0]) == ws.drives[ws.drive_idx[0]]


def test_short_drives_excluded(labelled):
    ws = make_windows(labelled, window_len=1000)
    assert len(ws) == 0
    assert ws.X.shape == (0, 1000, 16)


# ---------------------------------------------------------------
# Split integration — leakage must survive windowing
# ---------------------------------------------------------------

def test_windows_do_not_cross_split_boundaries(labelled):
    split = make_standard_split(labelled)
    parts = {
        p: make_windows(split.apply(labelled, p), stride=9)
        for p in ("train", "cal", "test")
    }
    keys = {p: {tuple(k) for k in w.drives} for p, w in parts.items()}
    assert not (keys["train"] & keys["cal"])
    assert not (keys["train"] & keys["test"])
    assert not (keys["cal"] & keys["test"])


def test_lomo_windows_respect_held_out_vendor(labelled):
    split = make_lomo_split(labelled, "C")
    for part in ("train", "cal"):
        w = make_windows(split.apply(labelled, part), stride=9)
        assert "C" not in set(np.unique(w.vendors))
    test = make_windows(split.apply(labelled, "test"), stride=9)
    assert set(np.unique(test.vendors)) == {"C"}


# ---------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------

def test_normalizer_standardises_training_data(ws):
    n = Normalizer.fit(ws)
    out = n.transform(ws)
    flat = out.X.reshape(-1, out.X.shape[-1])
    assert np.abs(flat.mean(axis=0)).max() < 1e-3
    assert np.abs(flat.std(axis=0) - 1).max() < 1e-2


def test_normalizer_fit_on_train_only(labelled):
    """
    Fitting on pooled data leaks test information into both the model
    and the conformal calibration scores.
    """
    split = make_standard_split(labelled)
    tr = make_windows(split.apply(labelled, "train"), stride=9)
    te = make_windows(split.apply(labelled, "test"), stride=9)

    n = Normalizer.fit(tr)
    te_n = n.transform(te)
    flat = te_n.X.reshape(-1, te_n.X.shape[-1])
    # test set is standardised by train stats, so it need not be exactly
    # zero-mean -- that difference is the point
    assert np.isfinite(flat).all()
    assert not np.allclose(flat.mean(axis=0), 0, atol=1e-6)


def test_normalizer_preserves_labels_and_provenance(ws):
    out = Normalizer.fit(ws).transform(ws)
    np.testing.assert_array_equal(out.y, ws.y)
    np.testing.assert_array_equal(out.drive_idx, ws.drive_idx)
    assert out.drives == ws.drives


def test_normalizer_rejects_feature_mismatch(ws, labelled):
    n = Normalizer.fit(ws)
    other = make_windows(labelled, features=COMMON_COLS[:4], stride=9)
    with pytest.raises(ValueError, match="feature mismatch"):
        n.transform(other)


def test_constant_feature_does_not_divide_by_zero(labelled):
    df = labelled.with_columns(pl.lit(1.0).alias("n_5"))
    ws = make_windows(df, stride=11)
    out = Normalizer.fit(ws).transform(ws)
    assert np.isfinite(out.X).all()


# ---------------------------------------------------------------
# Flattening — for the Random Forest baseline
# ---------------------------------------------------------------

def test_flatten_last_takes_final_timestep(ws):
    X, names = flatten(ws, "last")
    assert X.shape == (len(ws), len(ws.features))
    assert names == ws.features
    np.testing.assert_array_equal(X, ws.X[:, -1, :])


def test_flatten_full_concatenates_timesteps(ws):
    X, names = flatten(ws, "full")
    assert X.shape == (len(ws), ws.window_len * len(ws.features))
    assert len(names) == X.shape[1]


def test_flatten_stats_shape(ws):
    X, names = flatten(ws, "stats")
    assert X.shape == (len(ws), 6 * len(ws.features))
    assert len(names) == X.shape[1]
    assert len(set(names)) == len(names)


def test_flatten_unknown_mode_raises(ws):
    with pytest.raises(ValueError, match="unknown how"):
        flatten(ws, "nope")


# ---------------------------------------------------------------
# Determinism and reporting
# ---------------------------------------------------------------

def test_windowing_is_deterministic(labelled):
    a = make_windows(labelled, stride=7)
    b = make_windows(labelled, stride=7)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)
    assert a.drives == b.drives


def test_summary_reports_prevalence(ws):
    s = ws.summary()
    assert s["n_windows"] == len(ws)
    assert s["n_positive"] == int(ws.y.sum())
    assert 0 < s["positive_rate"] < 1
    assert "stride" in s and "positive_stride" in s
