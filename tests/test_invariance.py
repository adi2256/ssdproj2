"""
Tests for vendor-invariant feature representations.

Every experiment before this used flatten(ws, "last") -- the final
timestep's raw LEVEL. Levels are vendor-specific, so a model fitted on
them learns vendor-specific split points: AUC 0.73 within-fleet,
0.42-0.65 across manufacturers on the real data.

These tests pin two representations that remove level:

  flatten(ws, "dynamics")        slope, delta, volatility, change rate
  Normalizer(method="per_drive") scale each window against its own start

    python3 -m pytest tests/test_invariance.py -v
"""

import numpy as np
import pytest

from src.data.labels import build_labels
from src.data.splits import make_lomo_split
from src.data.synthetic import make_synthetic
from src.data.windowing import Normalizer, WindowSet, flatten, make_windows

W, F = 30, 3


def make_ws(X, vendors=None, y=None):
    n = len(X)
    return WindowSet(
        X=np.asarray(X, dtype=np.float32),
        y=(np.asarray(y, dtype=np.int8) if y is not None
           else np.zeros(n, dtype=np.int8)),
        drive_idx=np.arange(n),
        end_ds=np.zeros(n, dtype=object),
        drives=[(f"M{i}", i) for i in range(n)],
        vendors=(np.asarray(vendors, dtype=object) if vendors is not None
                 else np.array(["A"] * n, dtype=object)),
        features=[f"f{j}" for j in range(X.shape[2])],
        window_len=X.shape[1], stride=1, positive_stride=1,
    )


def two_encodings(n=40, scale=100.0, offset=5000.0, seed=0):
    """
    The same drives, encoded the way two different vendors would report
    them: one raw, one scaled and offset. Any representation claiming
    vendor invariance must map these to (nearly) the same thing.
    """
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.exponential(0.5, size=(n, W, F)), axis=1)
    X = np.concatenate([base, base * scale + offset]).astype(np.float32)
    vendors = np.array(["A"] * n + ["B"] * n, dtype=object)
    return make_ws(X, vendors), n


# ---------------------------------------------------------------
# THE POINT — invariance to vendor encoding
# ---------------------------------------------------------------

@pytest.mark.parametrize("method", ["rank", "per_drive"])
def test_normalizer_is_invariant_to_vendor_encoding(method):
    ws, n = two_encodings()
    out = Normalizer.fit(ws, method=method).transform(ws)
    gap = float(np.abs(out.X[:n] - out.X[n:]).mean())
    assert gap < 0.01, f"{method} left a gap of {gap}"


def test_zscore_is_not_invariant():
    """
    The baseline this work is replacing. Documented so the contrast is
    explicit rather than assumed.
    """
    ws, n = two_encodings()
    out = Normalizer.fit(ws, method="zscore").transform(ws)
    assert float(np.abs(out.X[:n] - out.X[n:]).mean()) > 1.0


def test_dynamics_under_per_drive_is_invariant():
    ws, n = two_encodings()
    out = Normalizer.fit(ws, method="per_drive").transform(ws)
    X, _ = flatten(out, "dynamics_only")
    assert float(np.abs(X[:n] - X[n:]).mean()) < 0.01


def test_per_drive_needs_no_training_statistics():
    """
    Scaling uses only each window's own opening days, so a test window's
    representation cannot depend on training data. That makes leakage
    structurally impossible for this method.
    """
    ws, n = two_encodings()
    a = Normalizer.fit(ws, method="per_drive").transform(ws)

    other = make_ws(ws.X * 3.0 + 17.0, ws.vendors)
    b = Normalizer.fit(other, method="per_drive").transform(ws)

    np.testing.assert_allclose(a.X, b.X, atol=1e-5)


# ---------------------------------------------------------------
# Dynamics features carry trend
# ---------------------------------------------------------------

def test_slope_separates_rising_from_flat():
    rng = np.random.default_rng(1)
    flat = rng.normal(size=(30, W, F))
    rising = flat + np.linspace(0, 5, W)[None, :, None]
    ws = make_ws(np.concatenate([rising, flat]),
                 y=np.array([1] * 30 + [0] * 30))

    X, names = flatten(ws, "dynamics")
    j = names.index("f0_slope")
    assert X[:30, j].mean() > X[30:, j].mean() + 0.1


def test_dynamics_detects_trend_that_last_timestep_misses():
    """
    Two drives end at the same value; one climbed to get there, one sat
    still. "last" cannot tell them apart. This is the case the whole
    change is motivated by.
    """
    rising = np.zeros((20, W, F), dtype=np.float32)
    rising += np.linspace(0, 10, W)[None, :, None]
    flat = np.full((20, W, F), 10.0, dtype=np.float32)
    ws = make_ws(np.concatenate([rising, flat]),
                 y=np.array([1] * 20 + [0] * 20))

    X_last, _ = flatten(ws, "last")
    assert np.allclose(X_last[:20], X_last[20:], atol=1e-4)

    X_dyn, names = flatten(ws, "dynamics")
    j = names.index("f0_slope")
    assert X_dyn[:20, j].mean() > X_dyn[20:, j].mean() + 0.1


def test_n_changes_is_scale_free():
    """
    SMART counters are step functions. How often a counter ticks is
    informative and survives any monotone rescaling, so it should be
    identical across vendor encodings.
    """
    ws, n = two_encodings()
    X, names = flatten(ws, "dynamics")
    j = names.index("f0_n_changes")
    np.testing.assert_allclose(X[:n, j], X[n:, j], atol=1e-6)


def test_rel_delta_is_scale_free_under_pure_scaling():
    rng = np.random.default_rng(2)
    base = np.cumsum(rng.exponential(1.0, size=(20, W, F)), axis=1) + 100
    ws = make_ws(np.concatenate([base, base * 50]).astype(np.float32),
                 np.array(["A"] * 20 + ["B"] * 20, dtype=object))
    X, names = flatten(ws, "dynamics")
    j = names.index("f0_rel_delta")
    # not exact -- the +1 in the denominator matters less as values grow
    assert abs(X[:20, j].mean() - X[20:, j].mean()) < 0.5


# ---------------------------------------------------------------
# Shape and hygiene
# ---------------------------------------------------------------

def test_dynamics_shape_and_names():
    ws, _ = two_encodings()
    X, names = flatten(ws, "dynamics")
    assert X.shape == (len(ws), 7 * F)
    assert len(names) == X.shape[1] == len(set(names))
    for suffix in ("slope", "delta", "rel_delta", "std",
                   "max_jump", "n_changes", "last"):
        assert any(n.endswith(f"_{suffix}") for n in names), suffix


def test_dynamics_only_drops_level():
    ws, _ = two_encodings()
    X, names = flatten(ws, "dynamics_only")
    assert X.shape == (len(ws), 6 * F)
    assert not any(n.endswith("_last") for n in names)


@pytest.mark.parametrize("how", ["dynamics", "dynamics_only"])
def test_dynamics_output_is_finite(how):
    ws, _ = two_encodings()
    ws.X[::7, :, 0] = 0.0                 # constant column, MAD zero
    ws.X[1::11, :, 1] = 1e12              # extreme magnitude
    X, _ = flatten(ws, how)
    assert np.isfinite(X).all()


def test_per_drive_handles_constant_window():
    ws = make_ws(np.ones((10, W, F), dtype=np.float32))
    out = Normalizer.fit(ws, method="per_drive").transform(ws)
    assert np.isfinite(out.X).all()
    assert np.allclose(out.X, 0.0)


def test_single_timestep_window_does_not_crash():
    ws = make_ws(np.ones((5, 1, F), dtype=np.float32))
    X, _ = flatten(ws, "dynamics")
    assert X.shape == (5, 7 * F)
    assert np.isfinite(X).all()


def test_unknown_method_raises():
    ws, _ = two_encodings()
    with pytest.raises(ValueError):
        Normalizer.fit(ws, method="nope")


def test_transform_preserves_labels_and_provenance():
    ws, _ = two_encodings()
    out = Normalizer.fit(ws, method="per_drive").transform(ws)
    np.testing.assert_array_equal(out.y, ws.y)
    np.testing.assert_array_equal(out.drive_idx, ws.drive_idx)
    assert out.drives == ws.drives
    assert out.features == ws.features


# ---------------------------------------------------------------
# End to end on the fixture
# ---------------------------------------------------------------

@pytest.mark.parametrize("method", ["zscore", "rank", "per_drive"])
@pytest.mark.parametrize("how", ["last", "dynamics", "dynamics_only"])
def test_combinations_run_on_real_pipeline(method, how):
    df, _ = make_synthetic(n_drives=120, seed=3)
    labelled, _ = build_labels(df)
    split = make_lomo_split(labelled, "C")

    tr = make_windows(split.apply(labelled, "train"), stride=15)
    te = make_windows(split.apply(labelled, "test"), stride=15)
    if len(tr) == 0 or len(te) == 0:
        pytest.skip("fixture too small for this stride")

    norm = Normalizer.fit(tr, method=method)
    Xtr, names = flatten(norm.transform(tr), how)
    Xte, _ = flatten(norm.transform(te), how)

    assert Xtr.shape[1] == Xte.shape[1] == len(names)
    assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()
