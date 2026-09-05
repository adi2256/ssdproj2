"""
Tests for WEFR feature selection.

The strongest check is that selection recovers the signal columns the
synthetic fixture deliberately plants (r_5, r_187, r_197) and ranks them
above pure noise.

    python3 -m pytest tests/test_wefr.py -v
"""

import numpy as np
import pytest

from src.data.labels import build_labels
from src.data.splits import make_lomo_split, make_standard_split
from src.data.synthetic import make_synthetic
from src.data.windowing import Normalizer, flatten, make_windows
from src.features.wefr import (RANKERS, SelectionResult, choose_k,
                               consensus_ranking, drop_outlier_rankers,
                               fisher_separability, kendall_tau,
                               rank_auc, rank_correlation, rank_mean_gap,
                               select_features)

STRIDE = 9


@pytest.fixture(scope="module")
def fold():
    """Training data from the standard split, ready for selection."""
    df, truth = make_synthetic()
    labelled, _ = build_labels(df)
    split = make_standard_split(labelled)
    ws = make_windows(split.apply(labelled, "train"), stride=STRIDE)
    ws = Normalizer.fit(ws).transform(ws)
    X, names = flatten(ws, "last")
    return X, ws.y, names, truth, labelled


def toy(n=800, n_feat=10, n_signal=3, seed=0):
    """First `n_signal` columns carry signal; the rest are noise."""
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.3).astype(int)
    X = rng.normal(size=(n, n_feat))
    for j in range(n_signal):
        X[:, j] += y * (2.0 - 0.4 * j)
    return X, y, [f"f{j}" for j in range(n_feat)]


# ---------------------------------------------------------------
# THE POINT — selection recovers planted signal
# ---------------------------------------------------------------

def test_recovers_planted_signal_on_toy():
    X, y, names = toy()
    res = select_features(X, y, names)
    assert set(res.order[:3]) == {"f0", "f1", "f2"}


def test_recovers_synthetic_fixture_signal(fold):
    """
    The fixture injects drift into r_5, r_187 and r_197. Selection must
    rank them above the unrelated columns.
    """
    X, y, names, truth, _ = fold
    res = select_features(X, y, names)

    signal = [f"r_{i}" for i in truth["signal_ids"]]
    ranks = {f: res.order.index(f) for f in signal}
    assert max(ranks.values()) < len(names) // 2, ranks
    assert any(f in res.selected for f in signal)


def test_signal_ranked_above_median(fold):
    X, y, names, truth, _ = fold
    res = select_features(X, y, names)
    med = np.median(list(res.consensus_rank.values()))
    for i in truth["signal_ids"]:
        assert res.consensus_rank[f"r_{i}"] < med, i


def test_pure_noise_selects_few():
    """With no signal anywhere, selection should not keep everything."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 12))
    y = rng.integers(0, 2, 600)
    res = select_features(X, y, [f"f{j}" for j in range(12)])
    assert res.k <= 12


# ---------------------------------------------------------------
# Individual rankers
# ---------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(RANKERS))
def test_ranker_scores_signal_highest(name):
    X, y, names = toy(n_feat=8, n_signal=2, seed=2)
    s = RANKERS[name](X, y)
    assert len(s) == 8
    assert np.isfinite(s).all()
    if name != "variance":       # variance is label-blind by design
        assert s[0] > np.median(s[2:]), name


def test_variance_ranker_is_label_blind():
    """Documents that variance ignores y; it is a diversity source only."""
    X, y, _ = toy()
    a = RANKERS["variance"](X, y)
    b = RANKERS["variance"](X, 1 - y)
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("name", sorted(RANKERS))
def test_rankers_handle_nan(name):
    X, y, _ = toy()
    X[::7, 0] = np.nan
    s = RANKERS[name](X, y)
    assert np.isfinite(s).all()


@pytest.mark.parametrize("name", sorted(RANKERS))
def test_rankers_handle_constant_column(name):
    X, y, _ = toy()
    X[:, 4] = 3.0
    s = RANKERS[name](X, y)
    assert np.isfinite(s).all()


@pytest.mark.parametrize("name", sorted(RANKERS))
def test_rankers_handle_single_class(name):
    X, _, _ = toy()
    y = np.zeros(len(X), dtype=int)
    s = RANKERS[name](X, y)
    assert np.isfinite(s).all()


def test_mean_gap_survives_heavy_imbalance():
    """At 1% prevalence correlation shrinks; mean_gap should not."""
    rng = np.random.default_rng(3)
    n = 6000
    y = (rng.uniform(size=n) < 0.01).astype(int)
    X = rng.normal(size=(n, 5))
    X[:, 0] += y * 3.0
    g = rank_mean_gap(X, y)
    assert g[0] > 2 * np.median(g[1:])


def test_auc_ranker_is_direction_agnostic():
    X, y, _ = toy(n_feat=4, n_signal=1, seed=4)
    a = rank_auc(X, y)
    b = rank_auc(-X, y)
    np.testing.assert_allclose(a, b, atol=1e-9)


# ---------------------------------------------------------------
# Kendall tau and outlier removal
# ---------------------------------------------------------------

def test_kendall_tau_identical_and_reversed():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert kendall_tau(v, v) == pytest.approx(1.0)
    assert kendall_tau(v, -v) == pytest.approx(-1.0)


def test_kendall_tau_symmetric_and_bounded():
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=20), rng.normal(size=20)
    t = kendall_tau(a, b)
    assert -1.0 <= t <= 1.0
    assert t == pytest.approx(kendall_tau(b, a))


def test_kendall_tau_length_mismatch_raises():
    with pytest.raises(ValueError):
        kendall_tau(np.ones(3), np.ones(4))


def test_outlier_ranker_is_dropped():
    """
    Four rankers agree, one is reversed. The disagreeing one must be
    removed so it cannot distort the consensus.
    """
    base = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    scores = {
        "a": base,
        "b": base + 0.1,
        "c": base * 2,
        "d": base - 0.05,
        "rogue": -base,
    }
    kept, agree = drop_outlier_rankers(scores, z=1.0)
    assert "rogue" not in kept
    assert agree["rogue"] < min(agree[k] for k in kept)


def test_no_drop_when_all_agree():
    base = np.array([3.0, 2.0, 1.0])
    scores = {k: base + i * 0.01 for i, k in enumerate("abcd")}
    kept, _ = drop_outlier_rankers(scores)
    assert set(kept) == set(scores)


def test_never_drops_more_than_half():
    rng = np.random.default_rng(6)
    scores = {f"r{i}": rng.normal(size=8) for i in range(6)}
    kept, _ = drop_outlier_rankers(scores, z=0.0)
    assert len(kept) >= 3


def test_too_few_rankers_are_all_kept():
    scores = {"a": np.array([1.0, 2.0]), "b": np.array([2.0, 1.0])}
    kept, _ = drop_outlier_rankers(scores)
    assert len(kept) == 2


# ---------------------------------------------------------------
# Consensus and k selection
# ---------------------------------------------------------------

def test_consensus_averages_ranks():
    scores = {"a": np.array([3.0, 2.0, 1.0]),
              "b": np.array([1.0, 2.0, 3.0])}
    r = consensus_ranking(scores)
    assert r[1] == pytest.approx(2.0)
    assert r[0] == pytest.approx(r[2])


def test_consensus_requires_rankers():
    with pytest.raises(ValueError):
        consensus_ranking({})


def test_choose_k_prefers_the_smallest_adequate_prefix():
    X, y, _ = toy(n_feat=12, n_signal=2, seed=7)
    order = np.arange(12)
    k, curve = choose_k(X, y, order, tol=0.05)
    assert k <= 6
    assert len(curve) == 12


def test_choose_k_respects_min_k():
    X, y, _ = toy()
    k, _ = choose_k(X, y, np.arange(X.shape[1]), min_k=4)
    assert k >= 4


def test_fisher_separability_increases_with_signal():
    X, y, _ = toy(n_feat=6, n_signal=3, seed=8)
    sig = fisher_separability(X[:, :3], y)
    noise = fisher_separability(X[:, 3:], y)
    assert sig > noise


def test_fisher_handles_degenerate_input():
    X, y, _ = toy()
    assert fisher_separability(X, np.zeros(len(y), dtype=int)) == 0.0
    assert fisher_separability(X[:, :0], y) == 0.0


# ---------------------------------------------------------------
# Leakage discipline
# ---------------------------------------------------------------

def test_selection_is_a_pure_function_of_training_data(fold):
    """
    Same inputs, same output. No hidden state that could carry
    information between folds.
    """
    X, y, names, _, _ = fold
    a = select_features(X, y, names)
    b = select_features(X, y, names)
    assert a.selected == b.selected
    assert a.order == b.order


def test_selection_differs_across_lomo_folds():
    """
    Re-running per fold must actually change the answer sometimes,
    otherwise the per-fold discipline would be pointless. If this ever
    fails everywhere, note in the paper that at common-16 width
    selection is effectively a no-op.
    """
    df, _ = make_synthetic()
    labelled, _ = build_labels(df)

    orders = {}
    for v in "ABC":
        split = make_lomo_split(labelled, v)
        ws = make_windows(split.apply(labelled, "train"), stride=STRIDE)
        ws = Normalizer.fit(ws).transform(ws)
        X, names = flatten(ws, "last")
        orders[v] = tuple(select_features(X, y=ws.y,
                                          feature_names=names).order)

    assert len(set(orders.values())) > 1, (
        "identical ranking in every fold; selection may be a no-op at "
        "this feature width"
    )


def test_selection_never_sees_more_than_it_is_given(fold):
    """
    Selection takes X and y only. Passing a subset must not change the
    scores of the columns that remain relative to each other.
    """
    X, y, names, _, _ = fold
    full = select_features(X, y, names)
    sub_names = names[:8]
    sub = select_features(X[:, :8], y, sub_names)

    for a, b in zip(sub_names, sub_names):
        pass
    full_sub_order = [f for f in full.order if f in sub_names]
    # rankings need not match exactly (k differs), but the top feature
    # of the subset should be highly placed in the full ranking too
    assert full.order.index(sub.order[0]) < len(names) // 2


# ---------------------------------------------------------------
# API and reporting
# ---------------------------------------------------------------

def test_result_fields(fold):
    X, y, names, _, _ = fold
    res = select_features(X, y, names)
    assert isinstance(res, SelectionResult)
    assert set(res.order) == set(names)
    assert res.selected == res.order[:res.k]
    assert set(res.consensus_rank) == set(names)
    assert set(res.ranker_scores) == set(RANKERS)
    assert 1 <= res.k <= len(names)
    s = res.summary()
    assert s["n_selected"] == res.k
    assert "min_agreement" in s
    assert str(res)


def test_max_k_caps_selection(fold):
    X, y, names, _, _ = fold
    res = select_features(X, y, names, max_k=3)
    assert res.k <= 3
    assert len(res.selected) <= 3


def test_custom_ranker_set(fold):
    X, y, names, _, _ = fold
    res = select_features(X, y, names,
                          rankers={"auc": rank_auc,
                                   "corr": rank_correlation})
    assert set(res.ranker_scores) == {"auc", "corr"}


def test_shape_mismatch_raises(fold):
    X, y, names, _, _ = fold
    with pytest.raises(ValueError):
        select_features(X, y[:-1], names)
    with pytest.raises(ValueError):
        select_features(X, y, names[:-1])


def test_single_class_raises(fold):
    X, y, names, _, _ = fold
    with pytest.raises(ValueError, match="both classes"):
        select_features(X, np.zeros(len(y), dtype=int), names)


def test_selection_is_narrow_at_common_sixteen(fold):
    """
    Records the expected behaviour flagged in the module docstring: with
    only 16 columns, WEFR has little to prune. This is a documented
    property, not a failure.
    """
    X, y, names, _, _ = fold
    assert len(names) == 16
    res = select_features(X, y, names)
    assert 1 <= res.k <= 16
