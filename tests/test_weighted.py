"""
Tests for weighted conformal prediction.

Two things must be true: with uniform weights it reduces EXACTLY to split
conformal, and under a known covariate shift with oracle weights it
restores coverage that split conformal loses. Estimated weights are then
checked to move coverage in the right direction.

    python3 -m pytest tests/test_weighted.py -v
"""

import numpy as np
import pytest

from src.conformal.split import (SplitConformal, conformal_quantile,
                                 coverage_report, empirical_coverage)
from src.conformal.weighted import (WeightedConformal, _auc,
                                    estimate_density_ratio,
                                    weighted_quantiles)
from tests.test_conformal import make_data, split_data


# ---------------------------------------------------------------
# A controlled covariate shift
# ---------------------------------------------------------------

def shifted_problem(n_cal, n_test, mu_cal=2.5, mu_test=0.5, sep=1.5,
                    seed=0):
    """
    P(Y|X) is shared. Only P(X) differs: calibration X ~ N(mu_cal, 1),
    test X ~ N(mu_test, 1). The decision boundary sits at x = 0.5, where
    the posterior is 0.5.

    Direction matters. Calibration is centred in the confident region
    (far from the boundary, low nonconformity scores) while test is
    centred ON the boundary (high scores). The calibration quantile is
    then too small for the test population and split conformal
    undercovers. A shift in the other direction would make split
    conformal over-cover, which is not the failure LOMO produces.

    The posterior used by the 'model' is the true one, so any coverage
    change is attributable to the shift alone.

    Oracle density ratio, closed form:
        w(x) = N(x; mu_test, 1) / N(x; mu_cal, 1)
             = exp((mu_test - mu_cal) x + (mu_cal^2 - mu_test^2) / 2)
    """
    rng = np.random.default_rng(seed)

    def posterior(x):
        return 1 / (1 + np.exp(-(sep * (x - 0.5))))  # shared P(Y|X)

    x_cal = rng.normal(mu_cal, 1, n_cal)
    x_te = rng.normal(mu_test, 1, n_test)
    y_cal = (rng.uniform(size=n_cal) < posterior(x_cal)).astype(int)
    y_te = (rng.uniform(size=n_test) < posterior(x_te)).astype(int)

    def oracle_w(x):
        return np.exp((mu_test - mu_cal) * x + (mu_cal**2 - mu_test**2) / 2)

    return (x_cal, y_cal, posterior(x_cal), oracle_w(x_cal),
            x_te, y_te, posterior(x_te), oracle_w(x_te))


# ---------------------------------------------------------------
# Weighted quantile
# ---------------------------------------------------------------

def test_uniform_weights_reduce_to_conformal_quantile():
    """The finite-sample correction falls out of the weighted formula."""
    rng = np.random.default_rng(0)
    for n in (20, 100, 1000):
        s = rng.uniform(size=n)
        q_std = conformal_quantile(s, 0.10)
        q_w = weighted_quantiles(s, np.ones(n), np.ones(5), 0.10)
        assert np.allclose(q_w, q_std), (n, q_w[0], q_std)


def test_weighted_quantile_is_per_test_point():
    rng = np.random.default_rng(1)
    s = rng.uniform(size=500)
    w_cal = np.ones(500)
    q = weighted_quantiles(s, w_cal, np.array([0.01, 1.0, 100.0]), 0.10)
    assert q.shape == (3,)
    # a heavier test weight demands more calibration mass -> higher quantile
    assert q[0] <= q[1] <= q[2]


def test_heavy_test_weight_gives_inf():
    """
    When the test point's own weight is large enough that the target
    mass cannot be reached by calibration alone, the honest answer is
    +inf.
    """
    s = np.linspace(0, 1, 50)
    q = weighted_quantiles(s, np.ones(50), np.array([1e6]), 0.10)
    assert q[0] == np.inf


def test_upweighting_high_score_points_raises_quantile():
    s = np.linspace(0, 1, 200)
    w_flat = np.ones(200)
    w_high = np.where(s > 0.7, 10.0, 1.0)
    q_flat = weighted_quantiles(s, w_flat, np.ones(1), 0.10)[0]
    q_high = weighted_quantiles(s, w_high, np.ones(1), 0.10)[0]
    assert q_high > q_flat


def test_weighted_quantile_rejects_bad_input():
    s = np.ones(5)
    with pytest.raises(ValueError):
        weighted_quantiles(s, np.ones(4), np.ones(1), 0.1)
    with pytest.raises(ValueError):
        weighted_quantiles(s, -np.ones(5), np.ones(1), 0.1)
    with pytest.raises(ValueError):
        weighted_quantiles(s, np.ones(5), np.ones(1), 1.0)


# ---------------------------------------------------------------
# Reduction to split conformal
# ---------------------------------------------------------------

@pytest.mark.parametrize("score_fn", ["lac", "aps"])
def test_uniform_weights_reproduce_split_conformal_sets(score_fn):
    _, y, p = make_data(3000, sep=1.2, seed=5)
    cp, cy, tp, _ = split_data(p, y, n_cal=1000, seed=5)

    s = SplitConformal(alpha=0.10, score_fn=score_fn, seed=5).fit(cp, cy) \
        .predict(tp)
    w = WeightedConformal(alpha=0.10, score_fn=score_fn, seed=5).fit(
        cp, cy, np.ones(len(cy))).predict(tp, np.ones(len(tp)))

    np.testing.assert_array_equal(s.sets, w.sets)
    assert np.allclose(w.qhat, s.qhat)


def test_scaling_all_weights_changes_nothing():
    _, y, p = make_data(2000, seed=6)
    cp, cy, tp, _ = split_data(p, y, n_cal=800, seed=6)
    rng = np.random.default_rng(6)
    w_cal = rng.uniform(0.5, 2.0, len(cy))
    w_te = rng.uniform(0.5, 2.0, len(tp))

    a = WeightedConformal(seed=6).fit(cp, cy, w_cal).predict(tp, w_te)
    b = WeightedConformal(seed=6).fit(cp, cy, 7 * w_cal).predict(tp, 7 * w_te)
    np.testing.assert_array_equal(a.sets, b.sets)


# ---------------------------------------------------------------
# THE POINT — coverage under covariate shift
# ---------------------------------------------------------------

def test_split_conformal_loses_coverage_under_shift():
    """Establishes the problem. If this passes trivially, the shift is too mild."""
    covs = []
    for seed in range(30):
        xc, yc, pc, _, xt, yt, pt, _ = shifted_problem(2000, 2000, seed=seed)
        r = SplitConformal(alpha=0.10, seed=seed).fit(pc, yc).predict(pt)
        covs.append(empirical_coverage(r, yt))
    assert np.mean(covs) < 0.87, np.mean(covs)


def test_oracle_weights_restore_coverage():
    """
    With the true density ratio, weighted conformal is exactly valid
    under covariate shift. This is Tibshirani et al.'s theorem, checked.
    """
    covs = []
    for seed in range(30):
        xc, yc, pc, wc, xt, yt, pt, wt = shifted_problem(2000, 2000, seed=seed)
        r = WeightedConformal(alpha=0.10, seed=seed).fit(pc, yc, wc) \
            .predict(pt, wt)
        covs.append(empirical_coverage(r, yt))
    assert np.mean(covs) >= 0.89, np.mean(covs)


def test_oracle_weights_cost_set_size():
    xc, yc, pc, wc, xt, yt, pt, wt = shifted_problem(2000, 2000, seed=3)
    s = SplitConformal(alpha=0.10, seed=3).fit(pc, yc).predict(pt)
    w = WeightedConformal(alpha=0.10, seed=3).fit(pc, yc, wc).predict(pt, wt)
    assert w.set_sizes.mean() >= s.set_sizes.mean()


def test_estimated_weights_improve_coverage_under_shift():
    """
    Estimated ratios are noisier than the oracle, so this asks only that
    they move coverage toward nominal, not that they hit it exactly.
    """
    split_cov, est_cov = [], []
    for seed in range(20):
        xc, yc, pc, _, xt, yt, pt, _ = shifted_problem(3000, 3000, seed=seed)
        s = SplitConformal(alpha=0.10, seed=seed).fit(pc, yc).predict(pt)
        w = WeightedConformal(alpha=0.10, seed=seed).fit_from_features(
            pc, yc, xc[:, None], xt[:, None]).predict_from_features(pt)
        split_cov.append(empirical_coverage(s, yt))
        est_cov.append(empirical_coverage(w, yt))
    assert np.mean(est_cov) > np.mean(split_cov) + 0.02
    assert np.mean(est_cov) >= 0.86


def test_no_shift_estimated_weights_are_harmless():
    """
    When there is no shift the estimated ratios should be ~1 and
    coverage should remain at nominal, not degrade.
    """
    covs = []
    for seed in range(20):
        _, y, p = make_data(4000, sep=1.2, seed=seed)
        rng = np.random.default_rng(seed)
        x = p + rng.normal(0, 0.05, len(p))          # a proxy feature
        cp, cy, tp, ty = split_data(p, y, n_cal=1500, seed=seed)
        xc, xt = split_data(x, y, n_cal=1500, seed=seed)[0::2]
        w = WeightedConformal(alpha=0.10, seed=seed).fit_from_features(
            cp, cy, xc[:, None], xt[:, None])
        assert w.weight_info_["domain_auc"] < 0.6
        covs.append(empirical_coverage(w.predict_from_features(tp), ty))
    assert np.mean(covs) >= 0.89


# ---------------------------------------------------------------
# Density-ratio estimation
# ---------------------------------------------------------------

def test_estimated_ratio_tracks_oracle_ordering():
    """Estimated weights need not match the oracle, but must rank like it."""
    xc, _, _, wc, xt, _, _, wt = shifted_problem(3000, 3000, seed=2)
    w_cal, w_test, info = estimate_density_ratio(
        xc[:, None], xt[:, None], clip=(1e-3, 1e3))
    rho = np.corrcoef(np.log(w_cal), np.log(wc))[0, 1]
    assert rho > 0.9, rho
    assert info["domain_auc"] > 0.7


def test_no_shift_gives_ratio_near_one_and_auc_near_half():
    rng = np.random.default_rng(0)
    xc, xt = rng.normal(size=(2000, 3)), rng.normal(size=(2000, 3))
    w_cal, w_test, info = estimate_density_ratio(xc, xt)
    assert abs(np.log(w_cal).mean()) < 0.15
    assert abs(info["domain_auc"] - 0.5) < 0.05


def test_clipping_is_enforced_and_reported():
    xc, _, _, _, xt, _, _, _ = shifted_problem(2000, 2000, mu_cal=4.0, seed=4)
    lo, hi = 0.2, 5.0
    w_cal, w_test, info = estimate_density_ratio(
        xc[:, None], xt[:, None], clip=(lo, hi))
    assert w_cal.min() >= lo and w_cal.max() <= hi
    assert w_test.min() >= lo and w_test.max() <= hi
    assert info["frac_clipped_lo"] + info["frac_clipped_hi"] > 0
    assert info["clip"] == (lo, hi)


def test_effective_sample_size_reported():
    xc, _, _, _, xt, _, _, _ = shifted_problem(1000, 1000, seed=5)
    _, _, info = estimate_density_ratio(xc[:, None], xt[:, None])
    assert 0 < info["ess_cal"] <= 1000


def test_density_ratio_rejects_bad_shapes():
    with pytest.raises(ValueError):
        estimate_density_ratio(np.ones(10), np.ones(10))
    with pytest.raises(ValueError):
        estimate_density_ratio(np.ones((10, 2)), np.ones((10, 3)))


def test_custom_classifier_is_used():
    class Const:
        def fit(self, X, y): return self
        def predict_proba(self, X): return np.full(len(X), 0.5)

    xc, xt = np.zeros((50, 1)), np.ones((50, 1))
    w_cal, w_test, info = estimate_density_ratio(xc, xt, classifier=Const())
    assert np.allclose(w_cal, 1.0) and np.allclose(w_test, 1.0)


def test_auc_helper():
    assert _auc(np.array([0.1, 0.9]), np.array([0, 1])) == 1.0
    assert _auc(np.array([0.9, 0.1]), np.array([0, 1])) == 0.0
    assert abs(_auc(np.array([0.5, 0.5]), np.array([0, 1])) - 0.5) < 1e-9


# ---------------------------------------------------------------
# API
# ---------------------------------------------------------------

def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        WeightedConformal().predict(np.array([0.5]), np.array([1.0]))
    with pytest.raises(RuntimeError):
        WeightedConformal().predict_from_features(np.array([0.5]))


def test_length_mismatches_raise():
    _, y, p = make_data(100, seed=1)
    with pytest.raises(ValueError):
        WeightedConformal().fit(p, y, np.ones(99))
    m = WeightedConformal().fit(p, y, np.ones(100))
    with pytest.raises(ValueError):
        m.predict(p, np.ones(99))


def test_result_carries_per_point_qhat_and_method():
    _, y, p = make_data(500, seed=2)
    cp, cy, tp, ty = split_data(p, y, n_cal=200, seed=2)
    r = WeightedConformal(seed=2).fit(cp, cy, np.ones(200)) \
        .predict(tp, np.ones(300))
    assert r.method == "weighted"
    assert np.asarray(r.qhat).shape == (300,)
    rep = coverage_report(r, ty)
    assert np.isfinite(rep["qhat"])


def test_allow_empty_false():
    _, y, p = make_data(2000, sep=6.0, seed=3)
    cp, cy, tp, _ = split_data(p, y, n_cal=700, seed=3)
    r = WeightedConformal(seed=3, allow_empty=False).fit(
        cp, cy, np.ones(700)).predict(tp, np.ones(1300))
    assert r.is_empty.sum() == 0


def test_deterministic():
    xc, yc, pc, _, xt, _, pt, _ = shifted_problem(800, 800, seed=9)
    a = WeightedConformal(seed=1).fit_from_features(
        pc, yc, xc[:, None], xt[:, None]).predict_from_features(pt)
    b = WeightedConformal(seed=1).fit_from_features(
        pc, yc, xc[:, None], xt[:, None]).predict_from_features(pt)
    np.testing.assert_array_equal(a.sets, b.sets)
