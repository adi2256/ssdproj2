"""
Validation of the split conformal implementation.

These tests are the deliverable, not the code. They establish that
coverage holds under exchangeability, so that undercoverage observed
later under LOMO can be attributed to distribution shift rather than to
a bug here.

Uses a synthetic classifier with tunable accuracy — this tests the
conformal wrapper, not any predictor.

    python3 -m pytest tests/test_conformal.py -v
"""

import numpy as np
import pytest

from src.conformal.scores import (aps_candidate_scores, aps_scores,
                                  lac_candidate_scores, lac_scores)
from src.conformal.split import (FAILURE, HEALTHY, SplitConformal,
                                 class_conditional_coverage,
                                 conformal_quantile, coverage_report,
                                 empirical_coverage)

SCORE_FNS = ["lac", "aps"]


# ---------------------------------------------------------------
# Synthetic classifier
# ---------------------------------------------------------------

def make_data(n, sep=2.0, prevalence=0.5, seed=0):
    """
    Two Gaussians separated by `sep`, with a Bayes-optimal posterior.

    Larger `sep` -> more separable -> smaller conformal sets.
    Because the posterior is exact, any coverage failure is the
    conformal layer's fault, not the classifier's.
    """
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < prevalence).astype(int)
    x = rng.normal(loc=y * sep, scale=1.0)

    log_ratio = (sep * x - sep**2 / 2) + np.log(
        prevalence / (1 - prevalence)
    )
    p1 = 1.0 / (1.0 + np.exp(-log_ratio))
    return x, y, p1


def split_data(probs, labels, n_cal, seed=0):
    """Random exchangeable calibration/test split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    cal, test = idx[:n_cal], idx[n_cal:]
    return probs[cal], labels[cal], probs[test], labels[test]


# ---------------------------------------------------------------
# The finite-sample quantile
# ---------------------------------------------------------------

def test_quantile_uses_n_plus_one_correction():
    """
    qhat must be the ceil((n+1)(1-alpha))/n empirical quantile. The
    plain (1-alpha) quantile undercovers by ~1/n, which on a small
    calibration set is indistinguishable from a mild shift effect.
    """
    scores = np.arange(1, 101, dtype=float)  # 1..100
    qhat = conformal_quantile(scores, alpha=0.10)
    # ceil(101 * 0.9) = 91 -> the 91st smallest score
    assert qhat == 91.0

    naive = float(np.quantile(scores, 0.90))
    assert qhat > naive, "correction must be strictly more conservative"


def test_quantile_returns_inf_when_n_too_small():
    """n < 1/alpha - 1 cannot support the level. Say so rather than lie."""
    assert conformal_quantile(np.arange(5.0), alpha=0.01) == float("inf")


def test_quantile_rejects_bad_input():
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), 0.1)
    with pytest.raises(ValueError):
        conformal_quantile(np.arange(10.0), alpha=0.0)
    with pytest.raises(ValueError):
        conformal_quantile(np.arange(10.0), alpha=1.0)


# ---------------------------------------------------------------
# THE CORE GUARANTEE
# ---------------------------------------------------------------

@pytest.mark.parametrize("score_fn", SCORE_FNS)
@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_marginal_coverage_under_exchangeability(score_fn, alpha):
    """
    The headline check. Averaged over repeated exchangeable splits,
    empirical coverage must reach 1 - alpha.

    If this fails, nothing downstream is interpretable.
    """
    covs = []
    for seed in range(40):
        _, y, p = make_data(4000, sep=1.5, seed=seed)
        cp, cy, tp, ty = split_data(p, y, n_cal=1000, seed=seed)

        cp_model = SplitConformal(alpha=alpha, score_fn=score_fn,
                                  seed=seed).fit(cp, cy)
        covs.append(empirical_coverage(cp_model.predict(tp), ty))

    mean_cov = float(np.mean(covs))
    assert mean_cov >= 1 - alpha - 0.01, (
        f"{score_fn} undercovers at alpha={alpha}: {mean_cov:.4f}"
    )
    assert mean_cov <= 1 - alpha + 0.06, (
        f"{score_fn} wildly over-covers at alpha={alpha}: {mean_cov:.4f}"
    )


@pytest.mark.parametrize("score_fn", SCORE_FNS)
def test_coverage_holds_on_tiny_calibration_set(score_fn):
    """
    n=50 is where a missing (n+1) correction shows up. Without it the
    shortfall is ~2% — small enough to be mistaken for a shift effect.
    """
    covs = []
    for seed in range(120):
        _, y, p = make_data(1200, sep=1.5, seed=seed)
        cp, cy, tp, ty = split_data(p, y, n_cal=50, seed=seed)
        m = SplitConformal(alpha=0.10, score_fn=score_fn,
                           seed=seed).fit(cp, cy)
        covs.append(empirical_coverage(m.predict(tp), ty))

    assert float(np.mean(covs)) >= 0.89


@pytest.mark.parametrize("score_fn", SCORE_FNS)
def test_coverage_independent_of_model_quality(score_fn):
    """
    Coverage is distribution-free: it must hold for a good model and a
    near-useless one. Only set size should change.
    """
    for sep in [0.2, 1.0, 3.0]:
        covs, sizes = [], []
        for seed in range(30):
            _, y, p = make_data(3000, sep=sep, seed=seed)
            cp, cy, tp, ty = split_data(p, y, n_cal=800, seed=seed)
            m = SplitConformal(alpha=0.10, score_fn=score_fn,
                               seed=seed).fit(cp, cy)
            r = m.predict(tp)
            covs.append(empirical_coverage(r, ty))
            sizes.append(r.set_sizes.mean())

        assert float(np.mean(covs)) >= 0.89, f"sep={sep} undercovers"
        if sep == 0.2:
            weak_size = float(np.mean(sizes))
        elif sep == 3.0:
            assert float(np.mean(sizes)) < weak_size, (
                "a better model must yield smaller sets"
            )


def test_coverage_holds_under_class_imbalance():
    """
    Marginal coverage must hold at the real fleet's ~1% failure rate.
    Class-conditional coverage is checked separately below.
    """
    covs = []
    for seed in range(40):
        _, y, p = make_data(8000, sep=2.0, prevalence=0.01, seed=seed)
        cp, cy, tp, ty = split_data(p, y, n_cal=2000, seed=seed)
        m = SplitConformal(alpha=0.10, seed=seed).fit(cp, cy)
        covs.append(empirical_coverage(m.predict(tp), ty))

    assert float(np.mean(covs)) >= 0.89


def test_lac_can_undercover_minority_class():
    """
    Documents a known LAC property rather than asserting correctness:
    marginal coverage can sit at nominal while failure-class coverage
    falls below it. This is why class-conditional coverage must be
    reported, and it motivates the Mondrian variant.
    """
    marg, fail_cov = [], []
    for seed in range(40):
        _, y, p = make_data(8000, sep=1.0, prevalence=0.02, seed=seed)
        cp, cy, tp, ty = split_data(p, y, n_cal=2000, seed=seed)
        r = SplitConformal(alpha=0.10, seed=seed).fit(cp, cy).predict(tp)
        marg.append(empirical_coverage(r, ty))
        cc = class_conditional_coverage(r, ty)
        if not np.isnan(cc[FAILURE]):
            fail_cov.append(cc[FAILURE])

    assert float(np.mean(marg)) >= 0.89
    assert len(fail_cov) > 0
    # No assertion on the gap's size — it is a measurement, not a target.


# ---------------------------------------------------------------
# Set behaviour
# ---------------------------------------------------------------

@pytest.mark.parametrize("score_fn", SCORE_FNS)
def test_lower_alpha_gives_larger_sets(score_fn):
    _, y, p = make_data(4000, sep=1.2, seed=11)
    cp, cy, tp, _ = split_data(p, y, n_cal=1000, seed=11)

    sizes = []
    for a in [0.30, 0.20, 0.10, 0.05]:
        m = SplitConformal(alpha=a, score_fn=score_fn, seed=11).fit(cp, cy)
        sizes.append(m.predict(tp).set_sizes.mean())

    assert all(sizes[i] <= sizes[i + 1] + 1e-9
               for i in range(len(sizes) - 1)), sizes


def test_perfect_classifier_never_yields_doubletons():
    """
    With a near-perfect model, LAC emits singletons and some EMPTY sets
    (both labels surprising) but never doubletons. Empty sets are
    correct LAC behaviour, not a bug — coverage is still met by the
    singletons.
    """
    _, y, p = make_data(4000, sep=8.0, seed=5)
    cp, cy, tp, ty = split_data(p, y, n_cal=1000, seed=5)
    r = SplitConformal(alpha=0.10, seed=5).fit(cp, cy).predict(tp)

    assert r.is_doubleton.mean() == 0.0
    assert r.is_singleton.mean() + r.is_empty.mean() == 1.0
    assert empirical_coverage(r, ty) >= 0.89


def test_allow_empty_false_removes_empty_sets():
    """
    An empty set is not actionable for an operator. allow_empty=False
    forces the argmax label in, which is conservative — coverage can
    only increase.
    """
    _, y, p = make_data(4000, sep=8.0, seed=5)
    cp, cy, tp, ty = split_data(p, y, n_cal=1000, seed=5)

    strict = SplitConformal(alpha=0.10, seed=5,
                            allow_empty=True).fit(cp, cy).predict(tp)
    forced = SplitConformal(alpha=0.10, seed=5,
                            allow_empty=False).fit(cp, cy).predict(tp)

    assert strict.is_empty.mean() > 0.0
    assert forced.is_empty.mean() == 0.0
    assert forced.is_singleton.mean() == 1.0
    assert (empirical_coverage(forced, ty)
            >= empirical_coverage(strict, ty))


def test_uninformative_classifier_yields_doubletons():
    """
    A model with no signal must say 'I don't know' rather than guess.
    This is the behaviour a point score cannot express.
    """
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=4000)
    p = np.full(4000, 0.5)
    cp, cy, tp, ty = split_data(p, y, n_cal=1000, seed=3)

    r = SplitConformal(alpha=0.10, seed=3).fit(cp, cy).predict(tp)
    assert r.is_doubleton.mean() > 0.95
    assert empirical_coverage(r, ty) >= 0.89


def test_doubleton_contains_both_labels():
    _, y, p = make_data(2000, sep=0.3, seed=7)
    cp, cy, tp, _ = split_data(p, y, n_cal=600, seed=7)
    r = SplitConformal(alpha=0.05, seed=7).fit(cp, cy).predict(tp)

    for row in np.asarray(r.as_labels(), dtype=object)[r.is_doubleton][:20]:
        assert set(row) == {HEALTHY, FAILURE}


def test_infinite_qhat_yields_full_sets():
    _, y, p = make_data(200, sep=1.0, seed=9)
    cp, cy, tp, ty = split_data(p, y, n_cal=5, seed=9)
    r = SplitConformal(alpha=0.01, seed=9).fit(cp, cy).predict(tp)

    assert r.qhat == float("inf")
    assert r.is_doubleton.all()
    assert empirical_coverage(r, ty) == 1.0


# ---------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------

def test_lac_score_is_one_minus_true_prob():
    p = np.array([0.9, 0.2, 0.5])
    y = np.array([1, 0, 1])
    np.testing.assert_allclose(lac_scores(p, y), [0.1, 0.2, 0.5])


def test_candidate_scores_match_true_label_scores():
    """cand[i, y_i] must equal the score computed for the true label."""
    _, y, p = make_data(500, seed=13)
    cand = lac_candidate_scores(p)
    np.testing.assert_allclose(
        cand[np.arange(len(y)), y], lac_scores(p, y)
    )

    cand_aps = aps_candidate_scores(p, randomized=False)
    s_aps = aps_scores(p, y, randomized=False)
    np.testing.assert_allclose(cand_aps[np.arange(len(y)), y], s_aps)


def test_aps_scores_are_bounded():
    _, y, p = make_data(500, seed=17)
    s = aps_scores(p, y, rng=np.random.default_rng(0))
    assert (s >= -1e-9).all() and (s <= 1 + 1e-9).all()


def test_scores_reject_invalid_probs():
    with pytest.raises(ValueError):
        lac_scores(np.array([1.5]), np.array([1]))
    with pytest.raises(ValueError):
        lac_scores(np.array([0.5]), np.array([2]))


# ---------------------------------------------------------------
# API
# ---------------------------------------------------------------

def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        SplitConformal().predict(np.array([0.5]))


def test_unknown_score_fn_raises():
    with pytest.raises(ValueError):
        SplitConformal(score_fn="nope")


def test_deterministic_for_fixed_seed():
    _, y, p = make_data(1000, seed=21)
    cp, cy, tp, _ = split_data(p, y, n_cal=300, seed=21)

    a = SplitConformal(alpha=0.1, score_fn="aps", seed=1).fit(cp, cy)
    b = SplitConformal(alpha=0.1, score_fn="aps", seed=1).fit(cp, cy)
    assert a.qhat_ == b.qhat_
    np.testing.assert_array_equal(a.predict(tp).sets, b.predict(tp).sets)


def test_coverage_report_shape():
    """One row of the LOMO results table."""
    _, y, p = make_data(3000, sep=1.5, prevalence=0.1, seed=23)
    cp, cy, tp, ty = split_data(p, y, n_cal=800, seed=23)
    r = SplitConformal(alpha=0.10, seed=23).fit(cp, cy).predict(tp)

    rep = coverage_report(r, ty)
    for k in ["alpha", "target_coverage", "empirical_coverage",
              "coverage_healthy", "coverage_failure", "avg_set_size",
              "singleton_rate", "doubleton_rate", "empty_rate",
              "qhat", "n_cal", "n_test", "n_failures_test", "score_fn"]:
        assert k in rep, k

    assert rep["target_coverage"] == 0.90
    assert rep["n_cal"] == 800
    assert 0.0 <= rep["empirical_coverage"] <= 1.0
    assert abs(
        rep["singleton_rate"] + rep["doubleton_rate"]
        + rep["empty_rate"] - 1.0
    ) < 1e-9


# ---------------------------------------------------------------
# Shift — the reason this project exists
# ---------------------------------------------------------------

def test_coverage_degrades_under_covariate_shift():
    """
    Sanity check that the harness can DETECT shift, using a synthetic
    shift where the mechanism is known.

    Calibrate on one distribution, test on another. Coverage should fall
    below nominal. This is the effect LOMO is meant to surface with a
    real manufacturer boundary; if this test ever stops failing to
    cover, the LOMO experiment would have no measurable signal.
    """
    _, y_cal, p_cal = make_data(3000, sep=2.0, prevalence=0.5, seed=31)

    # Shifted regime: different separation and prevalence, and the
    # calibration model's posterior is now miscalibrated for it.
    rng = np.random.default_rng(32)
    n = 3000
    y_test = (rng.uniform(size=n) < 0.3).astype(int)
    x_test = rng.normal(loc=y_test * 0.5, scale=1.0)
    log_ratio = (2.0 * x_test - 2.0**2 / 2)  # stale model
    p_test = 1.0 / (1.0 + np.exp(-log_ratio))

    m = SplitConformal(alpha=0.10, seed=31).fit(p_cal, y_cal)
    shifted = empirical_coverage(m.predict(p_test), y_test)

    _, y_iid, p_iid = make_data(3000, sep=2.0, prevalence=0.5, seed=33)
    iid = empirical_coverage(m.predict(p_iid), y_iid)

    assert iid >= 0.88, f"exchangeable case broken: {iid:.3f}"
    assert shifted < iid, (
        f"shift not detected: shifted={shifted:.3f} iid={iid:.3f}"
    )
