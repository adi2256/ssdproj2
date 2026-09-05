"""
Tests for Mondrian conformal prediction.

The headline check: under heavy class imbalance, split conformal covers
the failure class at ~15% while Mondrian-by-class restores it to nominal.
That is the measurement that made this module required.

    python3 -m pytest tests/test_mondrian.py -v
"""

import numpy as np
import pytest

from src.conformal.mondrian import MondrianConformal, group_conditional_coverage
from src.conformal.split import (FAILURE, HEALTHY, SplitConformal,
                                 class_conditional_coverage,
                                 coverage_report, empirical_coverage)
from tests.test_conformal import make_data, split_data


def make_grouped(n, seed, group_seps=(1.0, 2.0, 3.0),
                 group_prev=(0.10, 0.20, 0.35), group_share=(0.4, 0.3, 0.3)):
    """
    Three groups with different separability and prevalence, mirroring
    the real fleet: vendors differ in both P(X) and P(Y). The model's
    posterior is computed per group so it is correct within each -- any
    coverage failure is the conformal layer's.
    """
    rng = np.random.default_rng(seed)
    g = rng.choice(3, size=n, p=group_share)
    y = np.zeros(n, dtype=int)
    p = np.zeros(n)
    for k in range(3):
        m = g == k
        sep, prev = group_seps[k], group_prev[k]
        yk = (rng.uniform(size=m.sum()) < prev).astype(int)
        xk = rng.normal(loc=yk * sep, scale=1.0)
        lr = sep * xk - sep**2 / 2 + np.log(prev / (1 - prev))
        y[m] = yk
        p[m] = 1 / (1 + np.exp(-lr))
    groups = np.array(["A", "B", "C"])[g]
    return y, p, groups


# ---------------------------------------------------------------
# THE POINT — class-conditional coverage under imbalance
# ---------------------------------------------------------------

def test_split_undercovers_minority_class_at_low_prevalence():
    """Establishes the problem Mondrian exists to fix."""
    fail_cov = []
    for seed in range(30):
        _, y, p = make_data(8000, sep=1.0, prevalence=0.03, seed=seed)
        cp, cy, tp, ty = split_data(p, y, n_cal=2000, seed=seed)
        r = SplitConformal(alpha=0.10, seed=seed).fit(cp, cy).predict(tp)
        cc = class_conditional_coverage(r, ty)
        if not np.isnan(cc[FAILURE]):
            fail_cov.append(cc[FAILURE])
    assert np.mean(fail_cov) < 0.6, (
        "split conformal covers the minority class fine here; the "
        "fixture no longer demonstrates the problem"
    )


@pytest.mark.parametrize("score_fn", ["lac", "aps"])
@pytest.mark.parametrize("prevalence", [0.03, 0.10, 0.30])
def test_mondrian_class_restores_per_class_coverage(score_fn, prevalence):
    """
    Both classes must reach 1 - alpha, at every prevalence, for both
    score functions. This is the class-conditional guarantee.
    """
    covs = {HEALTHY: [], FAILURE: []}
    for seed in range(30):
        _, y, p = make_data(8000, sep=1.0, prevalence=prevalence, seed=seed)
        cp, cy, tp, ty = split_data(p, y, n_cal=2000, seed=seed)
        r = MondrianConformal(alpha=0.10, score_fn=score_fn, by="class",
                              seed=seed).fit(cp, cy).predict(tp)
        cc = class_conditional_coverage(r, ty)
        for k in (HEALTHY, FAILURE):
            if not np.isnan(cc[k]):
                covs[k].append(cc[k])

    for k in (HEALTHY, FAILURE):
        m = float(np.mean(covs[k]))
        assert m >= 0.89, f"class {k} at prev={prevalence}: {m:.4f}"


def test_mondrian_class_marginal_coverage_also_holds():
    """Per-class coverage implies marginal coverage."""
    covs = []
    for seed in range(20):
        _, y, p = make_data(6000, sep=1.2, prevalence=0.1, seed=seed)
        cp, cy, tp, ty = split_data(p, y, n_cal=1500, seed=seed)
        r = MondrianConformal(alpha=0.10, seed=seed).fit(cp, cy).predict(tp)
        covs.append(empirical_coverage(r, ty))
    assert np.mean(covs) >= 0.89


def test_mondrian_class_costs_set_size():
    """
    The stronger guarantee is paid for in larger sets. Report this;
    do not hide it.
    """
    _, y, p = make_data(8000, sep=1.0, prevalence=0.05, seed=3)
    cp, cy, tp, _ = split_data(p, y, n_cal=2000, seed=3)

    s = SplitConformal(alpha=0.10, seed=3).fit(cp, cy).predict(tp)
    m = MondrianConformal(alpha=0.10, seed=3).fit(cp, cy).predict(tp)
    assert m.set_sizes.mean() > s.set_sizes.mean()


def test_mondrian_class_thresholds_differ_per_class():
    _, y, p = make_data(4000, sep=1.0, prevalence=0.1, seed=5)
    cp, cy, _, _ = split_data(p, y, n_cal=1500, seed=5)
    m = MondrianConformal(alpha=0.10, seed=5).fit(cp, cy)
    assert set(m.qhat_) == {HEALTHY, FAILURE}
    assert m.qhat_[HEALTHY] != m.qhat_[FAILURE]
    assert m.cell_sizes_[HEALTHY] > m.cell_sizes_[FAILURE]


def test_mondrian_class_thin_minority_gives_inf_and_full_coverage():
    """
    Too few positives to support the level -> qhat_failure = inf -> the
    failure label is always included. Honest, and coverage is 1.0 for
    that class.
    """
    rng = np.random.default_rng(0)
    n = 300
    y = np.zeros(n, dtype=int)
    y[:4] = 1                       # 4 positives, alpha=0.1 needs >= 9
    p = rng.uniform(0.05, 0.95, n)
    m = MondrianConformal(alpha=0.10, seed=0).fit(p, y)
    assert m.qhat_[FAILURE] == float("inf")

    r = m.predict(rng.uniform(size=200))
    assert r.sets[:, FAILURE].all()


# ---------------------------------------------------------------
# Group-conditional
# ---------------------------------------------------------------

def test_mondrian_group_covers_each_known_group():
    """Per-vendor coverage on a standard (all vendors present) split."""
    per_group = {g: [] for g in "ABC"}
    for seed in range(25):
        y, p, g = make_grouped(9000, seed=seed)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        cal, te = idx[:3000], idx[3000:]
        r = MondrianConformal(alpha=0.10, by="group", seed=seed).fit(
            p[cal], y[cal], groups=g[cal]
        ).predict(p[te], groups=g[te])
        gc = group_conditional_coverage(r, y[te], g[te])
        for k in "ABC":
            per_group[k].append(gc[k]["coverage"])
    for k in "ABC":
        assert np.mean(per_group[k]) >= 0.89, (k, np.mean(per_group[k]))


def test_split_can_undercover_a_group_that_mondrian_group_covers():
    """
    With groups differing in difficulty, pooled split conformal
    over-covers the easy group and under-covers the hard one. Mondrian
    by group fixes both. This is the contrast the paper draws.
    """
    split_hard, mond_hard = [], []
    for seed in range(25):
        y, p, g = make_grouped(9000, seed=seed,
                               group_seps=(0.3, 2.0, 4.0),
                               group_prev=(0.5, 0.5, 0.5))
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        cal, te = idx[:3000], idx[3000:]

        s = SplitConformal(alpha=0.10, seed=seed).fit(p[cal], y[cal]) \
            .predict(p[te])
        m = MondrianConformal(alpha=0.10, by="group", seed=seed).fit(
            p[cal], y[cal], groups=g[cal]).predict(p[te], groups=g[te])

        split_hard.append(group_conditional_coverage(s, y[te], g[te])
                          ["A"]["coverage"])
        mond_hard.append(group_conditional_coverage(m, y[te], g[te])
                         ["A"]["coverage"])

    assert np.mean(mond_hard) >= 0.89
    assert np.mean(mond_hard) > np.mean(split_hard)


def test_mondrian_group_unseen_group_falls_back_and_is_flagged():
    """
    The LOMO situation: the test group has no calibration cell. With
    fallback="pooled" the point gets the pooled quantile and is flagged,
    so the report can say which coverage numbers carry a group guarantee
    and which do not.
    """
    y, p, g = make_grouped(6000, seed=7)
    cal = g != "C"
    te = g == "C"
    m = MondrianConformal(alpha=0.10, by="group", fallback="pooled",
                          seed=7).fit(p[cal], y[cal], groups=g[cal])
    assert "C" not in m.groups_seen_

    r = m.predict(p[te], groups=g[te])
    assert r.fallback is not None
    assert r.fallback.all()
    rep = coverage_report(r, y[te])
    assert rep["fallback_rate"] == 1.0
    assert rep["method"] == "mondrian_group"


def test_mondrian_group_fallback_inf_gives_full_sets():
    y, p, g = make_grouped(4000, seed=8)
    cal, te = g != "C", g == "C"
    m = MondrianConformal(alpha=0.10, by="group", fallback="inf",
                          seed=8).fit(p[cal], y[cal], groups=g[cal])
    r = m.predict(p[te], groups=g[te])
    assert r.is_doubleton.all()
    assert empirical_coverage(r, y[te]) == 1.0


def test_mondrian_group_no_fallback_flag_when_all_groups_seen():
    y, p, g = make_grouped(4000, seed=9)
    rng = np.random.default_rng(9)
    idx = rng.permutation(len(y))
    cal, te = idx[:1500], idx[1500:]
    r = MondrianConformal(alpha=0.10, by="group", seed=9).fit(
        p[cal], y[cal], groups=g[cal]).predict(p[te], groups=g[te])
    assert r.fallback is None


def test_mondrian_both_covers_each_group_class_cell():
    per_cell = {}
    for seed in range(20):
        y, p, g = make_grouped(15000, seed=seed)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        cal, te = idx[:6000], idx[6000:]
        r = MondrianConformal(alpha=0.10, by="both", seed=seed).fit(
            p[cal], y[cal], groups=g[cal]).predict(p[te], groups=g[te])
        gc = group_conditional_coverage(r, y[te], g[te])
        for k in "ABC":
            for name in ("healthy", "failure"):
                v = gc[k][f"coverage_{name}"]
                if not np.isnan(v):
                    per_cell.setdefault((k, name), []).append(v)
    for cell, vals in per_cell.items():
        assert np.mean(vals) >= 0.88, (cell, np.mean(vals))


# ---------------------------------------------------------------
# API
# ---------------------------------------------------------------

def test_group_methods_require_groups():
    _, y, p = make_data(500, seed=1)
    with pytest.raises(ValueError, match="requires groups"):
        MondrianConformal(by="group").fit(p, y)
    m = MondrianConformal(by="group").fit(p, y, groups=np.zeros(500))
    with pytest.raises(ValueError, match="requires groups"):
        m.predict(p)


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        MondrianConformal(by="nope")
    with pytest.raises(ValueError):
        MondrianConformal(fallback="nope")


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        MondrianConformal().predict(np.array([0.5]))


def test_cell_report_lists_every_cell():
    y, p, g = make_grouped(3000, seed=2)
    m = MondrianConformal(by="both").fit(p, y, groups=g)
    rep = m.cell_report()
    assert len(rep) == len(m.qhat_)
    for row in rep:
        assert {"cell", "n_cal", "qhat", "supported"} <= set(row)


def test_allow_empty_false_forces_nonempty():
    _, y, p = make_data(3000, sep=6.0, seed=4)
    cp, cy, tp, _ = split_data(p, y, n_cal=1000, seed=4)
    r = MondrianConformal(alpha=0.10, seed=4, allow_empty=False).fit(
        cp, cy).predict(tp)
    assert r.is_empty.sum() == 0


def test_coverage_report_shape(capsys):
    _, y, p = make_data(3000, prevalence=0.1, seed=6)
    cp, cy, tp, ty = split_data(p, y, n_cal=1000, seed=6)
    r = MondrianConformal(alpha=0.10, seed=6).fit(cp, cy).predict(tp)
    rep = coverage_report(r, ty)
    assert rep["method"] == "mondrian_class"
    assert np.isfinite(rep["qhat"])
    assert rep["fallback_rate"] == 0.0


def test_deterministic():
    y, p, g = make_grouped(3000, seed=11)
    a = MondrianConformal(by="group", seed=1).fit(p, y, groups=g) \
        .predict(p, groups=g)
    b = MondrianConformal(by="group", seed=1).fit(p, y, groups=g) \
        .predict(p, groups=g)
    np.testing.assert_array_equal(a.sets, b.sets)
