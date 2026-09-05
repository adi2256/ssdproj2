"""
End-to-end pipeline smoke test.

Runs synthetic -> labels -> splits -> windows -> normalise -> dummy
model -> conformal -> coverage report, for the standard split and all
three LOMO folds.

This is the test that catches interface mismatches between modules.
Every mismatch found here is one not found during a training run.

The model is a deliberate stand-in: logistic regression on the last
timestep. Nothing here is a claim about accuracy. What is being checked
is that the pieces compose, that leakage guards survive the full
pipeline, and that the LOMO harness produces a populated results table.

    python3 -m pytest tests/test_pipeline.py -v -s
"""

import numpy as np
import pytest

from src.conformal import METHODS, METHOD_INPUTS
from src.conformal.mondrian import group_conditional_coverage
from src.conformal.split import SplitConformal, coverage_report
from src.data.labels import build_labels
from src.data.splits import make_all_lomo_splits, make_standard_split
from src.data.synthetic import make_synthetic
from src.data.windowing import Normalizer, flatten, make_windows

STRIDE = 7


# ---------------------------------------------------------------
# Dummy model — a stand-in, not a baseline
# ---------------------------------------------------------------

class DummyModel:
    """
    Logistic regression by gradient descent on the final timestep.

    Deliberately minimal and dependency-free. Its only job is to emit
    probabilities so the conformal layer has something to wrap.
    """

    def __init__(self, lr=0.5, epochs=300, seed=0):
        self.lr, self.epochs, self.seed = lr, epochs, seed
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        self.w = rng.normal(0, 0.01, d)
        self.b = 0.0

        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            err = p - y
            self.w -= self.lr * (X.T @ err) / n
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._sigmoid(X @ self.w + self.b)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def run_fold(labelled, split, alpha=0.10, score_fn="lac", method="split",
             seed=0):
    """
    One fold end to end for one conformal method. Returns the coverage
    report row.

    Every method receives the same model probabilities. What differs is
    only how the calibration scores are turned into sets, so differences
    between rows are attributable to the conformal procedure alone.
    """
    parts = {}
    for name in ("train", "cal", "test"):
        parts[name] = make_windows(split.apply(labelled, name),
                                   stride=STRIDE)

    # Normalisation fit on TRAIN ONLY. Fitting on pooled data leaks into
    # both the model and the conformal calibration scores.
    norm = Normalizer.fit(parts["train"])
    parts = {k: norm.transform(v) for k, v in parts.items()}

    Xtr, _ = flatten(parts["train"], "last")
    Xca, _ = flatten(parts["cal"], "last")
    Xte, _ = flatten(parts["test"], "last")

    model = DummyModel().fit(Xtr, parts["train"].y)
    p_cal = model.predict_proba(Xca)
    p_te = model.predict_proba(Xte)

    cp = METHODS[method](alpha, score_fn, seed)
    needs = METHOD_INPUTS[method]

    if "groups" in needs:
        cp.fit(p_cal, parts["cal"].y, groups=parts["cal"].vendors)
        result = cp.predict(p_te, groups=parts["test"].vendors)
    elif "features" in needs:
        # Weighted conformal sees test FEATURES only, never test labels.
        cp.fit_from_features(p_cal, parts["cal"].y, Xca, Xte)
        result = cp.predict_from_features(p_te, Xte)
    else:
        cp.fit(p_cal, parts["cal"].y)
        result = cp.predict(p_te)

    rep = coverage_report(result, parts["test"].y)
    rep["split"] = split.name
    rep["n_train_windows"] = len(parts["train"])
    rep["test_prevalence"] = float(parts["test"].y.mean())
    if "features" in needs:
        rep["domain_auc"] = cp.weight_info_["domain_auc"]
    return rep


@pytest.fixture(scope="module")
def pipeline():
    df, _ = make_synthetic()
    labelled, _ = build_labels(df)
    standard = make_standard_split(labelled)
    lomo = make_all_lomo_splits(labelled)
    return labelled, standard, lomo


# ---------------------------------------------------------------
# The pipeline composes
# ---------------------------------------------------------------

def test_standard_fold_runs_end_to_end(pipeline):
    labelled, standard, _ = pipeline
    rep = run_fold(labelled, standard)

    assert rep["n_test"] > 0
    assert rep["n_cal"] > 0
    assert rep["n_failures_test"] > 0
    assert 0.0 <= rep["empirical_coverage"] <= 1.0
    assert rep["target_coverage"] == 0.90


@pytest.mark.parametrize("vendor", ["A", "B", "C"])
def test_lomo_fold_runs_end_to_end(pipeline, vendor):
    labelled, _, lomo = pipeline
    rep = run_fold(labelled, lomo[vendor])

    assert rep["split"] == f"lomo_{vendor}"
    assert rep["n_test"] > 0
    assert rep["n_failures_test"] > 0


def test_all_folds_produce_a_full_results_row(pipeline):
    """Every field the LOMO results table needs must be populated."""
    labelled, standard, lomo = pipeline
    required = [
        "split", "alpha", "target_coverage", "empirical_coverage",
        "coverage_healthy", "coverage_failure", "avg_set_size",
        "singleton_rate", "doubleton_rate", "empty_rate", "qhat",
        "n_cal", "n_test", "n_failures_test", "test_prevalence",
    ]
    for split in [standard, *lomo.values()]:
        rep = run_fold(labelled, split)
        for k in required:
            assert k in rep, f"{split.name} missing {k}"
            assert rep[k] is not None


# ---------------------------------------------------------------
# Leakage guards survive the full pipeline
# ---------------------------------------------------------------

@pytest.mark.parametrize("vendor", ["A", "B", "C"])
def test_no_held_out_vendor_windows_in_train_or_cal(pipeline, vendor):
    """
    The split-level guard is asserted in validate_split. This checks the
    property survives windowing, which is where a join could reintroduce
    it.
    """
    labelled, _, lomo = pipeline
    split = lomo[vendor]
    for part in ("train", "cal"):
        ws = make_windows(split.apply(labelled, part), stride=STRIDE)
        assert vendor not in set(np.unique(ws.vendors))

    te = make_windows(split.apply(labelled, "test"), stride=STRIDE)
    assert set(np.unique(te.vendors)) == {vendor}


def test_no_drive_appears_in_two_parts_after_windowing(pipeline):
    labelled, standard, _ = pipeline
    keys = {}
    for part in ("train", "cal", "test"):
        ws = make_windows(standard.apply(labelled, part), stride=STRIDE)
        keys[part] = {tuple(k) for k in ws.drives}

    assert not (keys["train"] & keys["cal"])
    assert not (keys["train"] & keys["test"])
    assert not (keys["cal"] & keys["test"])


def test_normalizer_stats_come_from_train_only(pipeline):
    labelled, standard, _ = pipeline
    tr = make_windows(standard.apply(labelled, "train"), stride=STRIDE)
    te = make_windows(standard.apply(labelled, "test"), stride=STRIDE)

    from_train = Normalizer.fit(tr)
    from_pooled = Normalizer.fit(
        make_windows(labelled, stride=STRIDE)
    )
    # If these were identical the fit would be pooling silently.
    assert not np.allclose(from_train.mean, from_pooled.mean)
    assert np.isfinite(from_train.transform(te).X).all()


# ---------------------------------------------------------------
# Sanity properties, not accuracy claims
# ---------------------------------------------------------------

def test_dummy_model_learns_the_injected_signal(pipeline):
    """
    The fixture injects drift into r_5, r_187 and r_197. If even a
    linear model on the last timestep cannot beat chance, the signal is
    not reaching the model and every downstream number is meaningless.
    """
    labelled, standard, _ = pipeline
    tr = make_windows(standard.apply(labelled, "train"), stride=STRIDE)
    te = make_windows(standard.apply(labelled, "test"), stride=STRIDE)
    norm = Normalizer.fit(tr)
    tr, te = norm.transform(tr), norm.transform(te)

    Xtr, _ = flatten(tr, "last")
    Xte, _ = flatten(te, "last")
    p = DummyModel().fit(Xtr, tr.y).predict_proba(Xte)

    pos, neg = p[te.y == 1], p[te.y == 0]
    assert len(pos) > 0 and len(neg) > 0
    assert pos.mean() > neg.mean(), (
        "model scores positives no higher than negatives; "
        "the injected signal is not reaching it"
    )


def test_coverage_is_reported_per_class(pipeline):
    """
    Class-conditional coverage must be present in every results row.
    Reporting only marginal coverage would repeat the exact mistake the
    paper criticises.
    """
    labelled, standard, _ = pipeline
    rep = run_fold(labelled, standard)
    assert not np.isnan(rep["coverage_healthy"])
    assert not np.isnan(rep["coverage_failure"])


def test_set_rates_sum_to_one(pipeline):
    labelled, standard, _ = pipeline
    rep = run_fold(labelled, standard)
    total = (rep["singleton_rate"] + rep["doubleton_rate"]
             + rep["empty_rate"])
    assert abs(total - 1.0) < 1e-9


def test_pipeline_is_deterministic(pipeline):
    labelled, standard, _ = pipeline
    a = run_fold(labelled, standard)
    b = run_fold(labelled, standard)
    assert a["empirical_coverage"] == b["empirical_coverage"]
    assert a["qhat"] == b["qhat"]


# ---------------------------------------------------------------
# All four conformal methods compose with the pipeline
# ---------------------------------------------------------------

@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_runs_on_standard_split(pipeline, method):
    labelled, standard, _ = pipeline
    rep = run_fold(labelled, standard, method=method)
    assert rep["method"].startswith(method.split("_")[0])
    assert 0.0 <= rep["empirical_coverage"] <= 1.0
    assert rep["n_failures_test"] > 0


@pytest.mark.parametrize("method", sorted(METHODS))
@pytest.mark.parametrize("vendor", ["A", "B", "C"])
def test_every_method_runs_on_every_lomo_fold(pipeline, method, vendor):
    labelled, _, lomo = pipeline
    rep = run_fold(labelled, lomo[vendor], method=method)
    assert rep["split"] == f"lomo_{vendor}"
    assert rep["n_test"] > 0


def test_mondrian_class_raises_failure_coverage_over_split(pipeline):
    """
    The reason Mondrian is required. On the same probabilities, per-class
    calibration must lift failure-class coverage relative to split
    conformal, on every fold.

    No per-fold floor is asserted. The guarantee is over repeated
    calibration draws, and each fold here is ONE draw from ~9 positive
    calibration drives whose windows are correlated (30 consecutive days
    of the same drive). Single folds legitimately land at 0.66 or 0.77.
    That is exactly why per-fold coverage is reported as a case study,
    not a population estimate. The mean across folds is checked instead.
    """
    labelled, standard, lomo = pipeline
    gains, mond = [], []
    for sp in [standard, *lomo.values()]:
        s = run_fold(labelled, sp, method="split")
        m = run_fold(labelled, sp, method="mondrian_class")
        assert m["coverage_failure"] > s["coverage_failure"], sp.name
        gains.append(m["coverage_failure"] - s["coverage_failure"])
        mond.append(m["coverage_failure"])
    assert np.mean(mond) >= 0.80, np.mean(mond)
    assert min(gains) > 0.2


def test_mondrian_group_flags_fallback_on_lomo_only(pipeline):
    """
    On the standard split every vendor has a calibration cell, so no
    fallback. On a LOMO fold the held-out vendor has none, so every test
    point is flagged. This is the contrast the paper reports.
    """
    labelled, standard, lomo = pipeline
    assert run_fold(labelled, standard,
                    method="mondrian_group")["fallback_rate"] == 0.0
    for v in "ABC":
        assert run_fold(labelled, lomo[v],
                        method="mondrian_group")["fallback_rate"] == 1.0


def test_weighted_reports_domain_auc(pipeline):
    """
    The domain classifier's AUC measures how separable calibration and
    test FEATURES are. Under real vendor shift it should be high.

    CAUTION, documented here because it will matter on real data: on the
    standard split -- exchangeable at drive level -- the AUC is NOT near
    0.5 on this fixture. With ~64 calibration drives and ~80 test
    drives, each carrying a drive-specific baseline and ~10 near-
    identical windows, the domain classifier memorises drive identity.
    The absolute AUC is therefore inflated by within-drive correlation
    whenever drive counts are small, and should be read relatively, not
    as a clean shift magnitude. The real fleet has thousands of drives
    per fold, which dilutes this, but the effect should be checked there
    too (compare domain AUC on the standard split against 0.5).
    """
    labelled, standard, lomo = pipeline
    std = run_fold(labelled, standard, method="weighted")["domain_auc"]
    lomo_auc = {v: run_fold(labelled, lomo[v], method="weighted")
                ["domain_auc"] for v in "ABC"}

    # Vendors B and C carry the largest synthetic offsets; their LOMO
    # folds must be more separable than the exchangeable split.
    assert lomo_auc["B"] > std and lomo_auc["C"] > std, (std, lomo_auc)
    assert max(lomo_auc.values()) > 0.9


def test_weighted_never_sees_test_labels(pipeline, monkeypatch):
    """
    Guard against a future refactor passing test labels into weight
    estimation. The fit path must only consume test features.
    """
    import src.conformal.weighted as wmod
    seen = {}
    orig = wmod.estimate_density_ratio

    def spy(X_cal, X_test, **kw):
        seen["shapes"] = (np.asarray(X_cal).shape, np.asarray(X_test).shape)
        return orig(X_cal, X_test, **kw)

    monkeypatch.setattr(wmod, "estimate_density_ratio", spy)
    labelled, _, lomo = pipeline
    run_fold(labelled, lomo["B"], method="weighted")
    # both arguments are 2-D feature matrices, not label vectors
    assert all(len(sh) == 2 for sh in seen["shapes"])


# ---------------------------------------------------------------
# Reporting helper — prints the table the paper needs
# ---------------------------------------------------------------

def test_print_results_table(pipeline, capsys):
    """
    Not an assertion of results, a shape check on the table. Run with
    -s to see it.
    """
    labelled, standard, lomo = pipeline
    splits = [standard, *lomo.values()]
    rows = [run_fold(labelled, sp, method=m)
            for m in sorted(METHODS) for sp in splits]

    with capsys.disabled():
        print()
        print(f"{'method':<15} {'split':<10} {'n_test':>7} {'prev':>6} "
              f"{'cov':>7} {'cov_hlt':>8} {'cov_fail':>9} {'size':>6} "
              f"{'fallbk':>7}")
        last = None
        for r in rows:
            if last is not None and r["method"] != last:
                print()
            last = r["method"]
            print(f"{r['method']:<15} {r['split']:<10} {r['n_test']:>7,} "
                  f"{r['test_prevalence']:>6.3f} "
                  f"{r['empirical_coverage']:>7.4f} "
                  f"{r['coverage_healthy']:>8.4f} "
                  f"{r['coverage_failure']:>9.4f} "
                  f"{r['avg_set_size']:>6.3f} "
                  f"{r['fallback_rate']:>7.2f}")
        print("\nDummy model on synthetic data. Shape check only — "
              "these are not results.")

    assert len(rows) == len(METHODS) * 4
