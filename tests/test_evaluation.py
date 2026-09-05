"""
Tests for models, metrics and the LOMO experiment runner.

    python3 -m pytest tests/test_evaluation.py -v
"""

import os
import shutil

import numpy as np
import polars as pl
import pytest

from src.data.labels import build_labels
from src.data.splits import make_lomo_split, make_standard_split
from src.data.synthetic import make_synthetic
from src.evaluation.lomo import (RunConfig, coverage_gap_table,
                                 prepare_fold, results_table, run_cell,
                                 run_experiment)
from src.evaluation.metrics import (auc, average_precision, confusion,
                                    drive_level_metrics, f_beta,
                                    fleet_metrics, pick_threshold)
from src.models import HAVE_TORCH, MODEL_INPUT, MODELS
from src.models.baseline import RandomForestBaseline


@pytest.fixture(scope="module")
def labelled():
    df, _ = make_synthetic()
    out, _ = build_labels(df)
    return out


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    return RunConfig(results_dir=str(tmp_path_factory.mktemp("res")),
                     models=("random_forest",), stride=9)


def toy_seq(n=500, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.2).astype(int)
    X = rng.normal(size=(n, 10, 4)).astype(np.float32)
    X[:, -4:, 0] += y[:, None] * 2.5
    return X, y


# ===============================================================
# METRICS
# ===============================================================

def test_auc_extremes():
    assert auc([0.1, 0.9], [0, 1]) == 1.0
    assert auc([0.9, 0.1], [0, 1]) == 0.0
    assert abs(auc([0.5, 0.5], [0, 1]) - 0.5) < 1e-9


def test_auc_single_class_is_nan():
    assert np.isnan(auc([0.1, 0.2], [0, 0]))


def test_average_precision_perfect_and_random():
    y = np.array([1, 1, 0, 0])
    assert average_precision([0.9, 0.8, 0.2, 0.1], y) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    y2 = (rng.uniform(size=4000) < 0.1).astype(int)
    ap = average_precision(rng.uniform(size=4000), y2)
    assert abs(ap - 0.1) < 0.05


def test_average_precision_beats_auc_as_imbalance_diagnostic():
    """
    At low prevalence a high AUC can hide poor precision. Both are
    reported for that reason.
    """
    rng = np.random.default_rng(1)
    n = 40000
    y = (rng.uniform(size=n) < 0.01).astype(int)
    # Overlapping Gaussians: good ranking, but at 1% prevalence the top
    # of the ranking is still mostly negatives.
    p = rng.normal(loc=y * 1.5, scale=1.0)
    assert auc(p, y) > 0.80
    assert average_precision(p, y) < 0.15


def test_confusion_counts():
    c = confusion([0.9, 0.4, 0.8, 0.1], [1, 1, 0, 0], 0.5)
    assert c == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_f_beta_half_favours_precision():
    """
    F0.5 weights precision above recall, matching the base paper: a
    false alarm burns a working drive and a technician's time.
    """
    y = np.array([1] * 10 + [0] * 90)
    precise = np.where(np.arange(100) < 5, 0.9, 0.1)      # 5/5 correct
    greedy = np.where(np.arange(100) < 40, 0.9, 0.1)      # 10 tp, 30 fp
    assert f_beta(precise, y, 0.5, 0.5) > f_beta(greedy, y, 0.5, 0.5)
    assert f_beta(greedy, y, 0.5, 2.0) > f_beta(precise, y, 0.5, 2.0)


def test_f_beta_zero_when_nothing_predicted():
    assert f_beta([0.1, 0.2], [1, 1], 0.9) == 0.0


def test_pick_threshold_uses_training_scores_only():
    """
    Rule 3. Tuning on test scores would report a best case nobody could
    reproduce, so pick_threshold is only ever given training data.
    """
    rng = np.random.default_rng(2)
    y = (rng.uniform(size=2000) < 0.2).astype(int)
    p = rng.uniform(size=2000) * 0.4 + y * 0.4
    thr = pick_threshold(p, y)
    assert p.min() <= thr <= p.max()
    assert f_beta(p, y, thr) >= f_beta(p, y, 0.5)


def test_pick_threshold_degenerate_inputs():
    assert pick_threshold([], []) == 0.5
    assert pick_threshold([0.3, 0.3], [1, 1]) == 0.5
    assert pick_threshold([0.4, 0.4], [1, 0]) == pytest.approx(0.4)


def test_fleet_metrics_fields():
    rng = np.random.default_rng(3)
    y = (rng.uniform(size=1000) < 0.1).astype(int)
    p = rng.uniform(size=1000) * 0.5 + y * 0.3
    m = fleet_metrics(p, y, 0.4)
    for k in ("precision", "recall", "f0.5", "f1", "auc",
              "avg_precision", "prevalence", "tp", "fp", "fn", "tn"):
        assert k in m
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == 1000


def test_drive_level_collapses_windows():
    """
    Operators replace drives, not windows. A drive with many windows
    would otherwise dominate one with few.
    """
    probs = np.array([0.1, 0.9, 0.2, 0.3])
    labels = np.array([0, 1, 0, 0])
    idx = np.array([0, 0, 1, 1])
    m = drive_level_metrics(probs, labels, idx, 0.5)
    assert m["n_drives"] == 2
    assert m["tp"] == 1 and m["tn"] == 1


# ===============================================================
# MODELS
# ===============================================================

def test_baseline_returns_one_dimensional_probabilities():
    """The conformal layer takes P(y=1), not sklearn's (n, 2)."""
    X, y = toy_seq()
    Xf = X[:, -1, :]
    p = RandomForestBaseline(n_estimators=40, seed=0).fit(Xf, y) \
        .predict_proba(Xf)
    assert p.ndim == 1 and len(p) == len(y)
    assert (p >= 0).all() and (p <= 1).all()


def test_baseline_learns_signal():
    X, y = toy_seq(seed=1)
    Xf = X[:, -1, :]
    m = RandomForestBaseline(n_estimators=80, seed=0).fit(Xf[:350], y[:350])
    p = m.predict_proba(Xf[350:])
    assert p[y[350:] == 1].mean() > p[y[350:] == 0].mean()


def test_baseline_handles_nan():
    X, y = toy_seq(seed=2)
    Xf = X[:, -1, :].copy()
    Xf[::13, 0] = np.nan
    p = RandomForestBaseline(n_estimators=30, seed=0).fit(Xf, y) \
        .predict_proba(Xf)
    assert np.isfinite(p).all()


def test_baseline_single_class_training_does_not_crash():
    """A thin fold can produce an all-negative training set."""
    X, _ = toy_seq(seed=3)
    Xf = X[:, -1, :]
    y = np.zeros(len(Xf), dtype=int)
    p = RandomForestBaseline(n_estimators=20, seed=0).fit(Xf, y) \
        .predict_proba(Xf)
    assert p.shape == (len(Xf),)
    assert np.isfinite(p).all()


def test_baseline_errors():
    X, y = toy_seq(seed=4)
    Xf = X[:, -1, :]
    with pytest.raises(RuntimeError):
        RandomForestBaseline().predict_proba(Xf)
    with pytest.raises(ValueError):
        RandomForestBaseline().fit(X, y)             # 3-D input
    m = RandomForestBaseline(n_estimators=10).fit(Xf, y)
    with pytest.raises(ValueError):
        m.predict_proba(Xf[:, :2])


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
def test_sequence_model_learns_temporal_signal():
    X, y = toy_seq(n=800, seed=5)
    from src.models.sequence import SequenceModel
    m = SequenceModel(seed=0, epochs=30).fit(X[:600], y[:600])
    p = m.predict_proba(X[600:])
    assert p.shape == (200,)
    assert p[y[600:] == 1].mean() > p[y[600:] == 0].mean()


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
def test_sequence_model_stays_small():
    """
    Must run on a laptop or free Colab tier, and with a few thousand
    failed drives a bigger model would overfit before it helped.
    """
    X, y = toy_seq(seed=6)
    from src.models.sequence import SequenceModel
    m = SequenceModel(seed=0, epochs=3).fit(X, y)
    assert m.n_parameters < 20000


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
def test_sequence_model_early_stops():
    X, y = toy_seq(seed=7)
    from src.models.sequence import SequenceModel
    m = SequenceModel(seed=0, epochs=200, patience=3).fit(X, y)
    assert len(m.history_) < 200


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
def test_sequence_model_errors():
    from src.models.sequence import SequenceModel
    X, y = toy_seq(seed=8)
    with pytest.raises(RuntimeError):
        SequenceModel().predict_proba(X)
    with pytest.raises(ValueError):
        SequenceModel(epochs=1).fit(X[:, -1, :], y)      # 2-D input


def test_model_registry_is_consistent():
    assert set(MODELS) == set(MODEL_INPUT)
    assert MODEL_INPUT["random_forest"] == "flat"
    assert MODEL_INPUT["gru"] == "seq"


# ===============================================================
# FOLD PREPARATION — the four rules
# ===============================================================

def test_prepare_fold_selects_features_within_the_fold(labelled, cfg):
    """
    Rule 1. Selection must run on this fold's training data only;
    selecting once on pooled data leaks held-out labels.
    """
    fold = prepare_fold(labelled, make_lomo_split(labelled, "C"), cfg)
    assert fold.selection is not None
    assert 1 <= len(fold.features) <= 16
    assert fold.train.X.shape[2] == len(fold.features)
    assert fold.cal.X.shape[2] == len(fold.features)
    assert fold.test.X.shape[2] == len(fold.features)


def test_feature_selection_differs_between_folds(labelled, cfg):
    a = prepare_fold(labelled, make_lomo_split(labelled, "A"), cfg)
    b = prepare_fold(labelled, make_lomo_split(labelled, "C"), cfg)
    assert a.features != b.features or a.selection.order != b.selection.order


def test_prepare_fold_can_disable_selection(labelled, tmp_path):
    c = RunConfig(results_dir=str(tmp_path), feature_selection=False,
                  stride=9)
    fold = prepare_fold(labelled, make_standard_split(labelled), c)
    assert len(fold.features) == 16
    assert fold.selection is None


def test_prepare_fold_rejects_held_out_vendor_contamination(labelled, cfg):
    """
    Rule 4, re-asserted after windowing because a join is where
    contamination could reappear.
    """
    good = make_lomo_split(labelled, "B")
    from src.data.splits import Split
    bad = Split(name="bad", train=good.train,
                cal=pl.concat([good.cal, good.test.head(3)]),
                test=good.test, held_out_vendor="B")
    with pytest.raises(AssertionError, match="held-out vendor"):
        prepare_fold(labelled, bad, cfg)


@pytest.mark.parametrize("vendor", ["A", "B", "C"])
def test_prepare_fold_test_is_only_held_out_vendor(labelled, cfg, vendor):
    fold = prepare_fold(labelled, make_lomo_split(labelled, vendor), cfg)
    assert set(np.unique(fold.test.vendors)) == {vendor}
    assert vendor not in set(np.unique(fold.train.vendors))
    assert vendor not in set(np.unique(fold.cal.vendors))


# ===============================================================
# CELLS AND SWEEP
# ===============================================================

@pytest.mark.parametrize("method",
                         ["split", "mondrian_class", "mondrian_group",
                          "weighted"])
def test_run_cell_produces_a_complete_row(labelled, cfg, method):
    split = make_standard_split(labelled)
    fold = prepare_fold(labelled, split, cfg)
    row = run_cell(fold, split, "random_forest", method, cfg)

    for k in ("split", "model", "conformal", "empirical_coverage",
              "coverage_failure", "avg_set_size", "win_f0.5", "win_auc",
              "drv_f0.5", "n_test_drives", "test_prevalence",
              "n_features"):
        assert k in row, k
    assert 0.0 <= row["empirical_coverage"] <= 1.0
    assert row["conformal"] == method


def test_weighted_cell_reports_domain_auc(labelled, cfg):
    split = make_lomo_split(labelled, "C")
    fold = prepare_fold(labelled, split, cfg)
    row = run_cell(fold, split, "random_forest", "weighted", cfg)
    assert "domain_auc" in row and 0.0 <= row["domain_auc"] <= 1.0
    assert "ess_cal" in row


def test_run_experiment_covers_every_combination(labelled, tmp_path):
    c = RunConfig(results_dir=str(tmp_path / "r1"),
                  models=("random_forest",), stride=11)
    df = run_experiment(labelled, c, verbose=False)
    assert df.height == 4 * len(c.methods)          # 4 splits
    assert set(df["split"].unique()) == {"standard", "lomo_A", "lomo_B",
                                         "lomo_C"}
    assert os.path.exists(os.path.join(c.results_dir, "results.csv"))


def test_run_experiment_resumes_from_disk(labelled, tmp_path, capsys):
    """
    Cloud notebook sessions are cut off after a fixed time, so a run must
    resume rather than restart.
    """
    c = RunConfig(results_dir=str(tmp_path / "r2"),
                  models=("random_forest",), methods=("split",), stride=11)
    first = run_experiment(labelled, c, verbose=False)
    n_files = len([f for f in os.listdir(c.results_dir)
                   if f.endswith(".json")])
    assert n_files == 4

    second = run_experiment(labelled, c, verbose=True)
    out = capsys.readouterr().out
    assert "[cached]" in out
    assert second.height == first.height


def test_run_experiment_can_skip_standard_split(labelled, tmp_path):
    c = RunConfig(results_dir=str(tmp_path / "r3"),
                  models=("random_forest",), methods=("split",), stride=11)
    df = run_experiment(labelled, c, include_standard=False, verbose=False)
    assert set(df["split"].unique()) == {"lomo_A", "lomo_B", "lomo_C"}


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
def test_sequence_model_runs_in_the_harness(labelled, tmp_path):
    c = RunConfig(results_dir=str(tmp_path / "r4"), models=("gru",),
                  methods=("split",), stride=15)
    split = make_standard_split(labelled)
    fold = prepare_fold(labelled, split, c)
    row = run_cell(fold, split, "gru", "split", c)
    assert row["model"] == "gru"
    assert 0.0 <= row["empirical_coverage"] <= 1.0


# ===============================================================
# RESULT TABLES
# ===============================================================

def test_results_table_columns(labelled, tmp_path):
    c = RunConfig(results_dir=str(tmp_path / "r5"),
                  models=("random_forest",),
                  methods=("split", "mondrian_class"), stride=11)
    df = run_experiment(labelled, c, verbose=False)
    t = results_table(df)
    for k in ("split", "model", "conformal", "empirical_coverage",
              "coverage_failure", "avg_set_size"):
        assert k in t.columns


def test_coverage_gap_table_compares_against_standard(labelled, tmp_path):
    """
    The headline result is the gap between coverage on an exchangeable
    split and coverage on a held-out manufacturer.
    """
    c = RunConfig(results_dir=str(tmp_path / "r6"),
                  models=("random_forest",), methods=("split",), stride=11)
    df = run_experiment(labelled, c, verbose=False)
    gap = coverage_gap_table(df)
    assert gap.height == 3                       # three LOMO folds
    assert "coverage_drop" in gap.columns
    assert "failure_coverage_drop" in gap.columns
    assert "standard" not in set(gap["split"].unique())


def test_mondrian_class_lifts_failure_coverage_across_folds(labelled,
                                                            tmp_path):
    """
    The reason Mondrian is required rather than optional. Checked as a
    mean across folds: a single fold calibrates on few positive drives
    whose windows are correlated, so per-fold values are noisy. This is
    also why per-fold coverage is reported as a case study rather than a
    population estimate.
    """
    c = RunConfig(results_dir=str(tmp_path / "r7"),
                  models=("random_forest",),
                  methods=("split", "mondrian_class"), stride=9)
    df = run_experiment(labelled, c, verbose=False)
    s = df.filter(pl.col("conformal") == "split")["coverage_failure"].mean()
    m = df.filter(pl.col("conformal") == "mondrian_class") \
        ["coverage_failure"].mean()
    assert m > s + 0.2, (s, m)


def test_set_size_is_the_price_of_the_guarantee(labelled, tmp_path):
    c = RunConfig(results_dir=str(tmp_path / "r8"),
                  models=("random_forest",),
                  methods=("split", "mondrian_class"), stride=9)
    df = run_experiment(labelled, c, verbose=False)
    s = df.filter(pl.col("conformal") == "split")["avg_set_size"].mean()
    m = df.filter(pl.col("conformal") == "mondrian_class") \
        ["avg_set_size"].mean()
    assert m > s


def test_results_are_deterministic(labelled, tmp_path):
    c1 = RunConfig(results_dir=str(tmp_path / "d1"),
                   models=("random_forest",), methods=("split",), stride=11)
    c2 = RunConfig(results_dir=str(tmp_path / "d2"),
                   models=("random_forest",), methods=("split",), stride=11)
    a = run_experiment(labelled, c1, verbose=False)
    b = run_experiment(labelled, c2, verbose=False)
    np.testing.assert_allclose(
        a.sort("split")["empirical_coverage"].to_numpy(),
        b.sort("split")["empirical_coverage"].to_numpy(),
    )
