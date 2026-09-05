"""
Tests for label construction.

    python3 -m pytest tests/test_labels.py -v
"""

import polars as pl
import pytest

from src.config import CFG, DRIVE_KEY
from src.data.labels import (LABEL_COL, VENDOR_COL, build_labels,
                             drive_table, label_summary)
from src.data.synthetic import make_synthetic


@pytest.fixture(scope="module")
def raw():
    return make_synthetic()


@pytest.fixture(scope="module")
def labelled(raw):
    df, _ = raw
    return build_labels(df)


# ---------------------------------------------------------------
# The rule
# ---------------------------------------------------------------

def test_label_is_strictly_inside_horizon(labelled):
    df, _ = labelled
    h = CFG.horizon_days

    pos = df.filter(pl.col(LABEL_COL) == 1)
    assert (pos["days_to_failure"] > 0).all()
    assert (pos["days_to_failure"] <= h).all()

    neg = df.filter(pl.col(LABEL_COL) == 0)
    outside = neg.filter(pl.col("days_to_failure").is_not_null())
    assert (outside["days_to_failure"] > h).all()


def test_failure_day_is_not_a_positive(raw):
    """
    dtf == 0 is the failure day itself. Under `0 < dtf <= h` it is not a
    positive, and being post-failure it is dropped entirely.
    """
    df, truth = raw
    assert df.filter(pl.col("days_to_failure") == 0).height == truth["n_failed"]

    out, _ = build_labels(df)
    assert out.filter(pl.col("days_to_failure") == 0).height == 0


def test_post_failure_rows_dropped(raw):
    df, truth = raw
    out, stats = build_labels(df)

    assert out.filter(pl.col("days_to_failure") < 0).height == 0
    # post-failure tail plus the failure day itself
    expected = truth["n_failed"] * (truth["post_failure_days"] + 1)
    assert stats.n_post_failure_dropped == expected


def test_post_failure_kept_when_disabled(raw):
    df, _ = raw
    out, stats = build_labels(df, drop_post_failure=False)
    assert stats.n_post_failure_dropped == 0
    assert out.filter(pl.col("days_to_failure") < 0).height > 0


def test_full_horizon_per_failed_drive(raw):
    """
    Every failed drive with enough history contributes exactly `horizon`
    positive rows.
    """
    df, truth = raw
    out, _ = build_labels(df)

    counts = (
        out.filter(pl.col(LABEL_COL) == 1)
        .group_by(DRIVE_KEY).len()
    )
    assert counts.height == truth["n_failed"]
    assert counts["len"].min() == truth["horizon_days"]
    assert counts["len"].max() == truth["horizon_days"]


# ---------------------------------------------------------------
# Censoring — the unobservable tail
# ---------------------------------------------------------------

def test_unobservable_tail_dropped_for_non_failed_drives(raw):
    """
    For a drive last seen at T that never failed, (T, T+horizon] is not
    observed, so the label is undefined. Those rows must be dropped, not
    labelled 0.
    """
    df, _ = raw
    out, stats = build_labels(df)
    assert stats.n_unobservable_dropped > 0

    h = CFG.horizon_days
    healthy = out.filter(pl.col("failure_time").is_null())
    per_drive = healthy.group_by(DRIVE_KEY).agg(
        pl.col("ds").max().alias("kept_last")
    )
    orig = (
        df.filter(pl.col("failure_time").is_null())
        .group_by(DRIVE_KEY).agg(pl.col("ds").max().alias("orig_last"))
    )
    joined = per_drive.join(orig, on=DRIVE_KEY, how="inner").with_columns(
        (pl.col("orig_last") - pl.col("kept_last"))
        .dt.total_days().alias("gap")
    )
    assert (joined["gap"] >= h).all()


def test_censored_and_survivor_drives_treated_identically(raw):
    """
    The rule is per drive's own last date, so a right-censored drive and
    one surviving to the global end get the same treatment. Nothing in
    the code should special-case censoring.
    """
    df, truth = raw
    out, _ = build_labels(df)
    h = CFG.horizon_days

    censored_keys = {(m, d) for m, d, _ in truth["censored"]}
    survivors = (
        df.filter(pl.col("failure_time").is_null())
        .group_by(DRIVE_KEY).agg(pl.col("ds").max().alias("last"))
        .filter(pl.col("last") == truth["global_end"])
    )
    surv_keys = set(map(tuple, survivors.select(DRIVE_KEY).iter_rows()))
    assert censored_keys and surv_keys

    for keys in (censored_keys, surv_keys):
        sub = out.filter(
            pl.struct(DRIVE_KEY).map_elements(
                lambda s: (s["model"], s["disk_id"]) in keys,
                return_dtype=pl.Boolean,
            )
        )
        orig = df.filter(
            pl.struct(DRIVE_KEY).map_elements(
                lambda s: (s["model"], s["disk_id"]) in keys,
                return_dtype=pl.Boolean,
            )
        )
        a = sub.group_by(DRIVE_KEY).agg(pl.col("ds").max().alias("k"))
        b = orig.group_by(DRIVE_KEY).agg(pl.col("ds").max().alias("o"))
        j = a.join(b, on=DRIVE_KEY).with_columns(
            (pl.col("o") - pl.col("k")).dt.total_days().alias("gap")
        )
        assert (j["gap"] >= h).all()


def test_unobservable_kept_when_disabled(raw):
    df, _ = raw
    _, stats = build_labels(df, drop_unobservable=False)
    assert stats.n_unobservable_dropped == 0


def test_no_positives_among_non_failed_drives(labelled):
    df, _ = labelled
    healthy = df.filter(pl.col("failure_time").is_null())
    assert int(healthy[LABEL_COL].sum()) == 0


# ---------------------------------------------------------------
# Short history
# ---------------------------------------------------------------

def test_short_history_drives_dropped(raw):
    df, _ = raw
    out, stats = build_labels(df, min_history=200)
    counts = out.group_by(DRIVE_KEY).len()
    if counts.height:
        assert counts["len"].min() >= 200
    assert stats.n_short_history_dropped > 0


def test_min_history_defaults_to_window_len(raw):
    df, _ = raw
    out, _ = build_labels(df)
    counts = out.group_by(DRIVE_KEY).len()
    assert counts["len"].min() >= CFG.window_len


# ---------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------

def test_row_accounting_balances(raw):
    """Every dropped row is attributed to exactly one reason."""
    df, _ = raw
    _, s = build_labels(df)
    assert (s.n_rows_in
            - s.n_post_failure_dropped
            - s.n_unobservable_dropped
            - s.n_short_history_dropped) == s.n_rows_out


def test_stats_positive_count_matches_frame(labelled):
    df, s = labelled
    assert s.n_positive == int(df[LABEL_COL].sum())
    assert 0 < s.positive_rate < 1


# ---------------------------------------------------------------
# Vendor and drive tables
# ---------------------------------------------------------------

def test_vendor_column_derived(labelled):
    df, _ = labelled
    assert set(df[VENDOR_COL].unique()) == {"A", "B", "C"}
    mismatch = df.filter(
        pl.col(VENDOR_COL) != pl.col("model").str.slice(1, 1)
    )
    assert mismatch.height == 0


def test_drive_table_is_one_row_per_drive(labelled):
    df, _ = labelled
    dt = drive_table(df)
    assert dt.height == df.select(DRIVE_KEY).unique().height
    assert dt.select(DRIVE_KEY).unique().height == dt.height


def test_drive_label_marks_any_positive(labelled):
    df, _ = labelled
    dt = drive_table(df)
    pos_drives = set(map(tuple, (
        df.filter(pl.col(LABEL_COL) == 1).select(DRIVE_KEY).unique()
        .iter_rows()
    )))
    marked = set(map(tuple, (
        dt.filter(pl.col("drive_label") == 1).select(DRIVE_KEY).iter_rows()
    )))
    assert pos_drives == marked


def test_label_shift_survives_labelling(labelled):
    """
    The A < B < C failure ordering must persist after the rule is
    applied, or LOMO loses the label-shift axis.
    """
    df, _ = labelled
    s = label_summary(df)
    rates = dict(zip(s[VENDOR_COL], s["drive_positive_pct"]))
    assert rates["A"] < rates["B"] < rates["C"]


def test_labelling_is_deterministic(raw):
    df, _ = raw
    a, sa = build_labels(df)
    b, sb = build_labels(df)
    assert a.equals(b)
    assert sa.as_dict() == sb.as_dict()
