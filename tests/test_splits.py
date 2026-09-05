"""
Tests for split construction.

The leakage assertions here are the highest-value tests in the project:
they encode the two things a reviewer will check hardest, so they cannot
silently regress.

    python3 -m pytest tests/test_splits.py -v
"""

import polars as pl
import pytest

from src.config import CFG, DRIVE_KEY
from src.data.labels import LABEL_COL, VENDOR_COL, build_labels, drive_table
from src.data.splits import (Split, make_all_lomo_splits, make_lomo_split,
                             make_standard_split, split_report,
                             validate_split)
from src.data.synthetic import make_synthetic


@pytest.fixture(scope="module")
def labelled():
    df, _ = make_synthetic()
    out, _ = build_labels(df)
    return out


@pytest.fixture(scope="module")
def standard(labelled):
    return make_standard_split(labelled)


@pytest.fixture(scope="module")
def lomo(labelled):
    return make_all_lomo_splits(labelled)


# ---------------------------------------------------------------
# INVARIANT 1 + 3 — drive-level disjointness
# ---------------------------------------------------------------

def test_standard_split_parts_are_disjoint(standard):
    tr, ca, te = (standard.keys(p) for p in ("train", "cal", "test"))
    assert not (tr & ca)
    assert not (tr & te)
    assert not (ca & te)


@pytest.mark.parametrize("vendor", ["A", "B", "C"])
def test_lomo_split_parts_are_disjoint(lomo, vendor):
    s = lomo[vendor]
    tr, ca, te = (s.keys(p) for p in ("train", "cal", "test"))
    assert not (tr & ca)
    assert not (tr & te)
    assert not (ca & te)


def test_no_drive_rows_shared_after_apply(standard, labelled):
    """
    The split is drive-level, so applying it to the row frame must also
    produce disjoint row sets. A row-level split would fail here.
    """
    parts = {p: standard.apply(labelled, p) for p in
             ("train", "cal", "test")}
    for a in parts:
        for b in parts:
            if a >= b:
                continue
            ka = set(map(tuple, parts[a].select(DRIVE_KEY).unique()
                         .iter_rows()))
            kb = set(map(tuple, parts[b].select(DRIVE_KEY).unique()
                         .iter_rows()))
            assert not (ka & kb), f"{a} and {b} share drives"


def test_every_drive_assigned_exactly_once(standard, labelled):
    tr, ca, te = (standard.keys(p) for p in ("train", "cal", "test"))
    allk = set(map(tuple, drive_table(labelled).select(DRIVE_KEY)
                   .iter_rows()))
    assert tr | ca | te == allk
    assert len(tr) + len(ca) + len(te) == len(allk)


# ---------------------------------------------------------------
# INVARIANT 2 — the held-out vendor must be absent from train AND cal
# ---------------------------------------------------------------

@pytest.mark.parametrize("vendor", ["A", "B", "C"])
def test_held_out_vendor_absent_from_train_and_cal(lomo, vendor):
    """
    The single most damaging failure mode. Calibration drives from the
    held-out vendor restore exchangeability and void the experiment.
    """
    s = lomo[vendor]
    assert vendor not in set(s.train[VENDOR_COL].unique())
    assert vendor not in set(s.cal[VENDOR_COL].unique())


@pytest.mark.parametrize("vendor", ["A", "B", "C"])
def test_test_set_is_exactly_the_held_out_vendor(lomo, vendor):
    s = lomo[vendor]
    assert set(s.test[VENDOR_COL].unique()) == {vendor}


@pytest.mark.parametrize("vendor", ["A", "B", "C"])
def test_lomo_test_contains_all_held_out_drives(lomo, labelled, vendor):
    s = lomo[vendor]
    expected = drive_table(labelled).filter(
        pl.col(VENDOR_COL) == vendor
    ).height
    assert s.test.height == expected


def test_validator_catches_held_out_vendor_in_cal(labelled):
    """The guard must actually fire, not just be present."""
    s = make_lomo_split(labelled, "C")
    contaminated = Split(
        name=s.name,
        train=s.train,
        cal=pl.concat([s.cal, s.test.head(1)]),
        test=s.test,
        held_out_vendor="C",
    )
    with pytest.raises(AssertionError, match="held-out vendor"):
        validate_split(contaminated)


def test_validator_catches_train_test_overlap(labelled):
    s = make_standard_split(labelled)
    bad = Split(name="bad", train=pl.concat([s.train, s.test.head(2)]),
                cal=s.cal, test=s.test)
    with pytest.raises(AssertionError, match="train and test"):
        validate_split(bad)


def test_validator_catches_train_cal_overlap(labelled):
    s = make_standard_split(labelled)
    bad = Split(name="bad", train=pl.concat([s.train, s.cal.head(2)]),
                cal=s.cal, test=s.test)
    with pytest.raises(AssertionError, match="train and cal"):
        validate_split(bad)


# ---------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------

def test_standard_split_preserves_vendor_mix(standard, labelled):
    """
    The standard split is the exchangeability-holds reference. Vendor mix
    must match across parts or the reference wobbles with the seed.
    """
    drives = drive_table(labelled)
    overall = (
        drives.group_by(VENDOR_COL).len()
        .with_columns(pl.col("len") / drives.height)
    )
    ref = dict(zip(overall[VENDOR_COL], overall["len"]))

    for part in ("train", "cal", "test"):
        f = getattr(standard, part)
        got = f.group_by(VENDOR_COL).len().with_columns(
            pl.col("len") / f.height
        )
        for v, share in zip(got[VENDOR_COL], got["len"]):
            assert abs(share - ref[v]) < 0.06, (part, v, share, ref[v])


def test_standard_split_preserves_failure_rate(standard, labelled):
    drives = drive_table(labelled)
    ref = float(drives["drive_label"].mean())
    for part in ("train", "cal", "test"):
        f = getattr(standard, part)
        assert abs(float(f["drive_label"].mean()) - ref) < 0.06, part


def test_every_part_contains_positives(standard):
    for part in ("train", "cal", "test"):
        assert int(getattr(standard, part)["drive_label"].sum()) > 0, part


@pytest.mark.parametrize("vendor", ["A", "B", "C"])
def test_lomo_calibration_contains_positives(lomo, vendor):
    """
    An all-negative calibration set makes class-conditional coverage
    undefined for the failure class — the class the paper cares about.
    """
    assert int(lomo[vendor].cal["drive_label"].sum()) > 0


# ---------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------

def test_standard_split_fractions_are_roughly_right(standard, labelled):
    total = drive_table(labelled).height
    assert abs(standard.test.height / total - CFG.test_frac) < 0.05

    non_test = standard.train.height + standard.cal.height
    assert abs(standard.cal.height / non_test - CFG.cal_frac) < 0.05


@pytest.mark.parametrize("vendor", ["A", "B", "C"])
def test_lomo_cal_frac_is_of_training_pool(lomo, vendor):
    s = lomo[vendor]
    pool = s.train.height + s.cal.height
    assert abs(s.cal.height / pool - CFG.cal_frac) < 0.05


# ---------------------------------------------------------------
# API
# ---------------------------------------------------------------

def test_unknown_vendor_raises(labelled):
    with pytest.raises(ValueError, match="not in data"):
        make_lomo_split(labelled, "Z")


def test_all_lomo_splits_covers_every_vendor(lomo):
    assert set(lomo) == {"A", "B", "C"}
    for v, s in lomo.items():
        assert s.held_out_vendor == v
        assert s.name == f"lomo_{v}"


def test_apply_returns_row_frame(standard, labelled):
    tr = standard.apply(labelled, "train")
    assert tr.height > 0
    assert LABEL_COL in tr.columns
    keys = set(map(tuple, tr.select(DRIVE_KEY).unique().iter_rows()))
    assert keys == standard.keys("train")


def test_split_report_shape(standard, lomo):
    rep = split_report({**lomo, "standard": standard})
    assert rep.height == 4 * 3
    assert set(rep["part"].unique()) == {"train", "cal", "test"}


def test_splits_are_deterministic(labelled):
    a = make_standard_split(labelled, seed=7)
    b = make_standard_split(labelled, seed=7)
    assert a.keys("train") == b.keys("train")
    assert a.keys("cal") == b.keys("cal")
    assert a.keys("test") == b.keys("test")


def test_different_seeds_give_different_splits(labelled):
    a = make_standard_split(labelled, seed=1)
    b = make_standard_split(labelled, seed=2)
    assert a.keys("test") != b.keys("test")


def test_lomo_is_seed_independent_for_test_set(labelled):
    """
    The LOMO test set is the whole held-out vendor, so it must not
    depend on the seed. Only the train/cal division does.
    """
    a = make_lomo_split(labelled, "B", seed=1)
    b = make_lomo_split(labelled, "B", seed=2)
    assert a.keys("test") == b.keys("test")
    assert a.keys("cal") != b.keys("cal")
