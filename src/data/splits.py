"""
Split construction.

Two split types:

STANDARD — stratified by (model, drive_label) over all vendors.
    The exchangeability-holds reference point. Conformal coverage must
    reach its nominal level here. Stratification keeps vendor mix and
    failure rate matched between train, cal and test, so the reference
    is as clean as possible and does not wobble with the seed.

LOMO — one vendor held out entirely as test; train and calibration
    drawn only from the other two.
    Exchangeability is broken by construction. Coverage is not expected
    to hold, and measuring the gap is the paper's primary result.

THREE INVARIANTS, asserted rather than assumed:

  1. Splits are grouped at drive level. A drive contributes many daily
     rows; a row-level split puts the same drive in train and test.
  2. Under LOMO the held-out vendor appears in NEITHER train NOR
     calibration. A calibration set containing held-out-vendor drives
     restores exchangeability and voids the experiment — this is the
     single most damaging thing that can go wrong here.
  3. Calibration is disjoint from train. Conformal requires scores from
     data the model did not fit.

Feature selection is NOT performed here, but note the related rule: it
must re-run inside each fold on training vendors only. Selecting once on
pooled data leaks held-out labels into the feature set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..config import CFG, DRIVE_KEY, VENDORS
from .labels import VENDOR_COL, drive_table


@dataclass(frozen=True)
class Split:
    """
    Drive-level split. Each frame holds the (model, disk_id) keys and
    metadata for its part.
    """
    name: str
    train: pl.DataFrame
    cal: pl.DataFrame
    test: pl.DataFrame
    held_out_vendor: str | None = None

    def keys(self, part: str) -> set[tuple]:
        frame = getattr(self, part)
        return set(map(tuple, frame.select(DRIVE_KEY).iter_rows()))

    def apply(self, labelled: pl.DataFrame, part: str) -> pl.DataFrame:
        """Filter a labelled row frame down to one part of the split."""
        return labelled.join(
            getattr(self, part).select(DRIVE_KEY), on=DRIVE_KEY, how="inner"
        )

    def summary(self) -> pl.DataFrame:
        rows = []
        for part in ("train", "cal", "test"):
            f = getattr(self, part)
            rows.append({
                "split": self.name,
                "part": part,
                "n_drives": f.height,
                "n_positive_drives": int(f["drive_label"].sum()),
                "vendors": ",".join(sorted(f[VENDOR_COL].unique())),
            })
        return pl.DataFrame(rows)


def _stratified_assign(drives: pl.DataFrame, fracs: dict[str, float],
                       strata: list[str], seed: int) -> pl.DataFrame:
    """
    Assign each drive to a part, sampling within each stratum so the
    stratum mix is preserved across parts.
    """
    rng = np.random.default_rng(seed)
    names = list(fracs)
    out = []

    for _, group in drives.group_by(strata, maintain_order=True):
        idx = rng.permutation(group.height)
        g = group[idx]

        bounds, acc = [], 0.0
        for nm in names[:-1]:
            acc += fracs[nm]
            bounds.append(int(round(acc * g.height)))
        starts = [0] + bounds
        ends = bounds + [g.height]

        for nm, s, e in zip(names, starts, ends):
            if e > s:
                out.append(g[s:e].with_columns(pl.lit(nm).alias("_part")))

    return pl.concat(out) if out else drives.with_columns(
        pl.lit(names[0]).alias("_part")
    )


def make_standard_split(
    labelled: pl.DataFrame,
    test_frac: float | None = None,
    cal_frac: float | None = None,
    seed: int | None = None,
) -> Split:
    """
    Stratified drive-level train/cal/test over all vendors.

    `cal_frac` is a fraction of the non-test drives, matching its LOMO
    meaning (a slice of the training population, not of the whole fleet).
    """
    test_frac = test_frac if test_frac is not None else CFG.test_frac
    cal_frac = cal_frac if cal_frac is not None else CFG.cal_frac
    seed = seed if seed is not None else CFG.seed

    drives = drive_table(labelled)
    fracs = {
        "test": test_frac,
        "cal": (1 - test_frac) * cal_frac,
        "train": (1 - test_frac) * (1 - cal_frac),
    }
    assigned = _stratified_assign(
        drives, fracs, strata=["model", "drive_label"], seed=seed
    )

    split = Split(
        name="standard",
        train=assigned.filter(pl.col("_part") == "train").drop("_part"),
        cal=assigned.filter(pl.col("_part") == "cal").drop("_part"),
        test=assigned.filter(pl.col("_part") == "test").drop("_part"),
    )
    validate_split(split)
    return split


def make_lomo_split(
    labelled: pl.DataFrame,
    held_out_vendor: str,
    cal_frac: float | None = None,
    seed: int | None = None,
) -> Split:
    """
    Hold out one vendor entirely. Train and calibration come only from
    the remaining vendors, stratified by (model, drive_label).
    """
    cal_frac = cal_frac if cal_frac is not None else CFG.cal_frac
    seed = seed if seed is not None else CFG.seed

    drives = drive_table(labelled)
    present = set(drives[VENDOR_COL].unique())
    if held_out_vendor not in present:
        raise ValueError(
            f"vendor {held_out_vendor!r} not in data; found {sorted(present)}"
        )

    test = drives.filter(pl.col(VENDOR_COL) == held_out_vendor)
    pool = drives.filter(pl.col(VENDOR_COL) != held_out_vendor)
    if pool.height == 0:
        raise ValueError(f"no training drives left after holding out "
                         f"{held_out_vendor!r}")

    assigned = _stratified_assign(
        pool, {"cal": cal_frac, "train": 1 - cal_frac},
        strata=["model", "drive_label"], seed=seed,
    )

    split = Split(
        name=f"lomo_{held_out_vendor}",
        train=assigned.filter(pl.col("_part") == "train").drop("_part"),
        cal=assigned.filter(pl.col("_part") == "cal").drop("_part"),
        test=test,
        held_out_vendor=held_out_vendor,
    )
    validate_split(split)
    return split


def make_all_lomo_splits(
    labelled: pl.DataFrame,
    vendors: list[str] | None = None,
    **kwargs,
) -> dict[str, Split]:
    """One LOMO split per vendor present in the data."""
    drives = drive_table(labelled)
    present = sorted(set(drives[VENDOR_COL].unique()))
    vendors = vendors if vendors is not None else present
    return {v: make_lomo_split(labelled, v, **kwargs) for v in vendors}


# ---------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------

def validate_split(split: Split) -> None:
    """
    Assert the three invariants. Called automatically on construction;
    call again after any manual edit to a split.
    """
    tr, ca, te = (split.keys(p) for p in ("train", "cal", "test"))

    # 2 first: it is the most damaging failure and the most specific
    # diagnosis. A held-out-vendor drive in cal would also trip the
    # disjointness check below, but with a far less useful message.
    if split.held_out_vendor is not None:
        v = split.held_out_vendor
        for part in ("train", "cal"):
            found = set(getattr(split, part)[VENDOR_COL].unique())
            if v in found:
                raise AssertionError(
                    f"{split.name}: held-out vendor {v!r} present in "
                    f"{part}. This restores exchangeability and voids "
                    f"the LOMO experiment."
                )
        test_vendors = set(split.test[VENDOR_COL].unique())
        if test_vendors != {v}:
            raise AssertionError(
                f"{split.name}: test must contain only vendor {v!r}, "
                f"found {sorted(test_vendors)}"
            )

    # 1 + 3: drive-level disjointness
    for a, b, msg in [
        (tr, ca, "train and cal share drives (conformal needs "
                 "calibration scores from unfitted data)"),
        (tr, te, "train and test share drives (drive-level leakage)"),
        (ca, te, "cal and test share drives (drive-level leakage)"),
    ]:
        overlap = a & b
        if overlap:
            raise AssertionError(
                f"{split.name}: {msg}; {len(overlap)} shared, "
                f"e.g. {sorted(overlap)[:3]}"
            )

    if not te:
        raise AssertionError(f"{split.name}: empty test set")
    if not ca:
        raise AssertionError(f"{split.name}: empty calibration set")


def split_report(splits: dict[str, Split] | Split) -> pl.DataFrame:
    """Stacked summaries, one block per split."""
    if isinstance(splits, Split):
        splits = {splits.name: splits}
    return pl.concat([s.summary() for s in splits.values()])
