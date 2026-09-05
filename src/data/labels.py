"""
Label construction.

Turns the raw parquet (SMART rows + joined failure_time) into a
supervised target.

RULE
    label = 1  if  0 < days_to_failure <= horizon
    label = 0  otherwise, where the outcome is actually observable

Three row classes are removed rather than labelled, because assigning
them a label would be a guess:

1. Post-failure and failure-day rows (days_to_failure <= 0).
   The drive has already failed; these rows are not predictions.
   The real fleet contains ~1.26M of them, so this is not a corner case.
   Note the strict inequality: the failure day itself (dtf == 0) is not
   a positive.

2. The final `horizon` days of any drive that did not fail.
   For a drive last seen at T, the interval (T, T + horizon] is not
   observed, so "did it fail within 30 days" has no answer. Labelling
   these 0 asserts survival that was never witnessed. This applies to
   BOTH right-censored drives and drives surviving to the global end —
   the same reasoning covers them, so the rule is uniform.

3. Drives with too little history to form a window (< window_len rows).

Consequence worth stating in the paper: censoring is not a nuisance to
be imputed away, it is a region where the label is undefined. Dropping
these rows costs `horizon` days per non-failed drive but keeps every
remaining label a fact.

The real fleet has 133,804 right-censored drives (28%), so the size of
this drop is material and should be reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..config import CFG, DRIVE_KEY

LABEL_COL = "label"
VENDOR_COL = "vendor"


def vendor_expr() -> pl.Expr:
    """Vendor is the 2nd character of the model code: MA1 -> A."""
    return pl.col("model").str.slice(1, 1).alias(VENDOR_COL)


@dataclass
class LabelStats:
    """Row accounting. Every dropped row is attributed to a reason."""
    n_rows_in: int
    n_post_failure_dropped: int
    n_unobservable_dropped: int
    n_short_history_dropped: int
    n_rows_out: int
    n_positive: int
    n_drives_in: int
    n_drives_out: int
    horizon_days: int

    @property
    def positive_rate(self) -> float:
        return self.n_positive / self.n_rows_out if self.n_rows_out else 0.0

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["positive_rate"] = self.positive_rate
        return d

    def __str__(self) -> str:
        return (
            f"rows in {self.n_rows_in:,} -> out {self.n_rows_out:,}\n"
            f"  dropped post-failure   : {self.n_post_failure_dropped:,}\n"
            f"  dropped unobservable   : {self.n_unobservable_dropped:,}\n"
            f"  dropped short history  : {self.n_short_history_dropped:,}\n"
            f"  positives              : {self.n_positive:,} "
            f"({100 * self.positive_rate:.4f}%)\n"
            f"  drives {self.n_drives_in:,} -> {self.n_drives_out:,}"
        )


def build_labels(
    df: pl.DataFrame,
    horizon_days: int | None = None,
    window_len: int | None = None,
    drop_post_failure: bool = True,
    drop_unobservable: bool = True,
    min_history: int | None = None,
) -> tuple[pl.DataFrame, LabelStats]:
    """
    Add `label` and `vendor`, dropping rows whose outcome is undefined.

    Parameters mirror the module docstring. Defaults come from CFG.
    Returns the labelled frame and a full row accounting.

    `min_history` defaults to `window_len`: a drive needs at least one
    full window to produce a single training example.
    """
    horizon = horizon_days if horizon_days is not None else CFG.horizon_days
    window = window_len if window_len is not None else CFG.window_len
    min_history = min_history if min_history is not None else window

    n_rows_in = df.height
    n_drives_in = df.select(DRIVE_KEY).unique().height

    out = df.with_columns(vendor_expr())
    # Use calendar days because SMART observations are daily.
    out = out.with_columns(
        (
            pl.col("failure_time").dt.date()
            - pl.col("ds").dt.date()
        )
        .dt.total_days()
        .cast(pl.Float64)
        .alias("days_to_failure")
    )
    # -- 1. post-failure and failure-day rows -------------------
    n_post = 0
    if drop_post_failure:
        mask = pl.col("days_to_failure") <= 0
        n_post = out.filter(mask.fill_null(False)).height
        out = out.filter(~mask.fill_null(False))

    # -- 2. unobservable tail of non-failed drives --------------
    #
    # For a drive that never failed, the last `horizon` days have no
    # observable outcome. Computed per drive from its own last date, so
    # right-censored drives and survivors are handled identically.
    n_unobs = 0
    if drop_unobservable:
        before = out.height
        out = (
            out.with_columns(
                pl.col("ds").max().over(DRIVE_KEY).alias("_last_ds")
            )
            .with_columns(
                (pl.col("_last_ds") - pl.col("ds"))
                .dt.total_days().alias("_days_to_last")
            )
            .filter(
                pl.col("failure_time").is_not_null()
                | (pl.col("_days_to_last") >= horizon)
            )
            .drop(["_last_ds", "_days_to_last"])
        )
        n_unobs = before - out.height

    # -- 3. the label -------------------------------------------
    out = out.with_columns(
        (
            (pl.col("days_to_failure") > 0)
            & (pl.col("days_to_failure") <= horizon)
        ).fill_null(False).cast(pl.Int8).alias(LABEL_COL)
    )

    # -- 4. drives too short to window --------------------------
    n_short = 0
    if min_history > 1:
        before = out.height
        out = (
            out.with_columns(pl.len().over(DRIVE_KEY).alias("_n"))
            .filter(pl.col("_n") >= min_history)
            .drop("_n")
        )
        n_short = before - out.height

    out = out.sort(DRIVE_KEY + ["ds"])

    stats = LabelStats(
        n_rows_in=n_rows_in,
        n_post_failure_dropped=n_post,
        n_unobservable_dropped=n_unobs,
        n_short_history_dropped=n_short,
        n_rows_out=out.height,
        n_positive=int(out[LABEL_COL].sum()) if out.height else 0,
        n_drives_in=n_drives_in,
        n_drives_out=out.select(DRIVE_KEY).unique().height,
        horizon_days=horizon,
    )
    return out, stats


def drive_table(labelled: pl.DataFrame) -> pl.DataFrame:
    """
    One row per drive: key, model, vendor, whether it ever carries a
    positive, and its row count.

    This is the unit splits operate on. Splitting on rows would leak a
    drive across train and test.
    """
    return (
        labelled.group_by(DRIVE_KEY)
        .agg(
            pl.col(VENDOR_COL).first(),
            pl.col(LABEL_COL).max().alias("drive_label"),
            pl.len().alias("n_rows"),
            pl.col("ds").min().alias("first_ds"),
            pl.col("ds").max().alias("last_ds"),
        )
        .sort(DRIVE_KEY)
    )


def label_summary(labelled: pl.DataFrame) -> pl.DataFrame:
    """Per-vendor row and drive counts under the applied rule."""
    drives = drive_table(labelled)
    return (
        drives.group_by(VENDOR_COL)
        .agg(
            pl.len().alias("n_drives"),
            pl.col("drive_label").sum().alias("n_positive_drives"),
            pl.col("n_rows").sum().alias("n_rows"),
        )
        .with_columns(
            (pl.col("n_positive_drives") / pl.col("n_drives") * 100)
            .round(4).alias("drive_positive_pct")
        )
        .sort(VENDOR_COL)
    )
