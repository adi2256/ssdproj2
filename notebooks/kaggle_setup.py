"""
Kaggle notebook setup.

Everything the notebook needs that is not part of `src/`. Previously
this lived only in notebook cells, so it vanished on every kernel
restart and could not be reviewed or version-controlled.

In a Kaggle notebook, cell 1 is:

    !rm -rf /kaggle/working/ssd && git clone -q <repo> /kaggle/working/ssd
    import sys; sys.path.insert(0, "/kaggle/working/ssd")
    import os; os.chdir("/kaggle/working/ssd")
    from notebooks.kaggle_setup import setup
    ctx = setup()

`ctx` carries SLIM, load_slim, collect and COLS.
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass
from typing import Callable

import polars as pl

from src.config import CFG, FLEET_DRIVES, FLEET_FAILED, FLEET_ROWS
from src.evaluation import RunConfig

# Columns that belong in every results table. Set size and singleton
# rate are not optional: coverage alone hides the case where a
# guarantee is met by widening sets until they carry no information.
RESULT_COLS = [
    "split", "model", "conformal",
    "win_auc", "drv_f0.5",
    "empirical_coverage", "coverage_healthy", "coverage_failure",
    "avg_set_size", "singleton_rate", "doubleton_rate", "empty_rate",
    "fallback_rate",
    "n_features", "n_dropped_constant", "normalize",
    "n_test_windows", "n_failures_test", "test_prevalence",
]


def collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Streaming collect across Polars 0.20 / 1.x APIs."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def find_slim(root: str = "/kaggle/input") -> str:
    """Locate the slim parquet export wherever Kaggle mounted it."""
    hits = sorted(glob.glob(os.path.join(root, "**", "*.parquet"),
                            recursive=True))
    if not hits:
        raise FileNotFoundError(
            f"no parquet under {root}. Attach the slim dataset to the "
            "notebook (Add Data)."
        )
    return os.path.join(os.path.dirname(hits[0]), "*.parquet")


def verify_data(slim_glob: str) -> None:
    """
    Confirm the upload matches the fleet totals in reports/.

    A truncated upload would silently change every downstream number,
    so this is an assertion rather than a print.
    """
    lf = pl.scan_parquet(slim_glob)
    s = collect(lf.select(
        pl.len().alias("rows"),
        pl.struct(["model", "disk_id"]).n_unique().alias("drives"),
    ))
    failed = collect(
        lf.filter(pl.col("failure_time").is_not_null())
          .select(pl.struct(["model", "disk_id"]).n_unique())
    ).item()

    print(f"  rows   {s['rows'][0]:>12,}  expect {FLEET_ROWS:,}")
    print(f"  drives {s['drives'][0]:>12,}  expect {FLEET_DRIVES:,}")
    print(f"  failed {failed:>12,}  expect {FLEET_FAILED:,}")

    if s["rows"][0] != FLEET_ROWS or failed != FLEET_FAILED:
        raise AssertionError(
            "slim export does not match reports/vendor_profile.csv — "
            "the upload is incomplete. Do not run experiments on it."
        )


def verify_code() -> None:
    """
    Fail loudly if the kernel is running a stale copy of src/.

    Python caches imported modules, so a fresh git clone does not
    replace what is already in memory. Restart the kernel if this
    raises.
    """
    fields = RunConfig.__dataclass_fields__
    missing = [f for f in ("normalize", "drop_constant_test_features")
               if f not in fields]
    if missing:
        raise AssertionError(
            f"RunConfig is missing {missing}. The kernel is running a "
            "stale src/ — restart the session and re-run this cell."
        )
    from src.data.windowing import (Normalizer, constant_features,  # noqa
                                    drop_features)
    if "method" not in Normalizer.__dataclass_fields__:
        raise AssertionError(
            "Normalizer has no `method` field — stale src/. Restart."
        )
    print("  code OK: normalize, drop_constant_test_features, rank scaler")


def make_loader(slim_glob: str) -> Callable:
    """Build load_slim() bound to this notebook's data path."""

    def load_slim(n_drives=None, keep_all_failed=None, seed=None):
        """
        Read the slim export into the frame build_labels() expects.

        Subsampling happens at DRIVE level and before labelling, so no
        drive is ever half-present.

        keep_all_failed retains every one of the 16,305 failed drives
        and thins only healthy ones. An earlier run used a 20,000-drive
        proportional sample, which left about 95 failed drives in each
        LOMO test fold; windows from one drive are 30 overlapping views
        of the same failure, so the effective sample size was the drive
        count, not the window count, and every held-out fold came out at
        AUC 0.50.
        """
        n_drives = n_drives if n_drives is not None else CFG.subset_n_drives
        keep_all_failed = (keep_all_failed if keep_all_failed is not None
                           else CFG.keep_all_failed)
        seed = seed if seed is not None else CFG.seed

        lf = pl.scan_parquet(slim_glob)
        if n_drives is None:
            print("loading whole fleet")
            return collect(lf)

        drives = collect(
            lf.select(["model", "disk_id", "failure_time"])
              .group_by(["model", "disk_id"])
              .agg(pl.col("failure_time").max().is_not_null().alias("failed"))
        )
        fail = drives.filter(pl.col("failed"))
        heal = drives.filter(~pl.col("failed"))

        if keep_all_failed:
            n_h = max(0, n_drives - fail.height)
            keep = pl.concat(
                [fail, heal.sample(n=min(n_h, heal.height), seed=seed)]
            )
            n_f = fail.height
        else:
            rate = fail.height / drives.height
            n_f = round(n_drives * rate)
            keep = pl.concat([
                fail.sample(n=n_f, seed=seed),
                heal.sample(n=n_drives - n_f, seed=seed),
            ])

        print(f"kept {keep.height:,} drives ({n_f:,} failed)")
        return collect(
            lf.join(keep.select(["model", "disk_id"]).lazy(),
                    on=["model", "disk_id"], how="inner")
        )

    return load_slim


@dataclass
class Context:
    slim: str
    load_slim: Callable
    collect: Callable
    cols: list


def setup(root: str = "/kaggle/input", verify: bool = True) -> Context:
    """Verify code and data, then return the notebook's helpers."""
    print("=" * 60)
    print("SETUP")
    print("=" * 60)

    verify_code()

    slim = find_slim(root)
    n = len(glob.glob(slim))
    print(f"  data: {n} parquet files")
    print(f"        {slim}")

    if verify:
        verify_data(slim)

    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(240)

    print("\nready\n")
    return Context(slim=slim, load_slim=make_loader(slim),
                   collect=collect, cols=RESULT_COLS)


def show(res: pl.DataFrame, cols: list | None = None) -> pl.DataFrame:
    """Select the reporting columns that exist in this result frame."""
    cols = cols or RESULT_COLS
    return res.select([c for c in cols if c in res.columns])
