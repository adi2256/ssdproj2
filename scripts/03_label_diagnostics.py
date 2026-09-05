"""
Label-rule diagnostics.

Run on the machine holding data/processed/alibaba_ssd_processed.parquet.
Writes reports/label_diagnostics.txt, which is small and committed —
that report is the contract that lets label construction be developed
without access to the parquet.

    python3 scripts/03_label_diagnostics.py

Answers four questions that determine the label rule:

  1. Does any drive report telemetry on its failure date? Decides
     whether the horizon rule needs `0 <` or `0 <=`.
  2. How much post-failure telemetry exists per drive? A drive still
     emitting SMART data long after its recorded failure is a dataset
     quirk a reviewer will ask about.
  3. How many failed drives have a full horizon of prior history?
     Drives failing early in 2018 have truncated windows and may need a
     minimum-history filter.
  4. How deep is right censoring? A drive stopping 3 days before the end
     is not the same as one stopping 400 days before.

Read-only. Does not modify the parquet.
"""

import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG, PARQUET  # noqa: E402

REPORTS = "reports"
OUT = os.path.join(REPORTS, "label_diagnostics.txt")

os.makedirs(REPORTS, exist_ok=True)
_lines = []


def note(s=""):
    print(s)
    _lines.append(str(s))


def collect(lf):
    """Streaming collect across Polars 0.20 / 1.x APIs."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def describe(series, label):
    d = series.describe()
    note(f"  {label}")
    for row in d.iter_rows():
        note(f"    {row[0]:<10} {row[1]}")


def main():
    if not os.path.exists(PARQUET):
        raise FileNotFoundError(
            f"{PARQUET} not found. Run scripts/01_preprocess.py first, "
            "or point PARQUET in src/config.py at the file."
        )

    lf = pl.scan_parquet(PARQUET)
    h = CFG.horizon_days

    note("=" * 64)
    note("LABEL DIAGNOSTICS")
    note(f"parquet : {PARQUET}")
    note(f"horizon : {h} days")
    note("=" * 64)

    # -----------------------------------------------------------
    # 1. days_to_failure boundary
    # -----------------------------------------------------------
    note()
    note("[1] days_to_failure boundary")
    b = collect(lf.select(
        (pl.col("days_to_failure") == 0).sum().alias("eq_0"),
        (pl.col("days_to_failure") < 0).sum().alias("lt_0"),
        (pl.col("days_to_failure") > 0).sum().alias("gt_0"),
        pl.col("days_to_failure").is_null().sum().alias("null"),
        pl.col("days_to_failure").min().alias("min"),
        pl.col("days_to_failure").max().alias("max"),
    ))
    r = b.row(0, named=True)
    note(f"  rows at dtf == 0 : {r['eq_0']:,}")
    note(f"  rows at dtf <  0 : {r['lt_0']:,}   (post-failure)")
    note(f"  rows at dtf >  0 : {r['gt_0']:,}")
    note(f"  rows null (healthy drives) : {r['null']:,}")
    note(f"  range            : [{r['min']}, {r['max']}]")
    if r["eq_0"] == 0:
        note("  -> no drive reports on its failure date; "
             "`0 < dtf <= h` and `0 <= dtf <= h` are equivalent here")
    else:
        note("  -> failure-date rows EXIST; the label rule must use "
             "`0 < dtf <= h` so the failure day is not a positive")

    # -----------------------------------------------------------
    # 2. post-failure telemetry
    # -----------------------------------------------------------
    note()
    note("[2] post-failure rows per failed drive")
    post = collect(
        lf.filter(pl.col("days_to_failure") < 0)
        .group_by(["model", "disk_id"]).len()
    )
    if post.height == 0:
        note("  none")
    else:
        note(f"  drives with post-failure telemetry: {post.height:,}")
        describe(post["len"], "rows per drive")
        note(f"  drives reporting >90 days after failure: "
             f"{post.filter(pl.col('len') > 90).height:,}")
        note(f"  drives reporting >180 days after failure: "
             f"{post.filter(pl.col('len') > 180).height:,}")

    # -----------------------------------------------------------
    # 3. horizon coverage
    # -----------------------------------------------------------
    note()
    note(f"[3] horizon coverage per failed drive (0 < dtf <= {h})")
    hz = collect(
        lf.filter((pl.col("days_to_failure") > 0)
                  & (pl.col("days_to_failure") <= h))
        .group_by(["model", "disk_id"]).len()
    )
    total_failed = collect(
        lf.filter(pl.col("failure_time").is_not_null())
        .select(["model", "disk_id"]).unique().select(pl.len())
    ).item()

    note(f"  failed drives total          : {total_failed:,}")
    note(f"  with >=1 horizon row         : {hz.height:,}")
    note(f"  with a FULL {h}-row horizon   : "
         f"{hz.filter(pl.col('len') == h).height:,}")
    note(f"  with 0 horizon rows          : {total_failed - hz.height:,}")
    if hz.height:
        describe(hz["len"], "horizon rows per drive")

    # -----------------------------------------------------------
    # 3b. pre-horizon history (window feasibility)
    # -----------------------------------------------------------
    note()
    note(f"[3b] total history per failed drive "
         f"(window_len={CFG.window_len})")
    hist = collect(
        lf.filter(pl.col("failure_time").is_not_null())
        .group_by(["model", "disk_id"]).len()
    )
    if hist.height:
        describe(hist["len"], "rows per failed drive")
        need = h + CFG.window_len
        note(f"  drives with < {need} rows (horizon + window): "
             f"{hist.filter(pl.col('len') < need).height:,}")

    # -----------------------------------------------------------
    # 4. censoring depth
    # -----------------------------------------------------------
    note()
    note("[4] censoring depth")
    last = collect(
        lf.group_by(["model", "disk_id"])
        .agg(pl.col("ds").max().alias("last_ds"),
             pl.col("failure_time").max().alias("failure_time"))
    )
    end = last["last_ds"].max()
    note(f"  global last date: {end}")

    gap = (
        last.filter(pl.col("failure_time").is_null())
        .with_columns(
            (pl.lit(end) - pl.col("last_ds"))
            .dt.total_days().alias("days_before_end")
        )
        .filter(pl.col("days_before_end") > 0)
    )
    note(f"  censored drives (healthy, ended early): {gap.height:,}")
    if gap.height:
        describe(gap["days_before_end"], "days before global end")
        for t in (7, 30, 90, 180, 365):
            note(f"  censored by >{t:>3} days: "
                 f"{gap.filter(pl.col('days_before_end') > t).height:,}")

    # -----------------------------------------------------------
    # 5. projected label counts
    # -----------------------------------------------------------
    note()
    note("[5] projected label counts under `0 < dtf <= h`, "
         "post-failure rows dropped")
    counts = collect(
        lf.filter(pl.col("days_to_failure").is_null()
                  | (pl.col("days_to_failure") >= 0))
        .with_columns(pl.col("model").str.slice(1, 1).alias("vendor"))
        .with_columns(
            ((pl.col("days_to_failure") > 0)
             & (pl.col("days_to_failure") <= h))
            .fill_null(False).cast(pl.Int8).alias("label")
        )
        .group_by("vendor")
        .agg(pl.len().alias("n_rows"),
             pl.col("label").sum().alias("n_positive"))
        .sort("vendor")
    )
    note("  vendor   rows            positives     rate")
    for row in counts.iter_rows(named=True):
        rate = 100 * row["n_positive"] / row["n_rows"]
        note(f"  {row['vendor']:<8} {row['n_rows']:>14,} "
             f"{row['n_positive']:>12,}   {rate:.4f}%")
    note("  -> row-level positive rate; drive-level rates are in "
         "reports/vendor_profile.csv")

    with open(OUT, "w") as fh:
        fh.write("\n".join(_lines) + "\n")
    note()
    note(f"Written to {OUT}")


if __name__ == "__main__":
    main()