"""
Slim export.

Reduces data/processed/alibaba_ssd_processed.parquet to a portable file
containing only what the experiments actually use. Run once on the
machine holding the full parquet; the output is small enough to upload
to shared storage, after which both machines can work on real data.

    python3 scripts/04_export_slim.py
    python3 scripts/04_export_slim.py --check     # verify afterwards

WHAT IS DROPPED AND WHY

  36 SMART columns  empty for every vendor (reports/attribute_availability.csv)
  50 SMART columns  populated by only one or two vendors

The second is the LOCKED common-16 feature policy, not a shortcut. Under
leave-one-manufacturer-out, a column only vendors A and B report is
entirely absent for vendor C at test time, and a column only C reports
was never seen during training. Sixteen columns survive in every fold,
so those are the sixteen the experiments can use.

  float64 -> float32   SMART values carry nowhere near 15 significant
                       digits; this halves the size with no loss that
                       matters.
  ZSTD compression     better ratio than snappy on this data.

WHAT IS KEPT

  model, disk_id       the drive key -- (model, disk_id), never disk_id
                       alone, since 119,213 disk_ids appear under more
                       than one model
  ds                   observation date
  failure_time         null for drives that never failed
  days_to_failure      recomputed here rather than carried over, so the
                       slim file is self-consistent
  the common-16        n_i and r_i for SMART IDs 5, 9, 12, 183, 184,
                       187, 197, 199

`failure_event` is deliberately NOT carried over. It equals
days_to_failure <= 0, a "has already failed" flag rather than the 30-day
target, and leaving it out removes any chance of training on it.

WHAT IS NOT DROPPED

No rows. Post-failure rows, censored drives and short-history drives all
survive; removing them is build_labels()'s job, governed by settings
that may still change. An export that silently pre-filtered rows would
make those settings a lie.

Output: data/processed/alibaba_ssd_slim/, partitioned by model, plus
reports/slim_export.txt for the shared record.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PARQUET, PROCESSED_DIR  # noqa: E402
from src.schema import COMMON_COLS, COMMON_IDS  # noqa: E402

OUT_DIR = PROCESSED_DIR / "alibaba_ssd_slim"
REPORTS = "reports"
REPORT = os.path.join(REPORTS, "slim_export.txt")

KEY = ["model", "disk_id"]
META = KEY + ["ds", "failure_time"]
KEEP = META + list(COMMON_COLS)

_lines: list[str] = []


def note(s=""):
    print(s)
    _lines.append(str(s))


def collect(lf):
    """Streaming collect across Polars 0.20 / 1.x APIs."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def dir_size(path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} TB"


# ---------------------------------------------------------------

def export(overwrite: bool = False, compression: str = "zstd") -> None:
    if not os.path.exists(PARQUET):
        raise FileNotFoundError(
            f"{PARQUET} not found. Run scripts/01_preprocess.py first, or "
            "point PARQUET in src/config.py at the file."
        )
    if os.path.exists(OUT_DIR):
        if not overwrite:
            raise FileExistsError(
                f"{OUT_DIR} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(OUT_DIR)

    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    lf = pl.scan_parquet(PARQUET)
    schema = lf.collect_schema()
    names = schema.names()

    missing = [c for c in KEEP if c not in names]
    if missing:
        raise ValueError(
            f"columns missing from the parquet: {missing}. If the schema "
            "changed, re-derive COMMON_IDS in src/schema.py from "
            "reports/attribute_availability.csv."
        )

    note("=" * 64)
    note("SLIM EXPORT")
    note("=" * 64)
    note(f"source      : {PARQUET}")
    note(f"destination : {OUT_DIR}")
    note(f"compression : {compression}")
    note()
    note(f"columns in  : {len(names)}")
    note(f"columns out : {len(KEEP)}  "
         f"({len(META)} metadata + {len(COMMON_COLS)} SMART)")
    note(f"common IDs  : {COMMON_IDS}")
    note()

    src_size = os.path.getsize(PARQUET)
    note(f"source size : {human(src_size)}")

    models = sorted(
        collect(lf.select("model").unique()).to_series().to_list()
    )
    note(f"drive models: {models}")
    note()

    t_start = time.time()
    total_rows = 0
    per_model = []

    # One model at a time. Predicate pushdown means only that model's row
    # groups are read, so peak memory stays proportional to the largest
    # single model rather than the whole 273M-row file.
    for m in models:
        t0 = time.time()
        part = (
            lf.filter(pl.col("model") == m)
            .select(KEEP)
            .with_columns(
                [pl.col(c).cast(pl.Float32) for c in COMMON_COLS]
            )
            .with_columns(
                (pl.col("failure_time") - pl.col("ds"))
                .dt.total_days().cast(pl.Float32)
                .alias("days_to_failure")
            )
            .sort(KEY + ["ds"])
        )
        df = collect(part)

        path = OUT_DIR / f"model={m}.parquet"
        df.write_parquet(path, compression=compression)

        n_drives = df.select(KEY).unique().height
        n_failed = (
            df.filter(pl.col("failure_time").is_not_null())
            .select(KEY).unique().height
        )
        size = os.path.getsize(path)
        total_rows += df.height
        per_model.append((m, df.height, n_drives, n_failed, size))

        note(f"  {m}: {df.height:>13,} rows  {n_drives:>7,} drives  "
             f"{n_failed:>6,} failed  {human(size):>10}  "
             f"({time.time() - t0:.0f}s)")
        del df

    out_size = dir_size(OUT_DIR)
    note()
    note(f"rows written: {total_rows:,}")
    note(f"output size : {human(out_size)}")
    note(f"reduction   : {src_size / max(out_size, 1):.1f}x")
    note(f"elapsed     : {time.time() - t_start:.0f}s")
    note()

    note("per model")
    note(f"  {'model':<6} {'rows':>14} {'drives':>9} {'failed':>8} "
         f"{'size':>11}")
    for m, r, d, f, s in per_model:
        note(f"  {m:<6} {r:>14,} {d:>9,} {f:>8,} {human(s):>11}")

    with open(REPORT, "w") as fh:
        fh.write("\n".join(_lines) + "\n")
    note()
    note(f"report written to {REPORT}")
    note()
    note("NEXT: run with --check to verify against "
         "reports/vendor_profile.csv, then upload the directory to "
         "shared storage.")


# ---------------------------------------------------------------

def check() -> bool:
    """
    Compare the slim export against the original profiling report.

    Drive counts, failure counts and date ranges must match exactly. If
    they do not, the export dropped something it should not have, and
    every experiment run on it would be quietly wrong.
    """
    if not os.path.exists(OUT_DIR):
        raise FileNotFoundError(f"{OUT_DIR} not found. Run the export first.")

    slim = pl.scan_parquet(str(OUT_DIR / "*.parquet"))
    got = collect(
        slim.group_by("model").agg(
            pl.col("disk_id").n_unique().alias("n_drives"),
            pl.len().alias("n_rows"),
            pl.col("ds").min().alias("first_ds"),
            pl.col("ds").max().alias("last_ds"),
            pl.col("disk_id")
              .filter(pl.col("failure_time").is_not_null())
              .n_unique().alias("n_failed_drives"),
        ).sort("model")
    )

    print("=" * 64)
    print("SLIM EXPORT CHECK")
    print("=" * 64)
    print(got)

    ok = True
    ref_path = os.path.join(REPORTS, "vendor_profile.csv")
    if os.path.exists(ref_path):
        ref = pl.read_csv(ref_path).sort("model")
        j = got.join(ref, on="model", how="inner", suffix="_ref")
        print()
        print(f"{'model':<6} {'drives':>9} {'ref':>9}  "
              f"{'failed':>8} {'ref':>8}  match")
        for r in j.iter_rows(named=True):
            d_ok = r["n_drives"] == r["n_drives_ref"]
            f_ok = r["n_failed_drives"] == r["n_failed_drives_ref"]
            ok = ok and d_ok and f_ok
            print(f"{r['model']:<6} {r['n_drives']:>9,} "
                  f"{r['n_drives_ref']:>9,}  "
                  f"{r['n_failed_drives']:>8,} "
                  f"{r['n_failed_drives_ref']:>8,}  "
                  f"{'OK' if d_ok and f_ok else 'MISMATCH'}")
        if j.height != got.height:
            ok = False
            print("\nWARNING: model list differs from vendor_profile.csv")
    else:
        print(f"\n{ref_path} not found; cannot cross-check. "
              "Run scripts/verify_and_profile.py first.")
        ok = False

    # The 16 SMART columns must be populated, and NaN must be counted as
    # missing: pd.to_numeric(errors='coerce') produced NaN, and Polars
    # treats NaN and null as different things. Checking is_null() alone
    # would report an entirely empty column as fully populated.
    print()
    print("column availability (non-missing fraction, NaN counted as missing)")
    avail = collect(
        slim.select([
            (1 - (pl.col(c).is_null()
                  | pl.col(c).is_nan().fill_null(True)).mean())
            .round(4).alias(c)
            for c in COMMON_COLS
        ])
    )
    row = avail.row(0, named=True)
    for c in COMMON_COLS:
        flag = "" if row[c] > 0.5 else "   <-- SPARSE"
        print(f"  {c:<8} {row[c]:.4f}{flag}")
        if row[c] <= 0.5:
            ok = False

    print()
    print("PASS" if ok else "FAIL -- do not upload until resolved")
    return ok


# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true",
                    help="verify an existing export instead of building one")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing export directory")
    ap.add_argument("--compression", default="zstd",
                    choices=["zstd", "snappy", "lz4", "gzip"])
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if check() else 1)
    export(overwrite=args.overwrite, compression=args.compression)


if __name__ == "__main__":
    main()