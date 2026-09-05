"""
Verification and profiling of alibaba_ssd_processed.parquet

Run BEFORE building labels or splits. Checks the failure modes that
corrupt downstream results silently, and produces the per-vendor
profiling table that decides whether LOMO is viable.

Outputs (reports/):
  column_order_check.txt
  row_reconciliation.txt
  vendor_profile.csv
  attribute_availability.csv
  verification_summary.txt
"""

import glob
import os

import polars as pl

RAW_DIRS = ["data/raw/smartlog2018ssd", "data/raw/smartlog2019ssd"]
FAILURE_GLOB = "data/raw/ssd_failure_label.csv/**/*.csv"
PARQUET = "data/processed/alibaba_ssd_processed.parquet"
REPORTS = "reports"

META_COLS = {"disk_id", "ds", "model", "failure_time",
             "days_to_failure", "failure_event"}

os.makedirs(REPORTS, exist_ok=True)
_summary = []


def note(line):
    print(line)
    _summary.append(line)


def collect(lf):
    """Streaming collect that works across Polars 0.20 / 1.x APIs."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def missing_expr(col, dtype):
    """
    Treat both null and NaN as missing.

    preprocess_data.py used pd.to_numeric(errors='coerce'), which
    produces NaN. Depending on the pandas/pyarrow versions those may
    land in Parquet as NaN values rather than nulls, and Polars keeps
    the two distinct. Checking is_null() alone would report ~100%
    availability for columns that are entirely empty.
    """
    e = pl.col(col).is_null()
    if dtype in (pl.Float32, pl.Float64):
        e = e | pl.col(col).is_nan().fill_null(True)
    return e


# ---------------------------------------------------------------
# CHECK 1 — column order across raw daily CSVs
#
# preprocess_data.py used table.cast(writer.schema, safe=False),
# which aligns fields BY POSITION. Any file with a different column
# order had its values written under the wrong names, silently.
# ---------------------------------------------------------------

def check_column_order():
    files = []
    for d in RAW_DIRS:
        files.extend(sorted(glob.glob(os.path.join(d, "*.csv"))))
    if not files:
        raise FileNotFoundError("No raw CSVs found for order check.")

    reference = pl.read_csv(files[0], n_rows=0).columns
    mismatches = []
    for f in files:
        cols = pl.read_csv(f, n_rows=0).columns
        if cols != reference:
            kind = ("REORDERED" if set(cols) == set(reference)
                    else "DIFFERENT COLUMN SET")
            mismatches.append((os.path.basename(f), kind))

    with open(f"{REPORTS}/column_order_check.txt", "w") as fh:
        fh.write(f"Files checked : {len(files)}\n")
        fh.write(f"Reference     : {os.path.basename(files[0])}\n")
        fh.write(f"Column count  : {len(reference)}\n\n")
        if mismatches:
            fh.write("!!! MISMATCHES — parquet is likely corrupted !!!\n")
            for name, kind in mismatches:
                fh.write(f"  {name}: {kind}\n")
        else:
            fh.write("OK: identical column order in every file.\n")

    if mismatches:
        note(f"[1] Column order: {len(mismatches)} MISMATCH(ES) "
             f"-- parquet suspect, re-ingest with name-based align")
    else:
        note(f"[1] Column order: OK across {len(files)} files "
             f"({len(reference)} columns)")
    return not mismatches


# ---------------------------------------------------------------
# CHECK 2 — row reconciliation (detects merge fan-out)
# ---------------------------------------------------------------

def check_row_counts(parquet_rows):
    patterns = [os.path.join(d, "*.csv") for d in RAW_DIRS]
    raw_rows = 0
    for p in patterns:
        raw_rows += collect(
            pl.scan_csv(p).select(pl.len())
        ).item()

    fail_files = glob.glob(FAILURE_GLOB, recursive=True)
    dup = 0
    if fail_files:
        fail = pl.read_csv(fail_files[0])
        dup = fail.height - fail.select(["model", "disk_id"]).unique().height

    delta = parquet_rows - raw_rows
    with open(f"{REPORTS}/row_reconciliation.txt", "w") as fh:
        fh.write(f"Raw CSV rows       : {raw_rows:,}\n")
        fh.write(f"Parquet rows       : {parquet_rows:,}\n")
        fh.write(f"Difference         : {delta:+,}\n")
        fh.write(f"Duplicate fail keys: {dup}\n")

    note(f"[2] Rows raw={raw_rows:,} parquet={parquet_rows:,} "
         f"delta={delta:+,} | dup failure keys={dup}")
    if delta > 0:
        note("    -> fan-out: duplicate (model, disk_id) in failure table")
    elif delta < 0:
        note("    -> rows dropped by dropna() in preprocessing")
    return delta, dup


# ---------------------------------------------------------------
# CHECK 3 — model code format, drive key, censoring, post-failure
# Single pass over the parquet.
# ---------------------------------------------------------------

def check_integrity(lf):
    models = collect(lf.select("model").unique()).to_series().to_list()
    note(f"[3] Drive models present: {sorted(models)}")
    bad = [m for m in models if m is None or len(str(m)) < 2]
    if bad:
        raise ValueError(
            f"Cannot derive vendor from model codes {bad}. "
            "Expected 'MA1'-style codes; fix vendor_expr() before use."
        )

    stats = collect(
        lf.select(
            pl.len().alias("n_rows"),
            (pl.col("days_to_failure") < 0).sum().alias("post_failure"),
            pl.col("failure_event").sum().alias("event_positives"),
            pl.col("ds").max().alias("global_end"),
        )
    )
    n_rows = stats["n_rows"][0]
    note(f"    Total rows                 : {n_rows:,}")
    note(f"    Rows after recorded failure: {stats['post_failure'][0]:,}")
    note(f"    failure_event == 1         : {stats['event_positives'][0]:,} "
         f"(a 'has failed' flag, NOT the 30-day target)")

    per_drive = collect(
        lf.group_by(["model", "disk_id"])
        .agg(pl.col("ds").max().alias("last_ds"))
    )
    end = stats["global_end"][0]
    censored = per_drive.filter(pl.col("last_ds") < end).height
    note(f"    Global last date           : {end}")
    note(f"    Drives ending early        : {censored:,} "
         f"(right-censored, not confirmed healthy)")

    collisions = (
        per_drive.group_by("disk_id").len()
        .filter(pl.col("len") > 1).height
    )
    note(f"    disk_id under >1 model     : {collisions:,} "
         f"({'key MUST be (model, disk_id)' if collisions else 'globally unique'})")
    return n_rows


# ---------------------------------------------------------------
# PROFILE — the numbers that decide whether LOMO is viable
# ---------------------------------------------------------------

def vendor_expr():
    return pl.col("model").str.slice(1, 1).alias("vendor")


def profile_vendors(lf):
    prof = collect(
        lf.with_columns(vendor_expr())
        .group_by(["vendor", "model"])
        .agg(
            pl.col("disk_id").n_unique().alias("n_drives"),
            pl.len().alias("n_rows"),
            pl.col("ds").min().alias("first_ds"),
            pl.col("ds").max().alias("last_ds"),
            pl.col("disk_id")
              .filter(pl.col("failure_time").is_not_null())
              .n_unique().alias("n_failed_drives"),
        )
        .with_columns(
            (pl.col("n_failed_drives") / pl.col("n_drives") * 100)
            .round(3).alias("failure_pct")
        )
        .sort(["vendor", "model"])
    )
    prof.write_csv(f"{REPORTS}/vendor_profile.csv")

    print("\n[PROFILE] per drive model:")
    print(prof)

    by_vendor = (
        prof.group_by("vendor")
        .agg(pl.col("n_drives").sum(), pl.col("n_failed_drives").sum())
        .sort("vendor")
    )
    print("\n[PROFILE] per vendor — LOMO viability:")
    print(by_vendor)

    note("")
    for r in by_vendor.iter_rows(named=True):
        v, d, f = r["vendor"], r["n_drives"], r["n_failed_drives"]
        verdict = ("OK" if f >= 500 else
                   "THIN — wide coverage band" if f >= 100 else
                   "TOO FEW — use leave-one-model-out instead")
        note(f"[VENDOR {v}] drives={d:,} failures={f:,} -> {verdict}")
    return prof


def attribute_availability(lf):
    schema = lf.collect_schema()
    smart = [c for c in schema.names() if c not in META_COLS]

    avail = collect(
        lf.with_columns(vendor_expr())
        .group_by("vendor")
        .agg([
            (1 - missing_expr(c, schema[c]).mean()).round(4).alias(c)
            for c in smart
        ])
    ).sort("vendor")

    avail.write_csv(f"{REPORTS}/attribute_availability.csv")

    long = avail.unpivot(index="vendor", variable_name="attr",
                         value_name="avail")
    per_vendor_usable = (
        long.filter(pl.col("avail") > 0.5)
        .group_by("vendor").len().sort("vendor")
    )
    note("")
    note(f"[ATTRS] {len(smart)} SMART columns; populated >50% of rows:")
    for r in per_vendor_usable.iter_rows(named=True):
        note(f"    vendor {r['vendor']}: {r['len']}")
    note("    -> divergence here IS the covariate shift evidence")
    return avail


if __name__ == "__main__":
    print("=" * 64)
    print("VERIFICATION")
    print("=" * 64)

    order_ok = check_column_order()

    lf = pl.scan_parquet(PARQUET)
    n_rows = check_integrity(lf)
    check_row_counts(n_rows)

    print("\n" + "=" * 64)
    print("PROFILING")
    print("=" * 64)

    profile_vendors(lf)
    attribute_availability(lf)

    with open(f"{REPORTS}/verification_summary.txt", "w") as fh:
        fh.write("\n".join(_summary) + "\n")

    print("\nReports written to ./reports/")
    if not order_ok:
        print("\n*** STOP: resolve column order before proceeding. ***")
