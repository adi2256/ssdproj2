"""
Synthetic fixture generator.

Produces a small dataframe matching the real Alibaba parquet schema, with
properties chosen in advance so downstream logic can be asserted against
known answers rather than eyeballed.

CALIBRATED TO THE REAL FLEET (reports/vendor_profile.csv, 2026-08-30).
Drive proportions and relative failure rates mirror the production data,
so anything tested here is tested against the right shape:

    vendor  drives    failures  rate     model rates
    A       158,379   2,253     1.42%    MA1 3.43%  MA2 0.75%
    B        93,212   2,411     2.59%    MB1 4.07%  MB2 1.24%
    C       223,465  11,641     5.21%    MC1 5.26%  MC2 4.75%

Absolute failure rates are scaled up by `failure_scale` so small fixtures
carry enough positives for fast tests. The RELATIVE ordering (A < B < C)
is preserved, because that label shift is half of what LOMO stresses.

Reproduced deliberately:
  - per-vendor attribute availability from the real profile, leaving
    exactly 16 columns common to all three vendors
  - missingness as NaN, not null (pd.to_numeric(errors='coerce'))
  - right-censored drives (~28% of the real fleet)
  - post-failure telemetry (~78 rows per failed drive in the real data)
  - label shift across vendors, not just covariate shift
  - monotonic drift in designated columns before failure

Usage:
    df, truth = make_synthetic()
    truth["failed"]           # [(model, disk_id, failure_date), ...]
    truth["signal_ids"]       # SMART IDs carrying the pre-failure signal
    truth["failures_by_vendor"]
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from ..config import CFG, DRIVE_MODELS
from ..schema import (ALL_COLS, COMMON_IDS, SMART_COLS, VENDOR_SMART_IDS,
                      vendor_cols)

START = date(2018, 1, 1)

# Signal columns must lie in the common-16 set, or the injected signal
# would be invisible in the folds that matter.
SIGNAL_IDS = [5, 187, 197]
assert all(i in COMMON_IDS for i in SIGNAL_IDS), (
    f"SIGNAL_IDS {SIGNAL_IDS} must be a subset of COMMON_IDS {COMMON_IDS}"
)

# Drive share per model, from reports/vendor_profile.csv.
# Vendor C is the LARGEST (47%), not the smallest. B is the smallest.
MODEL_WEIGHTS = {
    "MA1": 0.0841, "MA2": 0.2493,
    "MB1": 0.0935, "MB2": 0.1027,
    "MC1": 0.4203, "MC2": 0.0501,
}

# Per-model failure rate, from reports/vendor_profile.csv (failure_pct/100).
MODEL_FAILURE_RATE = {
    "MA1": 0.0343, "MA2": 0.0075,
    "MB1": 0.0407, "MB2": 0.0124,
    "MC1": 0.0526, "MC2": 0.0475,
}

# Real data has ~78 post-failure rows per failed drive: drives keep
# reporting after the recorded failure_time. labels.py must drop these,
# so the fixture has to contain them.
POST_FAILURE_DAYS = 20

# Per-vendor offset applied to shared columns, so the common-16 are not
# identically distributed across vendors. Without this the only shift
# would be structural and weighted conformal would have nothing to
# reweight.
VENDOR_OFFSET = {"A": 0.0, "B": 3.0, "C": -2.5}


def _vendor_of(model: str) -> str:
    return model[1]


def make_synthetic(
    n_drives: int | None = None,
    n_days: int | None = None,
    failure_scale: float | None = None,
    censor_rate: float | None = None,
    post_failure_days: int | None = None,
    seed: int | None = None,
) -> tuple[pl.DataFrame, dict]:
    """
    Build a synthetic dataset plus its ground truth.

    failure_scale multiplies every per-model rate. The default keeps the
    real relative ordering while producing enough positives to test on.
    """
    n_drives = n_drives if n_drives is not None else CFG.synth_n_drives
    n_days = n_days if n_days is not None else CFG.synth_days
    failure_scale = (failure_scale if failure_scale is not None
                     else CFG.synth_failure_scale)
    censor_rate = (censor_rate if censor_rate is not None
                   else CFG.synth_censor_rate)
    post_failure_days = (post_failure_days if post_failure_days is not None
                         else POST_FAILURE_DAYS)
    seed = seed if seed is not None else CFG.synth_seed

    rng = np.random.default_rng(seed)
    global_end = START + timedelta(days=n_days - 1)

    # -- allocate drives to models, mirroring real proportions --
    drives = []
    next_id = {m: 1000 for m in DRIVE_MODELS}
    for m in DRIVE_MODELS:
        k = max(3, int(round(n_drives * MODEL_WEIGHTS[m])))
        for _ in range(k):
            drives.append((m, next_id[m]))
            next_id[m] += 1

    # -- per-model failure assignment, preserving label shift ---
    fail_idx: set[int] = set()
    by_model: dict[str, list[int]] = {m: [] for m in DRIVE_MODELS}
    for i, (m, _) in enumerate(drives):
        by_model[m].append(i)

    for m, idxs in by_model.items():
        rate = min(0.9, MODEL_FAILURE_RATE[m] * failure_scale)
        k = max(1, int(round(len(idxs) * rate)))
        chosen = rng.choice(idxs, size=min(k, len(idxs)), replace=False)
        fail_idx.update(int(c) for c in chosen)

    remaining = [i for i in range(len(drives)) if i not in fail_idx]
    n_cens = int(round(len(drives) * censor_rate))
    cens_idx = {
        int(c) for c in rng.choice(remaining,
                                   size=min(n_cens, len(remaining)),
                                   replace=False)
    }

    failed, censored, frames = [], [], []

    for idx, (model, disk_id) in enumerate(drives):
        vendor = _vendor_of(model)
        live_cols = set(vendor_cols(vendor))

        # ---- lifespan -----------------------------------------
        if idx in fail_idx:
            # Fail late enough that a full horizon fits before it, and
            # early enough that post-failure rows still fit after it.
            lo = CFG.horizon_days + CFG.window_len + 5
            hi = max(lo + 1, n_days - post_failure_days)
            fail_day = int(rng.integers(lo, hi))
            last_day = min(n_days - 1, fail_day + post_failure_days)
            failure_date = START + timedelta(days=fail_day)
            failed.append((model, disk_id, failure_date))
        elif idx in cens_idx:
            last_day = int(rng.integers(n_days // 3, n_days - 10))
            failure_date = None
            censored.append((model, disk_id,
                             START + timedelta(days=last_day)))
        else:
            last_day = n_days - 1
            failure_date = None

        n_obs = last_day + 1
        dates = [START + timedelta(days=d) for d in range(n_obs)]

        # ---- SMART values -------------------------------------
        cols: dict[str, np.ndarray] = {}
        for c in SMART_COLS:
            if c not in live_cols:
                cols[c] = np.full(n_obs, np.nan)
                continue

            if c.startswith("n_"):
                base = rng.uniform(90, 100)
                series = base - np.linspace(0, rng.uniform(0, 4), n_obs)
                series += rng.normal(0, 0.4, n_obs)
            else:
                base = rng.uniform(0, 500)
                drift = np.cumsum(rng.exponential(0.5, n_obs))
                series = base + drift + rng.normal(0, 2, n_obs)

            cols[c] = series + VENDOR_OFFSET[vendor]

        # ---- inject pre-failure signal ------------------------
        if failure_date is not None:
            h = CFG.horizon_days
            fail_pos = (failure_date - START).days
            ramp = np.zeros(n_obs)
            lo = max(0, fail_pos - h)
            span = fail_pos - lo
            if span > 0:
                ramp[lo:fail_pos] = np.linspace(0, 1, span) ** 2
            ramp[fail_pos:] = 1.0  # stays elevated post-failure

            for sid in SIGNAL_IDS:
                nc, rc = f"n_{sid}", f"r_{sid}"
                if nc in live_cols:
                    cols[nc] = cols[nc] - 25 * ramp
                if rc in live_cols:
                    cols[rc] = cols[rc] + 400 * ramp

        # ---- sporadic missingness -----------------------------
        for c in live_cols:
            gaps = rng.random(n_obs) < 0.01
            cols[c] = np.where(gaps, np.nan, cols[c])

        frames.append(pl.DataFrame({
            "disk_id": np.full(n_obs, disk_id, dtype=np.int64),
            "ds": dates,
            "model": [model] * n_obs,
            **{c: cols[c].astype(np.float64) for c in SMART_COLS},
        }))

    df = pl.concat(frames, how="vertical").select(ALL_COLS)

    # -- join failure labels, mirroring 01_preprocess.py --------
    fail_schema = {"model": pl.String, "disk_id": pl.Int64,
                   "failure_time": pl.Date}
    if failed:
        fail_df = pl.DataFrame(
            {
                "model": [m for m, _, _ in failed],
                "disk_id": [d for _, d, _ in failed],
                "failure_time": [t for _, _, t in failed],
            },
            schema=fail_schema,
        )
    else:
        fail_df = pl.DataFrame(schema=fail_schema)

    df = (
        df.join(fail_df, on=["model", "disk_id"], how="left")
        .with_columns(
            (pl.col("failure_time") - pl.col("ds"))
            .dt.total_days().cast(pl.Float64).alias("days_to_failure")
        )
        .with_columns(
            (pl.col("days_to_failure") <= 0)
            .fill_null(False).cast(pl.Int8).alias("failure_event")
        )
    )

    failures_by_vendor = {"A": 0, "B": 0, "C": 0}
    for m, _, _ in failed:
        failures_by_vendor[_vendor_of(m)] += 1

    drives_by_vendor = {"A": 0, "B": 0, "C": 0}
    for m, _ in drives:
        drives_by_vendor[_vendor_of(m)] += 1

    truth = {
        "failed": failed,
        "censored": censored,
        "signal_ids": SIGNAL_IDS,
        "common_ids": list(COMMON_IDS),
        "vendor_smart_ids": VENDOR_SMART_IDS,
        "global_end": global_end,
        "n_drives": len(drives),
        "n_failed": len(failed),
        "drives_by_vendor": drives_by_vendor,
        "failures_by_vendor": failures_by_vendor,
        "post_failure_days": post_failure_days,
        "horizon_days": CFG.horizon_days,
        "failure_scale": failure_scale,
        "seed": seed,
    }
    return df, truth


def write_synthetic(path: str, **kwargs) -> dict:
    """Write a synthetic fixture to parquet. Returns ground truth."""
    df, truth = make_synthetic(**kwargs)
    df.write_parquet(path)
    return truth