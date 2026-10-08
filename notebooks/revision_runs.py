"""
Revision runs R1-R3 for the reviewer response.

Three reviewer requests are answered by this file. Two of them need no
extra training at all, which is the point of its design:

    REV-1  Platt / isotonic calibration baseline
    REV-2  drive-clustered bootstrap CIs for AUC and per-class coverage
    REV-3  healthy-drive subsampling sensitivity

REV-1 and REV-2 are functions of the per-window predictions, not of the
model. The existing pipeline computes those predictions and discards
them, so every question about them has so far required a full retrain.
This script dumps them once per configuration; REV-1 and REV-2 then run
off the dump in seconds, locally, with no Kaggle time at all.

REV-3 is the only one that genuinely needs retraining, because changing
the healthy-drive sampling ratio changes what the model is fit on. It is
therefore the expensive part: one `build()` per ratio.

WHAT GETS WRITTEN, per ratio tag

    preds_{tag}_cal.parquet    drive_idx, vendor, stage, y, p
    preds_{tag}_test.parquet   the same, plus set membership for each of
                               the four conformal methods at alpha=0.10
    preds_{tag}_train.parquet  y, p only -- needed to reproduce the
                               training-chosen operating threshold
    meta_{tag}.json            ratio, drive counts, window counts, AUC,
                               seed, and the config that produced them

The set-membership columns are `set_h_{method}` and `set_f_{method}`:
whether the healthy and the failure label are in that window's set. Both
false is an empty set, both true a doubleton. Keeping them as raw
booleans rather than a summary means the bootstrap can recompute any
coverage or composition statistic later without another run.

RATIOS

All 16,305 failed drives are kept at every ratio; only the healthy count
moves. `n_drives` is the total, so the healthy count is n_drives - 16,305
and the ratio quoted is healthy:failed.

    r1     32,610 drives     1:1     the aggressive end
    r2p7   60,000 drives     2.7:1   what the paper currently reports
    r5     97,830 drives     5:1     the conservative end

RUN ONE RATIO PER KERNEL SESSION. Windowing 97,830 drives holds several
GB of float32 at once, and Kaggle will OOM if a previous ratio's arrays
are still live. The script frees what it can between ratios, but a fresh
kernel is the only reliable reset.
"""

from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
import polars as pl

from notebooks.paper_results import METHODS, _fit_conformal, build
from src.data.labels import build_labels

MAIN_ALPHA = 0.10
SEED = 42
N_FAILED = 16_305

RATIOS = {
    "r1":    32_610,
    "r2p7":  60_000,
    "r5":    97_830,
}

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."


def _frame(ws_part, stage, y, p):
    """Per-window rows with the provenance the bootstrap needs."""
    return pl.DataFrame({
        "drive_idx": np.asarray(ws_part.drive_idx, dtype=np.int64),
        "vendor": [str(v) for v in np.asarray(ws_part.vendors)],
        "stage": np.asarray(stage, dtype=np.int16),
        "y": np.asarray(y, dtype=np.int8),
        "p": np.asarray(p, dtype=np.float64),
    })


def dump(ctx, tag: str, n_drives: int, alpha: float = MAIN_ALPHA) -> dict:
    """Write the three parquets and the meta json for one ratio."""
    cal = _frame(ctx["raw"]["cal"], ctx["stage"]["cal"],
                 ctx["y_cal"], ctx["p_cal"])
    test = _frame(ctx["raw"]["test"], ctx["stage"]["test"],
                  ctx["y_te"], ctx["p_te"])

    # Set membership per method. Stored raw so any coverage, set-size or
    # composition statistic can be recomputed downstream without a rerun.
    for method in METHODS:
        r = _fit_conformal(ctx, alpha, method)
        test = test.with_columns([
            pl.Series(f"set_h_{method}", r.sets[:, 0].astype(bool)),
            pl.Series(f"set_f_{method}", r.sets[:, 1].astype(bool)),
        ])
        if r.fallback is not None:
            test = test.with_columns(
                pl.Series(f"fb_{method}", np.asarray(r.fallback, dtype=bool))
            )

    train = pl.DataFrame({
        "y": np.asarray(ctx["y_tr"], dtype=np.int8),
        "p": np.asarray(ctx["p_tr"], dtype=np.float64),
    })

    cal.write_parquet(f"{OUT}/preds_{tag}_cal.parquet")
    test.write_parquet(f"{OUT}/preds_{tag}_test.parquet")
    train.write_parquet(f"{OUT}/preds_{tag}_train.parquet")

    n_heal = n_drives - N_FAILED
    meta = {
        "tag": tag,
        "n_drives": n_drives,
        "n_failed_drives": N_FAILED,
        "n_healthy_drives": n_heal,
        "healthy_to_failed": round(n_heal / max(N_FAILED, 1), 3),
        "seed": SEED,
        "alpha": alpha,
        "n_windows_train": int(len(ctx["y_tr"])),
        "n_windows_cal": int(len(ctx["y_cal"])),
        "n_windows_test": int(len(ctx["y_te"])),
        "n_fail_windows_test": int(ctx["y_te"].sum()),
        "test_prevalence": float(ctx["y_te"].mean()),
        "auc": float(ctx["auc"]),
        "methods": list(METHODS),
    }
    with open(f"{OUT}/meta_{tag}.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n  wrote preds_{tag}_{{cal,test,train}}.parquet  "
          f"({test.height:,} test rows)")
    print("  " + json.dumps(meta))
    return meta


def run_ratio(tag: str, load_slim, alpha: float = MAIN_ALPHA) -> dict:
    """
    Load at this ratio, label, build, dump.

    Everything after `build` is cheap; `build` is the ~20 minutes.
    """
    if tag not in RATIOS:
        raise KeyError(f"unknown tag {tag!r}; expected one of {list(RATIOS)}")
    n_drives = RATIOS[tag]
    heal = n_drives - N_FAILED

    print("=" * 64)
    print(f"RATIO {tag}: {n_drives:,} drives  "
          f"= {N_FAILED:,} failed + {heal:,} healthy  "
          f"({heal / max(N_FAILED, 1):.2f}:1)")
    print("=" * 64)

    t0 = time.time()
    raw = load_slim(n_drives=n_drives, keep_all_failed=True, seed=SEED)
    labelled, stats = build_labels(raw)
    print(stats)
    del raw
    gc.collect()

    ctx = build(labelled, seed=SEED)
    meta = dump(ctx, tag, n_drives, alpha=alpha)

    del ctx, labelled
    gc.collect()
    print(f"\n[{tag} done in {time.time() - t0:.0f}s]")
    return meta


if __name__ == "__main__":
    # In the notebook: run_ratio("r2p7", ctx_setup.load_slim)
    raise SystemExit("import and call run_ratio(tag, load_slim) instead")
