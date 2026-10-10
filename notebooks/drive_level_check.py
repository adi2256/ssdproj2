"""
Coverage at the exchangeable unit: one window per drive.

The split-conformal guarantee holds for a drive-level score, or for any
protocol that draws one window per drive in both calibration and test.
The paper's tables report window-level coverage, which averages over
drives contributing unequal numbers of dependent windows and is therefore
an empirical estimate, not the guaranteed quantity.

This script recomputes split-conformal (LAC) coverage on the released main
predictions with one randomly drawn window per calibration drive and per
test drive, repeated over B draws, so the guaranteed quantity can be read
beside the window-level one. It needs only the released parquet files.

    python -m notebooks.drive_level_check paper_results/final
"""

from __future__ import annotations

import math
import sys

import numpy as np
import polars as pl


def one_per_drive(drive_idx, rng):
    order = rng.permutation(len(drive_idx))
    _, first = np.unique(drive_idx[order], return_index=True)
    return order[first]


def run(res_dir="paper_results/final", tag="r2p7", alpha=0.10, B=200,
        seed=42):
    cal = pl.read_parquet(f"{res_dir}/preds_{tag}_cal.parquet")
    te = pl.read_parquet(f"{res_dir}/preds_{tag}_test.parquet")
    dc, pc, yc = (cal[c].to_numpy() for c in ("drive_idx", "p", "y"))
    dt, pt, yt, st = (te[c].to_numpy() for c in ("drive_idx", "p", "y",
                                                  "stage"))
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(B):
        ic, it = one_per_drive(dc, rng), one_per_drive(dt, rng)
        s = 1 - np.where(yc[ic] == 1, pc[ic], 1 - pc[ic])
        k = math.ceil((len(s) + 1) * (1 - alpha))
        q = np.sort(s)[k - 1] if k <= len(s) else np.inf
        inc_f, inc_h = (1 - pt[it]) <= q, pt[it] <= q
        hit = np.where(yt[it] == 1, inc_f, inc_h)
        f = yt[it] == 1
        rows.append({"draw": b, "coverage": hit.mean(),
                     "cov_failure": inc_f[f].mean() if f.any() else np.nan,
                     "n_fail_drives": int(f.sum()),
                     "set_size": (inc_f.astype(int) + inc_h).mean(),
                     "coverage_stage4": hit[st[it] == 4].mean()})
    df = pl.DataFrame(rows)
    print(f"{len(np.unique(dc)):,} calibration drives, "
          f"{len(np.unique(dt)):,} test drives, {B} draws")
    for c in ("coverage", "cov_failure", "set_size", "coverage_stage4"):
        v = df[c].to_numpy()
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        print(f"  {c:16s} mean {np.nanmean(v):.3f}  [{lo:.3f}, {hi:.3f}]")
    return df


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "paper_results/final")
