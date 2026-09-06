"""
Fast representation sweep.

WHY THE LAST RUN TOOK TWO HOURS

run_experiment calls prepare_fold for every configuration, and
prepare_fold re-windows the data every time. Windowing is a Python loop
over ~59,000 drives and dominates the runtime -- but windows do not
depend on the normaliser or the flattener at all. Nine configurations
therefore paid the windowing cost nine times over for identical output.

This script windows each fold ONCE, then sweeps every (normalize,
flatten) pair over the cached windows. It also drops everything not
needed to answer the one open question:

  - no conformal layer (we are testing the base model, not the wrapper)
  - no standard split by default (the gate is LOMO-A)
  - smaller forest, subsampled training windows
  - larger stride

Expect roughly 10-15 minutes for all nine cells instead of two hours.

WHAT IT ANSWERS

Does removing vendor-specific LEVEL restore cross-vendor transfer?

    zscore     keeps level and scale
    rank       removes scale        (previously moved lomo_C 0.37 -> 0.65)
    per_drive  removes level and scale, no fitted statistics

    last            level only        (what every prior run used)
    dynamics        trend + level
    dynamics_only   trend only

The gate is LOMO-A AUC >= 0.60. Currently 0.48 under (zscore, last),
0.596 under (zscore, dynamics_only).
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from src.data.splits import make_all_lomo_splits, make_standard_split
from src.data.windowing import (Normalizer, constant_features,
                                drop_features, flatten, make_windows)

# ---------------------------------------------------------------
# Knobs. Raise STRIDE or lower MAX_TRAIN if this is still too slow.
# ---------------------------------------------------------------
STRIDE = 60           # 30 -> 60 roughly halves the window count
MAX_TRAIN = 150_000   # subsample training windows; test stays whole
N_TREES = 60          # 200 -> 60; we need ranking, not a final model
SEED = 42

NORMALIZERS = ("zscore", "rank", "per_drive")
FLATTENS = ("last", "dynamics", "dynamics_only")


def auc(scores, labels) -> float:
    """Rank-based ROC AUC, no sklearn import needed."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True,
                               return_counts=True)
    if (counts > 1).any():
        ranks = np.bincount(inv, weights=ranks)[inv] / counts[inv]
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def window_fold(labelled, split):
    """
    Window a fold once. Everything downstream reuses this.

    Calibration is not needed here -- there is no conformal layer in
    this sweep -- so only train and test are built.
    """
    t0 = time.time()
    tr = make_windows(split.apply(labelled, "train"), stride=STRIDE)
    te = make_windows(split.apply(labelled, "test"), stride=STRIDE)

    # Subsample training windows, keeping EVERY positive. Positives are
    # 1.4% of windows and are the scarce resource; negatives are not.
    if len(tr) > MAX_TRAIN:
        rng = np.random.default_rng(SEED)
        pos = np.flatnonzero(tr.y == 1)
        neg = np.flatnonzero(tr.y == 0)
        n_neg = max(0, MAX_TRAIN - len(pos))
        keep = np.sort(np.concatenate([
            pos, rng.choice(neg, size=min(n_neg, len(neg)), replace=False)
        ]))
        tr = _subset(tr, keep)

    print(f"  {split.name}: train {len(tr):,} test {len(te):,} "
          f"({time.time() - t0:.0f}s)")
    return tr, te


def _subset(ws, idx):
    from src.data.windowing import WindowSet
    return WindowSet(
        X=ws.X[idx], y=ws.y[idx], drive_idx=ws.drive_idx[idx],
        end_ds=ws.end_ds[idx], drives=ws.drives, vendors=ws.vendors[idx],
        features=ws.features, window_len=ws.window_len,
        stride=ws.stride, positive_stride=ws.positive_stride,
    )


def evaluate(tr, te, method, how):
    """One (normalize, flatten) cell on already-windowed data."""
    from sklearn.ensemble import RandomForestClassifier

    norm = Normalizer.fit(tr, method=method)
    tr_n, te_n = norm.transform(tr), norm.transform(te)

    dead = constant_features(te_n)
    if dead and len(dead) < len(te_n.features):
        tr_n, te_n = drop_features(tr_n, dead), drop_features(te_n, dead)

    Xtr, names = flatten(tr_n, how)
    Xte, _ = flatten(te_n, how)

    m = RandomForestClassifier(
        n_estimators=N_TREES, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    ).fit(np.nan_to_num(Xtr), tr_n.y)

    p = m.predict_proba(np.nan_to_num(Xte))
    p = p[:, list(m.classes_).index(1)] if p.shape[1] > 1 else p[:, 0]
    return auc(p, te_n.y), len(names), len(dead)


def run(labelled, include_standard: bool = False):
    splits = {}
    if include_standard:
        splits["standard"] = make_standard_split(labelled, seed=SEED)
    splits.update(make_all_lomo_splits(labelled, seed=SEED))

    print("windowing once per fold ...")
    cached = {name: window_fold(labelled, sp) for name, sp in splits.items()}

    print(f"\n{'normalize':<11}{'flatten':<15}" +
          "".join(f"{k:>12}" for k in cached))
    rows = []
    for method in NORMALIZERS:
        for how in FLATTENS:
            row = {"normalize": method, "flatten": how}
            line = f"{method:<11}{how:<15}"
            for name, (tr, te) in cached.items():
                a, n_feat, n_dead = evaluate(tr, te, method, how)
                row[f"auc_{name}"] = round(a, 4)
                row[f"nfeat_{name}"] = n_feat
                line += f"{a:>12.3f}"
            rows.append(row)
            print(line, flush=True)

    res = pl.DataFrame(rows)

    lomo_cols = [c for c in res.columns
                 if c.startswith("auc_") and "standard" not in c]
    res = res.with_columns(
        pl.mean_horizontal([pl.col(c) for c in lomo_cols]).alias("auc_lomo_mean")
    )

    print("\n=== ranked by mean LOMO AUC ===")
    print(res.select(["normalize", "flatten"] + lomo_cols +
                     ["auc_lomo_mean"]).sort("auc_lomo_mean",
                                             descending=True))

    # make_all_lomo_splits keys by vendor letter, so the column is
    # auc_A rather than auc_lomo_A unless a prefix was used.
    col_a = "auc_lomo_A" if "auc_lomo_A" in res.columns else "auc_A"
    best_a = res.sort(col_a, descending=True).row(0, named=True)
    print(f"\nbest LOMO-A: {best_a[col_a]:.3f} "
          f"({best_a['normalize']}, {best_a['flatten']})")

    if best_a[col_a] >= 0.60:
        print("PASS -- level was the problem. Re-run the conformal sweep "
              "on this representation.")
    elif best_a[col_a] >= 0.55:
        print("PARTIAL -- moving in the right direction. Check whether "
              "lomo_C also moved; if it stays near 0.40, its failure is "
              "not a level problem and needs a separate diagnosis.")
    else:
        print("FAIL -- level is not the obstacle.")

    for path in ("/kaggle/working/repr_sweep.csv", "repr_sweep.csv"):
        try:
            res.write_csv(path)
            print(f"\nwritten to {path}")
            break
        except (FileNotFoundError, OSError):
            continue
    return res


if __name__ == "__main__":
    # `labelled` is expected in scope from the setup cell.
    res = run(labelled)          # noqa: F821
