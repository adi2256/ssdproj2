"""
Leave-one-drive-MODEL-out sweep.

THE QUESTION

Leave-one-VENDOR-out failed, and the fast representation sweep showed
why: three vendors, three different mechanisms.

    vendor A   raw levels outside the training range -> trend helps
               (zscore + dynamics: 0.586 -> 0.666)
    vendor C   counters on a different scale -> rank helps
               (rank + dynamics: 0.642; trend alone does nothing, 0.447)
    vendor B   two attributes correlate with failure in the opposite
               direction -> nothing tested moves it (0.52-0.59 across
               all nine combinations)

Best mean across all nine representation/normaliser pairs: 0.573. No
single representation repairs all three, because rank helps C and hurts
A while dynamics helps A and hurts C.

All three mechanisms are ENCODING differences, not hardware
differences. Within a vendor, encoding conventions are shared. So:

    Does the same model transfer between drive models of the SAME
    vendor?

If MA1 -> MA2 reaches 0.70 while A -> (B,C) sits at 0.59, encoding is
the obstacle and there is real transferable signal underneath. That
reopens the conformal experiment on six folds instead of three, at a
granularity that matches what data centres actually do -- new drive
models arrive constantly, new vendors rarely.

If within-vendor transfer ALSO fails, the shared attributes do not
carry enough signal in any form, and the honest output is a
characterisation of the three failure modes rather than a coverage
experiment.

TWO CONDITIONS

    all_others      train on the five other models (any vendor)
    same_vendor     train only on the held-out model's sibling

The second is the strict test. If same_vendor beats all_others, mixing
vendors into training is actively harmful, which is itself a result.

Runtime is roughly 25 minutes: six folds x two conditions, windowed
once each, three representations per fold.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from src.data.splits import make_all_lomm_splits
from src.data.windowing import (Normalizer, constant_features,
                                drop_features, flatten, make_windows)

STRIDE = 60
MAX_TRAIN = 150_000
N_TREES = 60
SEED = 42

# The three that each won a fold in the vendor-level sweep, so each
# mechanism is represented.
COMBOS = [
    ("zscore", "dynamics"),      # best for vendor A
    ("rank", "dynamics"),        # best for vendor C
    ("per_drive", "dynamics"),   # best mean overall (0.573)
    ("zscore", "last"),          # the original baseline, for reference
]


def auc(scores, labels) -> float:
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


def _subset(ws, idx):
    from src.data.windowing import WindowSet
    return WindowSet(
        X=ws.X[idx], y=ws.y[idx], drive_idx=ws.drive_idx[idx],
        end_ds=ws.end_ds[idx], drives=ws.drives, vendors=ws.vendors[idx],
        features=ws.features, window_len=ws.window_len,
        stride=ws.stride, positive_stride=ws.positive_stride,
    )


def window_fold(labelled, split):
    tr = make_windows(split.apply(labelled, "train"), stride=STRIDE)
    te = make_windows(split.apply(labelled, "test"), stride=STRIDE)

    # Keep every positive; thin only negatives. Positives are 1.4% of
    # windows and are the scarce resource.
    if len(tr) > MAX_TRAIN:
        rng = np.random.default_rng(SEED)
        pos = np.flatnonzero(tr.y == 1)
        neg = np.flatnonzero(tr.y == 0)
        n_neg = max(0, MAX_TRAIN - len(pos))
        keep = np.sort(np.concatenate(
            [pos, rng.choice(neg, size=min(n_neg, len(neg)), replace=False)]
        ))
        tr = _subset(tr, keep)
    return tr, te


def evaluate(tr, te, method, how):
    from sklearn.ensemble import RandomForestClassifier

    if len(tr) == 0 or len(te) == 0:
        return float("nan"), 0
    if len(np.unique(tr.y)) < 2 or len(np.unique(te.y)) < 2:
        return float("nan"), 0

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
    return auc(p, te_n.y), len(names)


def run(labelled, conditions=("all_others", "same_vendor")):
    rows = []

    for cond in conditions:
        same = cond == "same_vendor"
        print(f"\n{'=' * 64}\n{cond}\n{'=' * 64}")

        try:
            splits = make_all_lomm_splits(labelled, same_vendor_only=same)
        except ValueError as e:
            print(f"  skipped: {e}")
            continue

        for name, sp in splits.items():
            t0 = time.time()
            try:
                tr, te = window_fold(labelled, sp)
            except Exception as e:
                print(f"  {name}: windowing failed: {e}")
                continue

            n_fail_te = int(te.y.sum())
            line = (f"  {name}  train {len(tr):>7,} test {len(te):>7,} "
                    f"pos {n_fail_te:>5,} ({time.time() - t0:>3.0f}s) ")

            row = {"condition": cond, "held_out": name,
                   "n_train": len(tr), "n_test": len(te),
                   "n_pos_test": n_fail_te,
                   "vendor": name[1]}
            for method, how in COMBOS:
                a, _ = evaluate(tr, te, method, how)
                row[f"{method}_{how}"] = None if np.isnan(a) else round(a, 4)
                line += f"{method[:4]}/{how[:3]} {a:>6.3f}  "
            rows.append(row)
            print(line, flush=True)

    res = pl.DataFrame(rows)
    combo_cols = [f"{m}_{h}" for m, h in COMBOS]

    print("\n=== best representation per fold ===")
    best = res.with_columns(
        pl.max_horizontal([pl.col(c) for c in combo_cols]).alias("best_auc")
    ).select(["condition", "held_out", "vendor", "n_pos_test",
              "best_auc"] + combo_cols)
    print(best.sort(["condition", "held_out"]))

    print("\n=== mean best AUC per condition ===")
    print(best.group_by("condition").agg(
        pl.col("best_auc").mean().round(4).alias("mean_best"),
        pl.col("best_auc").min().round(4).alias("worst_fold"),
        pl.len().alias("n_folds"),
    ))

    # -- the gate ------------------------------------------------
    print("\n" + "=" * 64)
    for cond in best["condition"].unique():
        sub = best.filter(pl.col("condition") == cond)
        vals = [v for v in sub["best_auc"].to_list() if v is not None]
        if not vals:
            continue
        mean_v, min_v = float(np.mean(vals)), float(np.min(vals))
        print(f"{cond}: mean {mean_v:.3f}, worst fold {min_v:.3f}")

        if min_v >= 0.65:
            print("  PASS -- every fold transfers. The conformal "
                  "experiment can run on six folds.")
        elif mean_v >= 0.65:
            print("  PARTIAL -- most folds transfer. Report the failing "
                  "fold separately rather than averaging over it.")
        elif mean_v > 0.60:
            print("  MARGINAL -- better than vendor-level (0.573) but "
                  "thin. Compare against the vendor-level table before "
                  "deciding.")
        else:
            print("  FAIL -- within-vendor transfer is no better. The "
                  "shared attributes do not carry enough signal, and the "
                  "honest output is the three-mechanism characterisation.")

    print("\nCompare against vendor-level LOMO: mean 0.573, "
          "best single fold 0.666 (A, zscore+dynamics).")

    for path in ("/kaggle/working/lomm_sweep.csv", "lomm_sweep.csv"):
        try:
            res.write_csv(path)
            print(f"written to {path}")
            break
        except (FileNotFoundError, OSError):
            continue
    return res


if __name__ == "__main__":
    res = run(labelled)          # noqa: F821
