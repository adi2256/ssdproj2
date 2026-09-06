"""
Target-definition sweep.

THE QUESTION

Everything so far predicted the same thing: "does this drive fail in
the next 30 days", scored per 30-day window. Transfer failed at every
granularity and representation tried:

    within-fleet                     0.73
    held-out vendor (3 folds)        0.573 mean
    held-out drive model (6 folds)   0.597 mean
    held-out model, own vendor only  0.562 mean

Before concluding the shared attributes carry no transferable signal,
one thing has never been varied: THE TARGET ITSELF.

Related work does not use this target. Feature-selection work on the
Backblaze fleet predicts whether a drive fails by the END of a 90-day
observation period, scored once per DRIVE, with roughly a quarter of
drives positive. That is a different and much easier problem than
per-window 30-day prediction at 1.4% prevalence, and it is the setting
in which SMART attributes are usually reported to work.

Two axes:

    HORIZON     7, 15, 30, 60, 90 days
                A longer horizon means a weaker precursor requirement.
                If 30 days is too tight for the signal that exists,
                longer horizons should show it.

    GRANULARITY window-level vs drive-level
                Drive-level takes each drive's maximum score across all
                its windows and asks "will this drive fail at all in
                the observation period". Operators replace drives, not
                windows, so this is also the more meaningful unit --
                and it raises prevalence from ~1.4% to the drive
                failure rate, which is a substantially easier problem.

WHY THIS IS CHEAP

Horizon changes the LABEL, not the windows. The script windows once per
fold, joins days-to-failure onto each window, then derives every
horizon's labels arithmetically. Only the forest is refitted.

One caveat to state in any write-up: build_labels drops the final
`horizon` days of every non-failed drive as unobservable. To keep the
row set identical across horizons, labels are built ONCE at the largest
horizon, so shorter horizons inherit the larger drop. That is
conservative -- it costs a little data at short horizons -- but it
keeps the comparison clean.

Runtime: roughly 12 minutes for 3 folds x 5 horizons x 2 granularities.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from src.data.labels import build_labels
from src.data.splits import make_all_lomo_splits, make_standard_split
from src.data.windowing import (Normalizer, constant_features,
                                drop_features, flatten, make_windows)

HORIZONS = [7, 15, 30, 60, 90]
MAX_HORIZON = max(HORIZONS)

STRIDE = 60
MAX_TRAIN = 150_000
N_TREES = 60
SEED = 42

# Best performer from the representation sweep (mean LOMO 0.573).
NORMALIZE = "per_drive"
FLATTEN = "dynamics"


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


def attach_dtf(ws, labelled: pl.DataFrame) -> np.ndarray:
    """
    Days-to-failure at each window's END date.

    Joining back to the labelled frame avoids touching make_windows,
    and it is what makes the horizon sweep nearly free: with dtf in
    hand, every horizon's label is just (0 < dtf <= h).

    Healthy drives have a null dtf, which becomes +inf so they are
    negative at every horizon.
    """
    keys = pl.DataFrame({
        "model": [ws.drives[i][0] for i in ws.drive_idx],
        "disk_id": [ws.drives[i][1] for i in ws.drive_idx],
        "ds": list(ws.end_ds),
        "_row": np.arange(len(ws)),
    })
    joined = keys.join(
        labelled.select(["model", "disk_id", "ds", "days_to_failure"]),
        on=["model", "disk_id", "ds"], how="left",
    ).sort("_row")
    dtf = joined["days_to_failure"].to_numpy().astype(np.float64)
    return np.where(np.isfinite(dtf), dtf, np.inf)


def window_fold(labelled, split):
    t0 = time.time()
    parts = {}
    for part in ("train", "test"):
        ws = make_windows(split.apply(labelled, part), stride=STRIDE)
        parts[part] = (ws, attach_dtf(ws, labelled))

    # Subsample training windows. "Positive" here means positive at the
    # LARGEST horizon, so no horizon loses positives to the subsample.
    tr, tr_dtf = parts["train"]
    if len(tr) > MAX_TRAIN:
        rng = np.random.default_rng(SEED)
        pos = np.flatnonzero((tr_dtf > 0) & (tr_dtf <= MAX_HORIZON))
        neg = np.flatnonzero(~((tr_dtf > 0) & (tr_dtf <= MAX_HORIZON)))
        n_neg = max(0, MAX_TRAIN - len(pos))
        keep = np.sort(np.concatenate(
            [pos, rng.choice(neg, size=min(n_neg, len(neg)), replace=False)]
        ))
        parts["train"] = (_subset(tr, keep), tr_dtf[keep])

    print(f"  {split.name}: train {len(parts['train'][0]):,} "
          f"test {len(parts['test'][0]):,} ({time.time() - t0:.0f}s)")
    return parts


def prepare_matrices(parts):
    """Normalise and flatten once; reused for every horizon."""
    tr, tr_dtf = parts["train"]
    te, te_dtf = parts["test"]

    norm = Normalizer.fit(tr, method=NORMALIZE)
    tr_n, te_n = norm.transform(tr), norm.transform(te)

    dead = constant_features(te_n)
    if dead and len(dead) < len(te_n.features):
        tr_n, te_n = drop_features(tr_n, dead), drop_features(te_n, dead)

    Xtr, _ = flatten(tr_n, FLATTEN)
    Xte, _ = flatten(te_n, FLATTEN)
    return (np.nan_to_num(Xtr), tr_dtf, tr_n.drive_idx,
            np.nan_to_num(Xte), te_dtf, te_n.drive_idx)


def drive_level(scores, labels, drive_idx):
    """
    Collapse windows to drives: a drive's score is its maximum over
    windows, and it is positive if any of its windows is positive.

    Operators replace drives, not windows. This also lifts prevalence
    from the window rate to the drive rate.
    """
    order = np.argsort(drive_idx, kind="mergesort")
    d, s, l = drive_idx[order], scores[order], labels[order]
    bounds = np.flatnonzero(np.diff(d)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(d)]])
    ds = np.array([s[a:b].max() for a, b in zip(starts, ends)])
    dl = np.array([l[a:b].any() for a, b in zip(starts, ends)])
    return ds, dl


def run(labelled_raw, folds=("standard", "A", "B", "C")):
    from sklearn.ensemble import RandomForestClassifier

    print(f"building labels once at horizon={MAX_HORIZON} so the row set "
          f"is identical across horizons ...")
    labelled, stats = build_labels(labelled_raw, horizon_days=MAX_HORIZON)
    print(stats)

    splits = {}
    if "standard" in folds:
        splits["standard"] = make_standard_split(labelled, seed=SEED)
    lomo = make_all_lomo_splits(labelled, seed=SEED)
    splits.update({v: s for v, s in lomo.items() if v in folds})

    print("\nwindowing once per fold ...")
    cached = {n: prepare_matrices(window_fold(labelled, sp))
              for n, sp in splits.items()}

    rows = []
    print(f"\n{'fold':<10}{'horizon':>8}{'win_prev':>10}{'win_auc':>9}"
          f"{'drv_prev':>10}{'drv_auc':>9}{'n_pos':>8}")

    for name, (Xtr, tr_dtf, tr_di, Xte, te_dtf, te_di) in cached.items():
        for h in HORIZONS:
            ytr = ((tr_dtf > 0) & (tr_dtf <= h)).astype(int)
            yte = ((te_dtf > 0) & (te_dtf <= h)).astype(int)
            if ytr.sum() < 20 or yte.sum() < 20:
                print(f"{name:<10}{h:>8}   too few positives, skipped")
                continue

            m = RandomForestClassifier(
                n_estimators=N_TREES, min_samples_leaf=5,
                class_weight="balanced", random_state=SEED, n_jobs=-1,
            ).fit(Xtr, ytr)
            p = m.predict_proba(Xte)
            p = p[:, list(m.classes_).index(1)] if p.shape[1] > 1 else p[:, 0]

            w_auc = auc(p, yte)
            dp, dl = drive_level(p, yte.astype(bool), te_di)
            d_auc = auc(dp, dl)

            rows.append({
                "fold": name, "horizon": h,
                "win_prev": round(float(yte.mean()), 5),
                "win_auc": round(w_auc, 4),
                "drv_prev": round(float(dl.mean()), 5),
                "drv_auc": round(d_auc, 4),
                "n_pos_win": int(yte.sum()),
                "n_pos_drv": int(dl.sum()),
            })
            print(f"{name:<10}{h:>8}{yte.mean():>10.4f}{w_auc:>9.3f}"
                  f"{dl.mean():>10.4f}{d_auc:>9.3f}{int(dl.sum()):>8,}",
                  flush=True)

    res = pl.DataFrame(rows)

    print("\n=== drive-level AUC by horizon ===")
    print(res.pivot(values="drv_auc", index="horizon", on="fold"))

    print("\n=== window-level AUC by horizon ===")
    print(res.pivot(values="win_auc", index="horizon", on="fold"))

    # -- what changed, if anything ------------------------------
    lomo_rows = res.filter(pl.col("fold") != "standard")
    if lomo_rows.height:
        best = lomo_rows.sort("drv_auc", descending=True).row(0, named=True)
        print(f"\nbest LOMO drive-level: {best['drv_auc']:.3f} "
              f"(fold {best['fold']}, horizon {best['horizon']}d, "
              f"{best['n_pos_drv']:,} positive drives)")

        base = 0.597    # best mean from every prior configuration
        by_h = (lomo_rows.group_by("horizon")
                .agg(pl.col("drv_auc").mean().round(4).alias("mean_drv"),
                     pl.col("win_auc").mean().round(4).alias("mean_win"))
                .sort("horizon"))
        print("\nmean across LOMO folds by horizon:")
        print(by_h)

        top = float(by_h["mean_drv"].max())
        print(f"\nbest mean drive-level AUC: {top:.3f}  "
              f"(prior best, any configuration: {base:.3f})")
        if top >= 0.70:
            print("  The target definition was the obstacle. Rebuild the "
                  "conformal experiment at this horizon and granularity.")
        elif top >= 0.65:
            print("  Meaningful improvement. Worth pursuing, but check "
                  "whether it is driven by prevalence alone -- a higher "
                  "base rate makes AUC easier without more signal.")
        elif top > base + 0.03:
            print("  Small improvement, likely from prevalence rather "
                  "than signal. Not enough to rebuild around.")
        else:
            print("  No improvement. Target definition is not the "
                  "obstacle; the shared attributes do not support "
                  "transfer. Write the negative result.")

    for path in ("/kaggle/working/target_sweep.csv", "target_sweep.csv"):
        try:
            res.write_csv(path)
            print(f"\nwritten to {path}")
            break
        except (FileNotFoundError, OSError):
            continue
    return res


if __name__ == "__main__":
    # Pass the RAW frame (before build_labels) -- this script builds
    # labels itself at the largest horizon.
    res = run(df)          # noqa: F821
