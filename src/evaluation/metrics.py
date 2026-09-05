"""
Evaluation metrics.

Two families, and the distinction matters for the paper's argument.

FLEET METRICS (precision, recall, F0.5, AUC)
    What prior work reports. They describe average behaviour across the
    whole test population and say nothing about any individual drive.
    Included so the comparison with the base paper is fair.

    F0.5 weights precision above recall, following the base paper: a
    false alarm wastes a technician's time and a working drive, so
    operators tolerate missing some failures more readily than crying
    wolf.

SET METRICS (coverage, set size, singleton rate)
    Live in src/conformal/split.py, because they are properties of a
    prediction set rather than of a score. They carry a per-drive
    guarantee, which is the project's contribution.

THRESHOLDS
    Fleet metrics need a cut-off, and 0.5 is the wrong one here. Both
    models correct for class imbalance during training, which shifts the
    scale of the output probabilities. The threshold must be chosen on
    TRAINING data and then applied unchanged to test data. Choosing it
    on test data would report a best case that no operator could
    reproduce.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------
# Threshold-free
# ---------------------------------------------------------------

def auc(probs, labels) -> float:
    """Rank-based ROC AUC. No sklearn dependency."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")

    order = np.argsort(probs, kind="mergesort")
    ranks = np.empty(len(probs), dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1)
    vals, inv, counts = np.unique(probs, return_inverse=True,
                                  return_counts=True)
    if (counts > 1).any():
        sums = np.bincount(inv, weights=ranks)
        ranks = sums[inv] / counts[inv]
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(probs, labels) -> float:
    """
    Area under the precision-recall curve. More informative than AUC at
    the 1-5% prevalence of this fleet, where a high ROC AUC can coexist
    with poor precision.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    n1 = int(labels.sum())
    if n1 == 0:
        return float("nan")

    order = np.argsort(-probs, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / n1)


# ---------------------------------------------------------------
# Threshold-dependent
# ---------------------------------------------------------------

def confusion(probs, labels, threshold: float) -> dict:
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    pred = probs >= threshold
    return {
        "tp": int((pred & labels).sum()),
        "fp": int((pred & ~labels).sum()),
        "fn": int((~pred & labels).sum()),
        "tn": int((~pred & ~labels).sum()),
    }


def f_beta(probs, labels, threshold: float, beta: float = 0.5) -> float:
    """
    F-beta. beta < 1 favours precision; the base paper uses beta = 0.5.
    """
    c = confusion(probs, labels, threshold)
    prec = c["tp"] / (c["tp"] + c["fp"] + EPS)
    rec = c["tp"] / (c["tp"] + c["fn"] + EPS)
    if prec + rec < EPS:
        return 0.0
    b2 = beta ** 2
    return float((1 + b2) * prec * rec / (b2 * prec + rec + EPS))


def pick_threshold(train_probs, train_labels, beta: float = 0.5,
                   n_grid: int = 200) -> float:
    """
    Choose the cut-off maximising F-beta on TRAINING data.

    Never call this with test scores. Tuning the threshold on test data
    reports a best case that could not be reproduced in deployment.
    """
    p = np.asarray(train_probs, dtype=np.float64).ravel()
    y = np.asarray(train_labels).ravel().astype(bool)
    if len(p) == 0 or y.all() or (~y).all():
        return 0.5

    lo, hi = float(p.min()), float(p.max())
    if hi - lo < EPS:
        return float(lo)

    grid = np.linspace(lo, hi, n_grid)
    scores = [f_beta(p, y, t, beta) for t in grid]
    return float(grid[int(np.argmax(scores))])


# ---------------------------------------------------------------
# Report
# ---------------------------------------------------------------

def fleet_metrics(probs, labels, threshold: float,
                  beta: float = 0.5) -> dict:
    """
    One row of the accuracy comparison against the base paper.

    Reported alongside coverage, never instead of it. A strong F0.5 with
    poor failure-class coverage is exactly the situation this project
    exists to expose.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    c = confusion(probs, labels, threshold)
    prec = c["tp"] / (c["tp"] + c["fp"] + EPS)
    rec = c["tp"] / (c["tp"] + c["fn"] + EPS)

    return {
        "threshold": float(threshold),
        "precision": float(prec),
        "recall": float(rec),
        f"f{beta:g}": f_beta(probs, labels, threshold, beta),
        "f1": f_beta(probs, labels, threshold, 1.0),
        "auc": auc(probs, labels),
        "avg_precision": average_precision(probs, labels),
        "prevalence": float(labels.mean()) if len(labels) else float("nan"),
        **c,
    }


def drive_level_metrics(probs, labels, drive_idx,
                        threshold: float, beta: float = 0.5) -> dict:
    """
    Collapse windows to drives before scoring: a drive is flagged if ANY
    of its windows crosses the threshold, and is a true positive if any
    of its windows is labelled positive.

    Worth reporting alongside the window-level numbers. Operators
    replace drives, not windows, and a drive with 700 windows would
    otherwise dominate one with 40.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    drive_idx = np.asarray(drive_idx).ravel()

    uniq = np.unique(drive_idx)
    d_prob = np.array([probs[drive_idx == d].max() for d in uniq])
    d_lab = np.array([labels[drive_idx == d].any() for d in uniq])

    out = fleet_metrics(d_prob, d_lab.astype(int), threshold, beta)
    out["n_drives"] = int(len(uniq))
    return out
