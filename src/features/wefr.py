"""
WEFR feature selection.

Reproduces the ensemble feature ranking of Xu et al., "General Feature
Selection for Failure Prediction in Large-scale SSD Deployment"
(IEEE/IFIP DSN 2021), which is the base paper for this project.

THE PROCEDURE

1. Score every feature with several independent rankers. Each produces
   its own ordering; none is trusted alone.
2. Compare the rankers pairwise with Kendall's tau. A ranker that
   disagrees with the consensus is an outlier and is dropped, so one
   badly-behaved scorer cannot distort the result.
3. Average the surviving rankers' ranks into a consensus ordering.
4. Choose how many features to keep by walking down the consensus list
   and measuring a data-complexity criterion at each prefix. Take the
   smallest prefix that is within a tolerance of the best value seen,
   so ties resolve toward fewer features.

WHY IT IS A FUNCTION, NOT A FITTED OBJECT

    select_features(X, y, feature_names) -> SelectionResult

Selection MUST be re-run inside every LOMO fold, on the training
manufacturers only. Running it once on pooled data lets the held-out
manufacturer's labels influence which features are chosen, which is
leakage, and it is the most likely objection a reviewer will raise.
Keeping this a pure function of (X_train, y_train) makes the leak
structurally hard to introduce.

A NOTE ON SCALE

WEFR was designed to prune a large pool of SMART attributes. Under this
project's common-16 policy only 16 columns are available in every fold,
so selection has far less to do than in the original paper. Expect it to
retain most of them. That is worth a sentence in the paper rather than
being hidden: at this width, feature selection is close to a no-op, and
the reproduction's value is in the pipeline being faithful rather than
in the pruning being consequential.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------
# Rankers
#
# Each returns a per-feature importance, higher = more relevant.
# All are NaN-safe and constant-column-safe.
# ---------------------------------------------------------------

def _clean(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def rank_variance(X, y):
    """Spread alone. A feature that never moves cannot explain anything."""
    return _clean(X).std(axis=0)


def rank_correlation(X, y):
    """Absolute Pearson correlation with the outcome."""
    X = _clean(X)
    y = np.asarray(y, dtype=np.float64)
    yc = y - y.mean()
    Xc = X - X.mean(axis=0)
    denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum()) + EPS
    return np.abs((Xc * yc[:, None]).sum(axis=0) / denom)


def rank_mean_gap(X, y):
    """
    Standardised difference in means between failing and healthy rows.
    Robust under heavy imbalance, where correlation gets small.
    """
    X = _clean(X)
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return np.zeros(X.shape[1])
    a, b = X[y], X[~y]
    pooled = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2.0) + EPS
    return np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled


def rank_auc(X, y):
    """
    Per-feature AUC, folded to |AUC - 0.5| so a feature that ranks in
    either direction counts. Rank-based, so monotone transforms and
    outliers do not matter.
    """
    X = _clean(X)
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.zeros(X.shape[1])

    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        order = np.argsort(col, kind="mergesort")
        ranks = np.empty(len(col), dtype=np.float64)
        ranks[order] = np.arange(1, len(col) + 1)
        vals, inv, counts = np.unique(col, return_inverse=True,
                                      return_counts=True)
        if (counts > 1).any():
            sums = np.bincount(inv, weights=ranks)
            ranks = sums[inv] / counts[inv]
        auc = (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        out[j] = abs(auc - 0.5)
    return out


def rank_mutual_info(X, y, bins=10):
    """Mutual information against the binary outcome, on quantile bins."""
    X = _clean(X)
    y = np.asarray(y).astype(int)
    out = np.empty(X.shape[1])
    py = np.bincount(y, minlength=2) / len(y)
    hy = -np.sum([p * np.log(p) for p in py if p > 0])

    for j in range(X.shape[1]):
        col = X[:, j]
        qs = np.quantile(col, np.linspace(0, 1, bins + 1)[1:-1])
        b = np.digitize(col, np.unique(qs))
        joint = np.zeros((b.max() + 1, 2))
        np.add.at(joint, (b, y), 1)
        joint /= len(y)
        px = joint.sum(axis=1)
        mi = 0.0
        for bi in range(joint.shape[0]):
            for k in range(2):
                pxy = joint[bi, k]
                if pxy > 0:
                    mi += pxy * np.log(pxy / (px[bi] * py[k] + EPS))
        out[j] = mi / (hy + EPS)
    return out


RANKERS = {
    "variance": rank_variance,
    "correlation": rank_correlation,
    "mean_gap": rank_mean_gap,
    "auc": rank_auc,
    "mutual_info": rank_mutual_info,
}


# ---------------------------------------------------------------
# Kendall tau agreement
# ---------------------------------------------------------------

def kendall_tau(a, b) -> float:
    """
    Kendall's tau-b between two score vectors. No scipy dependency.

    O(n^2), which is fine at 16-105 features and is never called in a
    hot loop.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("score vectors must be the same length")
    n = len(a)
    if n < 2:
        return 1.0

    conc = disc = ties_a = ties_b = 0
    for i in range(n - 1):
        da = a[i + 1:] - a[i]
        db = b[i + 1:] - b[i]
        prod = da * db
        conc += int((prod > 0).sum())
        disc += int((prod < 0).sum())
        ties_a += int(((da == 0) & (db != 0)).sum())
        ties_b += int(((db == 0) & (da != 0)).sum())
        ties_a += int(((da == 0) & (db == 0)).sum())

    denom = np.sqrt((conc + disc + ties_a) * (conc + disc + ties_b))
    return float((conc - disc) / denom) if denom > 0 else 0.0


def drop_outlier_rankers(scores: dict, z: float = 2.0) -> tuple[dict, dict]:
    """
    Remove rankers whose mean agreement with the others is more than `z`
    standard deviations below the average agreement.

    Never removes more than half. With few rankers the spread estimate
    is unstable, so being conservative is deliberate.
    """
    names = sorted(scores)
    if len(names) < 3:
        return dict(scores), {n: 1.0 for n in names}

    agree = {}
    for n in names:
        taus = [kendall_tau(scores[n], scores[m]) for m in names if m != n]
        agree[n] = float(np.mean(taus))

    vals = np.array([agree[n] for n in names])
    mu, sd = vals.mean(), vals.std()
    if sd < EPS:
        return dict(scores), agree

    keep = {n: s for n, s in scores.items() if agree[n] >= mu - z * sd}
    if len(keep) < max(2, len(names) // 2):
        order = sorted(names, key=lambda n: -agree[n])
        keep = {n: scores[n] for n in order[:max(2, len(names) // 2)]}
    return keep, agree


# ---------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------

def _to_ranks(v):
    """Higher score -> rank 1. Average ranks for ties."""
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(-v, kind="mergesort")
    r = np.empty(len(v))
    r[order] = np.arange(1, len(v) + 1)
    vals, inv, counts = np.unique(-v, return_inverse=True,
                                  return_counts=True)
    if (counts > 1).any():
        sums = np.bincount(inv, weights=r)
        r = sums[inv] / counts[inv]
    return r


def consensus_ranking(scores: dict) -> np.ndarray:
    """Mean rank across rankers. Lower = more important."""
    if not scores:
        raise ValueError("no rankers supplied")
    return np.mean([_to_ranks(v) for v in scores.values()], axis=0)


# ---------------------------------------------------------------
# Complexity criterion — how many features to keep
# ---------------------------------------------------------------

def fisher_separability(X, y) -> float:
    """
    Multivariate Fisher-style separability of the two classes for a
    given feature subset. Higher is better separated.

    Uses only diagonal variances, so it stays stable when the subset is
    wide relative to the number of positives -- which is exactly the
    regime here, with a few hundred failed drives.
    """
    X = _clean(X)
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all() or X.shape[1] == 0:
        return 0.0
    a, b = X[y], X[~y]
    num = (a.mean(axis=0) - b.mean(axis=0)) ** 2
    den = a.var(axis=0) + b.var(axis=0) + EPS
    return float(np.sum(num / den))


def choose_k(X, y, order, criterion=fisher_separability,
             tol: float = 0.02, min_k: int = 1) -> tuple[int, list]:
    """
    Walk prefixes of `order`, evaluate the criterion, and return the
    SMALLEST prefix within `tol` (relative) of the best value.

    Preferring the smallest such prefix means ties break toward fewer
    features, which is the point of doing selection at all.
    """
    X = _clean(X)
    curve = []
    for k in range(min_k, X.shape[1] + 1):
        curve.append((k, float(criterion(X[:, order[:k]], y))))
    if not curve:
        return min_k, curve

    best = max(v for _, v in curve)
    if best <= 0:
        return min_k, curve
    for k, v in curve:
        if v >= best * (1.0 - tol):
            return k, curve
    return curve[-1][0], curve


# ---------------------------------------------------------------
# Result
# ---------------------------------------------------------------

@dataclass
class SelectionResult:
    selected: list[str]
    order: list[str]                  # all features, most important first
    consensus_rank: dict              # feature -> mean rank
    ranker_scores: dict               # ranker -> {feature: score}
    ranker_agreement: dict            # ranker -> mean Kendall tau
    dropped_rankers: list[str]
    k: int
    criterion_curve: list = field(default_factory=list)

    @property
    def indices(self) -> list[int]:
        return [self.order.index(f) for f in self.selected]

    def summary(self) -> dict:
        return {
            "n_features_in": len(self.order),
            "n_selected": self.k,
            "selected": list(self.selected),
            "dropped_rankers": list(self.dropped_rankers),
            "min_agreement": (min(self.ranker_agreement.values())
                              if self.ranker_agreement else float("nan")),
        }

    def __str__(self) -> str:
        s = self.summary()
        return (f"selected {s['n_selected']}/{s['n_features_in']} features\n"
                f"  top: {', '.join(self.order[:6])}\n"
                f"  dropped rankers: {s['dropped_rankers'] or 'none'}")


def select_features(
    X,
    y,
    feature_names: list[str],
    rankers: dict | None = None,
    kendall_z: float = 2.0,
    tol: float = 0.02,
    min_k: int = 1,
    max_k: int | None = None,
) -> SelectionResult:
    """
    Run the full WEFR selection on ONE fold's training data.

    X is (n_samples, n_features), already flattened if it came from
    windows. Never pass calibration or test data here.
    """
    X = _clean(X)
    y = np.asarray(y).ravel()
    if X.shape[0] != len(y):
        raise ValueError("X and y length mismatch")
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"{X.shape[1]} columns but {len(feature_names)} names"
        )
    if len(np.unique(y)) < 2:
        raise ValueError("y must contain both classes")

    rankers = rankers if rankers is not None else RANKERS
    raw = {name: np.asarray(fn(X, y), dtype=np.float64)
           for name, fn in rankers.items()}

    kept, agreement = drop_outlier_rankers(raw, z=kendall_z)
    dropped = sorted(set(raw) - set(kept))

    mean_rank = consensus_ranking(kept)
    order_idx = np.argsort(mean_rank, kind="mergesort")
    order = [feature_names[i] for i in order_idx]

    k, curve = choose_k(X, y, order_idx, tol=tol, min_k=min_k)
    if max_k is not None:
        k = min(k, max_k)

    return SelectionResult(
        selected=order[:k],
        order=order,
        consensus_rank={feature_names[i]: float(mean_rank[i])
                        for i in range(len(feature_names))},
        ranker_scores={n: dict(zip(feature_names, v))
                       for n, v in raw.items()},
        ranker_agreement=agreement,
        dropped_rankers=dropped,
        k=k,
        criterion_curve=curve,
    )
