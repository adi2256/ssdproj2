"""
Nonconformity scores.

A nonconformity score measures how poorly a candidate label fits a given
example. Split conformal thresholds these scores at a calibration
quantile to build prediction sets.

The score function is a free choice — coverage holds for any of them.
What changes is set size and how the sets behave under class imbalance,
which is exactly what makes the choice worth reporting in the paper.

Binary task: label 0 = healthy, 1 = fails within horizon.
`probs` is always P(y=1), shape (n,).
"""

from __future__ import annotations

import numpy as np

LABELS = (0, 1)


def _as_prob_matrix(probs: np.ndarray) -> np.ndarray:
    """(n,) of P(y=1) -> (n, 2) of [P(y=0), P(y=1)]."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if probs.ndim != 1:
        raise ValueError(f"expected 1-D probs, got shape {probs.shape}")
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("probs must lie in [0, 1]")
    return np.column_stack([1.0 - probs, probs])


# ---------------------------------------------------------------
# Least-ambiguous set / inverse-probability score
# ---------------------------------------------------------------

def lac_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    s_i = 1 - P(y_i | x_i)

    The standard baseline (Sadinle et al. 2019). Produces the smallest
    average sets for a given coverage level, but distributes coverage
    unevenly across classes — under heavy imbalance the minority class
    can be under-covered while marginal coverage still holds.

    That property matters here: failures are ~1% of drives, so marginal
    coverage can look fine while failure-class coverage is poor. Report
    class-conditional coverage alongside marginal, or use Mondrian
    conformal to enforce it.
    """
    p = _as_prob_matrix(probs)
    labels = np.asarray(labels, dtype=int).ravel()
    if labels.shape[0] != p.shape[0]:
        raise ValueError("probs and labels length mismatch")
    if not np.isin(labels, LABELS).all():
        raise ValueError("labels must be 0 or 1")
    return 1.0 - p[np.arange(len(labels)), labels]


def lac_candidate_scores(probs: np.ndarray) -> np.ndarray:
    """Score every candidate label. Returns (n, 2)."""
    return 1.0 - _as_prob_matrix(probs)


# ---------------------------------------------------------------
# Adaptive prediction sets (APS)
# ---------------------------------------------------------------

def aps_scores(probs: np.ndarray, labels: np.ndarray,
               rng: np.random.Generator | None = None,
               randomized: bool = True) -> np.ndarray:
    """
    Cumulative probability mass, most-likely class first, up to and
    including the true label (Romano et al. 2020).

    Gives better class-conditional coverage than LAC at the cost of
    larger sets. The randomized variant achieves exact rather than
    conservative coverage; disable it for deterministic tests.
    """
    p = _as_prob_matrix(probs)
    labels = np.asarray(labels, dtype=int).ravel()
    n = p.shape[0]

    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)

    rank = np.argmax(order == labels[:, None], axis=1)
    idx = np.arange(n)
    scores = cumsum[idx, rank]

    if randomized:
        rng = rng or np.random.default_rng(0)
        u = rng.uniform(size=n)
        scores = scores - u * sorted_p[idx, rank]
    return scores


def aps_candidate_scores(probs: np.ndarray,
                         rng: np.random.Generator | None = None,
                         randomized: bool = True) -> np.ndarray:
    """
    Score every candidate label under APS. Returns (n, 2).

    Must use the SAME randomization scheme as aps_scores. If calibration
    scores are randomized and candidate scores are not, the calibration
    scores are systematically smaller, qhat is too small, and the
    predictor undercovers — a bug that looks exactly like a shift effect.

    One shared u per row across candidate labels, matching the draw in
    aps_scores.
    """
    p = _as_prob_matrix(probs)
    n = p.shape[0]

    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)

    u = (rng or np.random.default_rng(0)).uniform(size=n) if randomized \
        else np.zeros(n)

    out = np.empty_like(p)
    idx = np.arange(n)
    for k in range(p.shape[1]):
        rank = np.argmax(order == k, axis=1)
        out[:, k] = cumsum[idx, rank] - u * sorted_p[idx, rank]
    return out


SCORE_FNS = {
    "lac": (lac_scores, lac_candidate_scores),
    "aps": (aps_scores, aps_candidate_scores),
}
