"""
Mondrian conformal prediction.

Split conformal guarantees MARGINAL coverage: averaged over all test
points, P(Y in C(X)) >= 1 - alpha. It says nothing about any subgroup.
At the real fleet's 1.4-5.2% failure rate, marginal coverage is
dominated by healthy drives and can sit at nominal while the failure
class -- the class an operator cares about -- is covered at 15%. That
measurement is what made this module required rather than optional.

Mondrian conformal (Vovk 2003) partitions the calibration set by a
taxonomy and computes a separate quantile per cell. Coverage then holds
WITHIN each cell:

    P(Y in C(X) | cell(X, Y) = k) >= 1 - alpha   for every k

Three taxonomies are provided:

  by="class"   cell = true label. Coverage holds for healthy AND for
               failed drives separately. At prediction time the label is
               unknown, so each candidate label k is tested against its
               own quantile qhat_k. This is the standard Mondrian
               construction for classification.

  by="group"   cell = an external group such as vendor. Coverage holds
               per vendor. Under LOMO the held-out vendor has no
               calibration cell by construction; see FALLBACK below.

  by="both"    cell = (group, class). Strongest guarantee, thinnest
               cells. With 3 positive drives in the lomo_C calibration
               set, expect +inf quantiles.

COST
    Each cell's quantile is set by fewer points, and rare cells get
    conservative (larger) quantiles. Sets grow. The paper reports this
    growth as the price of the stronger guarantee.

FALLBACK
    When a test point's group has no calibration cell, there is no valid
    quantile for it. Two choices:
      "pooled"  substitute the pooled split-conformal quantile. The
                point receives a marginal guarantee, not a group one,
                and is flagged in `result.fallback`.
      "inf"     emit the full set. Honest and useless.
    This is exactly the LOMO situation, and the contrast is the point:
    group-conditional coverage holds for vendors we have calibration for
    and cannot be provided for the one we do not.

THIN CELLS
    conformal_quantile returns +inf when a cell has fewer than
    1/alpha - 1 points. The affected label is then always included. This
    is the correct answer -- the data cannot support the level -- and
    the resulting set-size increase is reported, not hidden.
"""

from __future__ import annotations

import numpy as np

from .split import (HEALTHY, FAILURE, ConformalResult, candidate_scores,
                    conformal_quantile, force_nonempty, true_scores)

LABELS = (HEALTHY, FAILURE)
_NO_GROUP = "__none__"   # sentinel: by="class" has no group axis


class MondrianConformal:
    """
    Cell-conditional split conformal.

        cp = MondrianConformal(alpha=0.10, by="class").fit(p_cal, y_cal)
        result = cp.predict(p_test)

        cp = MondrianConformal(alpha=0.10, by="group").fit(
            p_cal, y_cal, groups=vendor_cal)
        result = cp.predict(p_test, groups=vendor_test)

    Calibration data must be disjoint from training data, and under LOMO
    must exclude the held-out vendor entirely.
    """

    def __init__(self, alpha: float = 0.10, score_fn: str = "lac",
                 by: str = "class", fallback: str = "pooled",
                 seed: int = 0, allow_empty: bool = True):
        if by not in ("class", "group", "both"):
            raise ValueError(f"by must be class|group|both, got {by!r}")
        if fallback not in ("pooled", "inf"):
            raise ValueError(f"fallback must be pooled|inf, got {fallback!r}")
        self.alpha = alpha
        self.score_fn = score_fn
        self.by = by
        self.fallback = fallback
        self.seed = seed
        self.allow_empty = allow_empty

        self.qhat_: dict | None = None
        self.qhat_pooled_: float | None = None
        self.n_cal_: int | None = None
        self.cell_sizes_: dict | None = None
        self.groups_seen_: set | None = None

    # -----------------------------------------------------------
    def _cell(self, label, group):
        if self.by == "class":
            return int(label)
        if self.by == "group":
            return group
        return (group, int(label))

    def fit(self, cal_probs, cal_labels, groups=None) -> "MondrianConformal":
        cal_probs = np.asarray(cal_probs, dtype=np.float64).ravel()
        cal_labels = np.asarray(cal_labels, dtype=int).ravel()
        n = len(cal_labels)

        if self.by in ("group", "both"):
            if groups is None:
                raise ValueError(f"by={self.by!r} requires groups")
            groups = np.asarray(groups).ravel()
            if len(groups) != n:
                raise ValueError("groups length mismatch")
            self.groups_seen_ = set(np.unique(groups).tolist())
        else:
            groups = np.full(n, _NO_GROUP, dtype=object)

        scores = true_scores(cal_probs, cal_labels, self.score_fn, self.seed)

        cells = {}
        for s, y, g in zip(scores, cal_labels, groups):
            cells.setdefault(self._cell(y, g), []).append(s)

        self.qhat_ = {k: conformal_quantile(np.asarray(v), self.alpha)
                      for k, v in cells.items()}
        self.cell_sizes_ = {k: len(v) for k, v in cells.items()}
        self.qhat_pooled_ = conformal_quantile(scores, self.alpha)
        self.n_cal_ = n
        return self

    # -----------------------------------------------------------
    def _threshold(self, label, group):
        """qhat for one (label, group) at prediction time."""
        key = self._cell(label, group)
        if key in self.qhat_:
            return self.qhat_[key], False

        # Cell missing. For by="class" every label was present at fit
        # time unless a class had zero calibration points.
        if self.by == "class":
            return float("inf"), True
        if self.fallback == "pooled":
            return self.qhat_pooled_, True
        return float("inf"), True

    def predict(self, probs, groups=None) -> ConformalResult:
        if self.qhat_ is None:
            raise RuntimeError("call fit() before predict()")
        probs = np.asarray(probs, dtype=np.float64).ravel()
        n = len(probs)

        if self.by in ("group", "both"):
            if groups is None:
                raise ValueError(f"by={self.by!r} requires groups")
            groups = np.asarray(groups).ravel()
            if len(groups) != n:
                raise ValueError("groups length mismatch")
        else:
            groups = np.full(n, _NO_GROUP, dtype=object)

        cand = candidate_scores(probs, self.score_fn, self.seed)
        sets = np.zeros((n, 2), dtype=bool)
        fallback = np.zeros(n, dtype=bool)

        # Vectorise over distinct groups; each group has at most two
        # thresholds (one per candidate label).
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            for k in LABELS:
                q, fb = self._threshold(k, g)
                sets[idx, k] = cand[idx, k] <= q
                if fb:
                    fallback[idx] = True

        if not self.allow_empty:
            sets = force_nonempty(sets, cand)

        return ConformalResult(
            sets=sets,
            qhat=dict(self.qhat_),
            alpha=self.alpha,
            score_fn=self.score_fn,
            n_cal=self.n_cal_,
            method=f"mondrian_{self.by}",
            fallback=fallback if fallback.any() else None,
        )

    # -----------------------------------------------------------
    def cell_report(self) -> list[dict]:
        """Per-cell calibration size and threshold, for the appendix."""
        if self.qhat_ is None:
            raise RuntimeError("call fit() first")
        return [
            {"cell": k, "n_cal": self.cell_sizes_[k], "qhat": q,
             "supported": np.isfinite(q)}
            for k, q in sorted(self.qhat_.items(), key=lambda kv: str(kv[0]))
        ]


# ---------------------------------------------------------------
# Group-conditional coverage diagnostic
# ---------------------------------------------------------------

def group_conditional_coverage(result: ConformalResult, labels,
                               groups) -> dict:
    """
    Coverage per group, and per (group, class).

    For the LOMO contrast: on the standard split this shows per-vendor
    coverage holding; on a LOMO fold the held-out vendor's coverage is
    the headline number.
    """
    labels = np.asarray(labels, dtype=int).ravel()
    groups = np.asarray(groups).ravel()
    hit = result.sets[np.arange(len(labels)), labels]

    out = {}
    for g in np.unique(groups):
        m = groups == g
        row = {"n": int(m.sum()), "coverage": float(hit[m].mean())}
        for k, name in ((HEALTHY, "healthy"), (FAILURE, "failure")):
            mk = m & (labels == k)
            row[f"coverage_{name}"] = (float(hit[mk].mean())
                                       if mk.any() else float("nan"))
            row[f"n_{name}"] = int(mk.sum())
        if result.fallback is not None:
            row["fallback_rate"] = float(result.fallback[m].mean())
        out[g] = row
    return out
