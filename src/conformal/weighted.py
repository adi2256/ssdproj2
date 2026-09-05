"""
Weighted conformal prediction under covariate shift.

Split conformal assumes calibration and test points are exchangeable.
LOMO breaks this by construction: calibration drives come from two
vendors, test drives from a third. Tibshirani et al. (2019) show that if
the shift is in the covariates only -- P_test(X) differs from P_cal(X)
but P(Y|X) is unchanged -- coverage can be restored by reweighting the
calibration scores by the density ratio

    w(x) = dP_test(x) / dP_cal(x).

Calibration points that look like the test distribution count more;
points that look nothing like it count less. The quantile becomes
test-point-specific:

    qhat(x) = inf { s : sum_{i: s_i <= s} p_i(x) >= 1 - alpha }

    p_i(x)  = w(x_i) / ( sum_j w(x_j) + w(x) )
    p_x     = w(x)   / ( sum_j w(x_j) + w(x) )        # weight on +inf

WHAT IT CAN AND CANNOT FIX
    The guarantee is exact when w is the true density ratio and the
    shift is pure covariate shift. The real fleet has label shift too
    (failure rate 1.42% -> 5.21% across vendors) and structural shift
    (different feature sets). Weighted conformal is not a complete
    remedy for LOMO; it is the standard shift-robust baseline, and the
    paper measures how much of the gap it closes.

WEIGHT ESTIMATION
    w is unknown and estimated by a domain classifier: fit
    P(source = test | x) on calibration features (label 0) against
    test features (label 1, no outcome labels needed), then

        w(x) = P(test | x) / P(cal | x)  *  n_cal / n_test.

    Only test FEATURES are used, never test labels, so this is legal
    at deployment time: the new vendor's SMART telemetry is available,
    its failures are not.

    Estimated ratios are noisy in the tails. Weights are clipped to
    [clip_lo, clip_hi]; without clipping a handful of extreme weights
    can dominate the quantile. Clipping trades some validity for
    stability and the clip range is reported.

    A numpy-only logistic regression is provided so the module has no
    dependency beyond numpy. Any callable returning P(test | x) can be
    substituted.
"""

from __future__ import annotations

import numpy as np

from .split import (ConformalResult, candidate_scores, force_nonempty,
                    true_scores)


# ---------------------------------------------------------------
# Weighted quantile
# ---------------------------------------------------------------

def weighted_quantiles(cal_scores: np.ndarray, cal_weights: np.ndarray,
                       test_weights: np.ndarray, alpha: float) -> np.ndarray:
    """
    One quantile per test point.

    Vectorised: sort calibration scores once, then for each test point
    find the first sorted score at which the normalised cumulative
    calibration weight reaches 1 - alpha. If it never does -- the test
    point's own weight is large enough that the mass on +inf is needed
    -- the quantile is +inf and the full set is returned.
    """
    cal_scores = np.asarray(cal_scores, dtype=np.float64).ravel()
    cal_weights = np.asarray(cal_weights, dtype=np.float64).ravel()
    test_weights = np.asarray(test_weights, dtype=np.float64).ravel()

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if cal_scores.shape != cal_weights.shape:
        raise ValueError("cal_scores and cal_weights length mismatch")
    if (cal_weights < 0).any() or (test_weights < 0).any():
        raise ValueError("weights must be non-negative")

    order = np.argsort(cal_scores)
    s_sorted = cal_scores[order]
    w_sorted = cal_weights[order]
    cum = np.cumsum(w_sorted)                       # (n_cal,)
    W = cum[-1]

    # For test point j: need cum[i] / (W + w_j) >= 1 - alpha
    #                 <=> cum[i] >= (1 - alpha) * (W + w_j)
    targets = (1.0 - alpha) * (W + test_weights)    # (n_test,)
    idx = np.searchsorted(cum, targets, side="left")

    q = np.full(len(test_weights), np.inf)
    ok = idx < len(s_sorted)
    q[ok] = s_sorted[idx[ok]]
    return q


# ---------------------------------------------------------------
# Density-ratio estimation via a domain classifier
# ---------------------------------------------------------------

class _Logistic:
    """Minimal L2-regularised logistic regression, numpy only."""

    def __init__(self, lr=0.1, epochs=500, l2=1e-3, seed=0):
        self.lr, self.epochs, self.l2, self.seed = lr, epochs, l2, seed
        self.mu = self.sd = self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        Z = (X - self.mu) / self.sd

        rng = np.random.default_rng(self.seed)
        n, d = Z.shape
        self.w = rng.normal(0, 0.01, d)
        for _ in range(self.epochs):
            p = self._sig(Z @ self.w + self.b)
            err = p - y
            self.w -= self.lr * ((Z.T @ err) / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        return self._sig(Z @ self.w + self.b)

    @staticmethod
    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def estimate_density_ratio(X_cal, X_test, classifier=None,
                           clip=(0.05, 20.0), seed=0):
    """
    Estimate w(x) = dP_test / dP_cal on both the calibration and test
    features via a domain classifier.

    Returns (w_cal, w_test, info). Uses test FEATURES only.

    `classifier` may be any object with fit(X, y) / predict_proba(X)
    returning P(y=1). Defaults to the numpy logistic regression.
    """
    X_cal = np.asarray(X_cal, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    if X_cal.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_cal and X_test must be 2-D (flatten first)")
    if X_cal.shape[1] != X_test.shape[1]:
        raise ValueError("feature dimension mismatch")

    n_cal, n_test = len(X_cal), len(X_test)
    X = np.vstack([X_cal, X_test])
    d = np.concatenate([np.zeros(n_cal), np.ones(n_test)])

    clf = classifier if classifier is not None else _Logistic(seed=seed)
    clf.fit(X, d)
    p = np.clip(clf.predict_proba(X), 1e-6, 1 - 1e-6)

    ratio = (p / (1.0 - p)) * (n_cal / n_test)
    lo, hi = clip
    w = np.clip(ratio, lo, hi)

    # How separable are the domains? AUC of the domain classifier is a
    # direct measure of covariate shift magnitude and is worth reporting.
    auc = _auc(p, d)

    info = {
        "domain_auc": auc,
        "clip": clip,
        "frac_clipped_lo": float((ratio < lo).mean()),
        "frac_clipped_hi": float((ratio > hi).mean()),
        "w_cal_mean": float(w[:n_cal].mean()),
        "w_test_mean": float(w[n_cal:].mean()),
        "ess_cal": float(w[:n_cal].sum() ** 2 / (w[:n_cal] ** 2).sum()),
    }
    return w[:n_cal], w[n_cal:], info


def _auc(scores, labels):
    """Rank-based AUC, no sklearn."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.empty_like(scores)
    order = np.argsort(scores)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True,
                               return_counts=True)
    if (counts > 1).any():
        sums = np.bincount(inv, weights=ranks)
        ranks = sums[inv] / counts[inv]
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------
# Weighted split conformal
# ---------------------------------------------------------------

class WeightedConformal:
    """
    Split conformal with importance-weighted calibration.

        cp = WeightedConformal(alpha=0.10).fit(p_cal, y_cal, w_cal)
        result = cp.predict(p_test, w_test)

    or let it estimate the weights from features:

        cp = WeightedConformal(alpha=0.10).fit_from_features(
            p_cal, y_cal, X_cal, X_test)
        result = cp.predict_from_features(p_test, X_test)

    With uniform weights this reduces exactly to split conformal
    (verified in tests).
    """

    def __init__(self, alpha: float = 0.10, score_fn: str = "lac",
                 clip=(0.05, 20.0), seed: int = 0,
                 allow_empty: bool = True):
        self.alpha = alpha
        self.score_fn = score_fn
        self.clip = clip
        self.seed = seed
        self.allow_empty = allow_empty

        self.cal_scores_ = None
        self.cal_weights_ = None
        self.n_cal_ = None
        self.weight_info_: dict | None = None
        self._w_test_cache = None

    def fit(self, cal_probs, cal_labels, cal_weights) -> "WeightedConformal":
        cal_probs = np.asarray(cal_probs, dtype=np.float64).ravel()
        cal_labels = np.asarray(cal_labels, dtype=int).ravel()
        cal_weights = np.asarray(cal_weights, dtype=np.float64).ravel()
        if len(cal_weights) != len(cal_labels):
            raise ValueError("cal_weights length mismatch")
        if (cal_weights < 0).any():
            raise ValueError("weights must be non-negative")

        self.cal_scores_ = true_scores(cal_probs, cal_labels,
                                       self.score_fn, self.seed)
        self.cal_weights_ = cal_weights
        self.n_cal_ = len(cal_labels)
        return self

    def fit_from_features(self, cal_probs, cal_labels, X_cal, X_test,
                          classifier=None) -> "WeightedConformal":
        w_cal, w_test, info = estimate_density_ratio(
            X_cal, X_test, classifier=classifier, clip=self.clip,
            seed=self.seed,
        )
        self.weight_info_ = info
        self._w_test_cache = w_test
        return self.fit(cal_probs, cal_labels, w_cal)

    def predict(self, probs, test_weights) -> ConformalResult:
        if self.cal_scores_ is None:
            raise RuntimeError("call fit() before predict()")
        probs = np.asarray(probs, dtype=np.float64).ravel()
        test_weights = np.asarray(test_weights, dtype=np.float64).ravel()
        if len(test_weights) != len(probs):
            raise ValueError("test_weights length mismatch")

        q = weighted_quantiles(self.cal_scores_, self.cal_weights_,
                               test_weights, self.alpha)      # (n_test,)
        cand = candidate_scores(probs, self.score_fn, self.seed)
        sets = cand <= q[:, None]

        if not self.allow_empty:
            sets = force_nonempty(sets, cand)

        return ConformalResult(
            sets=sets,
            qhat=q,
            alpha=self.alpha,
            score_fn=self.score_fn,
            n_cal=self.n_cal_,
            method="weighted",
        )

    def predict_from_features(self, probs, X_test=None) -> ConformalResult:
        """
        Use the test weights computed during fit_from_features. X_test
        is accepted for API symmetry; it must be the same array.
        """
        if self._w_test_cache is None:
            raise RuntimeError("call fit_from_features() first")
        if X_test is not None and len(X_test) != len(self._w_test_cache):
            raise ValueError("X_test differs from the array used at fit")
        return self.predict(probs, self._w_test_cache)
