"""
Random Forest baseline.

The back end used by the base paper (Xu et al., DSN 2021). Kept so the
reproduction is faithful and so the sequence model has something to be
compared against.

INTERFACE

Every model in this project exposes the same two methods:

    fit(X, y)               X is (n_samples, n_features)
    predict_proba(X) -> p   p is (n_samples,), P(fails within horizon)

A 1-D probability vector, not the (n, 2) matrix sklearn returns. The
conformal layer expects P(y=1) and derives the rest.

CLASS IMBALANCE

Failures are 1.4-5.2% of drives, so the default forest predicts
"healthy" for everything and looks accurate. `class_weight="balanced"`
is on by default. Note that this changes the SCALE of the output
probabilities: they no longer approximate the true failure rate.

That does not affect conformal coverage at all -- the guarantee holds
for any score function, calibrated or not, which is precisely the point
of using conformal prediction here. It does affect any metric that
thresholds the probability at 0.5, so F0.5 and similar should be
computed at a threshold chosen on training data, not at 0.5.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..config import CFG


class RandomForestBaseline:
    """
    Thin wrapper over sklearn's RandomForestClassifier.

        m = RandomForestBaseline(seed=0).fit(X_train, y_train)
        p = m.predict_proba(X_test)
    """

    name = "random_forest"

    def __init__(self, n_estimators: int = 200, max_depth: int | None = None,
                 min_samples_leaf: int = 2, class_weight="balanced",
                 seed: int = 0, n_jobs: int = -1):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=n_jobs,
        )
        self.model: RandomForestClassifier | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_: int | None = None

    def fit(self, X, y) -> "RandomForestBaseline":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel().astype(int)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.shape}")
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")

        self.n_features_ = X.shape[1]
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(np.nan_to_num(X), y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() before predict_proba()")
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"expected {self.n_features_} features, got {X.shape[1]}"
            )
        p = self.model.predict_proba(np.nan_to_num(X))
        # A fold whose training data has one class only still has to
        # return something usable rather than crash.
        if p.shape[1] == 1:
            only = int(self.classes_[0])
            return np.full(len(X), float(only))
        return p[:, list(self.classes_).index(1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        return self.model.feature_importances_


def make_baseline(seed: int = None, **kw) -> RandomForestBaseline:
    seed = seed if seed is not None else CFG.seed
    return RandomForestBaseline(seed=seed, **kw)
