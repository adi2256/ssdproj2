"""
Conformal prediction layer.

Four methods share one interface -- fit on calibration probabilities and
labels, predict sets on test probabilities -- and return a
ConformalResult that coverage_report() turns into one row of the results
table.

    split            marginal guarantee, exchangeability assumed
    mondrian_class   per-class guarantee (required: see checkpoint §8)
    mondrian_group   per-vendor guarantee; falls back for unseen vendors
    weighted         covariate-shift correction via density ratios

Use METHODS to iterate in the LOMO harness. Each entry is a factory
taking (alpha, score_fn, seed) and returning an unfitted predictor. The
harness supplies groups / features where the method needs them.
"""

from .mondrian import MondrianConformal, group_conditional_coverage
from .scores import SCORE_FNS, aps_scores, lac_scores
from .split import (ConformalResult, SplitConformal,
                    class_conditional_coverage, conformal_quantile,
                    coverage_report, empirical_coverage)
from .weighted import (WeightedConformal, estimate_density_ratio,
                       weighted_quantiles)

METHODS = {
    "split": lambda alpha, score_fn, seed:
        SplitConformal(alpha=alpha, score_fn=score_fn, seed=seed),
    "mondrian_class": lambda alpha, score_fn, seed:
        MondrianConformal(alpha=alpha, score_fn=score_fn, by="class",
                          seed=seed),
    "mondrian_group": lambda alpha, score_fn, seed:
        MondrianConformal(alpha=alpha, score_fn=score_fn, by="group",
                          fallback="pooled", seed=seed),
    "weighted": lambda alpha, score_fn, seed:
        WeightedConformal(alpha=alpha, score_fn=score_fn, seed=seed),
}

# What each method needs beyond (probs, labels) at fit / predict time.
METHOD_INPUTS = {
    "split": set(),
    "mondrian_class": set(),
    "mondrian_group": {"groups"},
    "weighted": {"features"},
}

__all__ = [
    "METHODS", "METHOD_INPUTS",
    "ConformalResult", "SplitConformal", "MondrianConformal",
    "WeightedConformal", "SCORE_FNS", "lac_scores", "aps_scores",
    "conformal_quantile", "weighted_quantiles", "estimate_density_ratio",
    "empirical_coverage", "class_conditional_coverage",
    "group_conditional_coverage", "coverage_report",
]
