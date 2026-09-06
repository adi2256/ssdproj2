"""
Wear-stage assignment.

The base paper (Xu et al., DSN 2021) found that the SMART attributes
informative for failure CHANGE as drives age, and added a wear-out
updating phase to its feature selection for that reason. This module
assigns each window a wear stage so the same question can be asked of
calibration: does a conformal guarantee fitted on the whole fleet hold
within each stage?

Two strategies:

    quantile    Equal-frequency bins on power-on hours. Simple,
                assumption-free, and every bin has the same calibration
                mass -- which matters because Mondrian coverage per
                cell depends on that cell's calibration count.

    changepoint Data-driven stage boundaries where the failure rate as
                a function of age changes regime. Closer to what WEFR
                actually does and to the bathtub curve's three phases.
                Implemented as binary segmentation on the failure-rate
                curve, which needs no external dependency.

Age is SMART 9 (power-on hours), column `r_9`. It is populated for
every vendor in this fleet and, unlike most shared attributes, is
monotone within a drive by construction. Stage is read at the END of
the window, matching where the label is read.

LEAKAGE
    Stage boundaries -- both quantile edges and change points -- are
    fit on the TRAINING split only and applied unchanged to calibration
    and test. Fitting them on pooled data would let test-set age
    structure shape the taxonomy the guarantee is stated over.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.windowing import WindowSet

AGE_COL = "r_9"


@dataclass
class StageMap:
    """Fitted stage boundaries. Apply with .assign()."""
    edges: np.ndarray              # interior boundaries, ascending
    strategy: str
    n_stages: int
    fit_on: int
    diagnostics: dict = field(default_factory=dict)

    def assign(self, age: np.ndarray) -> np.ndarray:
        """Map ages to integer stages 0..n_stages-1."""
        age = np.asarray(age, dtype=np.float64)
        return np.searchsorted(self.edges, age, side="right").astype(int)

    def labels(self) -> list[str]:
        lo = ["-inf"] + [f"{e:.0f}" for e in self.edges]
        hi = [f"{e:.0f}" for e in self.edges] + ["inf"]
        return [f"s{i}[{a},{b})" for i, (a, b) in enumerate(zip(lo, hi))]


# ---------------------------------------------------------------
# Age extraction
# ---------------------------------------------------------------

def window_age(ws: WindowSet, col: str = AGE_COL) -> np.ndarray:
    """
    Power-on hours at the END of each window, in the window's own
    units (raw, before any normaliser). Pass the RAW WindowSet, not a
    normalised one -- normalisation would destroy the age scale.
    """
    if col not in ws.features:
        raise ValueError(f"{col!r} not in window features {ws.features}")
    j = ws.features.index(col)
    age = ws.X[:, -1, j].astype(np.float64)
    if not np.isfinite(age).all():
        # Forward-fill already ran in make_windows; anything still
        # non-finite is a drive with no age reading at all.
        age = np.where(np.isfinite(age), age, np.nan)
    return age


# ---------------------------------------------------------------
# Strategy 1: quantile bins
# ---------------------------------------------------------------

def fit_quantile_stages(age: np.ndarray, n_stages: int = 5) -> StageMap:
    a = np.asarray(age, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("no finite ages to fit on")
    qs = np.linspace(0, 1, n_stages + 1)[1:-1]
    edges = np.unique(np.quantile(a, qs))
    return StageMap(edges=edges, strategy="quantile",
                    n_stages=len(edges) + 1, fit_on=int(a.size),
                    diagnostics={"quantiles": qs.tolist()})


# ---------------------------------------------------------------
# Strategy 2: change points on the failure-rate curve
# ---------------------------------------------------------------

def _failure_rate_curve(age, y, n_bins: int = 60):
    """
    Failure rate as a function of age, on equal-frequency age bins so
    every point has comparable statistical weight.
    """
    a = np.asarray(age, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(a)
    a, y = a[m], y[m]
    order = np.argsort(a, kind="mergesort")
    a, y = a[order], y[order]
    bins = np.array_split(np.arange(a.size), n_bins)
    centres = np.array([a[b].mean() for b in bins if b.size])
    rates = np.array([y[b].mean() for b in bins if b.size])
    return centres, rates


def _binary_segment(signal: np.ndarray, n_splits: int,
                    min_len: int = 5) -> list[int]:
    """
    Greedy binary segmentation: repeatedly split the segment whose
    split most reduces within-segment squared error. Returns split
    indices into `signal`. No external dependency.
    """
    def cost(seg):
        return float(((seg - seg.mean()) ** 2).sum()) if seg.size else 0.0

    segments = [(0, len(signal))]
    splits = []
    for _ in range(n_splits):
        best = None
        for si, (lo, hi) in enumerate(segments):
            if hi - lo < 2 * min_len:
                continue
            base = cost(signal[lo:hi])
            for k in range(lo + min_len, hi - min_len + 1):
                gain = base - cost(signal[lo:k]) - cost(signal[k:hi])
                if best is None or gain > best[0]:
                    best = (gain, si, k)
        if best is None:
            break
        _, si, k = best
        lo, hi = segments.pop(si)
        segments.extend([(lo, k), (k, hi)])
        segments.sort()
        splits.append(k)
    return sorted(splits)


def fit_changepoint_stages(age: np.ndarray, y: np.ndarray,
                           n_stages: int = 3, n_bins: int = 60) -> StageMap:
    """
    Stage boundaries where the failure-rate-vs-age curve changes
    regime. n_stages=3 targets the bathtub's three phases; the
    boundaries are learned, not assumed.
    """
    centres, rates = _failure_rate_curve(age, y, n_bins=n_bins)
    if centres.size < 2 * n_stages:
        raise ValueError("too few age bins for the requested stages")
    splits = _binary_segment(rates, n_splits=n_stages - 1)
    # boundary = midpoint between the adjacent bin centres
    edges = np.array([(centres[k - 1] + centres[k]) / 2 for k in splits])
    return StageMap(
        edges=edges, strategy="changepoint",
        n_stages=len(edges) + 1, fit_on=int(np.isfinite(age).sum()),
        diagnostics={"bin_centres": centres.tolist(),
                     "bin_rates": rates.tolist(),
                     "split_bins": splits},
    )


# ---------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------

def fit_stages(ws_train: WindowSet, strategy: str = "quantile",
               n_stages: int = 5) -> StageMap:
    """Fit on the TRAINING window set only."""
    age = window_age(ws_train)
    if strategy == "quantile":
        return fit_quantile_stages(age, n_stages=n_stages)
    if strategy == "changepoint":
        return fit_changepoint_stages(age, ws_train.y, n_stages=n_stages)
    raise ValueError(f"strategy must be quantile|changepoint, got {strategy!r}")


def stage_table(stage: np.ndarray, y: np.ndarray, smap: StageMap):
    """Per-stage count and failure rate -- the bathtub, if it's there."""
    import polars as pl
    rows = []
    for s in range(smap.n_stages):
        m = stage == s
        rows.append({
            "stage": s, "label": smap.labels()[s],
            "n": int(m.sum()),
            "n_pos": int(y[m].sum()) if m.any() else 0,
            "rate": float(y[m].mean()) if m.any() else float("nan"),
        })
    return pl.DataFrame(rows)
