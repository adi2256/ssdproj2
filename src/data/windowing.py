"""
Trajectory windowing.

Turns labelled daily SMART rows into fixed-length sequences, the actual
model input.

    (n_samples, window_len, n_features)

WINDOW SEMANTICS
    A window ending at day T is "everything known as of T". Its label is
    the label of row T: does this drive fail within `horizon` days of T.
    The label is therefore taken from the LAST row of the window, never
    the first and never an aggregate. Taking it from anywhere else
    silently shifts the prediction horizon.

STRIDE
    `stride` subsamples windows along time. `positive_stride` defaults
    to `stride`, i.e. NO oversampling of the positive class.

DATE GAPS
    Drives skip days. Windows are built over a REINDEXED continuous
    daily range per drive, so a window always spans exactly
    `window_len` calendar days.

MISSING VALUES
    Reindexing introduces gaps, and the raw data already has NaN. Values
    are forward-filled within a drive, then zero-filled at the head. A
    window is rejected if the fraction of originally-present values
    falls below `min_valid_frac`.

NORMALISATION
    Statistics are fit on TRAINING drives only and applied unchanged to
    calibration and test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..config import CFG, DRIVE_KEY
from ..schema import COMMON_COLS
from .labels import LABEL_COL, VENDOR_COL


@dataclass
class WindowSet:
    """
    Windowed data plus the provenance needed to group by drive.

    X          (n, window_len, n_features) float32
    y          (n,) int8
    drive_idx  (n,) int  -- index into `drives`
    end_ds     (n,) date -- last day of each window
    drives     list[(model, disk_id)]
    vendors    (n,) str  -- vendor of each window's drive
    features   list[str]
    """

    X: np.ndarray
    y: np.ndarray
    drive_idx: np.ndarray
    end_ds: np.ndarray
    drives: list[tuple]
    vendors: np.ndarray
    features: list[str]
    window_len: int
    stride: int
    positive_stride: int

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean()) if len(self) else 0.0

    @property
    def n_drives(self) -> int:
        return len(self.drives)

    def drive_keys(self) -> np.ndarray:
        """(n,) array of (model, disk_id) tuples, one per window."""
        return np.array(
            [self.drives[i] for i in self.drive_idx],
            dtype=object,
        )

    def summary(self) -> dict:
        return {
            "n_windows": len(self),
            "n_drives": self.n_drives,
            "n_features": len(self.features),
            "window_len": self.window_len,
            "n_positive": int(self.y.sum()),
            "positive_rate": self.positive_rate,
            "stride": self.stride,
            "positive_stride": self.positive_stride,
        }

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"{s['n_windows']:,} windows from {s['n_drives']:,} drives\n"
            f"  shape      : {self.X.shape}\n"
            f"  positives  : {s['n_positive']:,} "
            f"({100 * s['positive_rate']:.4f}%)\n"
            f"  stride     : {s['stride']} "
            f"(positives {s['positive_stride']})"
        )


def _reindex_drive(
    g: pl.DataFrame,
    features: list[str],
) -> tuple:
    """
    Reindex one drive to a continuous daily range.

    Returns
    -------
    values:
        Filled SMART values.
    labels:
        Labels, with -1 for undefined/reindexed rows.
    dates:
        Calendar dates.
    present_mask:
        Boolean mask indicating original non-null SMART values
        BEFORE filling.
    """

    # ------------------------------------------------------------
    # NORMALISE ds TO CALENDAR DATE
    #
    # Real Alibaba telemetry can contain datetime values such as:
    #     2019-01-01 03:08:11
    #
    # Synthetic/test fixtures can already contain:
    #     2019-01-01
    #
    # Therefore we handle BOTH types.
    # ------------------------------------------------------------

    ds_dtype = g.schema["ds"]

    if ds_dtype == pl.Date:
        # Already a calendar date.
        first = g["ds"].min()
        last = g["ds"].max()

    elif ds_dtype == pl.Datetime:
        # Convert datetime -> calendar date.
        first = g["ds"].min().date()
        last = g["ds"].max().date()

        g = g.with_columns(
            pl.col("ds").dt.date().alias("ds")
        )

    else:
        raise TypeError(
            f"Unsupported ds dtype: {ds_dtype!r}. "
            "Expected pl.Date or pl.Datetime."
        )

    # ------------------------------------------------------------
    # COMPLETE DAILY CALENDAR
    # ------------------------------------------------------------

    full = pl.DataFrame(
        {
            "ds": pl.date_range(
                first,
                last,
                interval="1d",
                eager=True,
            )
        }
    )

    # ------------------------------------------------------------
    # LEFT JOIN ORIGINAL OBSERVATIONS ONTO COMPLETE CALENDAR
    # ------------------------------------------------------------

    j = (
        full
        .join(
            g,
            on="ds",
            how="left",
        )
        .sort("ds")
    )

    # ------------------------------------------------------------
    # EXTRACT SMART VALUES
    # ------------------------------------------------------------

    vals = (
        j
        .select(features)
        .to_numpy()
        .astype(np.float64)
    )

    # IMPORTANT:
    # Record which SMART cells were genuinely present BEFORE
    # forward filling.
    present = np.isfinite(vals)

    # ------------------------------------------------------------
    # FORWARD FILL SMART VALUES
    #
    # Missing values inside a drive are filled using the most recent
    # available value.
    #
    # Leading missing values are then zero-filled.
    # ------------------------------------------------------------

    for c in range(vals.shape[1]):

        col = vals[:, c]

        idx = np.where(
            np.isfinite(col),
            np.arange(len(col)),
            0,
        )

        np.maximum.accumulate(
            idx,
            out=idx,
        )

        filled = col[idx]

        filled[~np.isfinite(filled)] = 0.0

        vals[:, c] = filled

    # ------------------------------------------------------------
    # LABELS
    #
    # Rows introduced by reindexing have no label.
    # These MUST remain undefined (-1), not healthy (0).
    # ------------------------------------------------------------

    labels = (
        j[LABEL_COL]
        .fill_null(-1)
        .to_numpy()
        .astype(np.int64)
    )

    # `ds` is now guaranteed to be Date.
    dates = j["ds"].to_list()

    return vals, labels, dates, present


def make_windows(
    labelled: pl.DataFrame,
    features: list[str] | None = None,
    window_len: int | None = None,
    stride: int | None = None,
    positive_stride: int | None = None,
    min_valid_frac: float | None = None,
) -> WindowSet:
    """
    Build fixed-length windows from labelled daily rows.

    `features` defaults to the common-16 set, per the locked feature
    policy: only those columns are populated by every vendor, so only
    they are usable in every LOMO fold.

    Rows whose label is undefined after reindexing can appear inside
    a window as input values, but a window is never emitted for an
    undefined end row.
    """

    # ------------------------------------------------------------
    # DEFAULT CONFIGURATION
    # ------------------------------------------------------------

    features = (
        list(features)
        if features is not None
        else list(COMMON_COLS)
    )

    window_len = (
        window_len
        if window_len is not None
        else CFG.window_len
    )

    stride = (
        stride
        if stride is not None
        else CFG.stride
    )

    min_valid_frac = (
        min_valid_frac
        if min_valid_frac is not None
        else CFG.min_valid_frac
    )

    # No positive oversampling by default.
    positive_stride = (
        positive_stride
        if positive_stride is not None
        else stride
    )

    # ------------------------------------------------------------
    # CHECK FEATURES
    # ------------------------------------------------------------

    missing = [
        c
        for c in features
        if c not in labelled.columns
    ]

    if missing:
        raise ValueError(
            f"features not in frame: {missing[:5]}"
        )

    # ------------------------------------------------------------
    # OUTPUT ACCUMULATORS
    # ------------------------------------------------------------

    Xs = []
    ys = []
    didx = []
    ends = []
    vends = []

    drives: list[tuple] = []

    # ------------------------------------------------------------
    # PROCESS EACH DRIVE INDEPENDENTLY
    # ------------------------------------------------------------

    for key, g in (
        labelled
        .sort(DRIVE_KEY + ["ds"])
        .group_by(
            DRIVE_KEY,
            maintain_order=True,
        )
    ):

        g = g.sort("ds")

        vendor = g[VENDOR_COL][0]

        vals, labels, dates, present = _reindex_drive(
            g,
            features,
        )

        n = vals.shape[0]

        # Drive must have at least one complete window.
        if n < window_len:
            continue

        # Index of this drive in the drives list.
        di = len(drives)

        drives.append(tuple(key))

        emitted = False

        # --------------------------------------------------------
        # CANDIDATE WINDOW END POSITIONS
        # --------------------------------------------------------

        for end in range(
            window_len - 1,
            n,
        ):

            lab = labels[end]

            # End row has no observed label.
            if lab < 0:
                continue

            # ----------------------------------------------------
            # STRIDE
            #
            # Normal windows use `stride`.
            # Positive windows use `positive_stride`.
            #
            # Default positive_stride == stride, so there is
            # NO positive oversampling.
            # ----------------------------------------------------

            step = (
                positive_stride
                if lab == 1
                else stride
            )

            # Phase stride from the first possible end position.
            if (
                end - (window_len - 1)
            ) % step != 0:
                continue

            # ----------------------------------------------------
            # WINDOW BOUNDARIES
            # ----------------------------------------------------

            s = end - window_len + 1

            win_present = present[
                s:end + 1
            ]

            # ----------------------------------------------------
            # MINIMUM ORIGINAL DATA VALIDITY
            #
            # This uses the ORIGINAL presence mask, BEFORE filling.
            # ----------------------------------------------------

            if (
                win_present.mean()
                < min_valid_frac
            ):
                continue

            # ----------------------------------------------------
            # SAVE WINDOW
            # ----------------------------------------------------

            Xs.append(
                vals[
                    s:end + 1
                ]
            )

            ys.append(lab)

            didx.append(di)

            ends.append(
                dates[end]
            )

            vends.append(vendor)

            emitted = True

        # --------------------------------------------------------
        # If this drive produced no valid window, remove it from
        # provenance.
        # --------------------------------------------------------

        if not emitted:
            drives.pop()

    # ------------------------------------------------------------
    # EMPTY WINDOW SET
    # ------------------------------------------------------------

    if not Xs:

        return WindowSet(
            X=np.zeros(
                (
                    0,
                    window_len,
                    len(features),
                ),
                dtype=np.float32,
            ),
            y=np.zeros(
                0,
                dtype=np.int8,
            ),
            drive_idx=np.zeros(
                0,
                dtype=int,
            ),
            end_ds=np.zeros(
                0,
                dtype=object,
            ),
            drives=[],
            vendors=np.zeros(
                0,
                dtype=object,
            ),
            features=features,
            window_len=window_len,
            stride=stride,
            positive_stride=positive_stride,
        )

    # ------------------------------------------------------------
    # FINAL WINDOW SET
    # ------------------------------------------------------------

    return WindowSet(
        X=np.stack(
            Xs
        ).astype(
            np.float32
        ),

        y=np.asarray(
            ys,
            dtype=np.int8,
        ),

        drive_idx=np.asarray(
            didx,
            dtype=int,
        ),

        end_ds=np.asarray(
            ends,
            dtype=object,
        ),

        drives=drives,

        vendors=np.asarray(
            vends,
            dtype=object,
        ),

        features=features,

        window_len=window_len,

        stride=stride,

        positive_stride=positive_stride,
    )


# ---------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------

@dataclass
class Normalizer:
    """
    Feature scaling, fit on training windows only.

    Fitting on pooled data leaks test information into both the model
    and the conformal calibration scores.

    TWO METHODS

    "zscore" (default)
        Subtract the training mean, divide by the training standard
        deviation. Correct when calibration and test come from the same
        population.

    "rank"
        Map each value to its percentile WITHIN THE GROUP IT BELONGS TO
        (vendor, by default), then centre on 0.5. Fit stores the
        training quantile grid per feature per group; unseen groups fall
        back to the pooled training grid.

    WHY "rank" EXISTS

    Measured on the real fleet, holding out vendor A:

        feature   train_mean   test_mean   test_std   out_of_range
        r_9         0.08        3.49        1.16        0.268
        n_183      -0.99       -2.04        2.20        0.186
        r_12       -0.01        0.64       15.74        0.002
        r_199       0.00        1.05       44.06        0.001

    r_9 is power-on-hours. Vendor A's drives sit 3.5 training standard
    deviations older, with 27% outside the training range entirely.
    r_12 and r_199 are raw counters whose spread differs by more than an
    order of magnitude between vendors.

    Under z-scoring, a Random Forest trained on B+C returns the nearest
    leaf for out-of-range inputs, so vendor A's scores shift wholesale:
    calibration scores clustered at median 0.049 while vendor A's
    clustered at 0.195. The conformal threshold calibrated on B+C
    (qhat = 0.181) then rejected BOTH labels for most vendor-A drives,
    producing 84% empty prediction sets -- a mechanical failure that
    looks like a distribution-shift result but is not one.

    Rank normalisation asks "is this drive's power-on-hours high FOR ITS
    VENDOR", which is the question an operator onboarding a new
    manufacturer would actually ask. It also removes the counter-scaling
    problem, since percentiles are scale-free.

    This is a preprocessing choice that must be stated in the paper,
    not hidden. Report results under both methods.
    """

    mean: np.ndarray
    std: np.ndarray
    features: list[str]
    method: str = "zscore"
    # rank mode: group -> (n_features, n_grid) sorted training values
    grids: dict | None = None
    pooled_grid: np.ndarray | None = None
    n_grid: int = 1000

    # -----------------------------------------------------------
    @classmethod
    def fit(cls, ws: WindowSet, method: str = "zscore",
            groups: np.ndarray | None = None, eps: float = 1e-8,
            n_grid: int = 1000) -> "Normalizer":
        """
        `groups` defaults to ws.vendors for method="rank". Pass an
        explicit array to group by something else (drive model, say).
        """
        if method not in ("zscore", "rank"):
            raise ValueError(f"method must be zscore|rank, got {method!r}")

        flat = ws.X.reshape(-1, ws.X.shape[-1])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < eps] = 1.0

        grids, pooled = None, None
        if method == "rank":
            g = groups if groups is not None else ws.vendors
            # one label per window, expanded to every timestep
            g_flat = np.repeat(np.asarray(g), ws.X.shape[1])
            qs = np.linspace(0.0, 1.0, n_grid)

            pooled = np.quantile(flat, qs, axis=0).T          # (F, n_grid)
            grids = {}
            for key in np.unique(g_flat):
                sub = flat[g_flat == key]
                if len(sub) < 2:
                    grids[key] = pooled
                else:
                    grids[key] = np.quantile(sub, qs, axis=0).T

        return cls(mean=mean, std=std, features=list(ws.features),
                   method=method, grids=grids, pooled_grid=pooled,
                   n_grid=n_grid)

    # -----------------------------------------------------------
    def transform(self, ws: WindowSet,
                  groups: np.ndarray | None = None) -> WindowSet:
        if list(ws.features) != list(self.features):
            raise ValueError(
                "feature mismatch between normalizer and window set"
            )

        if self.method == "zscore":
            out = (ws.X - self.mean) / self.std
        else:
            out = self._rank_transform(ws, groups)

        return WindowSet(
            X=out.astype(np.float32),
            y=ws.y,
            drive_idx=ws.drive_idx,
            end_ds=ws.end_ds,
            drives=ws.drives,
            vendors=ws.vendors,
            features=ws.features,
            window_len=ws.window_len,
            stride=ws.stride,
            positive_stride=ws.positive_stride,
        )

    def _rank_transform(self, ws, groups):
        g = groups if groups is not None else ws.vendors
        g = np.asarray(g)
        n, w, f = ws.X.shape
        out = np.empty_like(ws.X, dtype=np.float64)

        for key in np.unique(g):
            idx = np.flatnonzero(g == key)
            # An unseen group has no training grid -- fall back to the
            # pooled one. Under LOMO the held-out vendor is always
            # unseen, so this is the normal path for the test set and
            # is exactly the deployment situation: a new manufacturer
            # arrives and you have no history for it.
            grid = self.grids.get(key, self.pooled_grid)
            block = ws.X[idx].reshape(-1, f)
            ranked = np.empty_like(block, dtype=np.float64)
            for j in range(f):
                pos = np.searchsorted(grid[j], block[:, j], side="left")
                ranked[:, j] = pos / max(len(grid[j]) - 1, 1)
            out[idx] = ranked.reshape(len(idx), w, f)

        return out - 0.5          # centre on zero


def constant_features(ws: WindowSet, eps: float = 1e-6) -> list[str]:
    """
    Features with (near) zero variance in this window set.

    On the real fleet, holding out vendor A, SEVEN of the sixteen shared
    SMART columns are constant for vendor A: n_9, n_12, n_184, n_187,
    n_197, n_199 and r_187, with r_184 near-constant at std 0.02.

    A feature the model split on during training but which never varies
    at test time contributes nothing but the leaf value it happened to
    land in. Dropping them is both a modelling fix and a reportable
    finding about how little the "shared" attributes actually share.
    """
    flat = ws.X.reshape(-1, ws.X.shape[-1])
    std = flat.std(axis=0)
    return [f for f, s in zip(ws.features, std) if s < eps]


def drop_features(ws: WindowSet, drop: list[str]) -> WindowSet:
    """Return a copy of `ws` without the named features."""
    keep = [f for f in ws.features if f not in set(drop)]
    idx = [ws.features.index(f) for f in keep]
    return WindowSet(
        X=ws.X[:, :, idx], y=ws.y, drive_idx=ws.drive_idx,
        end_ds=ws.end_ds, drives=ws.drives, vendors=ws.vendors,
        features=keep, window_len=ws.window_len, stride=ws.stride,
        positive_stride=ws.positive_stride,
    )


# ---------------------------------------------------------------
# Flattening, for the Random Forest baseline
# ---------------------------------------------------------------

def flatten(
    ws: WindowSet,
    how: str = "last",
) -> tuple[np.ndarray, list[str]]:
    """
    Collapse the time axis for models that take tabular input.

    "last"  -- the final timestep only, closest to WEFR's per-day rows
    "stats" -- mean, std, min, max, last and first-to-last delta
    "full"  -- concatenate every timestep
    """

    n, w, f = ws.X.shape

    # ------------------------------------------------------------
    # LAST TIMESTEP
    # ------------------------------------------------------------

    if how == "last":

        return (
            ws.X[:, -1, :],
            list(ws.features),
        )

    # ------------------------------------------------------------
    # FULL WINDOW
    # ------------------------------------------------------------

    if how == "full":

        names = [
            f"{c}_t{t}"
            for t in range(w)
            for c in ws.features
        ]

        return (
            ws.X.reshape(
                n,
                w * f,
            ),
            names,
        )

    # ------------------------------------------------------------
    # SUMMARY STATISTICS
    # ------------------------------------------------------------

    if how == "stats":

        parts = [
            ws.X.mean(1),
            ws.X.std(1),
            ws.X.min(1),
            ws.X.max(1),
            ws.X[:, -1, :],
            (
                ws.X[:, -1, :]
                - ws.X[:, 0, :]
            ),
        ]

        names = [
            f"{c}_{s}"
            for s in (
                "mean",
                "std",
                "min",
                "max",
                "last",
                "delta",
            )
            for c in ws.features
        ]

        return (
            np.concatenate(
                parts,
                axis=1,
            ),
            names,
        )

    raise ValueError(
        f"unknown how={how!r}"
    )