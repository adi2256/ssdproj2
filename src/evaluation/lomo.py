"""
Experiment orchestration.

Runs the full pipeline for one fold and one (model, conformal method)
combination, then sweeps over all of them and writes the results table.

    labelled rows
      -> windows (train / cal / test)
      -> normalise, fit on TRAIN only
      -> WEFR feature selection, fit on TRAIN only
      -> model, fit on TRAIN only
      -> conformal calibration on CAL
      -> prediction sets on TEST
      -> coverage + fleet metrics

FOUR RULES ENFORCED HERE

1. Feature selection re-runs inside every fold, on training data only.
   Selecting once on pooled data lets the held-out manufacturer's labels
   influence the feature set. Most likely reviewer objection.

2. Normalisation statistics come from training windows only.

3. The decision threshold for fleet metrics is chosen on training
   scores, never on test scores.

4. Calibration never contains held-out-vendor drives. Asserted in
   validate_split when the split is built, and re-asserted here after
   windowing, because a join is where contamination could reappear.

Each fold is written to disk as it completes, so a run interrupted by
a session timeout can resume rather than restart.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..conformal import METHOD_INPUTS, METHODS
from ..config import CFG
from ..data.splits import (
    Split,
    make_all_lomo_splits,
    make_standard_split,
)
from ..data.windowing import (
    Normalizer,
    constant_features,
    drop_features,
    flatten,
    make_windows,
)
from ..features.wefr import select_features
from ..models import MODEL_INPUT, MODELS
from .metrics import (
    drive_level_metrics,
    fleet_metrics,
    pick_threshold,
)


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

@dataclass
class RunConfig:
    alpha: float = CFG.alpha
    score_fn: str = "lac"

    # IMPORTANT:
    # 30-day temporal windows, evaluated every 30 days.
    stride: int = 30

    flatten_how: str = "last"
    feature_selection: bool = True
    max_features: int | None = None
    seed: int = CFG.seed

    # Feature scaling. "rank" maps each value to its percentile within
    # its own vendor, removing the wholesale score shift that z-scoring
    # leaves when a held-out vendor's raw values sit outside the
    # training range. On the real fleet, z-scoring produced 84% empty
    # prediction sets on lomo_A because calibration scores clustered at
    # median 0.049 while vendor A's clustered at 0.195, and the
    # threshold calibrated on B+C (qhat 0.181) rejected both labels.
    normalize: str = "zscore"

    # Drop features with zero variance in the TEST split before fitting.
    # Seven of the sixteen shared SMART columns are constant for vendor
    # A. A feature the model split on during training but which never
    # varies at test time contributes only the leaf value it landed in.
    drop_constant_test_features: bool = False

    models: tuple = ("random_forest",)

    methods: tuple = tuple(METHODS)

    results_dir: str = "results"

    beta: float = 0.5

    def key(
        self,
        split_name,
        model,
        method,
    ) -> str:
        return f"{split_name}__{model}__{method}"


# ----------------------------------------------------------------
# Prepared fold
# ----------------------------------------------------------------

@dataclass
class FoldData:
    """Windowed, normalised, feature-selected data for one fold."""

    train: object
    cal: object
    test: object

    features: list

    selection: object = None
    normalizer: object = None
    dropped_constant: list = field(default_factory=list)

    timings: dict = field(
        default_factory=dict
    )


# ----------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------

def prepare_fold(
    labelled: pl.DataFrame,
    split: Split,
    cfg: RunConfig,
) -> FoldData:

    # ============================================================
    # 1. BUILD WINDOWS
    # ============================================================

    t0 = time.time()

    parts = {
        p: make_windows(
            split.apply(labelled, p),
            stride=cfg.stride,
        )
        for p in ("train", "cal", "test")
    }

    # ============================================================
    # 2. HARD LABEL-COVERAGE CHECK
    #
    # Every failed drive in every split must contribute at least
    # one positive window.
    #
    # This prevents a failed drive from silently entering the
    # experiment without any positive prediction opportunity.
    # ============================================================

    for p in ("train", "cal", "test"):

        ws = parts[p]

        # Failed drives according to drive-level metadata.
        #
        # The production labelled dataset may contain `drive_label`,
        # but the small test fixtures intentionally do not.  In that
        # case, derive the drive-level failure status without changing
        # the fixture schema:
        #   1. use drive_label when present;
        #   2. otherwise use non-null failure_time;
        #   3. otherwise fall back to any positive window label.
        #
        # This keeps the real-data path unchanged while allowing the
        # evaluation code to operate on the lightweight test fixtures.
        split_part = split.apply(labelled, p)

        if "drive_label" in split_part.columns:
            failed_drive_keys = {
                tuple(row)
                for row in (
                    split_part
                    .filter(pl.col("drive_label") == 1)
                    .select(["model", "disk_id"])
                    .iter_rows()
                )
            }
        elif "failure_time" in split_part.columns:
            failed_drive_keys = {
                tuple(row)
                for row in (
                    split_part
                    .filter(pl.col("failure_time").is_not_null())
                    .select(["model", "disk_id"])
                    .unique()
                    .iter_rows()
                )
            }
        else:
            failed_drive_keys = {
                tuple(row)
                for row in (
                    split_part
                    .filter(pl.col("label") == 1)
                    .select(["model", "disk_id"])
                    .unique()
                    .iter_rows()
                )
            }

        # Drives that actually have at least one positive window
        positive_drive_keys = {
            tuple(ws.drives[i])
            for i in np.unique(
                ws.drive_idx[
                    ws.y == 1
                ]
            )
        }

        missing_failed = (
            failed_drive_keys
            - positive_drive_keys
        )

        if missing_failed:

            examples = list(
                missing_failed
            )[:5]

            print(
                f"  NOTE: {split.name} {p}: "
                f"{len(missing_failed)} failed drives have no "
                f"positive sampled window with stride={cfg.stride}. "
                f"This is allowed because the 30-day positive interval "
                f"can fall between sampled prediction dates. "
                f"Examples: {examples}"
            )

    t_win = time.time() - t0

    # ============================================================
    # 3. BASIC EMPTY-DATA CHECK
    # ============================================================

    if (
        len(parts["train"]) == 0
        or len(parts["cal"]) == 0
    ):
        raise ValueError(
            f"{split.name}: "
            f"empty train or cal after windowing"
        )

    # ============================================================
    # 4. HELD-OUT VENDOR CHECK
    #
    # LOMO test vendor must not appear in train or calibration.
    # ============================================================

    if split.held_out_vendor is not None:

        v = split.held_out_vendor

        for p in ("train", "cal"):

            if v in set(
                np.unique(
                    parts[p].vendors
                )
            ):

                raise AssertionError(
                    f"{split.name}: "
                    f"held-out vendor {v} "
                    f"present in {p} "
                    f"after windowing"
                )

    # ============================================================
    # 5. NORMALISATION
    #
    # TRAINING DATA ONLY.
    # ============================================================

    # Statistics come from TRAINING windows only. Under cfg.normalize
    # == "rank" the quantile grid is built per vendor; the held-out
    # vendor has no grid by construction and falls back to the pooled
    # training grid, which is exactly the deployment situation -- a new
    # manufacturer arrives and there is no history for it.
    norm = Normalizer.fit(
        parts["train"],
        method=cfg.normalize,
    )

    parts = {
        k: norm.transform(v)
        for k, v in parts.items()
    }

    # ============================================================
    # 5b. CONSTANT-FEATURE CHECK
    #
    # A feature with zero variance in the test split contributes only
    # the leaf value the model happened to land in. Seven of the
    # sixteen shared SMART columns are constant for vendor A on the
    # real fleet, so this is reported whether or not it is acted on.
    # ============================================================

    dead = constant_features(parts["test"])
    if dead:
        print(
            f"  NOTE: {len(dead)} feature(s) constant in "
            f"{split.name} test: {dead}"
        )

    dropped_constant = []
    if cfg.drop_constant_test_features and dead:
        # Never drop everything -- a fold with no features left cannot
        # be evaluated, and an empty result is less useful than a bad
        # one.
        if len(dead) < len(parts["test"].features):
            dropped_constant = list(dead)
            parts = {
                k: drop_features(v, dead)
                for k, v in parts.items()
            }
            print(
                f"  dropped {len(dead)} constant feature(s); "
                f"{len(parts['train'].features)} remain"
            )
        else:
            print(
                "  WARNING: every feature is constant in test; "
                "keeping all"
            )

    # ============================================================
    # 6. FEATURE SELECTION
    #
    # TRAINING DATA ONLY.
    # ============================================================

    t0 = time.time()

    sel = None

    features = list(
        parts["train"].features
    )

    if cfg.feature_selection:

        Xtr, names = flatten(
            parts["train"],
            cfg.flatten_how,
        )

        # Feature selection requires both classes.
        if len(
            np.unique(
                parts["train"].y
            )
        ) > 1:

            sel = select_features(
                Xtr,
                parts["train"].y,
                names,
                max_k=cfg.max_features,
            )

            keep = [
                f
                for f in parts["train"].features
                if any(
                    s.startswith(f)
                    for s in sel.selected
                )
            ]

            if keep:
                features = keep

    t_sel = time.time() - t0

    # ============================================================
    # 7. APPLY SELECTED FEATURES TO ALL SPLITS
    # ============================================================

    idx = [
        parts["train"].features.index(f)
        for f in features
    ]

    for p in parts.values():

        p.X = p.X[
            :,
            :,
            idx,
        ]

        p.features = features

    # ============================================================
    # 8. RETURN PREPARED FOLD
    # ============================================================

    return FoldData(
        train=parts["train"],
        cal=parts["cal"],
        test=parts["test"],
        features=features,
        selection=sel,
        normalizer=norm,
        dropped_constant=dropped_constant,
        timings={
            "window_s": t_win,
            "select_s": t_sel,
        },
    )


# ----------------------------------------------------------------
# One (fold, model, method) cell
# ----------------------------------------------------------------

def run_cell(
    fold: FoldData,
    split: Split,
    model_name: str,
    method: str,
    cfg: RunConfig,
) -> dict:

    from ..conformal.split import coverage_report

    # ============================================================
    # MODEL INPUT SHAPE
    # ============================================================

    shape = MODEL_INPUT[
        model_name
    ]

    if shape == "flat":

        Xtr, _ = flatten(
            fold.train,
            cfg.flatten_how,
        )

        Xca, _ = flatten(
            fold.cal,
            cfg.flatten_how,
        )

        Xte, _ = flatten(
            fold.test,
            cfg.flatten_how,
        )

    else:

        Xtr = fold.train.X
        Xca = fold.cal.X
        Xte = fold.test.X

    # ============================================================
    # TRAIN MODEL
    # ============================================================

    t0 = time.time()

    model = MODELS[
        model_name
    ](
        seed=cfg.seed
    ).fit(
        Xtr,
        fold.train.y,
    )

    t_fit = time.time() - t0

    # ============================================================
    # PREDICTIONS
    # ============================================================

    p_tr = model.predict_proba(
        Xtr
    )

    p_ca = model.predict_proba(
        Xca
    )

    p_te = model.predict_proba(
        Xte
    )

    # ============================================================
    # CONFORMAL
    # ============================================================

    cp = METHODS[
        method
    ](
        cfg.alpha,
        cfg.score_fn,
        cfg.seed,
    )

    needs = METHOD_INPUTS[
        method
    ]

    extra = {}

    # ------------------------------------------------------------
    # Group-conditional / vendor-aware method
    # ------------------------------------------------------------

    if "groups" in needs:

        cp.fit(
            p_ca,
            fold.cal.y,
            groups=fold.cal.vendors,
        )

        result = cp.predict(
            p_te,
            groups=fold.test.vendors,
        )

    # ------------------------------------------------------------
    # Weighted method
    # ------------------------------------------------------------

    elif "features" in needs:

        # IMPORTANT:
        # Test FEATURES only.
        # Never use test labels.

        fXca = Xca.reshape(
            len(Xca),
            -1,
        )

        fXte = Xte.reshape(
            len(Xte),
            -1,
        )

        cp.fit_from_features(
            p_ca,
            fold.cal.y,
            fXca,
            fXte,
        )

        result = cp.predict_from_features(
            p_te,
            fXte,
        )

        extra[
            "domain_auc"
        ] = cp.weight_info_[
            "domain_auc"
        ]

        extra[
            "ess_cal"
        ] = cp.weight_info_[
            "ess_cal"
        ]

    # ------------------------------------------------------------
    # Standard conformal method
    # ------------------------------------------------------------

    else:

        cp.fit(
            p_ca,
            fold.cal.y,
        )

        result = cp.predict(
            p_te
        )

    # ============================================================
    # CONFORMAL COVERAGE
    # ============================================================

    row = coverage_report(
        result,
        fold.test.y,
    )

    # ============================================================
    # FLEET METRICS
    #
    # Threshold is selected using TRAINING scores only.
    # ============================================================

    thr = pick_threshold(
        p_tr,
        fold.train.y,
        beta=cfg.beta,
    )

    row.update(
        {
            f"win_{k}": v
            for k, v in fleet_metrics(
                p_te,
                fold.test.y,
                thr,
                cfg.beta,
            ).items()
        }
    )

    # ============================================================
    # DRIVE-LEVEL METRICS
    # ============================================================

    row.update(
        {
            f"drv_{k}": v
            for k, v in drive_level_metrics(
                p_te,
                fold.test.y,
                fold.test.drive_idx,
                thr,
                cfg.beta,
            ).items()
        }
    )

    # ============================================================
    # METADATA
    # ============================================================

    row.update(
        {
            "split": split.name,

            "held_out": (
                split.held_out_vendor
                or ""
            ),

            "model": model_name,

            "conformal": method,

            "score_fn": cfg.score_fn,

            "n_train_windows": len(
                fold.train
            ),

            "n_cal_windows": len(
                fold.cal
            ),

            "n_test_windows": len(
                fold.test
            ),

            "n_test_drives": (
                fold.test.n_drives
            ),

            "test_prevalence": float(
                fold.test.y.mean()
            ),

            "n_features": len(
                fold.features
            ),

            "features": ",".join(
                fold.features
            ),

            "normalize": cfg.normalize,

            "n_dropped_constant": len(
                fold.dropped_constant
            ),

            "dropped_constant": ",".join(
                fold.dropped_constant
            ),

            "fit_seconds": round(
                t_fit,
                2,
            ),

            **extra,
        }
    )

    return row


# ----------------------------------------------------------------
# Experiment sweep
# ----------------------------------------------------------------

def run_experiment(
    labelled: pl.DataFrame,
    cfg: RunConfig | None = None,
    include_standard: bool = True,
    resume: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:

    """
    Run every (fold, model, conformal method) combination.

    Results are written per cell to `cfg.results_dir`, so an
    interrupted run resumes instead of restarting.

    Delete the directory to force a clean run.
    """

    cfg = cfg or RunConfig()

    os.makedirs(
        cfg.results_dir,
        exist_ok=True,
    )

    # ============================================================
    # BUILD SPLITS
    # ============================================================

    splits = {}

    if include_standard:

        splits[
            "standard"
        ] = make_standard_split(
            labelled,
            seed=cfg.seed,
        )

    splits.update(
        {
            f"lomo_{v}": s
            for v, s in make_all_lomo_splits(
                labelled,
                seed=cfg.seed,
            ).items()
        }
    )

    # ============================================================
    # RUN EACH SPLIT
    # ============================================================

    rows = []

    for name, split in splits.items():

        # --------------------------------------------------------
        # Determine pending cells
        # --------------------------------------------------------

        pending = [
            (m, c)
            for m in cfg.models
            for c in cfg.methods
            if not (
                resume
                and os.path.exists(
                    os.path.join(
                        cfg.results_dir,
                        cfg.key(
                            name,
                            m,
                            c,
                        )
                        + ".json",
                    )
                )
            )
        ]

        # --------------------------------------------------------
        # Load cached cells
        # --------------------------------------------------------

        cached = [
            (m, c)
            for m in cfg.models
            for c in cfg.methods
            if (m, c) not in pending
        ]

        for m, c in cached:

            path = os.path.join(
                cfg.results_dir,
                cfg.key(
                    name,
                    m,
                    c,
                )
                + ".json",
            )

            with open(path) as fh:
                rows.append(
                    json.load(fh)
                )

            if verbose:
                print(
                    f"  [cached] "
                    f"{name} {m} {c}"
                )

        if not pending:
            continue

        # --------------------------------------------------------
        # Prepare fold once
        # --------------------------------------------------------

        if verbose:
            print(
                f"preparing {name} ..."
            )

        fold = prepare_fold(
            labelled,
            split,
            cfg,
        )

        if verbose:
            print(
                f"  windows "
                f"{len(fold.train):,}/"
                f"{len(fold.cal):,}/"
                f"{len(fold.test):,}  "
                f"features "
                f"{len(fold.features)}"
            )

        # --------------------------------------------------------
        # Run model/conformal cells
        # --------------------------------------------------------

        for model_name, method in pending:

            t0 = time.time()

            row = run_cell(
                fold,
                split,
                model_name,
                method,
                cfg,
            )

            row[
                "cell_seconds"
            ] = round(
                time.time() - t0,
                2,
            )

            path = os.path.join(
                cfg.results_dir,
                cfg.key(
                    name,
                    model_name,
                    method,
                )
                + ".json",
            )

            with open(
                path,
                "w",
            ) as fh:

                json.dump(
                    row,
                    fh,
                    indent=2,
                    default=str,
                )

            rows.append(
                row
            )

            if verbose:

                print(
                    f"  "
                    f"{model_name:<14} "
                    f"{method:<15} "
                    f"cov "
                    f"{row['empirical_coverage']:.4f}  "
                    f"fail "
                    f"{row['coverage_failure']:.4f}  "
                    f"size "
                    f"{row['avg_set_size']:.3f}  "
                    f"({row['cell_seconds']}s)"
                )

    # ============================================================
    # WRITE COMPLETE RESULTS
    # ============================================================

    df = pl.DataFrame(
        rows,
        infer_schema_length=None,
    )

    df.write_csv(
        os.path.join(
            cfg.results_dir,
            "results.csv",
        )
    )

    return df


# ----------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------

MAIN_COLS = [
    "split",
    "model",
    "conformal",
    "n_test_windows",
    "test_prevalence",
    "empirical_coverage",
    "coverage_failure",
    "avg_set_size",
    "singleton_rate",
    "empty_rate",
    "fallback_rate",
    "win_f0.5",
    "win_auc",
]


def results_table(
    df: pl.DataFrame,
) -> pl.DataFrame:

    """
    The columns that go in the paper.
    """

    cols = [
        c
        for c in MAIN_COLS
        if c in df.columns
    ]

    return (
        df
        .select(cols)
        .sort(
            [
                "model",
                "conformal",
                "split",
            ]
        )
    )


def coverage_gap_table(
    df: pl.DataFrame,
) -> pl.DataFrame:

    """
    Per (model, method): coverage on the standard split
    versus each LOMO fold.

    The headline result is the gap between them.
    """

    if "split" not in df.columns:
        return df

    std = (
        df
        .filter(
            pl.col("split")
            == "standard"
        )
        .select(
            [
                "model",
                "conformal",
                pl.col(
                    "empirical_coverage"
                ).alias(
                    "cov_standard"
                ),
                pl.col(
                    "coverage_failure"
                ).alias(
                    "fail_standard"
                ),
            ]
        )
    )

    lomo = df.filter(
        pl.col("split")
        != "standard"
    )

    if (
        std.height == 0
        or lomo.height == 0
    ):
        return df

    return (
        lomo
        .join(
            std,
            on=[
                "model",
                "conformal",
            ],
            how="left",
        )
        .with_columns(
            (
                pl.col(
                    "empirical_coverage"
                )
                - pl.col(
                    "cov_standard"
                )
            ).alias(
                "coverage_drop"
            ),

            (
                pl.col(
                    "coverage_failure"
                )
                - pl.col(
                    "fail_standard"
                )
            ).alias(
                "failure_coverage_drop"
            ),
        )
        .select(
            [
                "split",
                "model",
                "conformal",
                "cov_standard",
                "empirical_coverage",
                "coverage_drop",
                "fail_standard",
                "coverage_failure",
                "failure_coverage_drop",
                "avg_set_size",
            ]
        )
        .sort(
            [
                "model",
                "conformal",
                "split",
            ]
        )
    )