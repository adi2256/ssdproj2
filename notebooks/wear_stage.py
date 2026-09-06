"""
Wear-stage conditional validity.

THE QUESTION

The base paper found that the SMART attributes informative for failure
change as drives age, and built wear-out updating into its feature
selection. Nobody has asked the follow-on question of the guarantee:

    A conformal predictor calibrated on the whole fleet promises 90%
    coverage on average. Does it deliver 90% for YOUNG drives and for
    OLD drives separately, or is the average hiding a stage where it
    quietly fails?

The bathtub curve says failure rate differs sharply by age. That is
label shift across age strata, within a vendor, with consistent
encoding -- so the model works (within-fleet AUC 0.73) and any coverage
gap is a real finding rather than a broken predictor.

FOUR CONDITIONS, one model, same probabilities:

    split           fleet-wide calibration, the baseline
    mondrian_class  per-class -- already known to fix failure coverage
    mondrian_stage  per-wear-stage -- the conformal analogue of WEFR's
                    wear-out updating
    mondrian_both   per (stage, class) -- the full guarantee

Each reported PER STAGE, with set size, because a guarantee met by
widening sets to the maximum carries no information.

Then LEAVE-ONE-STAGE-OUT: calibrate on four stages, test on the fifth.
This is the deployment question -- a fleet that has never had drives
this old has no calibration data for them -- and it is LOMO on an axis
where the model actually transfers.

RUNTIME

Windows once per split, then every condition reuses the same
probabilities. Roughly 10 minutes for the standard split and 20 for
leave-one-stage-out.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from src.conformal.mondrian import MondrianConformal, group_conditional_coverage
from src.conformal.split import SplitConformal, coverage_report
from src.data.splits import Split, make_standard_split
from src.data.windowing import (Normalizer, constant_features,
                                drop_features, flatten, make_windows)
from src.features.wear import fit_stages, stage_table, window_age

STRIDE = 30
MAX_TRAIN = 200_000
N_TREES = 100
SEED = 42
ALPHA = 0.10

NORMALIZE = "per_drive"
FLATTEN = "dynamics"
N_STAGES = 5
STAGE_STRATEGY = "quantile"      # or "changepoint"


def _subset(ws, idx):
    from src.data.windowing import WindowSet
    return WindowSet(
        X=ws.X[idx], y=ws.y[idx], drive_idx=ws.drive_idx[idx],
        end_ds=ws.end_ds[idx], drives=ws.drives, vendors=ws.vendors[idx],
        features=ws.features, window_len=ws.window_len,
        stride=ws.stride, positive_stride=ws.positive_stride,
    )


def window_parts(labelled, split, parts=("train", "cal", "test")):
    out = {}
    for p in parts:
        ws = make_windows(split.apply(labelled, p), stride=STRIDE)
        if p == "train" and len(ws) > MAX_TRAIN:
            rng = np.random.default_rng(SEED)
            pos = np.flatnonzero(ws.y == 1)
            neg = np.flatnonzero(ws.y == 0)
            keep = np.sort(np.concatenate(
                [pos, rng.choice(neg, size=min(MAX_TRAIN - len(pos), len(neg)),
                                 replace=False)]))
            ws = _subset(ws, keep)
        out[p] = ws
    return out


def fit_and_score(raw):
    """
    Model on normalised features; STAGES on raw age. Returns
    probabilities and stage assignments for cal and test.
    """
    from sklearn.ensemble import RandomForestClassifier

    # Stage boundaries from TRAINING ages only.
    smap = fit_stages(raw["train"], strategy=STAGE_STRATEGY,
                      n_stages=N_STAGES)
    stages = {p: smap.assign(window_age(raw[p])) for p in raw}

    norm = Normalizer.fit(raw["train"], method=NORMALIZE)
    nm = {p: norm.transform(ws) for p, ws in raw.items()}
    dead = constant_features(nm["test"])
    if dead and len(dead) < len(nm["test"].features):
        nm = {p: drop_features(ws, dead) for p, ws in nm.items()}

    X = {p: np.nan_to_num(flatten(ws, FLATTEN)[0]) for p, ws in nm.items()}

    m = RandomForestClassifier(
        n_estimators=N_TREES, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    ).fit(X["train"], raw["train"].y)

    def proba(Xp):
        p = m.predict_proba(Xp)
        return p[:, list(m.classes_).index(1)] if p.shape[1] > 1 else p[:, 0]

    return {
        "p_cal": proba(X["cal"]), "y_cal": raw["cal"].y,
        "p_te": proba(X["test"]), "y_te": raw["test"].y,
        "st_cal": stages["cal"], "st_te": stages["test"],
        "smap": smap,
    }


def conformal_by_stage(S, alpha=ALPHA):
    """All four conditions on the same probabilities; per-stage report."""
    st_cal, st_te = S["st_cal"], S["st_te"]
    conds = {
        "split": SplitConformal(alpha=alpha, seed=SEED)
                 .fit(S["p_cal"], S["y_cal"]),
        "mondrian_class": MondrianConformal(alpha=alpha, by="class", seed=SEED)
                 .fit(S["p_cal"], S["y_cal"]),
        "mondrian_stage": MondrianConformal(alpha=alpha, by="group", seed=SEED)
                 .fit(S["p_cal"], S["y_cal"], groups=st_cal),
        "mondrian_both": MondrianConformal(alpha=alpha, by="both", seed=SEED)
                 .fit(S["p_cal"], S["y_cal"], groups=st_cal),
    }

    rows = []
    for name, cp in conds.items():
        if name in ("mondrian_stage", "mondrian_both"):
            r = cp.predict(S["p_te"], groups=st_te)
        else:
            r = cp.predict(S["p_te"])

        overall = coverage_report(r, S["y_te"])
        per = group_conditional_coverage(r, S["y_te"], st_te)
        for s, d in per.items():
            m = st_te == s
            rows.append({
                "method": name, "stage": int(s),
                "n": d["n"], "n_fail": d["n_failure"],
                "prevalence": round(d["n_failure"] / max(d["n"], 1), 5),
                "coverage": round(d["coverage"], 4),
                "cov_failure": round(d["coverage_failure"], 4),
                "cov_healthy": round(d["coverage_healthy"], 4),
                "set_size": round(float(r.set_sizes[m].mean()), 4),
                "singleton": round(float(r.is_singleton[m].mean()), 4),
                "fallback": round(float(r.fallback[m].mean()), 4)
                            if r.fallback is not None else 0.0,
            })
        rows.append({
            "method": name, "stage": -1, "n": overall["n_test"],
            "n_fail": overall["n_failures_test"],
            "prevalence": round(overall["n_failures_test"]
                                / max(overall["n_test"], 1), 5),
            "coverage": round(overall["empirical_coverage"], 4),
            "cov_failure": round(overall["coverage_failure"], 4),
            "cov_healthy": round(overall["coverage_healthy"], 4),
            "set_size": round(overall["avg_set_size"], 4),
            "singleton": round(overall["singleton_rate"], 4),
            "fallback": round(overall["fallback_rate"], 4),
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------
# Experiment 1: within-fleet, per-stage coverage
# ---------------------------------------------------------------

def run_within_fleet(labelled):
    print("=" * 64 + "\nWITHIN-FLEET, PER-STAGE COVERAGE\n" + "=" * 64)
    split = make_standard_split(labelled, seed=SEED)
    t0 = time.time()
    raw = window_parts(labelled, split)
    print(f"windows train {len(raw['train']):,} cal {len(raw['cal']):,} "
          f"test {len(raw['test']):,} ({time.time() - t0:.0f}s)")

    S = fit_and_score(raw)
    print(f"\nstage map ({S['smap'].strategy}): {S['smap'].labels()}")
    print("\nfailure rate by stage on TEST -- the bathtub, if present:")
    print(stage_table(S["st_te"], S["y_te"], S["smap"]))

    from src.evaluation.metrics import auc
    print(f"\nwithin-fleet AUC: {auc(S['p_te'], S['y_te']):.4f}")

    res = conformal_by_stage(S)
    res = res.with_columns(pl.lit("standard").alias("split"))

    print("\n=== coverage by stage (target 0.90) ===")
    print(res.filter(pl.col("stage") >= 0)
             .pivot(values="coverage", index="stage", on="method"))
    print("\n=== FAILURE-class coverage by stage ===")
    print(res.filter(pl.col("stage") >= 0)
             .pivot(values="cov_failure", index="stage", on="method"))
    print("\n=== set size by stage (max 2) ===")
    print(res.filter(pl.col("stage") >= 0)
             .pivot(values="set_size", index="stage", on="method"))

    # -- the headline number --------------------------------------
    sp = res.filter((pl.col("method") == "split") & (pl.col("stage") >= 0))
    spread = float(sp["coverage"].max() - sp["coverage"].min())
    worst = sp.sort("coverage").row(0, named=True)
    print(f"\nsplit conformal: coverage ranges "
          f"{sp['coverage'].min():.3f}-{sp['coverage'].max():.3f} "
          f"across stages (spread {spread:.3f}); worst stage "
          f"{worst['stage']} at prevalence {worst['prevalence']:.4f}")
    if spread > 0.05:
        print("  -> fleet-wide calibration is NOT uniform over wear stage.")
    else:
        print("  -> coverage is roughly uniform over stage. Check "
              "cov_failure -- that is where the gap usually hides.")
    return res, S


# ---------------------------------------------------------------
# Experiment 2: leave-one-stage-out
# ---------------------------------------------------------------

def make_stage_split(labelled, raw_all, stages_all, held_out: int,
                     cal_frac: float = 0.2):
    """
    Hold out one wear stage entirely: test = every window in that
    stage; train + cal = the rest, split at DRIVE level.

    Windows of one drive span stages as it ages, so a drive can appear
    in test (its old windows) and train (its young ones). That is the
    honest deployment situation -- the drive existed, its early life
    was seen, its late life has not been -- but it is not drive-disjoint,
    and the write-up must say so. The drive-disjoint variant is stricter
    and is left as a flag.
    """
    is_test = stages_all == held_out
    rest = np.flatnonzero(~is_test)
    rng = np.random.default_rng(SEED)
    drives_rest = np.unique(raw_all.drive_idx[rest])
    rng.shuffle(drives_rest)
    n_cal = int(round(cal_frac * len(drives_rest)))
    cal_drives = set(drives_rest[:n_cal].tolist())
    cal = np.array([i for i in rest if raw_all.drive_idx[i] in cal_drives])
    tr = np.array([i for i in rest if raw_all.drive_idx[i] not in cal_drives])
    te = np.flatnonzero(is_test)
    return {"train": _subset(raw_all, tr), "cal": _subset(raw_all, cal),
            "test": _subset(raw_all, te)}


def run_leave_one_stage_out(labelled):
    print("\n" + "=" * 64 + "\nLEAVE-ONE-STAGE-OUT\n" + "=" * 64)
    split = make_standard_split(labelled, seed=SEED)
    t0 = time.time()
    # Window the whole thing once, then carve by stage.
    ws_all = make_windows(
        labelled.join(pl.concat([split.train, split.cal, split.test])
                      .select(["model", "disk_id"]),
                      on=["model", "disk_id"], how="inner"),
        stride=STRIDE)
    print(f"windows {len(ws_all):,} ({time.time() - t0:.0f}s)")

    smap = fit_stages(ws_all, strategy=STAGE_STRATEGY, n_stages=N_STAGES)
    st_all = smap.assign(window_age(ws_all))
    print(f"stages: {smap.labels()}")

    from src.evaluation.metrics import auc
    rows = []
    for held in range(smap.n_stages):
        raw = make_stage_split(labelled, ws_all, st_all, held)
        if len(raw["test"]) == 0 or raw["test"].y.sum() < 20:
            print(f"  stage {held}: too few test positives, skipped")
            continue
        if len(raw["train"]) > MAX_TRAIN:
            rng = np.random.default_rng(SEED)
            pos = np.flatnonzero(raw["train"].y == 1)
            neg = np.flatnonzero(raw["train"].y == 0)
            keep = np.sort(np.concatenate(
                [pos, rng.choice(neg, size=min(MAX_TRAIN - len(pos), len(neg)),
                                 replace=False)]))
            raw["train"] = _subset(raw["train"], keep)

        S = fit_and_score(raw)
        # Under hold-out, mondrian_stage has no cell for the test stage
        # and falls back to pooled -- that fallback rate IS the result.
        S["st_cal"] = smap.assign(window_age(raw["cal"]))
        S["st_te"] = np.full(len(raw["test"]), held)

        a = auc(S["p_te"], S["y_te"])
        for name, cp in {
            "split": SplitConformal(alpha=ALPHA, seed=SEED)
                     .fit(S["p_cal"], S["y_cal"]),
            "mondrian_class": MondrianConformal(alpha=ALPHA, by="class",
                                                seed=SEED)
                     .fit(S["p_cal"], S["y_cal"]),
            "mondrian_stage": MondrianConformal(alpha=ALPHA, by="group",
                                                seed=SEED)
                     .fit(S["p_cal"], S["y_cal"], groups=S["st_cal"]),
        }.items():
            r = (cp.predict(S["p_te"], groups=S["st_te"])
                 if name == "mondrian_stage" else cp.predict(S["p_te"]))
            rep = coverage_report(r, S["y_te"])
            rows.append({
                "held_out_stage": held, "label": smap.labels()[held],
                "method": name, "auc": round(a, 4),
                "n_test": rep["n_test"], "n_fail": rep["n_failures_test"],
                "prevalence": round(rep["n_failures_test"]
                                    / max(rep["n_test"], 1), 5),
                "coverage": round(rep["empirical_coverage"], 4),
                "cov_failure": round(rep["coverage_failure"], 4),
                "set_size": round(rep["avg_set_size"], 4),
                "fallback": round(rep["fallback_rate"], 4),
            })
        last = [r for r in rows if r["held_out_stage"] == held]
        print(f"  stage {held} {smap.labels()[held]:<22} auc {a:.3f}  " +
              "  ".join(f"{r['method'][:8]} cov {r['coverage']:.3f}/"
                        f"fail {r['cov_failure']:.3f}" for r in last),
              flush=True)

    res = pl.DataFrame(rows)
    print("\n=== leave-one-stage-out: coverage ===")
    print(res.pivot(values="coverage", index="held_out_stage", on="method"))
    print("\n=== leave-one-stage-out: failure-class coverage ===")
    print(res.pivot(values="cov_failure", index="held_out_stage", on="method"))
    print("\n=== AUC per held-out stage (does the model transfer "
          "across age?) ===")
    print(res.filter(pl.col("method") == "split")
             .select(["held_out_stage", "label", "auc", "n_fail"]))
    return res


def run(labelled, leave_one_out: bool = True):
    r1, _ = run_within_fleet(labelled)
    r2 = run_leave_one_stage_out(labelled) if leave_one_out else None
    for name, r in (("wear_within", r1), ("wear_loso", r2)):
        if r is None:
            continue
        for path in (f"/kaggle/working/{name}.csv", f"{name}.csv"):
            try:
                r.write_csv(path); print(f"written {path}"); break
            except (FileNotFoundError, OSError):
                continue
    return r1, r2


if __name__ == "__main__":
    r1, r2 = run(labelled)          # noqa: F821
