"""
Validation battery for the wear-stage result.

WHAT WE HAVE

Within-fleet, standard split, split conformal on the oldest drives:

    stage 4  coverage 0.9609 (target 0.90)  failure coverage 0.2956

Leave-one-stage-out, monotone across five folds:

    held out    coverage   failure coverage   AUC
    0 youngest    0.854          0.877        0.61
    1             0.887          0.878        0.65
    2             0.894          0.806        0.58
    3             0.917          0.754        0.67
    4 oldest      0.973          0.050        0.61

Mechanism: stage 4 has the LOWEST failure rate (0.69% vs 2.40% at
stage 2) because drives reaching 28,000+ hours are survivors. Pooled
calibration is dominated by mid-life drives, so the threshold is far
too loose for the oldest cohort. Only the joint (stage, class)
taxonomy repairs it -- 0.2956 -> 0.8978.

WHAT COULD STILL KILL IT

Six threats, each with a run below. None is expensive. All of them
should be answered BEFORE any prose is written, because two of them
could invalidate the headline.

    V1  seed stability          is the monotone trend reproducible?
    V2  stage definition        is it wear, or an artefact of quantile bins?
    V3  drive disjointness      LOSO lets a drive's young windows train
                                and its old windows test
    V4  alpha sensitivity       does it hold at 0.05 and 0.20?
    V5  prevalence confound     is this just "low prevalence breaks
                                conformal", with age incidental?
    V6  model dependence        does it survive a different classifier?

V3 and V5 are the dangerous ones. V5 in particular: if a random
low-prevalence stratum shows the same collapse, the finding is about
prevalence rather than wear, and the framing changes completely.

Run order matters. V1 and V3 first -- if either fails there is no
result to defend.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from src.conformal.mondrian import MondrianConformal, group_conditional_coverage
from src.conformal.split import SplitConformal, coverage_report
from src.data.splits import make_standard_split
from src.data.windowing import (Normalizer, WindowSet, constant_features,
                                drop_features, flatten, make_windows)
from src.evaluation.metrics import auc
from src.features.wear import fit_stages, stage_table, window_age

STRIDE = 30
MAX_TRAIN = 200_000
N_TREES = 100
ALPHA = 0.10
NORMALIZE = "per_drive"
FLATTEN = "dynamics"
N_STAGES = 5


def _subset(ws, idx):
    return WindowSet(
        X=ws.X[idx], y=ws.y[idx], drive_idx=ws.drive_idx[idx],
        end_ds=ws.end_ds[idx], drives=ws.drives, vendors=ws.vendors[idx],
        features=ws.features, window_len=ws.window_len,
        stride=ws.stride, positive_stride=ws.positive_stride,
    )


def _cap_train(ws, seed, cap=MAX_TRAIN):
    if len(ws) <= cap:
        return ws
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(ws.y == 1)
    neg = np.flatnonzero(ws.y == 0)
    keep = np.sort(np.concatenate(
        [pos, rng.choice(neg, size=min(cap - len(pos), len(neg)),
                         replace=False)]))
    return _subset(ws, keep)


def _fit_predict(raw, seed, model="rf"):
    """Model on normalised dynamics features. Returns cal/test scores."""
    norm = Normalizer.fit(raw["train"], method=NORMALIZE)
    nm = {p: norm.transform(w) for p, w in raw.items()}
    dead = constant_features(nm["test"])
    if dead and len(dead) < len(nm["test"].features):
        nm = {p: drop_features(w, dead) for p, w in nm.items()}
    X = {p: np.nan_to_num(flatten(w, FLATTEN)[0]) for p, w in nm.items()}

    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(
            n_estimators=N_TREES, min_samples_leaf=5,
            class_weight="balanced", random_state=seed, n_jobs=-1)
    elif model == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, class_weight="balanced",
                               random_state=seed))
    elif model == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(
            max_iter=150, random_state=seed,
            class_weight="balanced")
    else:
        raise ValueError(model)

    m.fit(X["train"], raw["train"].y)

    def proba(Xp):
        p = m.predict_proba(Xp)
        return p[:, 1] if p.shape[1] > 1 else p[:, 0]

    return proba(X["cal"]), proba(X["test"])


def _report(p_cal, y_cal, p_te, y_te, g_cal, g_te, alpha=ALPHA, seed=0):
    """All four conformal conditions, reported per group."""
    conds = {
        "split": (SplitConformal(alpha=alpha, seed=seed)
                  .fit(p_cal, y_cal), None),
        "mondrian_class": (MondrianConformal(alpha=alpha, by="class",
                                             seed=seed)
                           .fit(p_cal, y_cal), None),
        "mondrian_stage": (MondrianConformal(alpha=alpha, by="group",
                                             seed=seed)
                           .fit(p_cal, y_cal, groups=g_cal), g_te),
        "mondrian_both": (MondrianConformal(alpha=alpha, by="both",
                                            seed=seed)
                          .fit(p_cal, y_cal, groups=g_cal), g_te),
    }
    rows = []
    for name, (cp, grp) in conds.items():
        r = cp.predict(p_te, groups=grp) if grp is not None else cp.predict(p_te)
        per = group_conditional_coverage(r, y_te, g_te)
        for s, d in per.items():
            m = g_te == s
            rows.append({
                "method": name, "stage": int(s), "n": d["n"],
                "prevalence": round(d["n_failure"] / max(d["n"], 1), 5),
                "coverage": round(d["coverage"], 4),
                "cov_failure": round(d["coverage_failure"], 4),
                "set_size": round(float(r.set_sizes[m].mean()), 4),
            })
    return pl.DataFrame(rows)


def _oldest(rep, method="split"):
    """
    The oldest-stage row, whatever N_STAGES is set to. Hardcoding
    stage 4 breaks the moment the bin count changes, which V2 does
    deliberately.
    """
    rows = rep.filter(pl.col("method") == method).sort("stage")
    return rows.row(-1, named=True) if rows.height else None


# ===============================================================
# V1 — seed stability
# ===============================================================

def v1_seed_stability(labelled, seeds=(42, 7, 13)):
    """
    Five folds is enough for a trend only if the trend reproduces.
    Re-runs the within-fleet experiment at three seeds; the split, the
    forest and the training subsample all change.

    KILL CONDITION: if stage 4 failure coverage under split conformal
    varies by more than ~0.15 across seeds, the headline is noise.
    """
    print("=" * 64 + "\nV1  SEED STABILITY\n" + "=" * 64)
    out = []
    for sd in seeds:
        t0 = time.time()
        sp = make_standard_split(labelled, seed=sd)
        raw = {p: make_windows(sp.apply(labelled, p), stride=STRIDE)
               for p in ("train", "cal", "test")}
        raw["train"] = _cap_train(raw["train"], sd)

        smap = fit_stages(raw["train"], strategy="quantile",
                          n_stages=N_STAGES)
        g = {p: smap.assign(window_age(w)) for p, w in raw.items()}
        p_cal, p_te = _fit_predict(raw, sd)

        rep = _report(p_cal, raw["cal"].y, p_te, raw["test"].y,
                      g["cal"], g["test"], seed=sd)
        rep = rep.with_columns(pl.lit(sd).alias("seed"),
                               pl.lit(round(auc(p_te, raw["test"].y), 4))
                               .alias("auc"))
        out.append(rep)
        s4 = _oldest(rep)
        print(f"  seed {sd}: auc {auc(p_te, raw['test'].y):.3f}  "
              f"oldest-stage cov {s4['coverage']:.4f} "
              f"fail {s4['cov_failure']:.4f}  ({time.time()-t0:.0f}s)",
              flush=True)

    res = pl.concat(out)
    oldest = int(res["stage"].max())
    s4 = res.filter((pl.col("method") == "split")
                    & (pl.col("stage") == oldest))
    spread = float(s4["cov_failure"].max() - s4["cov_failure"].min())
    print(f"\noldest-stage failure coverage across seeds: "
          f"{s4['cov_failure'].to_list()}  spread {spread:.4f}")
    print("  STABLE" if spread < 0.15 else "  UNSTABLE -- headline is noise")
    return res


# ===============================================================
# V2 — stage definition
# ===============================================================

def v2_stage_definition(labelled, seed=42):
    """
    Quantile bins are arbitrary. If the effect is about WEAR it should
    survive data-driven boundaries and a different bin count. If it
    only appears at 5 equal-frequency bins, it is a binning artefact.
    """
    print("\n" + "=" * 64 + "\nV2  STAGE DEFINITION\n" + "=" * 64)
    sp = make_standard_split(labelled, seed=seed)
    raw = {p: make_windows(sp.apply(labelled, p), stride=STRIDE)
           for p in ("train", "cal", "test")}
    raw["train"] = _cap_train(raw["train"], seed)
    p_cal, p_te = _fit_predict(raw, seed)

    out = []
    for strategy, k in [("quantile", 3), ("quantile", 5), ("quantile", 8),
                        ("changepoint", 3), ("changepoint", 5)]:
        try:
            smap = fit_stages(raw["train"], strategy=strategy, n_stages=k)
        except ValueError as e:
            print(f"  {strategy}/{k}: {e}")
            continue
        g = {p: smap.assign(window_age(w)) for p, w in raw.items()}
        rep = _report(p_cal, raw["cal"].y, p_te, raw["test"].y,
                      g["cal"], g["test"], seed=seed)
        rep = rep.with_columns(pl.lit(f"{strategy}_{k}").alias("stages"))
        out.append(rep)

        sp_rows = rep.filter(pl.col("method") == "split")
        oldest = sp_rows.sort("stage").row(-1, named=True)
        print(f"  {strategy}/{k}: edges {smap.edges.round(0).tolist()}  "
              f"oldest-stage cov {oldest['coverage']:.4f} "
              f"fail {oldest['cov_failure']:.4f} "
              f"prev {oldest['prevalence']:.4f}", flush=True)

    res = pl.concat(out)
    print("\nIf the oldest stage under-covers failures in EVERY row, the "
          "effect is about wear, not binning.")
    return res


# ===============================================================
# V3 — drive disjointness
# ===============================================================

def v3_drive_disjoint(labelled, seed=42):
    """
    THE MOST DANGEROUS THREAT.

    In the leave-one-stage-out run, windows are carved by stage, so a
    drive's young windows can sit in train while its old windows sit in
    test. That is arguably the honest deployment situation, but it is
    not drive-disjoint, and a reviewer will ask whether the model is
    simply memorising drives.

    This variant assigns each DRIVE to the stage it spends the most
    windows in, then holds out whole drives. Stricter, cleaner, and if
    the monotone trend survives it, the objection is closed.

    KILL CONDITION: if the trend disappears under drive-disjoint
    splitting, the leave-one-stage-out result cannot be used.
    """
    print("\n" + "=" * 64 + "\nV3  DRIVE-DISJOINT LEAVE-ONE-STAGE-OUT\n"
          + "=" * 64)
    sp = make_standard_split(labelled, seed=seed)
    keys = pl.concat([sp.train, sp.cal, sp.test]).select(["model", "disk_id"])
    ws = make_windows(labelled.join(keys, on=["model", "disk_id"],
                                    how="inner"), stride=STRIDE)
    smap = fit_stages(ws, strategy="quantile", n_stages=N_STAGES)
    st = smap.assign(window_age(ws))

    # Each drive -> its modal stage.
    n_dr = len(ws.drives)
    counts = np.zeros((n_dr, smap.n_stages), dtype=int)
    np.add.at(counts, (ws.drive_idx, st), 1)
    drive_stage = counts.argmax(axis=1)
    print(f"  drives per modal stage: "
          f"{np.bincount(drive_stage, minlength=smap.n_stages).tolist()}")

    rng = np.random.default_rng(seed)
    rows = []
    for held in range(smap.n_stages):
        te_drives = np.flatnonzero(drive_stage == held)
        rest = np.flatnonzero(drive_stage != held)
        if len(te_drives) == 0:
            continue
        rng.shuffle(rest)
        n_cal = int(round(0.2 * len(rest)))
        cal_d, tr_d = set(rest[:n_cal].tolist()), set(rest[n_cal:].tolist())

        idx_te = np.flatnonzero(np.isin(ws.drive_idx, te_drives))
        idx_cal = np.flatnonzero(np.isin(ws.drive_idx, list(cal_d)))
        idx_tr = np.flatnonzero(np.isin(ws.drive_idx, list(tr_d)))
        raw = {"train": _cap_train(_subset(ws, idx_tr), seed),
               "cal": _subset(ws, idx_cal), "test": _subset(ws, idx_te)}
        if raw["test"].y.sum() < 20 or raw["cal"].y.sum() < 10:
            print(f"  stage {held}: too few positives, skipped")
            continue

        p_cal, p_te = _fit_predict(raw, seed)
        a = auc(p_te, raw["test"].y)
        for name, cp, grp in [
            ("split", SplitConformal(alpha=ALPHA, seed=seed)
             .fit(p_cal, raw["cal"].y), None),
            ("mondrian_class", MondrianConformal(alpha=ALPHA, by="class",
                                                 seed=seed)
             .fit(p_cal, raw["cal"].y), None),
        ]:
            r = cp.predict(p_te)
            rep = coverage_report(r, raw["test"].y)
            rows.append({
                "held_out_stage": held, "method": name, "auc": round(a, 4),
                "n_test_drives": len(te_drives),
                "n_fail": rep["n_failures_test"],
                "prevalence": round(rep["n_failures_test"]
                                    / max(rep["n_test"], 1), 5),
                "coverage": round(rep["empirical_coverage"], 4),
                "cov_failure": round(rep["coverage_failure"], 4),
                "set_size": round(rep["avg_set_size"], 4),
            })
        last = [r for r in rows if r["held_out_stage"] == held]
        print(f"  stage {held}: auc {a:.3f}  " +
              "  ".join(f"{r['method'][:8]} cov {r['coverage']:.3f}/"
                        f"fail {r['cov_failure']:.3f}" for r in last),
              flush=True)

    res = pl.DataFrame(rows)
    print("\n=== drive-disjoint LOSO ===")
    print(res.filter(pl.col("method") == "split")
             .select(["held_out_stage", "n_test_drives", "prevalence",
                      "auc", "coverage", "cov_failure"]))
    print("\nCompare against the window-carved version: coverage rose "
          "0.854 -> 0.973 and failure coverage fell 0.877 -> 0.050.")
    return res


# ===============================================================
# V4 — alpha sensitivity
# ===============================================================

def v4_alpha(labelled, seed=42, alphas=(0.05, 0.10, 0.20)):
    """
    A guarantee that only misbehaves at one confidence level is a
    curiosity. One that misbehaves across levels is a property.
    """
    print("\n" + "=" * 64 + "\nV4  ALPHA SENSITIVITY\n" + "=" * 64)
    sp = make_standard_split(labelled, seed=seed)
    raw = {p: make_windows(sp.apply(labelled, p), stride=STRIDE)
           for p in ("train", "cal", "test")}
    raw["train"] = _cap_train(raw["train"], seed)
    smap = fit_stages(raw["train"], strategy="quantile", n_stages=N_STAGES)
    g = {p: smap.assign(window_age(w)) for p, w in raw.items()}
    p_cal, p_te = _fit_predict(raw, seed)

    out = []
    for a in alphas:
        rep = _report(p_cal, raw["cal"].y, p_te, raw["test"].y,
                      g["cal"], g["test"], alpha=a, seed=seed)
        rep = rep.with_columns(pl.lit(a).alias("alpha"))
        out.append(rep)
        s4 = _oldest(rep)
        print(f"  alpha {a}: target {1-a:.2f}  oldest-stage cov "
              f"{s4['coverage']:.4f}  fail {s4['cov_failure']:.4f}",
              flush=True)
    return pl.concat(out)


# ===============================================================
# V5 — prevalence confound
# ===============================================================

def v5_prevalence_confound(labelled, seed=42):
    """
    THE OTHER DANGEROUS THREAT.

    Stage 4 has the lowest prevalence (0.69% vs 2.40% at stage 2). Is
    the collapse about WEAR, or simply about low prevalence with age
    incidental?

    Control: build five RANDOM strata matched to the wear stages'
    prevalences by subsampling positives, with no age structure at all.
    If the low-prevalence random stratum shows the same collapse, the
    finding is "conformal breaks at low prevalence" -- true, but
    already known, and the wear framing would be wrong.

    If the random stratum does NOT collapse, prevalence alone does not
    explain it and the wear-specific mechanism survives.
    """
    print("\n" + "=" * 64 + "\nV5  PREVALENCE CONFOUND\n" + "=" * 64)
    sp = make_standard_split(labelled, seed=seed)
    raw = {p: make_windows(sp.apply(labelled, p), stride=STRIDE)
           for p in ("train", "cal", "test")}
    raw["train"] = _cap_train(raw["train"], seed)
    smap = fit_stages(raw["train"], strategy="quantile", n_stages=N_STAGES)
    g_cal = smap.assign(window_age(raw["cal"]))
    g_te = smap.assign(window_age(raw["test"]))
    p_cal, p_te = _fit_predict(raw, seed)

    y_te = raw["test"].y
    target_prev = [float(y_te[g_te == s].mean()) for s in range(smap.n_stages)]
    print(f"  wear-stage prevalences: "
          f"{[round(p, 5) for p in target_prev]}")

    # Random strata with the same prevalences, no age structure.
    rng = np.random.default_rng(seed)
    n_per = len(y_te) // smap.n_stages
    pos_pool = list(np.flatnonzero(y_te == 1))
    neg_pool = list(np.flatnonzero(y_te == 0))
    rng.shuffle(pos_pool)
    rng.shuffle(neg_pool)

    g_rand = np.full(len(y_te), -1)
    pi = ni = 0
    for s, prev in enumerate(target_prev):
        n_pos = min(int(round(prev * n_per)), len(pos_pool) - pi)
        n_neg = min(n_per - n_pos, len(neg_pool) - ni)
        take = pos_pool[pi:pi + n_pos] + neg_pool[ni:ni + n_neg]
        pi += n_pos
        ni += n_neg
        g_rand[np.array(take, dtype=int)] = s

    keep = g_rand >= 0
    rep = _report(p_cal, raw["cal"].y, p_te[keep], y_te[keep],
                  g_cal, g_rand[keep], seed=seed)
    sp_rows = rep.filter(pl.col("method") == "split").sort("stage")
    print("\n  RANDOM strata matched on prevalence (no age structure):")
    print(sp_rows.select(["stage", "n", "prevalence", "coverage",
                          "cov_failure", "set_size"]))
    print("\n  For comparison, the real wear stages gave stage-4 "
          "coverage 0.9609 / failure coverage 0.2956.")
    print("  If the matched random stratum ALSO collapses, the finding "
          "is about prevalence and the wear framing must change.")
    return rep


# ===============================================================
# V6 — model dependence
# ===============================================================

def v6_model(labelled, seed=42, models=("rf", "gbm", "logreg")):
    """
    Conformal coverage is distribution-free, so the guarantee should
    not depend on the classifier. If the stage-4 collapse appears with
    all three, it is a property of the calibration, not the model.
    """
    print("\n" + "=" * 64 + "\nV6  MODEL DEPENDENCE\n" + "=" * 64)
    sp = make_standard_split(labelled, seed=seed)
    raw = {p: make_windows(sp.apply(labelled, p), stride=STRIDE)
           for p in ("train", "cal", "test")}
    raw["train"] = _cap_train(raw["train"], seed)
    smap = fit_stages(raw["train"], strategy="quantile", n_stages=N_STAGES)
    g = {p: smap.assign(window_age(w)) for p, w in raw.items()}

    out = []
    for mdl in models:
        try:
            p_cal, p_te = _fit_predict(raw, seed, model=mdl)
        except Exception as e:
            print(f"  {mdl}: {e}")
            continue
        rep = _report(p_cal, raw["cal"].y, p_te, raw["test"].y,
                      g["cal"], g["test"], seed=seed)
        rep = rep.with_columns(pl.lit(mdl).alias("model"),
                               pl.lit(round(auc(p_te, raw["test"].y), 4))
                               .alias("auc"))
        out.append(rep)
        s4 = _oldest(rep)
        print(f"  {mdl}: auc {auc(p_te, raw['test'].y):.3f}  oldest-stage "
              f"cov {s4['coverage']:.4f} fail {s4['cov_failure']:.4f}",
              flush=True)
    return pl.concat(out) if out else pl.DataFrame()


# ===============================================================

def run_all(labelled, which=("v1", "v3", "v5", "v2", "v4", "v6")):
    """
    Order is deliberate: V1 and V3 can kill the result, V5 can force a
    reframe. Run those three first; if any fails, the rest is wasted.
    """
    fns = {"v1": v1_seed_stability, "v2": v2_stage_definition,
           "v3": v3_drive_disjoint, "v4": v4_alpha,
           "v5": v5_prevalence_confound, "v6": v6_model}
    results = {}
    for key in which:
        t0 = time.time()
        results[key] = fns[key](labelled)
        print(f"\n[{key} done in {time.time()-t0:.0f}s]\n")
        for path in (f"/kaggle/working/{key}.csv", f"{key}.csv"):
            try:
                results[key].write_csv(path)
                break
            except (FileNotFoundError, OSError):
                continue
    return results


if __name__ == "__main__":
    results = run_all(labelled)          # noqa: F821
