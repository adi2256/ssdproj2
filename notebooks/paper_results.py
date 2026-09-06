"""
Paper-ready results for the wear-stage finding.

VALIDATED (all six gates passed, twice)

    V1  seed spread 0.0235 across 42/7/13        STABLE
    V3  drive-disjoint LOSO, stage 4:            0.983 cov / 0.077 fail
    V5  random strata at matched prevalence:     0.919 cov / 0.874 fail
        -> prevalence does NOT explain it
    V2  oldest stage under-covers in all 5 binning schemes
    V4  holds across alpha
    V6  holds across RF / GBM / logistic

V5 is the one that makes this novel: at 0.63% prevalence with no age
structure, failure coverage is 0.874. At 0.69% prevalence WITH age
structure, it is 0.296. Same prevalence, three times the coverage.

WHAT THIS SCRIPT PRODUCES

    R1  reliability curves -- empirical vs nominal coverage per stage
        across alpha. Licenses reading stage 4 as a calibration effect
        rather than an implementation error. Not optional.
    R2  set-size accounting -- every coverage number needs its cost
        beside it. Mondrian-both reaches 0.898 failure coverage at set
        size 1.687 on a two-label problem: 84% of the uninformative
        maximum.
    R3  vendor interaction -- is the effect universal or driven by one
        vendor?
    R4  operating points -- what the coverage failure costs in alerts
        per 1,000 drives at a threshold chosen on training data.
    R5  clustered standard errors -- windows within a drive are not
        independent; naive binomial intervals are far too tight.

Windows once, reused throughout. Roughly 20 minutes.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from src.conformal.mondrian import (MondrianConformal,
                                    group_conditional_coverage)
from src.conformal.split import SplitConformal, coverage_report
from src.data.splits import make_standard_split
from src.data.windowing import (Normalizer, WindowSet, constant_features,
                                drop_features, flatten, make_windows)
from src.evaluation.metrics import auc, f_beta, pick_threshold
from src.features.wear import fit_stages, stage_table, window_age

STRIDE = 30
MAX_TRAIN = 200_000
N_TREES = 100
SEED = 42
NORMALIZE = "per_drive"
FLATTEN = "dynamics"
N_STAGES = 5
ALPHAS = (0.02, 0.05, 0.10, 0.20, 0.30)
MAIN_ALPHA = 0.10


# ---------------------------------------------------------------
# Shared setup — windowed and scored once
# ---------------------------------------------------------------

def _subset(ws, idx):
    return WindowSet(
        X=ws.X[idx], y=ws.y[idx], drive_idx=ws.drive_idx[idx],
        end_ds=ws.end_ds[idx], drives=ws.drives, vendors=ws.vendors[idx],
        features=ws.features, window_len=ws.window_len,
        stride=ws.stride, positive_stride=ws.positive_stride,
    )


def build(labelled, seed=SEED):
    """Window, stage, fit, score. Everything below reuses this."""
    from sklearn.ensemble import RandomForestClassifier

    t0 = time.time()
    sp = make_standard_split(labelled, seed=seed)
    raw = {p: make_windows(sp.apply(labelled, p), stride=STRIDE)
           for p in ("train", "cal", "test")}

    if len(raw["train"]) > MAX_TRAIN:
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(raw["train"].y == 1)
        neg = np.flatnonzero(raw["train"].y == 0)
        keep = np.sort(np.concatenate(
            [pos, rng.choice(neg, size=min(MAX_TRAIN - len(pos), len(neg)),
                             replace=False)]))
        raw["train"] = _subset(raw["train"], keep)

    smap = fit_stages(raw["train"], strategy="quantile", n_stages=N_STAGES)
    stage = {p: smap.assign(window_age(w)) for p, w in raw.items()}

    norm = Normalizer.fit(raw["train"], method=NORMALIZE)
    nm = {p: norm.transform(w) for p, w in raw.items()}
    dead = constant_features(nm["test"])
    if dead and len(dead) < len(nm["test"].features):
        nm = {p: drop_features(w, dead) for p, w in nm.items()}
    X = {p: np.nan_to_num(flatten(w, FLATTEN)[0]) for p, w in nm.items()}

    m = RandomForestClassifier(
        n_estimators=N_TREES, min_samples_leaf=5, class_weight="balanced",
        random_state=seed, n_jobs=-1).fit(X["train"], raw["train"].y)

    def proba(Xp):
        p = m.predict_proba(Xp)
        return p[:, 1] if p.shape[1] > 1 else p[:, 0]

    ctx = {
        "smap": smap, "stage": stage, "raw": raw,
        "p_tr": proba(X["train"]), "p_cal": proba(X["cal"]),
        "p_te": proba(X["test"]),
        "y_tr": raw["train"].y, "y_cal": raw["cal"].y, "y_te": raw["test"].y,
        "drive_te": raw["test"].drive_idx,
        "vendor_te": raw["test"].vendors,
    }
    ctx["auc"] = auc(ctx["p_te"], ctx["y_te"])
    print(f"built in {time.time()-t0:.0f}s | test {len(ctx['y_te']):,} "
          f"windows | AUC {ctx['auc']:.4f}")
    print(f"stages: {smap.labels()}")
    print(stage_table(stage["test"], ctx["y_te"], smap))
    return ctx


def _fit_conformal(ctx, alpha, method, seed=SEED):
    g_cal, g_te = ctx["stage"]["cal"], ctx["stage"]["test"]
    if method == "split":
        cp = SplitConformal(alpha=alpha, seed=seed).fit(ctx["p_cal"],
                                                        ctx["y_cal"])
        return cp.predict(ctx["p_te"])
    by = {"mondrian_class": "class", "mondrian_stage": "group",
          "mondrian_both": "both"}[method]
    cp = MondrianConformal(alpha=alpha, by=by, seed=seed)
    if by == "class":
        return cp.fit(ctx["p_cal"], ctx["y_cal"]).predict(ctx["p_te"])
    return (cp.fit(ctx["p_cal"], ctx["y_cal"], groups=g_cal)
              .predict(ctx["p_te"], groups=g_te))


METHODS = ("split", "mondrian_class", "mondrian_stage", "mondrian_both")


# ---------------------------------------------------------------
# R5 — clustered standard errors
# ---------------------------------------------------------------

def cluster_se(hit: np.ndarray, drive_idx: np.ndarray) -> float:
    """
    Standard error of a coverage estimate, clustering by drive.

    Windows from one drive are ~30 overlapping views of the same
    trajectory. A naive binomial SE treats them as independent and is
    far too tight -- with 39,000 windows from perhaps 1,300 drives it
    understates the interval by roughly sqrt(30).

    Cluster-robust: variance of the per-drive mean, divided by the
    number of drives.
    """
    hit = np.asarray(hit, dtype=np.float64)
    d = np.asarray(drive_idx)
    order = np.argsort(d, kind="mergesort")
    d, h = d[order], hit[order]
    bounds = np.flatnonzero(np.diff(d)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(d)]])
    per_drive = np.array([h[a:b].mean() for a, b in zip(starts, ends)])
    n = len(per_drive)
    return float(per_drive.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")


# ---------------------------------------------------------------
# R1 — reliability curves
# ---------------------------------------------------------------

def r1_reliability(ctx):
    """
    Empirical vs nominal coverage per stage across alpha.

    This is what licenses reading the stage-4 result as a calibration
    effect. If coverage tracked nominal everywhere except stage 4 at
    one alpha, the obvious explanation would be a bug. Tracking nominal
    on stages 0-3 across five alphas, and diverging on stage 4 at every
    alpha, is a property.
    """
    print("\n" + "=" * 64 + "\nR1  RELIABILITY CURVES\n" + "=" * 64)
    g_te, y_te, drv = ctx["stage"]["test"], ctx["y_te"], ctx["drive_te"]
    rows = []
    for a in ALPHAS:
        for method in METHODS:
            r = _fit_conformal(ctx, a, method)
            hit = r.sets[np.arange(len(y_te)), y_te]
            per = group_conditional_coverage(r, y_te, g_te)
            for s, d in per.items():
                m = g_te == s
                rows.append({
                    "alpha": a, "target": round(1 - a, 3), "method": method,
                    "stage": int(s), "n": d["n"],
                    "prevalence": round(d["n_failure"] / max(d["n"], 1), 5),
                    "coverage": round(d["coverage"], 4),
                    "cov_se": round(cluster_se(hit[m], drv[m]), 5),
                    "cov_failure": round(d["coverage_failure"], 4),
                    "gap": round(d["coverage"] - (1 - a), 4),
                    "fail_gap": round(d["coverage_failure"] - (1 - a), 4),
                    "set_size": round(float(r.set_sizes[m].mean()), 4),
                })
        print(f"  alpha {a} done", flush=True)

    res = pl.DataFrame(rows)
    sp = res.filter(pl.col("method") == "split")

    print("\n=== split conformal: coverage - target, by stage ===")
    print(sp.pivot(values="gap", index="stage", on="alpha"))
    print("\n=== split conformal: FAILURE coverage - target, by stage ===")
    print(sp.pivot(values="fail_gap", index="stage", on="alpha"))

    oldest = int(res["stage"].max())
    o = sp.filter(pl.col("stage") == oldest)
    print(f"\noldest stage ({oldest}) at every alpha:")
    print(o.select(["alpha", "target", "coverage", "cov_se",
                    "cov_failure", "set_size"]))
    print("\nIf `gap` is near zero for stages 0-3 and positive for the "
          "oldest at EVERY alpha, the divergence is a calibration "
          "property, not an artefact of one confidence level.")
    return res


# ---------------------------------------------------------------
# R2 — set-size accounting
# ---------------------------------------------------------------

def r2_set_size(ctx, alpha=MAIN_ALPHA):
    """
    Every coverage number with its cost.

    Maximum set size on a two-label problem is 2. A method that reaches
    nominal coverage at 1.9 is answering "could be either" almost
    always, which is a refusal rather than a prediction. Reporting
    coverage without set size cannot distinguish the two.
    """
    print("\n" + "=" * 64 + "\nR2  SET-SIZE ACCOUNTING\n" + "=" * 64)
    g_te, y_te, drv = ctx["stage"]["test"], ctx["y_te"], ctx["drive_te"]
    rows = []
    for method in METHODS:
        r = _fit_conformal(ctx, alpha, method)
        hit = r.sets[np.arange(len(y_te)), y_te]
        per = group_conditional_coverage(r, y_te, g_te)
        for s, d in per.items():
            m = g_te == s
            rows.append({
                "method": method, "stage": int(s), "n": d["n"],
                "n_fail": d["n_failure"],
                "prevalence": round(d["n_failure"] / max(d["n"], 1), 5),
                "coverage": round(d["coverage"], 4),
                "cov_se": round(cluster_se(hit[m], drv[m]), 5),
                "cov_failure": round(d["coverage_failure"], 4),
                "cov_healthy": round(d["coverage_healthy"], 4),
                "set_size": round(float(r.set_sizes[m].mean()), 4),
                "pct_of_max": round(float(r.set_sizes[m].mean()) / 2 * 100, 1),
                "singleton": round(float(r.is_singleton[m].mean()), 4),
                "doubleton": round(float(r.is_doubleton[m].mean()), 4),
                "empty": round(float(r.is_empty[m].mean()), 4),
            })

    res = pl.DataFrame(rows)
    print("\n=== coverage / failure coverage / set size, alpha=0.10 ===")
    for method in METHODS:
        sub = res.filter(pl.col("method") == method)
        print(f"\n{method}")
        print(sub.select(["stage", "prevalence", "coverage", "cov_se",
                          "cov_failure", "set_size", "pct_of_max",
                          "doubleton"]))

    oldest = int(res["stage"].max())
    o = res.filter(pl.col("stage") == oldest)
    print(f"\n=== the repair and its cost, oldest stage ===")
    print(o.select(["method", "coverage", "cov_failure", "set_size",
                    "pct_of_max", "doubleton"]))
    return res


# ---------------------------------------------------------------
# R3 — vendor interaction
# ---------------------------------------------------------------

def r3_vendor(ctx, alpha=MAIN_ALPHA):
    """
    Is the effect universal, or driven by one vendor?

    If only one vendor shows it, the claim narrows to that vendor's
    fleet and the paper must say so. If all three show it, the claim is
    about wear rather than about a manufacturer.
    """
    print("\n" + "=" * 64 + "\nR3  VENDOR INTERACTION\n" + "=" * 64)
    g_te, y_te = ctx["stage"]["test"], ctx["y_te"]
    ven = np.asarray(ctx["vendor_te"])
    drv = ctx["drive_te"]

    rows = []
    for method in ("split", "mondrian_both"):
        r = _fit_conformal(ctx, alpha, method)
        hit = r.sets[np.arange(len(y_te)), y_te]
        for v in np.unique(ven):
            for s in np.unique(g_te):
                m = (ven == v) & (g_te == s)
                if m.sum() < 200:
                    continue
                fm = m & (y_te == 1)
                rows.append({
                    "method": method, "vendor": str(v), "stage": int(s),
                    "n": int(m.sum()), "n_fail": int(fm.sum()),
                    "prevalence": round(float(y_te[m].mean()), 5),
                    "coverage": round(float(hit[m].mean()), 4),
                    "cov_se": round(cluster_se(hit[m], drv[m]), 5),
                    "cov_failure": (round(float(hit[fm].mean()), 4)
                                    if fm.sum() else float("nan")),
                    "set_size": round(float(r.set_sizes[m].mean()), 4),
                })

    res = pl.DataFrame(rows)
    sp = res.filter(pl.col("method") == "split")
    print("\n=== split conformal, failure coverage by vendor x stage ===")
    print(sp.pivot(values="cov_failure", index="stage", on="vendor"))
    print("\n=== prevalence by vendor x stage ===")
    print(sp.pivot(values="prevalence", index="stage", on="vendor"))
    print("\n=== n failures by vendor x stage ===")
    print(sp.pivot(values="n_fail", index="stage", on="vendor"))
    print("\nIf the oldest stage under-covers in all three vendor "
          "columns, the effect is about wear. If one column drives it, "
          "narrow the claim.")
    return res


# ---------------------------------------------------------------
# R4 — operating points
# ---------------------------------------------------------------

def r4_operating(ctx, alpha=MAIN_ALPHA):
    """
    Coverage is the guarantee; operators act on alerts.

    Threshold chosen on TRAINING scores, then applied unchanged. For
    each stage: precision, recall, F0.5, and alerts per 1,000 drives --
    which is what a technician's workload actually looks like.
    """
    print("\n" + "=" * 64 + "\nR4  OPERATING POINTS\n" + "=" * 64)
    thr = pick_threshold(ctx["p_tr"], ctx["y_tr"], beta=0.5)
    print(f"  threshold from TRAINING scores: {thr:.4f}")

    g_te, y_te, p_te, drv = (ctx["stage"]["test"], ctx["y_te"],
                             ctx["p_te"], ctx["drive_te"])
    r = _fit_conformal(ctx, alpha, "split")

    rows = []
    for s in np.unique(g_te):
        m = g_te == s
        p, y = p_te[m], y_te[m]
        pred = p >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        n_drives = len(np.unique(drv[m]))
        alert_drives = len(np.unique(drv[m][pred]))
        rows.append({
            "stage": int(s), "n_windows": int(m.sum()),
            "n_drives": n_drives,
            "prevalence": round(float(y.mean()), 5),
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4),
            "f0.5": round(f_beta(p, y, thr, 0.5), 4),
            "alerts_per_1k_drives": round(1000 * alert_drives
                                          / max(n_drives, 1), 1),
            "conformal_cov_failure": round(
                float(r.sets[np.arange(len(y_te)), y_te][m & (y_te == 1)]
                      .mean()) if (m & (y_te == 1)).sum() else float("nan"),
                4),
        })

    res = pl.DataFrame(rows)
    print(res)
    print("\nThe last two columns are the point: at the oldest stage the "
          "thresholded model still alerts, but the conformal set stops "
          "containing the true label. An operator reading the guarantee "
          "would trust it least where it is least deserved.")
    return res


# ---------------------------------------------------------------

def run_all(labelled, seed=SEED):
    ctx = build(labelled, seed=seed)
    out = {}
    for key, fn in (("r1", r1_reliability), ("r2", r2_set_size),
                    ("r3", r3_vendor), ("r4", r4_operating)):
        t0 = time.time()
        out[key] = fn(ctx)
        print(f"\n[{key} done in {time.time()-t0:.0f}s]")
        for path in (f"/kaggle/working/{key}_results.csv",
                     f"{key}_results.csv"):
            try:
                out[key].write_csv(path)
                break
            except (FileNotFoundError, OSError):
                continue
    out["ctx"] = ctx
    return out


if __name__ == "__main__":
    out = run_all(labelled)          # noqa: F821
