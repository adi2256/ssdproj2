"""
Within-vendor wear-stage analysis.

WHY THIS RUN EXISTS

R3 showed that wear stage and vendor identity are confounded in this
fleet:

    failure coverage        A        B        C
      stage 0             --      0.714    0.876
      stage 1             --      0.935    0.893
      stage 2           1.000     0.805    0.929
      stage 3           0.065     0.778    0.926
      stage 4           0.292     0.727      --

    n failures            A        B        C
      stage 2               1      118      793
      stage 3              31      144      510
      stage 4             284       11       --

Vendor A supplies 284 of roughly 295 stage-4 failures; vendor C has no
drives that old at all. So the pooled stage-4 result is very nearly
vendor A's stage-4 result, and "oldest stage" cannot be distinguished
from "vendor A's old drives" using the fleet-wide table.

VENDOR C IS THE INFORMATIVE CASE

C has the most failures of any vendor (2,142 across its stages) and
shows NO collapse anywhere: 0.876, 0.893, 0.929, 0.926 across stages
0-3. If ageing degraded coverage as a general mechanism, C should show
a gradient over its own age range. It shows none -- if anything
coverage rises slightly with age.

Two readings, and this run distinguishes them:

  (a) Wear is real but only bites at extreme age. C never reaches
      stage 4, so it never enters the regime where coverage fails.
      Under this reading C's flatness is expected and says nothing.

  (b) There is no general wear effect. Vendor A simply has poorly
      calibrated scores, and because A dominates the oldest stage the
      effect masquerades as a wear gradient.

WHAT THIS RUN DOES

For each vendor separately, fit stages on THAT VENDOR'S OWN age
distribution, so every vendor gets five stages spanning its own range
rather than the fleet's. Then measure per-stage coverage within vendor.

    If vendor A alone shows a monotone decline over its own stages,
    wear is doing work independent of vendor identity.

    If A is uniformly poor at every one of its own stages, the effect
    is a vendor calibration problem and the wear framing is wrong.

    If B and C also show declines over their own ranges, wear is
    general and the fleet-wide table was merely underpowered.

Also reports the age OVERLAP between vendors, which is what makes the
fleet-wide stage bins misleading in the first place.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from src.conformal.mondrian import (MondrianConformal,
                                    group_conditional_coverage)
from src.conformal.split import SplitConformal
from src.data.splits import make_standard_split
from src.data.windowing import (Normalizer, WindowSet, constant_features,
                                drop_features, flatten, make_windows)
from src.evaluation.metrics import auc
from src.features.wear import fit_stages, window_age

STRIDE = 30
MAX_TRAIN = 200_000
N_TREES = 100
SEED = 42
NORMALIZE = "per_drive"
FLATTEN = "dynamics"
ALPHAS = (0.05, 0.10, 0.20)
MAIN_ALPHA = 0.10
MIN_FAIL_PER_CELL = 30      # below this a per-stage coverage number is noise


def _subset(ws, idx):
    return WindowSet(
        X=ws.X[idx], y=ws.y[idx], drive_idx=ws.drive_idx[idx],
        end_ds=ws.end_ds[idx], drives=ws.drives, vendors=ws.vendors[idx],
        features=ws.features, window_len=ws.window_len,
        stride=ws.stride, positive_stride=ws.positive_stride,
    )


def _cluster_se(hit, drive_idx):
    hit = np.asarray(hit, dtype=np.float64)
    d = np.asarray(drive_idx)
    o = np.argsort(d, kind="mergesort")
    d, h = d[o], hit[o]
    b = np.flatnonzero(np.diff(d)) + 1
    st, en = np.concatenate([[0], b]), np.concatenate([b, [len(d)]])
    pd_ = np.array([h[a:c].mean() for a, c in zip(st, en)])
    return (float(pd_.std(ddof=1) / np.sqrt(len(pd_)))
            if len(pd_) > 1 else float("nan"))


# ---------------------------------------------------------------
# Age overlap — why fleet-wide bins mislead
# ---------------------------------------------------------------

def age_overlap(labelled, seed=SEED):
    """
    Per-vendor age distribution. If the vendors occupy different parts
    of the age axis, a fleet-wide stage bin is partly a vendor label.
    """
    print("=" * 64 + "\nAGE OVERLAP BY VENDOR\n" + "=" * 64)
    sp = make_standard_split(labelled, seed=seed)
    ws = make_windows(sp.apply(labelled, "test"), stride=STRIDE)
    age, ven = window_age(ws), np.asarray(ws.vendors)

    rows = []
    for v in np.unique(ven):
        a = age[ven == v]
        a = a[np.isfinite(a)]
        rows.append({
            "vendor": str(v), "n_windows": int(a.size),
            "p05": round(float(np.quantile(a, 0.05)), 0),
            "p25": round(float(np.quantile(a, 0.25)), 0),
            "median": round(float(np.median(a)), 0),
            "p75": round(float(np.quantile(a, 0.75)), 0),
            "p95": round(float(np.quantile(a, 0.95)), 0),
            "max": round(float(a.max()), 0),
        })
    res = pl.DataFrame(rows)
    print(res)
    print("\nIf the p05-p95 ranges barely overlap, a fleet-wide stage "
          "bin is close to a vendor indicator, and the two effects "
          "cannot be separated by the pooled table.")
    return res


# ---------------------------------------------------------------
# Per-vendor pipeline
# ---------------------------------------------------------------

def build_vendor(labelled, vendor, n_stages=5, seed=SEED):
    """
    Train, calibrate and test entirely within one vendor, with stages
    fit on that vendor's own age distribution.
    """
    from sklearn.ensemble import RandomForestClassifier

    sub = labelled.filter(pl.col("vendor") == vendor)
    sp = make_standard_split(sub, seed=seed)
    raw = {p: make_windows(sp.apply(sub, p), stride=STRIDE)
           for p in ("train", "cal", "test")}
    if min(len(w) for w in raw.values()) == 0:
        return None
    if raw["cal"].y.sum() < 20 or raw["test"].y.sum() < 40:
        print(f"  vendor {vendor}: too few positives "
              f"(cal {int(raw['cal'].y.sum())}, "
              f"test {int(raw['test'].y.sum())}) -- skipped")
        return None

    if len(raw["train"]) > MAX_TRAIN:
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(raw["train"].y == 1)
        neg = np.flatnonzero(raw["train"].y == 0)
        keep = np.sort(np.concatenate(
            [pos, rng.choice(neg, size=min(MAX_TRAIN - len(pos), len(neg)),
                             replace=False)]))
        raw["train"] = _subset(raw["train"], keep)

    # Stages relative to THIS vendor's own ages.
    smap = fit_stages(raw["train"], strategy="quantile", n_stages=n_stages)
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

    return {
        "vendor": vendor, "smap": smap, "stage": stage, "raw": raw,
        "p_cal": proba(X["cal"]), "p_te": proba(X["test"]),
        "y_cal": raw["cal"].y, "y_te": raw["test"].y,
        "drive_te": raw["test"].drive_idx,
        "auc": auc(proba(X["test"]), raw["test"].y),
    }


def vendor_stage_coverage(c, alphas=ALPHAS):
    """Per-stage coverage within one vendor, across alpha."""
    rows = []
    g_cal, g_te = c["stage"]["cal"], c["stage"]["test"]
    for a in alphas:
        for name, cp, grp in (
            ("split", SplitConformal(alpha=a, seed=SEED)
             .fit(c["p_cal"], c["y_cal"]), None),
            ("mondrian_both", MondrianConformal(alpha=a, by="both",
                                                seed=SEED)
             .fit(c["p_cal"], c["y_cal"], groups=g_cal), g_te),
        ):
            r = cp.predict(c["p_te"], groups=grp) if grp is not None \
                else cp.predict(c["p_te"])
            hit = r.sets[np.arange(len(c["y_te"])), c["y_te"]]
            per = group_conditional_coverage(r, c["y_te"], g_te)
            for s, d in per.items():
                m = g_te == s
                rows.append({
                    "vendor": c["vendor"], "alpha": a, "method": name,
                    "stage": int(s), "n": d["n"], "n_fail": d["n_failure"],
                    "prevalence": round(d["n_failure"] / max(d["n"], 1), 5),
                    "coverage": round(d["coverage"], 4),
                    "cov_se": round(_cluster_se(hit[m], c["drive_te"][m]), 5),
                    "cov_failure": round(d["coverage_failure"], 4),
                    "fail_gap": round(d["coverage_failure"] - (1 - a), 4),
                    "set_size": round(float(r.set_sizes[m].mean()), 4),
                    "reliable": d["n_failure"] >= MIN_FAIL_PER_CELL,
                })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------

def run(labelled, vendors=("A", "B", "C"), n_stages=5, seed=SEED):
    overlap = age_overlap(labelled, seed=seed)

    out = []
    for v in vendors:
        print("\n" + "=" * 64 + f"\nVENDOR {v} -- stages on its own age range"
              + "\n" + "=" * 64)
        t0 = time.time()
        c = build_vendor(labelled, v, n_stages=n_stages, seed=seed)
        if c is None:
            continue
        print(f"  train {len(c['raw']['train']):,} cal {len(c['raw']['cal']):,} "
              f"test {len(c['raw']['test']):,} | AUC {c['auc']:.4f} "
              f"({time.time()-t0:.0f}s)")
        print(f"  stage edges: {c['smap'].edges.round(0).tolist()}")

        res = vendor_stage_coverage(c)
        out.append(res.with_columns(pl.lit(round(c["auc"], 4)).alias("auc")))

        main = res.filter((pl.col("alpha") == MAIN_ALPHA)
                          & (pl.col("method") == "split")).sort("stage")
        print(f"\n  split conformal, alpha={MAIN_ALPHA}:")
        print(main.select(["stage", "n", "n_fail", "prevalence", "coverage",
                           "cov_se", "cov_failure", "reliable"]))

        rel = main.filter(pl.col("reliable"))
        if rel.height >= 2:
            first, last = rel.row(0, named=True), rel.row(-1, named=True)
            trend = last["cov_failure"] - first["cov_failure"]
            print(f"  failure coverage, youngest reliable stage "
                  f"{first['stage']} -> oldest {last['stage']}: "
                  f"{first['cov_failure']:.3f} -> {last['cov_failure']:.3f} "
                  f"({trend:+.3f})")
        else:
            print(f"  fewer than 2 stages reach {MIN_FAIL_PER_CELL} "
                  f"failures -- no within-vendor trend can be read")

    if not out:
        print("\nno vendor produced enough positives")
        return None, overlap

    res = pl.concat(out)

    print("\n" + "=" * 64 + "\nWITHIN-VENDOR FAILURE COVERAGE, alpha=0.10"
          + "\n" + "=" * 64)
    main = res.filter((pl.col("alpha") == MAIN_ALPHA)
                      & (pl.col("method") == "split"))
    print("\nall cells:")
    print(main.pivot(values="cov_failure", index="stage", on="vendor"))
    print("\nfailures per cell (cells below "
          f"{MIN_FAIL_PER_CELL} are not readable):")
    print(main.pivot(values="n_fail", index="stage", on="vendor"))
    print("\nreliable cells only:")
    print(main.filter(pl.col("reliable"))
              .pivot(values="cov_failure", index="stage", on="vendor"))

    print("\n" + "-" * 64)
    print("READING THIS TABLE")
    print("  A declines over its OWN stages      -> wear is real, "
          "independent of vendor")
    print("  A is uniformly poor at every stage  -> vendor calibration "
          "problem, wear framing is wrong")
    print("  B and C also decline               -> wear is general; the "
          "fleet table was underpowered")
    print("  Only A shows anything              -> narrow the claim to "
          "A's fleet and say so")

    for name, r in (("within_vendor_wear", res), ("age_overlap", overlap)):
        for path in (f"/kaggle/working/{name}.csv", f"{name}.csv"):
            try:
                r.write_csv(path)
                break
            except (FileNotFoundError, OSError):
                continue
    return res, overlap


if __name__ == "__main__":
    res, overlap = run(labelled)          # noqa: F821
