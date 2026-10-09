"""
Revision analysis: everything the reviewer response needs, computed off
the prediction dumps written by revision_runs.py.

    REV-1  Platt / isotonic baseline             platt_*.csv, reliability_*.csv
    REV-2  drive-clustered bootstrap CIs         boot_pooled.csv, boot_wv.csv
    REV-3  healthy-sampling-ratio sensitivity    ratio_summary.csv

One call does all of it, and re-running it is safe:

    from notebooks.revision_analysis import run_all, analyse
    run_all(C.load_slim)      # trains whatever dumps are missing (~20 min)
    analyse()                 # no training; ~5-10 min of bootstrap

WHAT THE BOOTSTRAP RESAMPLES

Drives, not windows, and calibration drives as well as test drives. The
conformal thresholds are recomputed from each resampled calibration set,
so the interval reflects calibration-set variability -- which is where
conformal coverage actually varies -- and not only test-set noise. The
classifier itself is held fixed: the intervals are conditional on the
trained model. That is stated in the paper, not hidden.

To make the recomputation trustworthy, the thresholds are first
recomputed at unit weights and compared set-for-set against the sets
the src/ conformal code wrote into the dump. Any mismatch raises.

WHAT THE PLATT / ISOTONIC BASELINE CAN AND CANNOT SHOW

Both are monotone maps of the score. Any single threshold on the
calibrated probability is a threshold on the raw score, so at a matched
alert budget they flag exactly the same windows (isotonic up to ties).
The per-stage recall they deliver is therefore fixed by the ranking, and
calibration cannot move it. That is the structural argument in the
paper; this file measures it rather than asserting it.
"""

from __future__ import annotations

import json
import os
import time
import zipfile

import numpy as np
import polars as pl

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
ALPHA = 0.10
SEED = 42
B_DEFAULT = 2000
POOLED_METHODS = ("split", "mondrian_class", "mondrian_stage", "mondrian_both")
WV_METHODS = ("split", "mondrian_both")
TAGS = ("r1", "r2p7", "r5")
RATIO = {"r1": 1.0, "r2p7": 2.68, "r5": 5.0}


# =================================================================
# Orchestration
# =================================================================

def _have(tag: str) -> bool:
    return os.path.exists(f"{OUT}/meta_{tag}.json")


def run_all(load_slim) -> None:
    """
    Train and dump whatever is missing. Resumable: if the kernel dies,
    restart, re-run setup, call again -- finished runs are skipped.
    """
    import gc
    from notebooks.revision_runs import run_ratio, run_within_vendor

    t0 = time.time()
    for tag in ("r2p7", "r1", "r5"):
        if _have(tag):
            print(f"[skip {tag}: dump exists]")
            continue
        run_ratio(tag, load_slim)
        gc.collect()
    if _have("wv"):
        print("[skip within-vendor: dump exists]")
    else:
        run_within_vendor(load_slim)
    print(f"\n[run_all done in {time.time() - t0:.0f}s]")


# =================================================================
# Conformal recomputation (LAC), weight-aware
# =================================================================

def _cand(p):
    """Candidate scores with the exact float ops of src/conformal."""
    pm_h = 1.0 - p
    return 1.0 - pm_h, 1.0 - p          # (healthy, failure)


def _true_score(p, y):
    h, f = _cand(p)
    return np.where(y == 1, f, h)


class _Cells:
    """Calibration scores pre-sorted within cells; weighted quantiles."""

    def __init__(self, score, key):
        self.order = np.lexsort((score, key))
        k = key[self.order]
        self.s = score[self.order]
        self.keys, self.a = np.unique(k, return_index=True)
        self.b = np.r_[self.a[1:], len(k)]

    def quantiles(self, w, alpha):
        ws = w[self.order]
        out = {}
        for key, a, b in zip(self.keys, self.a, self.b):
            cw = np.cumsum(ws[a:b])
            n = int(round(cw[-1])) if b > a else 0
            if n == 0:
                continue                      # cell absent in this draw
            k = int(np.ceil((n + 1) * (1.0 - alpha)))
            if k > n:
                out[int(key)] = np.inf
            else:
                out[int(key)] = float(self.s[a + np.searchsorted(cw, k)])
        return out


class _Conformal:
    """All four methods for one cal/test pair, thresholds per draw."""

    def __init__(self, cal, test, methods, n_stages, alpha=ALPHA):
        self.alpha, self.methods, self.S = alpha, methods, n_stages
        pc = cal["p"].to_numpy()
        yc = cal["y"].to_numpy().astype(int)
        gc_ = cal["stage"].to_numpy().astype(int)
        sc = _true_score(pc, yc)
        self.cells = {
            "split": _Cells(sc, np.zeros(len(sc), dtype=int)),
            "mondrian_class": _Cells(sc, yc),
            "mondrian_stage": _Cells(sc, gc_),
            "mondrian_both": _Cells(sc, gc_ * 2 + yc),
        }
        self.ch, self.cf = _cand(test["p"].to_numpy())
        self.st = test["stage"].to_numpy().astype(int)

    def sets(self, method, wc):
        a, S = self.alpha, self.S
        pooled = self.cells["split"].quantiles(wc, a).get(0, np.inf)
        q = self.cells[method].quantiles(wc, a)
        if method == "split":
            qh = qf = np.full(S, pooled)
        elif method == "mondrian_class":
            qh = np.full(S, q.get(0, np.inf))
            qf = np.full(S, q.get(1, np.inf))
        elif method == "mondrian_stage":
            qh = qf = np.array([q.get(s, pooled) for s in range(S)])
        else:
            qh = np.array([q.get(2 * s, pooled) for s in range(S)])
            qf = np.array([q.get(2 * s + 1, pooled) for s in range(S)])
        return self.ch <= qh[self.st], self.cf <= qf[self.st]


# =================================================================
# Weighted statistics
# =================================================================

class _AUC:
    """Weighted ROC AUC with ties at half credit; O(n) per draw."""

    def __init__(self, p, y):
        self.u = np.unique(p, return_inverse=True)[1]
        self.U = int(self.u.max()) + 1
        self.y = y.astype(float)

    def __call__(self, w):
        pos = np.bincount(self.u, w * self.y, self.U)
        neg = np.bincount(self.u, w * (1 - self.y), self.U)
        below = np.cumsum(neg) - neg
        den = pos.sum() * neg.sum()
        return float((pos * (below + 0.5 * neg)).sum() / den) if den else np.nan


STATS = ("coverage", "cov_healthy", "cov_failure", "set_size",
         "singleton", "doubleton", "empty", "alert_rate")


def _group_stats(g, G, w, y, inc_h, inc_f):
    """All STATS for every group of one grouping, as an (len(STATS), G) array."""
    yf = y.astype(float)
    hit = np.where(y == 1, inc_f, inc_h)
    size = inc_h.astype(int) + inc_f.astype(int)
    bc = lambda v: np.bincount(g, w * v, G)      # noqa: E731
    W, Wf = bc(np.ones_like(yf)), bc(yf)
    Wh = W - Wf
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.vstack([
            bc(hit) / W,
            bc((1 - yf) * inc_h) / Wh,
            bc(yf * inc_f) / Wf,
            bc(size) / W,
            bc(size == 1) / W,
            bc(size == 2) / W,
            bc(size == 0) / W,
            bc(inc_f) / W,
        ])


def _drive_codes(d):
    u, inv = np.unique(np.asarray(d), return_inverse=True)
    return inv, len(u)


# =================================================================
# Bootstrap
# =================================================================

def bootstrap(cal, test, methods, B=B_DEFAULT, alpha=ALPHA, seed=SEED,
              with_vendor=True, label=""):
    """
    Drive-clustered bootstrap over calibration AND test drives.

    Returns a long frame: method, grouping, group, stat, point, lo, hi,
    se, n_windows, n_fail_windows, n_nan (draws where the cell had no
    failures, so the statistic was undefined).
    """
    t0 = time.time()
    S = int(max(cal["stage"].max(), test["stage"].max())) + 1
    cp = _Conformal(cal, test, methods, S, alpha)
    y = test["y"].to_numpy().astype(int)
    st = cp.st

    groupings = {"all": (np.zeros(len(y), dtype=int), 1, ["all"]),
                 "stage": (st, S, [str(s) for s in range(S)])}
    if with_vendor and "vendor" in test.columns:
        vend = test["vendor"].to_numpy()
        vlev = sorted(set(vend.tolist()))
        vcode = np.searchsorted(np.array(vlev), vend)
        groupings["vendor"] = (vcode, len(vlev), vlev)
        groupings["vendor_stage"] = (vcode * S + st, len(vlev) * S,
                                     [f"{v}{s}" for v in vlev for s in range(S)])

    # ---- self-check against the sets src/ wrote --------------------
    one_c = np.ones(cal.height)
    for m in methods:
        ih, if_ = cp.sets(m, one_c)
        bad = int((ih != test[f"set_h_{m}"].to_numpy()).sum()
                  + (if_ != test[f"set_f_{m}"].to_numpy()).sum())
        if bad:
            raise AssertionError(
                f"{label} {m}: recomputed sets differ from the dump in "
                f"{bad} entries -- bootstrap would not match the paper")
    print(f"  [{label}] recomputed sets match dump for {len(methods)} methods")

    auc = _AUC(test["p"].to_numpy(), y)

    def one_draw(wc, wt):
        vals = [np.array([auc(wt)])]
        for m in methods:
            ih, if_ = cp.sets(m, wc)
            for (g, G, _) in groupings.values():
                vals.append(_group_stats(g, G, wt, y, ih, if_).ravel())
        return np.concatenate(vals)

    point = one_draw(one_c, np.ones(len(y)))

    rng = np.random.default_rng(seed)
    ci, nc = _drive_codes(cal["drive_idx"].to_numpy())
    ti, nt = _drive_codes(test["drive_idx"].to_numpy())
    draws = np.empty((B, len(point)))
    for b in range(B):
        wc = np.bincount(rng.integers(0, nc, nc), minlength=nc)[ci].astype(float)
        wt = np.bincount(rng.integers(0, nt, nt), minlength=nt)[ti].astype(float)
        draws[b] = one_draw(wc, wt)
        if (b + 1) % 500 == 0:
            print(f"  [{label}] {b + 1}/{B} draws ({time.time() - t0:.0f}s)",
                  flush=True)

    lo = np.nanpercentile(draws, 2.5, axis=0)
    hi = np.nanpercentile(draws, 97.5, axis=0)
    se = np.nanstd(draws, axis=0, ddof=1)
    nnan = np.isnan(draws).sum(axis=0)

    # ---- unpack into a long frame -------------------------------
    rows = [{"method": "-", "grouping": "all", "group": "all", "stat": "auc",
             "n_windows": len(y), "n_fail_windows": int(y.sum())}]
    for m in methods:
        for gname, (g, G, labels) in groupings.items():
            nw = np.bincount(g, minlength=G)
            nf = np.bincount(g, y, minlength=G).astype(int)
            for s in STATS:
                for gi in range(G):
                    rows.append({"method": m, "grouping": gname,
                                 "group": labels[gi], "stat": s,
                                 "n_windows": int(nw[gi]),
                                 "n_fail_windows": int(nf[gi])})
    out = pl.DataFrame(rows).with_columns([
        pl.Series("point", point), pl.Series("lo", lo), pl.Series("hi", hi),
        pl.Series("se", se), pl.Series("n_nan", nnan.astype(int)),
    ])
    print(f"  [{label}] bootstrap done: {B} draws, {nc:,} cal drives, "
          f"{nt:,} test drives, {time.time() - t0:.0f}s")
    return out


# =================================================================
# REV-1: Platt / isotonic
# =================================================================

def _ece(pred, y, bins=15):
    """Equal-mass bins: stable at 1-5% prevalence, unlike equal-width."""
    order = np.argsort(pred, kind="mergesort")
    parts = np.array_split(order, bins)
    return float(sum(len(ix) / len(pred) * abs(pred[ix].mean() - y[ix].mean())
                     for ix in parts if len(ix)))


def _reliability(pred, y, bins=15):
    order = np.argsort(pred, kind="mergesort")
    return [(float(pred[ix].mean()), float(y[ix].mean()), len(ix))
            for ix in np.array_split(order, bins) if len(ix)]


def platt_isotonic(tag="r2p7", alpha=ALPHA):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score

    cal = pl.read_parquet(f"{OUT}/preds_{tag}_cal.parquet")
    te = pl.read_parquet(f"{OUT}/preds_{tag}_test.parquet")
    pc, yc = cal["p"].to_numpy(), cal["y"].to_numpy().astype(int)
    pt, yt = te["p"].to_numpy(), te["y"].to_numpy().astype(int)
    st = te["stage"].to_numpy().astype(int)
    S = int(st.max()) + 1

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pc)[:, None], yc)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pc, yc)
    preds = {
        "raw": (pt, pc),
        "platt": (platt.predict_proba(logit(pt)[:, None])[:, 1],
                  platt.predict_proba(logit(pc)[:, None])[:, 1]),
        "isotonic": (iso.predict(pt), iso.predict(pc)),
    }

    # ---- calibration quality, global and per stage -------------
    qual, rel = [], []
    for name, (q, _) in preds.items():
        qual.append({"tag": tag, "scorer": name, "stage": "all",
                     "brier": float(np.mean((q - yt) ** 2)),
                     "ece": _ece(q, yt),
                     "logloss": float(log_loss(yt, np.clip(q, 1e-6, 1 - 1e-6))),
                     "auc": float(roc_auc_score(yt, q)),
                     "mean_pred": float(q.mean()), "obs_rate": float(yt.mean())})
        for s in range(S):
            m = st == s
            qual.append({"tag": tag, "scorer": name, "stage": str(s),
                         "brier": float(np.mean((q[m] - yt[m]) ** 2)),
                         "ece": _ece(q[m], yt[m]),
                         "logloss": float("nan"), "auc": float("nan"),
                         "mean_pred": float(q[m].mean()),
                         "obs_rate": float(yt[m].mean())})
        for i, (mp, obs, n) in enumerate(_reliability(q, yt)):
            rel.append({"tag": tag, "scorer": name, "bin": i,
                        "mean_pred": mp, "obs_rate": obs, "n": n})

    # ---- decision rules: per-stage failure recall ---------------
    # Rule A: alert budget matched (on test) to Mondrian-both's
    #         fleet-wide alert rate. Favours the baseline.
    # Rule B: one threshold set on CAL for 90% pooled failure recall,
    #         the calibrated-probability analogue of class-Mondrian.
    budget = float(te["set_f_mondrian_both"].to_numpy().mean())
    dec = []

    def _rows(rule, scorer, alert):
        for s in ["all"] + list(range(S)):
            m = np.ones(len(yt), bool) if s == "all" else st == s
            f = m & (yt == 1)
            dec.append({"tag": tag, "rule": rule, "scorer": scorer,
                        "stage": str(s),
                        "fail_recall": float(alert[f].mean()),
                        "alert_rate": float(alert[m].mean()),
                        "n_fail_windows": int(f.sum())})

    for name, (q, qc) in preds.items():
        t = np.quantile(q, 1 - budget)
        _rows("A_matched_budget", name, q >= t)
        tc = np.quantile(qc[yc == 1], alpha)       # 90% of cal failures above
        _rows("B_cal_90pct_recall", name, q >= tc)
    for m in ("split", "mondrian_class", "mondrian_both"):
        _rows("conformal_set_contains_failure", m,
              te[f"set_f_{m}"].to_numpy())

    pl.DataFrame(qual).write_csv(f"{OUT}/platt_quality_{tag}.csv")
    pl.DataFrame(rel).write_csv(f"{OUT}/reliability_{tag}.csv")
    dec_df = pl.DataFrame(dec)
    dec_df.write_csv(f"{OUT}/platt_decision_{tag}.csv")

    print(f"\n=== REV-1 [{tag}] calibration quality (test) ===")
    print(pl.DataFrame(qual).filter(pl.col("stage") == "all")
          .select(["scorer", "brier", "ece", "logloss", "auc",
                   "mean_pred", "obs_rate"]))
    print(f"\n=== REV-1 [{tag}] per-stage failure recall "
          f"(budget matched = {budget:.4f}) ===")
    print(dec_df.pivot(values="fail_recall", index=["rule", "scorer"],
                       on="stage"))
    return dec_df


# =================================================================
# Driver
# =================================================================

def _fmt(r):
    return f"{r['point']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]"


def analyse(B=B_DEFAULT, alpha=ALPHA, zip_name="revision_results.zip"):
    t0 = time.time()
    missing = [t for t in TAGS if not _have(t)] + ([] if _have("wv") else ["wv"])
    if missing:
        print(f"WARNING: missing dumps {missing}; analysing what exists")

    # ---- REV-2 + REV-3: pooled runs --------------------------------
    pooled = []
    for tag in TAGS:
        if not _have(tag):
            continue
        cal = pl.read_parquet(f"{OUT}/preds_{tag}_cal.parquet")
        te = pl.read_parquet(f"{OUT}/preds_{tag}_test.parquet")
        b = bootstrap(cal, te, POOLED_METHODS, B=B, alpha=alpha,
                      label=tag).with_columns(pl.lit(tag).alias("tag"))
        pooled.append(b)
    boot = pl.concat(pooled)
    boot.write_csv(f"{OUT}/boot_pooled.csv")

    # ---- REV-2: within-vendor ------------------------------------
    wv = []
    for v in ("A", "B", "C"):
        pc, pt = f"{OUT}/preds_wv{v}_cal.parquet", f"{OUT}/preds_wv{v}_test.parquet"
        if not (os.path.exists(pc) and os.path.exists(pt)):
            continue
        b = bootstrap(pl.read_parquet(pc), pl.read_parquet(pt), WV_METHODS,
                      B=B, alpha=alpha, with_vendor=False, label=f"wv{v}")
        wv.append(b.with_columns(pl.lit(v).alias("vendor")))
    if wv:
        pl.concat(wv).write_csv(f"{OUT}/boot_wv.csv")

    # ---- REV-1 -----------------------------------------------------
    for tag in TAGS:
        if _have(tag):
            platt_isotonic(tag, alpha)

    # ---- REV-3 summary ----------------------------------------------
    keep = boot.filter(
        ((pl.col("stat") == "auc"))
        | ((pl.col("grouping").is_in(["all", "stage"]))
           & (pl.col("stat").is_in(["coverage", "cov_failure", "set_size"]))
           & (pl.col("method").is_in(["split", "mondrian_both"])))
    )
    keep.write_csv(f"{OUT}/ratio_summary.csv")

    # ---- console summary -------------------------------------------
    def get(df, **kw):
        q = df
        for k, v in kw.items():
            q = q.filter(pl.col(k) == v)
        return q.row(0, named=True) if q.height else None

    print("\n" + "=" * 72)
    print("REV-3  RATIO SENSITIVITY (95% drive-clustered bootstrap CI)")
    print("=" * 72)
    for tag in TAGS:
        sub = boot.filter(pl.col("tag") == tag)
        if not sub.height:
            continue
        a = get(sub, stat="auc")
        sc = get(sub, method="split", grouping="all", stat="coverage")
        sf = get(sub, method="split", grouping="all", stat="cov_failure")
        mf = get(sub, method="mondrian_both", grouping="all", stat="cov_failure")
        print(f"{tag:5s} {RATIO[tag]:.2f}:1  AUC {_fmt(a)}  "
              f"split cov {_fmt(sc)}  split fail {_fmt(sf)}  "
              f"MB fail {_fmt(mf)}")

    sub = boot.filter(pl.col("tag") == "r2p7")
    if sub.height:
        print("\n" + "=" * 72)
        print("TABLE I (r2p7, regenerated): by wear stage")
        print("=" * 72)
        print(f"{'st':>3} {'n':>7} {'nf':>5} {'prev%':>6}  "
              f"{'split cov':>22} {'split fail':>22} {'MB fail':>22} {'MB size':>6}")
        S = int(sub.filter(pl.col("grouping") == "stage")["group"]
                .cast(pl.Int32).max()) + 1
        for s in range(S):
            r = get(sub, method="split", grouping="stage", group=str(s),
                    stat="coverage")
            sf = get(sub, method="split", grouping="stage", group=str(s),
                     stat="cov_failure")
            mf = get(sub, method="mondrian_both", grouping="stage",
                     group=str(s), stat="cov_failure")
            ms = get(sub, method="mondrian_both", grouping="stage",
                     group=str(s), stat="set_size")
            print(f"{s:>3} {r['n_windows']:>7,} {r['n_fail_windows']:>5,} "
                  f"{100 * r['n_fail_windows'] / r['n_windows']:>6.2f}  "
                  f"{_fmt(r):>22} {_fmt(sf):>22} {_fmt(mf):>22} "
                  f"{ms['point']:>6.3f}")

    if wv:
        w = pl.concat(wv)
        print("\n" + "=" * 72)
        print("TABLE III (within-vendor): failure coverage by own-age stage")
        print("=" * 72)
        for v in ("A", "B", "C"):
            sv = w.filter(pl.col("vendor") == v)
            if not sv.height:
                continue
            a = get(sv, stat="auc")
            print(f"vendor {v}  AUC {_fmt(a)}")
            for s in range(5):
                sp = get(sv, method="split", grouping="stage", group=str(s),
                         stat="cov_failure")
                mb = get(sv, method="mondrian_both", grouping="stage",
                         group=str(s), stat="cov_failure")
                if sp is None:
                    continue
                print(f"   stage {s}  nf {sp['n_fail_windows']:>5,}  "
                      f"split {_fmt(sp)}  MB {_fmt(mb)}")

    # ---- bundle ---------------------------------------------------
    files = [f for f in sorted(os.listdir(OUT))
             if f.endswith((".csv", ".json"))
             or (f.startswith("preds_") and f.endswith(".parquet"))]
    with zipfile.ZipFile(f"{OUT}/{zip_name}", "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f"{OUT}/{f}", arcname=f)
    size = os.path.getsize(f"{OUT}/{zip_name}") / 1e6
    print(f"\nwrote {OUT}/{zip_name} ({len(files)} files, {size:.1f} MB)")
    print(f"[analyse done in {time.time() - t0:.0f}s]")


if __name__ == "__main__":
    analyse()
