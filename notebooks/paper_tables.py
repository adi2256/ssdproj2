"""
Generate every results table, figure and in-text number of the paper
from the revision dumps -- nothing is typed in by hand.

    python -m notebooks.paper_tables RESULTS_DIR OUT_DIR [--edges e1,e2,e3,e4]

RESULTS_DIR holds what revision_analysis.analyse() bundled into
revision_results.zip. OUT_DIR receives:

    numbers.tex            \\newcommand macros for every in-text number
    tab_*.tex              table bodies, \\input by the manuscript
    fig_*.pdf              figures
    numbers.json           the same numbers, machine-readable

--edges are the fleet-wide wear-stage edges (power-on hours) printed by
run_ratio("r2p7") as `stages: [...]`; they are only needed for the
age-overlap figure, which shows where those edges fall.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import polars as pl

from notebooks.revision_analysis import _Conformal

MAIN = "r2p7"
METHOD_LABEL = {"split": "Split", "mondrian_class": "Mond.\\ class",
                "mondrian_stage": "Mond.\\ stage",
                "mondrian_both": "Mond.\\ cls$\\times$stg"}
MIN_FAIL = 30          # cells below this are greyed and not interpreted


# =================================================================
# Formatting
# =================================================================

def f3(x):
    return "---" if x is None or (isinstance(x, float) and math.isnan(x)) \
        else f"{x:.3f}"


def nz(x):
    """.917 rather than 0.917, for compact intervals."""
    s = f"{x:.3f}"
    return s[1:] if s.startswith("0.") else s


def _n(x):
    return f"{int(x):,}".replace(",", "{,}")


def ci(lo, hi):
    if any(v is None or math.isnan(v) for v in (lo, hi)):
        return ""
    return f"{{\\scriptsize[{nz(lo)},{nz(hi)}]}}"


def macro_name(key):
    """numbers.tex macro names must be letters only."""
    out, up = [], False
    for ch in key:
        if ch.isalpha():
            out.append(ch.upper() if up else ch)
            up = False
        elif ch.isdigit():
            out.append("ABCDEFGHIJ"[int(ch)])
            up = False
        else:
            up = True
    return "N" + "".join(out)


# =================================================================
# Loading
# =================================================================

class Results:
    def __init__(self, d):
        self.d = d
        self.boot = pl.read_csv(f"{d}/boot_pooled.csv",
                                schema_overrides={"group": pl.Utf8})
        self.wv = pl.read_csv(f"{d}/boot_wv.csv",
                              schema_overrides={"group": pl.Utf8})
        self.meta = {t: json.load(open(f"{d}/meta_{t}.json"))
                     for t in ("r1", "r2p7", "r5")}
        self.meta_wv = json.load(open(f"{d}/meta_wv.json"))

    def pq(self, name):
        return pl.read_parquet(f"{self.d}/{name}.parquet")

    def get(self, df=None, **kw):
        q = self.boot if df is None else df
        for k, v in kw.items():
            q = q.filter(pl.col(k) == v)
        if q.height != 1:
            raise KeyError(f"{kw}: {q.height} rows")
        return q.row(0, named=True)

    def b(self, tag=MAIN, method="-", grouping="all", group="all",
          stat="auc"):
        return self.get(tag=tag, method=method, grouping=grouping,
                        group=str(group), stat=stat)


# =================================================================
# Tables
# =================================================================

def tab_main(R, N):
    """Table I: standard split by fleet-wide wear quintile."""
    rows = []
    for s in range(5):
        cov = R.b(method="split", grouping="stage", group=s, stat="coverage")
        sf = R.b(method="split", grouping="stage", group=s, stat="cov_failure")
        ss = R.b(method="split", grouping="stage", group=s, stat="set_size")
        cf = R.b(method="mondrian_class", grouping="stage", group=s,
                 stat="cov_failure")
        cs = R.b(method="mondrian_class", grouping="stage", group=s,
                 stat="set_size")
        bf = R.b(method="mondrian_both", grouping="stage", group=s,
                 stat="cov_failure")
        bs = R.b(method="mondrian_both", grouping="stage", group=s,
                 stat="set_size")
        prev = 100 * cov["n_fail_windows"] / cov["n_windows"]
        rows.append(
            f"{s} & {prev:.2f}\\% & {_n(cov['n_fail_windows'])} & "
            f"{f3(cov['point'])} {ci(cov['lo'], cov['hi'])} & "
            f"{f3(sf['point'])} {ci(sf['lo'], sf['hi'])} & {f3(ss['point'])} & "
            f"{f3(cf['point'])} {ci(cf['lo'], cf['hi'])} & {f3(cs['point'])} & "
            f"{f3(bf['point'])} {ci(bf['lo'], bf['hi'])} & {f3(bs['point'])} \\\\")
        N[f"prev_s{s}"] = f"{prev:.2f}"
        N[f"nfail_s{s}"] = f"{cov['n_fail_windows']:,}".replace(",", "{,}")
        N[f"nwin_s{s}"] = f"{cov['n_windows']:,}".replace(",", "{,}")
        for k, r in (("splitcov", cov), ("splitfail", sf), ("classfail", cf),
                     ("bothfail", bf)):
            N[f"{k}_s{s}"] = f3(r["point"])
            N[f"{k}_s{s}_lo"] = f3(r["lo"])
            N[f"{k}_s{s}_hi"] = f3(r["hi"])
        N[f"splitsize_s{s}"] = f3(ss["point"])
        N[f"classsize_s{s}"] = f3(cs["point"])
        N[f"bothsize_s{s}"] = f3(bs["point"])
    return "\n".join(rows)


def tab_vendorstage(R, N):
    """Table II: split failure coverage by vendor x fleet-wide stage."""
    out = []
    for v in ("A", "B", "C"):
        top, bot = [], []
        for s in range(5):
            r = R.b(method="split", grouping="vendor_stage", group=f"{v}{s}",
                    stat="cov_failure")
            n = r["n_fail_windows"]
            if r["n_windows"] == 0 or n == 0:
                top.append("---" if r["n_windows"] == 0 else "\\grey{(0)}")
                bot.append("")
            elif n < MIN_FAIL:
                top.append(f"\\grey{{{f3(r['point'])}\\,({n})}}")
                bot.append("")
            else:
                top.append(f"{f3(r['point'])}\\,({n})")
                bot.append(ci(r["lo"], r["hi"]))
            N[f"vs_{v}{s}_fail"] = f3(r["point"])
            N[f"vs_{v}{s}_n"] = str(n)
        out.append(f"{v} & " + " & ".join(top) + " \\\\")
        out.append(" & " + " & ".join(bot) + " \\\\")
    return "\n".join(out)


def tab_within(R, N):
    """Table III: within-vendor audit, vendor-relative quintiles."""
    out = []
    for v in ("A", "B", "C"):
        sub = R.wv.filter(pl.col("vendor") == v)
        g = lambda **kw: R.get(sub, **kw)                    # noqa: E731
        auc = g(method="-", stat="auc")
        l1, l2, l3 = [], [], []
        for s in range(5):
            mc = g(method="split", grouping="stage", group=str(s),
                   stat="coverage")
            fc = g(method="split", grouping="stage", group=str(s),
                   stat="cov_failure")
            n = fc["n_fail_windows"]
            grey = n < MIN_FAIL
            wrap = (lambda t: f"\\grey{{{t}}}") if grey else (lambda t: t)
            l1.append(wrap(f"${f3(mc['point'])}$"))
            l2.append(wrap(f"${nz(fc['point'])}\\,({n})$"))
            l3.append("" if grey else ci(fc["lo"], fc["hi"]))
            N[f"wv_{v}{s}_marg"] = f3(mc["point"])
            N[f"wv_{v}{s}_fail"] = f3(fc["point"])
            N[f"wv_{v}{s}_lo"] = f3(fc["lo"])
            N[f"wv_{v}{s}_hi"] = f3(fc["hi"])
            N[f"wv_{v}{s}_n"] = str(n)
            mb = g(method="mondrian_both", grouping="stage", group=str(s),
                   stat="cov_failure")
            N[f"wv_{v}{s}_mb"] = f3(mb["point"])
        N[f"wv_{v}_auc"] = f3(auc["point"])
        N[f"wv_{v}_auc_lo"] = f3(auc["lo"])
        N[f"wv_{v}_auc_hi"] = f3(auc["hi"])
        out.append(f"\\multirow{{3}}{{*}}{{{v}}} & " + " & ".join(l1)
                   + f" & \\multirow{{3}}{{*}}{{\\shortstack{{{f3(auc['point'])}"
                   f"\\\\{ci(auc['lo'], auc['hi'])}}}}} \\\\")
        out.append(" & " + " & ".join(l2) + " & \\\\")
        out.append(" & " + " & ".join(l3) + " & \\\\")
        if v != "C":
            out.append("\\addlinespace[2pt]")
    return "\n".join(out)


def tab_cells(R, N):
    """Table IV: Mondrian class x stage cell sizes, counted exactly."""
    cal, te = R.pq(f"preds_{MAIN}_cal"), R.pq(f"preds_{MAIN}_test")
    rows, tot = [], np.zeros(4, dtype=int)
    for s in range(5):
        c = cal.filter(pl.col("stage") == s)
        t = te.filter(pl.col("stage") == s)
        v = [int((c["y"] == 0).sum()), int((c["y"] == 1).sum()),
             int((t["y"] == 0).sum()), int((t["y"] == 1).sum())]
        tot += v
        rows.append(f"{s} & " + " & ".join(f"{x:,}".replace(",", "{,}")
                                          for x in v) + " \\\\")
        N[f"calfail_s{s}"] = f"{v[1]:,}".replace(",", "{,}")
    rows.append("\\midrule")
    rows.append("Total & " + " & ".join(f"{x:,}".replace(",", "{,}")
                                       for x in tot) + " \\\\")
    cf = [int(((cal["stage"] == s) & (cal["y"] == 1)).sum()) for s in range(5)]
    N["calfail_min"] = f"{min(cf):,}".replace(",", "{,}")
    N["calfail_max"] = f"{max(cf):,}".replace(",", "{,}")
    N["ncal_win"] = f"{cal.height:,}".replace(",", "{,}")
    N["ntest_win"] = f"{te.height:,}".replace(",", "{,}")
    N["ntest_fail"] = f"{int(te['y'].sum()):,}".replace(",", "{,}")
    return "\n".join(rows)


def tab_composition(R, N):
    """Table V: prediction-set composition."""
    want = [("split", s) for s in range(5)] + [
        ("mondrian_class", 4), ("mondrian_stage", 4),
        ("mondrian_both", 0), ("mondrian_both", 4)]
    rows, last = [], None
    for m, s in want:
        g = lambda st: R.b(method=m, grouping="stage", group=s, stat=st)  # noqa
        sing, doub, emp, size = (g("singleton")["point"],
                                 g("doubleton")["point"],
                                 g("empty")["point"], g("set_size")["point"])
        lab = METHOD_LABEL[m] if m != last else ""
        if m != last and last is not None:
            rows.append("\\addlinespace[1.5pt]")
        last = m
        rows.append(f"{lab} & {s} & {100*sing:.1f} & {100*doub:.1f} & "
                    f"{100*emp:.1f} & {size:.3f} \\\\")
        N[f"comp_{m}_s{s}_sing"] = f"{100*sing:.1f}"
        N[f"comp_{m}_s{s}_doub"] = f"{100*doub:.1f}"
        N[f"comp_{m}_s{s}_empty"] = f"{100*emp:.1f}"
        N[f"comp_{m}_s{s}_size"] = f"{size:.3f}"
    return "\n".join(rows)


def tab_platt(R, N):
    """Table VI (new): conventional calibration vs conformal sets."""
    q = pl.read_csv(f"{R.d}/platt_quality_{MAIN}.csv",
                    schema_overrides={"stage": pl.Utf8})
    d = pl.read_csv(f"{R.d}/platt_decision_{MAIN}.csv",
                    schema_overrides={"stage": pl.Utf8})

    def rec(rule, scorer, stage):
        return R.get(d, rule=rule, scorer=scorer, stage=str(stage))

    rows = []
    order = [("raw", "Raw RF"), ("platt", "Platt"),
             ("isotonic", "Isotonic")]
    for sc, lab in order:
        qq = R.get(q, scorer=sc, stage="all")
        N[f"pl_{sc}_brier"] = f"{qq['brier']:.4f}"
        N[f"pl_{sc}_ece"] = f"{qq['ece']:.4f}"
        N[f"pl_{sc}_ece_s"] = (f"{qq['ece']:.2f}" if qq["ece"] >= 0.1
                               else f"{qq['ece']:.3f}")
        N[f"pl_{sc}_meanpred"] = f"{qq['mean_pred']:.3f}"
        for rule, rl in (("A_matched_budget", "A"),
                         ("B_cal_90pct_recall", "B")):
            a = rec(rule, sc, "all")
            r4 = rec(rule, sc, 4)
            cells = [f"{rec(rule, sc, s)['fail_recall']:.3f}" for s in range(5)]
            rows.append(
                f"{lab if rl == 'A' else ''} & {rl} & {qq['ece']:.4f} & "
                f"{a['alert_rate']:.3f} & {a['fail_recall']:.3f} & "
                + " & ".join(cells) + " \\\\")
            N[f"pl_{sc}_{rl}_alert"] = f"{a['alert_rate']:.3f}"
            N[f"pl_{sc}_{rl}_recall"] = f"{a['fail_recall']:.3f}"
            N[f"pl_{sc}_{rl}_s4"] = f"{r4['fail_recall']:.3f}"
            for s in range(5):
                N[f"pl_{sc}_{rl}_s{s}"] = f"{rec(rule, sc, s)['fail_recall']:.3f}"
        rows.append("\\addlinespace[1.5pt]")
    for m in ("split", "mondrian_class", "mondrian_both"):
        a = rec("conformal_set_contains_failure", m, "all")
        cells = [f"{rec('conformal_set_contains_failure', m, s)['fail_recall']:.3f}"
                 for s in range(5)]
        rows.append(f"{METHOD_LABEL[m]} & --- & --- & "
                    f"{a['alert_rate']:.3f} & {a['fail_recall']:.3f} & "
                    + " & ".join(cells) + " \\\\")
        N[f"cp_{m}_alert"] = f"{a['alert_rate']:.3f}"
    N["pl_max_s4"] = max((N[f"pl_{sc}_{rl}_s4"] for sc in ("platt", "isotonic")
                          for rl in ("A", "B")), key=float)
    # per-stage calibration-in-the-large after Platt
    for s in range(5):
        r = R.get(q, scorer="platt", stage=str(s))
        N[f"pl_platt_s{s}_mean"] = f"{100*r['mean_pred']:.2f}"
        N[f"pl_platt_s{s}_obs"] = f"{100*r['obs_rate']:.2f}"
        N[f"pl_platt_s{s}_ece"] = f"{r['ece']:.4f}"
    eces = [R.get(q, scorer="platt", stage=str(s))["ece"] for s in range(5)]
    N["pl_platt_stage_ece_max"] = f"{max(eces):.4f}"
    return "\n".join(rows)


def tab_ratio(R, N):
    """Table VII (new): healthy-drive sampling ratio."""
    rows = []
    lab = {"r1": "1:1", "r2p7": "2.68:1", "r5": "5:1"}
    for t in ("r1", "r2p7", "r5"):
        m = R.meta[t]
        g = lambda **kw: R.b(tag=t, **kw)                     # noqa: E731
        auc = g()
        sc = g(method="split", stat="coverage")
        sf = g(method="split", stat="cov_failure")
        s4 = g(method="split", grouping="stage", group=4, stat="cov_failure")
        bf = g(method="mondrian_both", stat="cov_failure")
        b4 = g(method="mondrian_both", grouping="stage", group=4,
               stat="cov_failure")
        bs = g(method="mondrian_both", stat="set_size")
        rows.append(
            f"{lab[t]} & {_n(m['n_drives'])} & {_n(m['n_windows_test'])} & "
            f"{_n(m['n_fail_windows_test'])} & {100*m['test_prevalence']:.2f}\\% & "
            f"{f3(auc['point'])} {ci(auc['lo'], auc['hi'])} & "
            f"{f3(sc['point'])} & {f3(sf['point'])} & "
            f"{f3(s4['point'])} {ci(s4['lo'], s4['hi'])} & "
            f"{f3(bf['point'])} & {f3(b4['point'])} {ci(b4['lo'], b4['hi'])} & "
            f"{f3(bs['point'])} \\\\")
        for k, r in (("auc", auc), ("splitcov", sc), ("splitfail", sf),
                     ("split4", s4), ("bothfail", bf), ("both4", b4),
                     ("bothsize", bs)):
            N[f"ratio_{t}_{k}"] = f3(r["point"])
            N[f"ratio_{t}_{k}_lo"] = f3(r["lo"])
            N[f"ratio_{t}_{k}_hi"] = f3(r["hi"])
        N[f"ratio_{t}_prev"] = f"{100*m['test_prevalence']:.2f}"
        f4 = R.pq(f"preds_{t}_test").filter((pl.col("stage") == 4)
                                            & (pl.col("y") == 1))
        N[f"ratio_{t}_s4A"] = f"{100 * (f4['vendor'] == 'A').mean():.1f}"
    return "\n".join(rows)


# =================================================================
# Derived statistics
# =================================================================

def confound_stats(R, N):
    """Vendor x stage association on test failure windows."""
    te = R.pq(f"preds_{MAIN}_test").filter(pl.col("y") == 1)
    vs = sorted(te["vendor"].unique().to_list())
    tab = np.array([[int(((te["vendor"] == v) & (te["stage"] == s)).sum())
                     for s in range(5)] for v in vs], dtype=float)
    n = tab.sum()
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / n
    chi2 = float(((tab - exp) ** 2 / np.where(exp > 0, exp, 1)).sum())
    dof = (tab.shape[0] - 1) * (tab.shape[1] - 1)
    V = math.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    pj = tab / n
    pv, ps = pj.sum(1), pj.sum(0)
    nzm = pj > 0
    mi = float((pj[nzm] * np.log(pj[nzm] / np.outer(pv, ps)[nzm])).sum())
    hv = float(-(pv[pv > 0] * np.log(pv[pv > 0])).sum())
    N.update({
        "conf_n": f"{int(n):,}".replace(",", "{,}"),
        "conf_chi": f"{chi2:.1f}", "conf_dof": str(dof),
        "conf_V": f"{V:.3f}", "conf_Vtwo": f"{V:.2f}",
        "conf_mi": f"{mi:.3f}", "conf_hv": f"{hv:.3f}",
        "conf_frac": f"{100 * mi / hv:.1f}",
    })
    # stage purity
    for s in range(5):
        col = tab[:, s]
        if col.sum():
            k = int(col.argmax())
            N[f"conf_s{s}_top"] = vs[k]
            N[f"conf_s{s}_toppct"] = f"{100 * col[k] / col.sum():.1f}"
            for i, v in enumerate(vs):
                N[f"conf_{v}{s}"] = str(int(col[i]))
    fleet = tab.sum(1) / n
    worst = min(np.abs(tab[:, s] / tab[:, s].sum() - fleet).max()
                for s in range(5) if tab[:, s].sum())
    N["conf_min_dev_pp"] = f"{100 * worst:.0f}"
    return tab, vs


def per_stage_auc(R, N):
    from sklearn.metrics import roc_auc_score
    te = R.pq(f"preds_{MAIN}_test")
    for s in range(5):
        t = te.filter(pl.col("stage") == s)
        N[f"auc_s{s}"] = f"{roc_auc_score(t['y'], t['p']):.3f}"


def operating_points(R, N):
    """Section V-F: threshold on training scores maximising F0.5."""
    from src.evaluation.metrics import pick_threshold
    tr, te = R.pq(f"preds_{MAIN}_train"), R.pq(f"preds_{MAIN}_test")
    thr = pick_threshold(tr["p"].to_numpy(), tr["y"].to_numpy(), beta=0.5)
    N["op_thr"] = f"{thr:.3f}"
    for s in range(5):
        t = te.filter(pl.col("stage") == s)
        pred = t["p"].to_numpy() >= thr
        y = t["y"].to_numpy()
        d = t["drive_idx"].to_numpy()
        rec = pred[y == 1].mean() if (y == 1).any() else float("nan")
        n_dr = len(np.unique(d))
        al = len(np.unique(d[pred]))
        N[f"op_s{s}_recall"] = f"{rec:.3f}"
        N[f"op_s{s}_alerts"] = f"{1000 * al / max(n_dr, 1):.1f}"


def alpha_sweep(R, N, out):
    """Fig. 2: split conformal failure coverage - target, by stage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal, te = R.pq(f"preds_{MAIN}_cal"), R.pq(f"preds_{MAIN}_test")
    alphas = [0.02, 0.05, 0.10, 0.20, 0.30]
    y = te["y"].to_numpy()
    st = te["stage"].to_numpy()
    gaps = {s: [] for s in range(5)}
    marg = []
    for a in alphas:
        cp = _Conformal(cal, te, ("split",), 5, a)
        ih, if_ = cp.sets("split", np.ones(cal.height))
        hit = np.where(y == 1, if_, ih)
        for s in range(5):
            m = (st == s) & (y == 1)
            gaps[s].append(float(if_[m].mean()) - (1 - a))
            if s < 4:                       # youngest four quintiles
                mm = st == s
                marg.append(float(hit[mm].mean()) - (1 - a))
    N["sweep_s4_min"] = f"{-max(gaps[4]):.2f}"
    N["sweep_s4_max"] = f"{-min(gaps[4]):.2f}"
    young = [g for s in range(4) for g in gaps[s]]
    N["sweep_young_absmax"] = f"{max(abs(g) for g in young):.2f}"
    N["sweep_young_marg_absmax"] = f"{max(abs(g) for g in marg):.2f}"
    N["sweep_young_marg_min"] = f"{min(marg):+.3f}"   # most under target
    N["sweep_young_marg_max"] = f"{max(marg):+.3f}"
    N["sweep_young_marg_under"] = f"{max(0.0, -min(marg)):.3f}"
    # match the existing figure's look
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for s in range(5):
        ax.plot(alphas, gaps[s], "o-", ms=3, lw=1.2, color=cols[s],
                label=f"stage {s}" + (" (oldest)" if s == 4 else ""))
    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:.2f}" for a in alphas])
    ax.minorticks_off()
    ax.set_xlabel(r"nominal miscoverage $\alpha$", fontsize=7)
    ax.set_ylabel("failure-class coverage $-$ target", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=5.5, ncol=2, frameon=False, loc="center right")
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{out}/fig_reliability.pdf")
    plt.close(fig)


def fig_setsize(R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    cols = {"split": "#1f77b4", "mondrian_class": "#ff7f0e",
            "mondrian_stage": "#2ca02c", "mondrian_both": "#d62728"}
    names = {"split": "Split", "mondrian_class": "Mondrian (class)",
             "mondrian_stage": "Mondrian (stage)",
             "mondrian_both": r"Mondrian (class$\times$stage)"}
    for m in cols:
        f = R.b(method=m, grouping="stage", group=4, stat="cov_failure")
        s = R.b(method=m, grouping="stage", group=4, stat="set_size")
        ax.errorbar(s["point"], f["point"],
                    yerr=[[f["point"] - f["lo"]], [f["hi"] - f["point"]]],
                    xerr=[[s["point"] - s["lo"]], [s["hi"] - s["point"]]],
                    fmt="o", ms=4, capsize=2, color=cols[m], label=names[m])
    ax.axhline(0.9, color="k", ls="--", lw=0.8)
    ax.text(0.82, 0.92, r"target $1-\alpha=0.90$", fontsize=5.5)
    ax.axvline(2.0, color="grey", ls=":", lw=0.8)
    ax.text(1.80, 1.02, "max set size", fontsize=5.5, color="grey")
    ax.set_xlim(0.8, 2.05)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("average prediction-set size", fontsize=7)
    ax.set_ylabel("failure-class coverage\n(oldest quintile)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{out}/fig_setsize.pdf")
    plt.close(fig)


def fig_within(R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    cols = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    off = {"A": -0.08, "B": 0.0, "C": 0.08}
    for v in cols:
        sub = R.wv.filter(pl.col("vendor") == v)
        xs, ys, lo, hi, ok = [], [], [], [], []
        for s in range(5):
            r = R.get(sub, method="split", grouping="stage", group=str(s),
                      stat="cov_failure")
            xs.append(s + off[v]); ys.append(r["point"])
            lo.append(r["point"] - r["lo"]); hi.append(r["hi"] - r["point"])
            ok.append(r["n_fail_windows"] >= MIN_FAIL)
        ax.plot(xs, ys, "-", lw=1.1, color=cols[v], label=f"vendor {v}")
        for x, yv, l, h, k in zip(xs, ys, lo, hi, ok):
            ax.errorbar(x, yv, yerr=[[l], [h]], fmt="o", ms=3.5, capsize=1.5,
                        lw=0.8, color=cols[v],
                        mfc=cols[v] if k else "white")
    ax.axhline(0.9, color="k", ls="--", lw=0.8)
    ax.text(-0.15, 0.85, r"target $1-\alpha=0.90$", fontsize=5.5)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(range(5))
    ax.set_xlabel("vendor-relative wear quintile", fontsize=7)
    ax.set_ylabel("failure-class coverage", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=3)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{out}/fig_within_vendor.pdf")
    plt.close(fig)


def fig_age(R, N, out, edges):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ov = pl.read_csv(f"{R.d}/age_overlap.csv")
    cols = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    for r in ov.iter_rows(named=True):
        v = r["vendor"]
        for k in ("p05", "p25", "median", "p75", "p95"):
            N[f"age_{v}_{k}"] = f"{int(r[k]):,}".replace(",", "{,}")
    if not edges:
        print("  (no --edges: age-overlap figure not regenerated)")
        return
    fig, ax = plt.subplots(figsize=(3.4, 1.35))
    for i, v in enumerate(("C", "B", "A")):
        r = R.get(ov, vendor=v)
        ax.plot([r["p05"], r["p95"]], [i, i], color="k", lw=0.8)
        ax.barh(i, r["p75"] - r["p25"], left=r["p25"], height=0.45,
                color=cols[v])
        ax.plot([r["median"]] * 2, [i - 0.22, i + 0.22], color="k", lw=1)
    for e in edges:
        ax.axvline(e, color="k", ls=":", lw=0.7)
    ax.text(edges[-1], 2.45, "fleet-wide stage-4 edge", fontsize=5,
            color="grey", ha="left")
    ax.set_yticks(range(3))
    ax.set_yticklabels(["vendor C", "vendor B", "vendor A"], fontsize=6)
    ax.set_ylim(-0.5, 2.7)
    ax.set_xlabel("power-on hours at window end", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{out}/fig_age_overlap.pdf")
    plt.close(fig)
    for i, e in enumerate(edges):
        N[f"edge{i + 1}"] = f"{int(e):,}".replace(",", "{,}")


# =================================================================

def main(res_dir, out, edges=None):
    os.makedirs(out, exist_ok=True)
    R = Results(res_dir)
    N = {}
    tables = {
        "tab_main": tab_main(R, N),
        "tab_vendorstage": tab_vendorstage(R, N),
        "tab_within": tab_within(R, N),
        "tab_cells": tab_cells(R, N),
        "tab_composition": tab_composition(R, N),
        "tab_platt": tab_platt(R, N),
        "tab_ratio": tab_ratio(R, N),
    }
    for k, v in tables.items():
        with open(f"{out}/{k}.tex", "w") as fh:
            fh.write(v + "\n")
    confound_stats(R, N)
    per_stage_auc(R, N)
    operating_points(R, N)
    alpha_sweep(R, N, out)
    fig_setsize(R, out)
    fig_within(R, out)
    fig_age(R, N, out, edges)

    a = R.b()
    N.update({"auc": f3(a["point"]), "auc_lo": f3(a["lo"]),
              "auc_hi": f3(a["hi"]),
              "auctwo": f"{a['point']:.2f}"})
    for m in ("split", "mondrian_class", "mondrian_stage", "mondrian_both"):
        for st in ("coverage", "cov_failure", "set_size", "alert_rate"):
            r = R.b(method=m, stat=st)
            N[f"all_{m}_{st}"] = f3(r["point"])
            N[f"all_{m}_{st}_lo"] = f3(r["lo"])
            N[f"all_{m}_{st}_hi"] = f3(r["hi"])
    # within-vendor ranges used in the abstract / text
    wv_fail = [float(N[f"wv_{v}{s}_fail"]) for v in "ABC" for s in range(5)]
    wv_hi = [float(N[f"wv_{v}{s}_hi"]) for v in "ABC" for s in range(5)]
    wv_marg = [float(N[f"wv_{v}{s}_marg"]) for v in "ABC" for s in range(5)]
    wv_mb = [float(N[f"wv_{v}{s}_mb"]) for v in "ABC" for s in range(5)]
    N.update({"wv_fail_min": f"{min(wv_fail):.2f}",
              "wv_fail_max": f"{max(wv_fail):.2f}",
              "wv_hi_max": f"{max(wv_hi):.3f}",
              "wv_marg_min": f"{min(wv_marg):.2f}",
              "wv_marg_max": f"{max(wv_marg):.2f}",
              "wv_mb_min": f"{min(wv_mb):.2f}",
              "wv_mb_max": f"{max(wv_mb):.2f}"})
    # ranges quoted in the text
    young = range(4)
    for k in ("splitfail", "splitsize", "bothsize", "classfail"):
        vals = [float(N[f"{k}_s{s}"]) for s in young]
        N[f"young_{k}_min"] = f"{min(vals):.3f}"
        N[f"young_{k}_max"] = f"{max(vals):.3f}"
    cov = [float(N[f"splitcov_s{s}"]) for s in range(5)]
    N["splitcov_min"], N["splitcov_max"] = f"{min(cov):.3f}", f"{max(cov):.3f}"
    st4 = R.b(method="mondrian_stage", grouping="stage", group=4,
              stat="cov_failure")
    N["stagefail_s4"] = f3(st4["point"])
    N["both_s4_pctmax"] = f"{100 * float(N['bothsize_s4']) / 2:.0f}"
    relA = [float(N[f"wv_A{s}_fail"]) for s in range(5)
            if int(N[f"wv_A{s}_n"]) >= MIN_FAIL]
    N["wv_A_young"] = str(min(s for s in range(5)
                              if int(N[f"wv_A{s}_n"]) >= MIN_FAIL))
    for k in ("fail", "lo", "hi", "n"):
        N[f"wv_Ay_{k}"] = N[f"wv_A{N['wv_A_young']}_{k}"]
    N["wv_A_rel_min"], N["wv_A_rel_max"] = f"{min(relA):.2f}", f"{max(relA):.2f}"
    sizes = []
    for v in "ABC":
        sub = R.wv.filter(pl.col("vendor") == v)
        for s in range(5):
            sizes.append(R.get(sub, method="mondrian_both", grouping="stage",
                               group=str(s), stat="set_size")["point"])
        mv = R.meta_wv["vendors"][v]
        N[f"wv_{v}_ntrain"] = _n(mv["n_windows_train"])
        N[f"wv_{v}_ncal"] = _n(mv["n_windows_cal"])
        N[f"wv_{v}_ntest"] = _n(mv["n_windows_test"])
    N["wv_mbsize_min"], N["wv_mbsize_max"] = f"{min(sizes):.2f}", f"{max(sizes):.2f}"

    # direction claims the text makes -- checked, not assumed
    def sep(v, a, b):
        """interval of quintile a entirely below interval of quintile b"""
        return float(N[f"wv_{v}{a}_hi"]) < float(N[f"wv_{v}{b}_lo"])
    checks = {
        # youngest quintile with >= MIN_FAIL failures vs the oldest
        "A_old_above_young": sep("A", min(s for s in range(5)
                                          if int(N[f"wv_A{s}_n"]) >= MIN_FAIL), 4),
        "B_old_below_young": sep("B", 4, 0),
        "C_old_below_young": sep("C", 4, 0),
        "wv_no_cell_reaches_target": float(N["wv_hi_max"]) < 0.90,
        "marginal_within_0.09_everywhere": max(
            [abs(float(N[f"wv_{v}{s}_marg"]) - 0.9) for v in "ABC" for s in range(5)]
            + [abs(R.b(tag=t, method="split", grouping="stage", group=s,
                       stat="coverage")["point"] - 0.9)
               for t in ("r1", "r2p7", "r5") for s in range(5)]) <= 0.09,
        "every_stage_marginal_ci_reaches_target": all(
            float(N[f"splitcov_s{s}_hi"]) >= 0.9 for s in range(5)),
        "both_s4_ci_covers_target": float(N["bothfail_s4_lo"]) <= 0.90
                                     <= float(N["bothfail_s4_hi"]),
    }
    N["checks"] = json.dumps(checks)
    print("claim checks:", checks)

    # binomial-vs-bootstrap width on the key cell
    f4 = R.b(method="split", grouping="stage", group=4, stat="cov_failure")
    p, n = f4["point"], f4["n_fail_windows"]
    binom = 1.96 * math.sqrt(p * (1 - p) / n)
    N["s4_boot_halfwidth"] = f"{(f4['hi'] - f4['lo']) / 2:.3f}"
    N["s4_binom_halfwidth"] = f"{binom:.3f}"
    N["s4_width_ratio"] = f"{(f4['hi'] - f4['lo']) / 2 / binom:.1f}"

    with open(f"{out}/numbers.tex", "w") as fh:
        fh.write("% generated by notebooks/paper_tables.py -- do not edit\n"
                 "% use as \\R{key}; an unknown key is a hard error\n"
                 "\\providecommand{\\R}[1]{\\ifcsname R:#1\\endcsname"
                 "\\csname R:#1\\endcsname\\else"
                 "\\errmessage{unknown result key #1}\\fi}\n")
        for k in sorted(N):
            if k == "checks":
                continue
            fh.write(f"\\expandafter\\def\\csname R:{k}\\endcsname{{{N[k]}}}\n")
    with open(f"{out}/numbers.json", "w") as fh:
        json.dump(N, fh, indent=1, sort_keys=True)
    names = [macro_name(k) for k in N]
    assert len(set(names)) == len(names), "macro name collision"
    print(f"wrote {len(tables)} tables, {len(N)} numbers, figures -> {out}")
    return N


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--edges", default="")
    a = ap.parse_args()
    edges = [float(e) for e in a.edges.split(",") if e]
    if not edges:   # newer dumps record them in meta
        edges = json.load(open(f"{a.results}/meta_{MAIN}.json")).get(
            "stage_edges") or None
    main(a.results, a.out, edges)
