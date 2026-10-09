"""
Revision batch: every remaining experiment, re-run on the fixed sampler.

Closes the open items of the reviewer response:

    main     deterministic re-run of r2p7 / r1 / r5 / within-vendor dumps
    analyse  bootstrap + Platt/isotonic on those dumps
    models   within-DRIVE-MODEL re-audit (6 populations instead of 3)
    robust   stage hold-out, binning/change-point, prevalence control,
             classifier families (validate_wear V2 V3 V5 V6)
    repr     held-out-vendor representation sweep (Fig. 6)
    lomm     held-out-drive-model sweep
    shift    conformal under vendor shift: split / Mondrian / weighted
    seeds    the main configuration at 10 seeds -- retraining variability

    from notebooks.revision_batch import run_batch
    run_batch(C.load_slim)

RESUMABLE. Each step writes `done_<step>` to the output directory when it
finishes and is skipped next time; `seeds` also resumes per seed. If the
kernel dies, restart, re-run setup, call run_batch again.

At the end everything is zipped to batch_results.zip, together with a
pip freeze of the environment so the paper can cite exact versions.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import time
import zipfile

import numpy as np
import polars as pl

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
SEED = 42
N_DRIVES = 60_000
ALPHA = 0.10
SEEDS = (42, 0, 1, 2, 3, 4, 5, 6, 7, 8)
STEPS = ("main", "analyse", "models", "robust", "repr", "lomm", "shift",
         "seeds")


def _done(step):
    return os.path.exists(f"{OUT}/done_{step}")


def _mark(step, t0):
    with open(f"{OUT}/done_{step}", "w") as fh:
        fh.write(f"{time.time() - t0:.0f}s\n")
    print(f"\n[{step} done in {time.time() - t0:.0f}s]\n", flush=True)


def _load(load_slim, seed=SEED, n_drives=N_DRIVES):
    from src.data.labels import build_labels
    raw = load_slim(n_drives=n_drives, keep_all_failed=True, seed=seed)
    labelled, stats = build_labels(raw)
    print(stats)
    del raw
    gc.collect()
    return labelled


def _write(df, name):
    df.write_csv(f"{OUT}/{name}")
    print(f"  wrote {name} ({df.height} rows)")


# =================================================================
# models: within-drive-model re-audit
# =================================================================

def build_group(labelled, col, value, n_stages=5, seed=SEED):
    """build_vendor, generalised to any population column."""
    from sklearn.ensemble import RandomForestClassifier
    from notebooks import within_vendor_wear as W

    sub = labelled.filter(pl.col(col) == value)
    sp = W.make_standard_split(sub, seed=seed)
    raw = {p: W.make_windows(sp.apply(sub, p), stride=W.STRIDE)
           for p in ("train", "cal", "test")}
    if min(len(w) for w in raw.values()) == 0:
        return None
    if raw["cal"].y.sum() < 20 or raw["test"].y.sum() < 40:
        print(f"  {value}: too few positives (cal {int(raw['cal'].y.sum())},"
              f" test {int(raw['test'].y.sum())}) -- skipped")
        return None
    if len(raw["train"]) > W.MAX_TRAIN:
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(raw["train"].y == 1)
        neg = np.flatnonzero(raw["train"].y == 0)
        keep = np.sort(np.concatenate(
            [pos, rng.choice(neg, size=min(W.MAX_TRAIN - len(pos), len(neg)),
                             replace=False)]))
        raw["train"] = W._subset(raw["train"], keep)

    smap = W.fit_stages(raw["train"], strategy="quantile", n_stages=n_stages)
    stage = {p: smap.assign(W.window_age(w)) for p, w in raw.items()}
    norm = W.Normalizer.fit(raw["train"], method=W.NORMALIZE)
    nm = {p: norm.transform(w) for p, w in raw.items()}
    dead = W.constant_features(nm["test"])
    if dead and len(dead) < len(nm["test"].features):
        nm = {p: W.drop_features(w, dead) for p, w in nm.items()}
    X = {p: np.nan_to_num(W.flatten(w, W.FLATTEN)[0]) for p, w in nm.items()}
    m = RandomForestClassifier(
        n_estimators=W.N_TREES, min_samples_leaf=5, class_weight="balanced",
        random_state=seed, n_jobs=-1).fit(X["train"], raw["train"].y)

    def proba(Xp):
        p = m.predict_proba(Xp)
        return p[:, 1] if p.shape[1] > 1 else p[:, 0]

    p_cal, p_te = proba(X["cal"]), proba(X["test"])
    return {"smap": smap, "stage": stage, "raw": raw,
            "p_cal": p_cal, "p_te": p_te,
            "y_cal": raw["cal"].y, "y_te": raw["test"].y,
            "drive_cal": raw["cal"].drive_idx,
            "drive_te": raw["test"].drive_idx,
            "auc": W.auc(p_te, raw["test"].y)}


def step_models(labelled):
    """Dump per-drive-model predictions in the within-vendor format."""
    from src.conformal.mondrian import MondrianConformal
    from src.conformal.split import SplitConformal

    meta = {"seed": SEED, "alpha": ALPHA, "models": {}}
    for mdl in sorted(labelled["model"].unique().to_list()):
        t0 = time.time()
        c = build_group(labelled, "model", mdl)
        if c is None:
            meta["models"][mdl] = {"skipped": True}
            continue
        g_cal, g_te = c["stage"]["cal"], c["stage"]["test"]
        cal = pl.DataFrame({"drive_idx": np.asarray(c["drive_cal"], np.int64),
                            "stage": np.asarray(g_cal, np.int16),
                            "y": np.asarray(c["y_cal"], np.int8),
                            "p": np.asarray(c["p_cal"], np.float64)})
        te = pl.DataFrame({"drive_idx": np.asarray(c["drive_te"], np.int64),
                           "stage": np.asarray(g_te, np.int16),
                           "y": np.asarray(c["y_te"], np.int8),
                           "p": np.asarray(c["p_te"], np.float64)})
        sp = SplitConformal(alpha=ALPHA, seed=SEED).fit(
            c["p_cal"], c["y_cal"]).predict(c["p_te"])
        mb = (MondrianConformal(alpha=ALPHA, by="both", seed=SEED)
              .fit(c["p_cal"], c["y_cal"], groups=g_cal)
              .predict(c["p_te"], groups=g_te))
        for name, r in (("split", sp), ("mondrian_both", mb)):
            te = te.with_columns([
                pl.Series(f"set_h_{name}", r.sets[:, 0].astype(bool)),
                pl.Series(f"set_f_{name}", r.sets[:, 1].astype(bool))])
        cal.write_parquet(f"{OUT}/preds_wm{mdl}_cal.parquet")
        te.write_parquet(f"{OUT}/preds_wm{mdl}_test.parquet")
        meta["models"][mdl] = {
            "auc": float(c["auc"]),
            "stage_edges": [float(e) for e in c["smap"].edges],
            "n_windows_cal": int(len(c["y_cal"])),
            "n_windows_test": int(len(c["y_te"])),
            "n_fail_windows_test": int(c["y_te"].sum())}
        print(f"  {mdl}: AUC {c['auc']:.4f}, test {len(c['y_te']):,} windows, "
              f"{int(c['y_te'].sum())} failures ({time.time() - t0:.0f}s)",
              flush=True)
        del c
        gc.collect()
    with open(f"{OUT}/meta_wm.json", "w") as fh:
        json.dump(meta, fh, indent=2)


# =================================================================
# seeds: retraining variability of the main configuration
# =================================================================

def _seed_rows(ctx, seed):
    from notebooks.paper_results import METHODS, _fit_conformal
    y, g = ctx["y_te"], ctx["stage"]["test"]
    ven = np.asarray([str(v) for v in np.asarray(ctx["vendor_te"])])
    rows = [{"seed": seed, "method": "-", "stage": "all", "stat": "auc",
             "value": float(ctx["auc"])}]
    f4 = (g == 4) & (y == 1)
    rows.append({"seed": seed, "method": "-", "stage": "4",
                 "stat": "vendorA_share_fail",
                 "value": float((ven[f4] == "A").mean()) if f4.any() else np.nan})
    for m in METHODS:
        r = _fit_conformal(ctx, ALPHA, m, seed=SEED)
        hit = r.sets[np.arange(len(y)), y]
        size = r.sets.sum(1)
        for s in ["all"] + list(range(5)):
            mk = np.ones(len(y), bool) if s == "all" else g == s
            fk = mk & (y == 1)
            rows += [
                {"seed": seed, "method": m, "stage": str(s), "stat": "coverage",
                 "value": float(hit[mk].mean())},
                {"seed": seed, "method": m, "stage": str(s),
                 "stat": "cov_failure",
                 "value": float(r.sets[fk, 1].mean()) if fk.any() else np.nan},
                {"seed": seed, "method": m, "stage": str(s), "stat": "set_size",
                 "value": float(size[mk].mean())},
                {"seed": seed, "method": m, "stage": str(s), "stat": "n_fail",
                 "value": float(fk.sum())},
            ]
    return rows


def step_seeds(load_slim, seeds=SEEDS):
    from notebooks.paper_results import build
    path = f"{OUT}/seeds.csv"
    have = set()
    if os.path.exists(path):
        have = set(pl.read_csv(path)["seed"].unique().to_list())
    for s in seeds:
        if s in have:
            print(f"  [skip seed {s}]")
            continue
        t0 = time.time()
        print(f"\n--- seed {s} ---", flush=True)
        labelled = _load(load_slim, seed=s)
        ctx = build(labelled, seed=s)
        new = pl.DataFrame(_seed_rows(ctx, s))
        old = pl.read_csv(path) if os.path.exists(path) else None
        (pl.concat([old, new]) if old is not None else new).write_csv(path)
        print(f"  seed {s}: AUC {ctx['auc']:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
        del labelled, ctx
        gc.collect()


# =================================================================
# shift: conformal behaviour under held-out vendor
# =================================================================

def step_shift(labelled):
    from src.evaluation import RunConfig
    from src.evaluation.lomo import run_experiment
    cfg = RunConfig(normalize="rank", seed=SEED,
                    methods=("split", "mondrian_class", "weighted"),
                    results_dir=f"{OUT}/shift_cells")
    res = run_experiment(labelled, cfg, include_standard=True, resume=True)
    _write(res, "shift_rank.csv")


# =================================================================

def _bundle():
    try:
        freeze = subprocess.run(["pip", "freeze"], capture_output=True,
                                text=True, timeout=120).stdout
        with open(f"{OUT}/pip_freeze.txt", "w") as fh:
            fh.write(freeze)
    except Exception as e:                       # noqa: BLE001
        print(f"  pip freeze failed: {e}")
    keep = (".csv", ".json", ".txt")
    files = [f for f in sorted(os.listdir(OUT))
             if (f.endswith(keep) and not f.startswith("done_"))
             or (f.startswith("preds_") and f.endswith(".parquet"))]
    zp = f"{OUT}/batch_results.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f"{OUT}/{f}", arcname=f)
        sc = f"{OUT}/shift_cells"
        if os.path.isdir(sc):
            for f in sorted(os.listdir(sc)):
                z.write(f"{sc}/{f}", arcname=f"shift_cells/{f}")
    print(f"\nwrote {zp} ({len(files)} files, "
          f"{os.path.getsize(zp) / 1e6:.1f} MB)")


def run_batch(load_slim, steps=STEPS, B=1000):
    t_all = time.time()
    labelled = None

    def lab():
        nonlocal labelled
        if labelled is None:
            labelled = _load(load_slim)
        return labelled

    for step in steps:
        if _done(step):
            print(f"[skip {step}: done]")
            continue
        print("\n" + "#" * 64 + f"\n# {step}\n" + "#" * 64, flush=True)
        t0 = time.time()
        if step == "main":
            from notebooks.revision_analysis import run_all
            run_all(load_slim)
        elif step == "analyse":
            from notebooks.revision_analysis import analyse
            analyse(B=B)
        elif step == "models":
            step_models(lab())
        elif step == "robust":
            from notebooks.validate_wear import run_all as vw
            vw(lab(), which=("v3", "v5", "v2", "v6"))
        elif step == "repr":
            from notebooks.fast_repr_sweep import run as repr_run
            repr_run(lab())
        elif step == "lomm":
            from notebooks.lomm_sweep import run as lomm_run
            lomm_run(lab())
        elif step == "shift":
            step_shift(lab())
        elif step == "seeds":
            labelled = None              # free before loading 10 more
            gc.collect()
            step_seeds(load_slim)
        else:
            raise KeyError(step)
        _mark(step, t0)

    _bundle()
    print(f"[batch finished in {(time.time() - t_all) / 60:.0f} min]")
