"""
Per-run manifests, split assignments, and a regeneration check.

Answers three reviewer requests:

  1. Release the exact sampled drives and random seeds for every experiment.
     For every drive sample the paper uses (three sampling ratios at seed 42,
     and the main configuration at seeds 0-8), this writes the sampled
     (model, disk_id) list with its fingerprint, whether each drive survived
     labelling, and its part (train / cal / test) in every split derived from
     that sample: the standard split, the within-vendor and within-drive-model
     splits, the three held-out-vendor splits and the drive-disjoint
     wear-stage hold-outs.

  2. Identify which results regenerate from the released artifacts.
     For every saved prediction dump (r1, r2p7, r5, wvA/B/C) the test and
     calibration windows are rebuilt from the manifest and compared,
     window by window, with the released parquet files: same drive index,
     same label, same count. A mismatch is reported, never hidden.

  3. Re-run the per-vendor transfer diagnostics on the deterministic sampler
     (range, scale and correlation-sign checks behind Sect. 5.5).

Usage on Kaggle (Save Version -> Save & Run All):

    from notebooks.export_manifests import run
    run(C.load_slim)

Writes manifests.zip to the output directory. Runtime is dominated by
loading each sample and windowing the 60k-drive sample once.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
import zipfile

import numpy as np
import polars as pl

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
MAN = f"{OUT}/manifests"
REL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper_results", "final")

SEED = 42
SAMPLES = ([("r2p7", 60_000, 42), ("r1", 32_610, 42), ("r5", 97_830, 42)]
           + [(f"seed{s}", 60_000, s) for s in range(9)])
KEY = ["model", "disk_id"]
DIAG_FEATS = ("r_9", "r_12", "r_199", "n_183")
SIGN_FEATS = ("r_9", "r_12", "r_199")
N_SIGN_BOOT = 500


def fingerprint(keys: pl.DataFrame) -> str:
    """Same definition as the sampler's printed fingerprint."""
    return hashlib.sha1(
        "|".join(sorted(f"{m}:{d}" for m, d in keys.select(KEY).iter_rows()))
        .encode()).hexdigest()[:12]


def _parts(split) -> pl.DataFrame:
    return pl.concat([getattr(split, p).select(KEY).with_columns(
        pl.lit(p).alias("part")) for p in ("train", "cal", "test")])


def _add_parts(man: pl.DataFrame, split, col: str) -> pl.DataFrame:
    return man.join(_parts(split).rename({"part": col}), on=KEY, how="left")


# -----------------------------------------------------------------
# sample + split manifests
# -----------------------------------------------------------------

def sample_manifest(load_slim, tag, n, seed):
    from src.data.labels import build_labels, drive_table
    from src.data.splits import make_standard_split

    raw = load_slim(n_drives=n, keep_all_failed=True, seed=seed)
    sampled = (raw.group_by(KEY)
                  .agg(pl.col("failure_time").max().is_not_null()
                       .alias("failed"))
                  .sort(KEY))
    fp = fingerprint(sampled)
    labelled, stats = build_labels(raw)
    del raw
    gc.collect()

    dt = drive_table(labelled).select(KEY + ["vendor", "drive_label"])
    man = (sampled.with_columns(pl.col("model").str.slice(1, 1).alias("vendor"))
                  .join(dt.select(KEY + ["drive_label"]), on=KEY, how="left")
                  .with_columns(pl.col("drive_label").is_not_null()
                                .alias("in_labelled")))
    man = _add_parts(man, make_standard_split(labelled, seed=seed),
                     "part_standard")
    info = {
        "tag": tag, "n_drives_requested": n, "seed": seed,
        "keep_all_failed": True,
        "n_sampled": sampled.height,
        "n_failed_sampled": int(sampled["failed"].sum()),
        "n_after_labelling": dt.height,
        "fingerprint": fp,
        "labelling": {k: (int(v) if isinstance(v, (int, np.integer)) else
                          float(v) if isinstance(v, (float, np.floating))
                          else str(v))
                      for k, v in (stats.__dict__.items()
                                   if hasattr(stats, "__dict__")
                                   else dict(stats).items())},
        "standard_split": {"test_frac": 0.20, "cal_frac_of_rest": 0.20,
                           "strata": ["model", "drive_label"], "seed": seed},
    }
    print(f"[{tag}] n={n:,} seed={seed} fingerprint {fp} | "
          f"{info['n_failed_sampled']:,} failed | "
          f"{dt.height:,} after labelling", flush=True)
    return labelled, man, info


def main_extra_splits(labelled, man):
    """Splits derived from the 60k seed-42 sample beyond the standard one."""
    from src.data.splits import make_lomo_split, make_standard_split

    vend = []
    for v in sorted(labelled["vendor"].unique().to_list()):
        sub = labelled.filter(pl.col("vendor") == v)
        vend.append(_parts(make_standard_split(sub, seed=SEED)))
    man = man.join(pl.concat(vend).rename({"part": "part_within_vendor"}),
                   on=KEY, how="left")

    mods = []
    for m in sorted(labelled["model"].unique().to_list()):
        sub = labelled.filter(pl.col("model") == m)
        mods.append(_parts(make_standard_split(sub, seed=SEED)))
    man = man.join(pl.concat(mods).rename({"part": "part_within_model"}),
                   on=KEY, how="left")

    for v in sorted(labelled["vendor"].unique().to_list()):
        man = _add_parts(man, make_lomo_split(labelled, v, seed=SEED),
                         f"part_holdout_vendor_{v}")
    return man


def stage_holdout_manifest(ws, man):
    """
    Re-create validate_wear.v3 assignments exactly: modal stage per drive,
    then for each held-out stage the same rng draw for the cal/train cut.
    """
    from notebooks.validate_wear import N_STAGES
    from src.features.wear import fit_stages, window_age

    smap = fit_stages(ws, strategy="quantile", n_stages=N_STAGES)
    st = smap.assign(window_age(ws))
    n_dr = len(ws.drives)
    counts = np.zeros((n_dr, smap.n_stages), dtype=int)
    np.add.at(counts, (ws.drive_idx, st), 1)
    drive_stage = counts.argmax(axis=1)

    rng = np.random.default_rng(SEED)
    cols = {"model": [d[0] for d in ws.drives],
            "disk_id": [d[1] for d in ws.drives],
            "modal_stage": drive_stage.astype(np.int16)}
    for held in range(smap.n_stages):
        te = np.flatnonzero(drive_stage == held)
        rest = np.flatnonzero(drive_stage != held)
        part = np.full(n_dr, "", dtype=object)
        if len(te) == 0:
            cols[f"part_holdout_stage_{held}"] = part
            continue
        rng.shuffle(rest)
        n_cal = int(round(0.2 * len(rest)))
        part[te] = "test"
        part[rest[:n_cal]] = "cal"
        part[rest[n_cal:]] = "train"
        cols[f"part_holdout_stage_{held}"] = part
    df = pl.DataFrame(cols).with_columns(
        pl.col("model").cast(man["model"].dtype),
        pl.col("disk_id").cast(man["disk_id"].dtype))
    return man.join(df, on=KEY, how="left"), [float(e) for e in smap.edges]


# -----------------------------------------------------------------
# regeneration check against the released predictions
# -----------------------------------------------------------------

def _check_dump(ws, path):
    if not os.path.exists(path):
        return {"file": os.path.basename(path), "status": "missing"}
    d = pl.read_parquet(path)
    res = {"file": os.path.basename(path), "n_released": d.height,
           "n_rebuilt": len(ws)}
    same_n = d.height == len(ws)
    res["same_count"] = bool(same_n)
    if same_n:
        res["same_drive_idx"] = bool(
            np.array_equal(d["drive_idx"].to_numpy(),
                           np.asarray(ws.drive_idx, np.int64)))
        res["same_labels"] = bool(
            np.array_equal(d["y"].to_numpy().astype(int),
                           np.asarray(ws.y).astype(int)))
    res["status"] = ("match" if same_n and res.get("same_drive_idx")
                     and res.get("same_labels") else "MISMATCH")
    return res


def _drive_index(ws):
    return pl.DataFrame({"drive_idx": np.arange(len(ws.drives)),
                         "model": [d[0] for d in ws.drives],
                         "disk_id": [d[1] for d in ws.drives]})


def verify_dumps(labelled, tag, seed, rel_dir):
    """Rebuild cal/test windows of a pooled dump and compare."""
    from notebooks.paper_results import STRIDE
    from src.data.splits import make_standard_split
    from src.data.windowing import make_windows

    sp = make_standard_split(labelled, seed=seed)
    out = []
    for part in ("cal", "test"):
        ws = make_windows(sp.apply(labelled, part), stride=STRIDE)
        r = _check_dump(ws, f"{rel_dir}/preds_{tag}_{part}.parquet")
        r.update({"dump": tag, "part": part})
        out.append(r)
        _drive_index(ws).write_csv(
            f"{MAN}/drive_index_{tag}_{part}.csv")
        print(f"  verify {tag}/{part}: {r['status']} "
              f"({r.get('n_released')} vs {r.get('n_rebuilt')} windows)",
              flush=True)
    return out


def verify_within_vendor(labelled, rel_dir):
    from notebooks.within_vendor_wear import STRIDE
    from src.data.splits import make_standard_split
    from src.data.windowing import make_windows

    out = []
    for v in sorted(labelled["vendor"].unique().to_list()):
        sub = labelled.filter(pl.col("vendor") == v)
        sp = make_standard_split(sub, seed=SEED)
        for part in ("cal", "test"):
            ws = make_windows(sp.apply(sub, part), stride=STRIDE)
            r = _check_dump(ws, f"{rel_dir}/preds_wv{v}_{part}.parquet")
            r.update({"dump": f"wv{v}", "part": part})
            out.append(r)
            _drive_index(ws).write_csv(f"{MAN}/drive_index_wv{v}_{part}.csv")
            print(f"  verify wv{v}/{part}: {r['status']}", flush=True)
    return out


# -----------------------------------------------------------------
# transfer diagnostics (Sect. 5.5), deterministic sampler
# -----------------------------------------------------------------

def _last_level(ws, feat):
    j = ws.features.index(feat)
    return np.asarray(ws.X[:, -1, j], dtype=float)


def transfer_diagnostics(labelled, ws):
    """
    Per held-out vendor: how far its raw final-day levels sit from the
    training vendors' (fraction outside the training range, SD ratio),
    computed on windows of the held-out-vendor split. Per vendor: sign of
    the window-level Pearson correlation between each raw attribute and
    the label, with a drive-clustered bootstrap.
    """
    from src.data.splits import make_lomo_split

    keys = np.array([f"{m}:{d}" for m, d in ws.drives], dtype=object)
    wkey = keys[ws.drive_idx]
    vend = np.asarray([str(v) for v in ws.vendors])
    rows = []
    for v in sorted(set(vend)):
        sp = make_lomo_split(labelled, v, seed=SEED)
        tr_keys = set(f"{m}:{d}" for m, d in sp.train.select(KEY).iter_rows())
        tr = np.isin(wkey, list(tr_keys))
        te = vend == v
        for f in DIAG_FEATS:
            if f not in ws.features:
                continue
            x = _last_level(ws, f)
            a, b = x[tr], x[te]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) == 0 or len(b) == 0:
                continue
            sd = a.std() if a.std() > 0 else np.nan
            rows.append({
                "held_out_vendor": v, "feature": f,
                "n_train_windows": int(len(a)), "n_test_windows": int(len(b)),
                "frac_outside_train_range":
                    float(((b < a.min()) | (b > a.max())).mean()),
                "test_mean_in_train_sd": float((b.mean() - a.mean()) / sd),
                "sd_ratio_test_over_train": float(b.std() / sd),
            })
    diag = pl.DataFrame(rows)

    rng = np.random.default_rng(SEED)
    srows = []
    y_all = np.asarray(ws.y, dtype=float)
    for v in sorted(set(vend)):
        m = vend == v
        d_idx = ws.drive_idx[m]
        uniq, inv = np.unique(d_idx, return_inverse=True)
        nd = len(uniq)
        y = y_all[m]
        W = rng.multinomial(nd, np.full(nd, 1.0 / nd), size=N_SIGN_BOOT)
        for f in SIGN_FEATS:
            if f not in ws.features:
                continue
            x = _last_level(ws, f)[m]
            ok = np.isfinite(x)
            xx, yy, ii = x[ok], y[ok], inv[ok]

            def sums(arr):
                return np.bincount(ii, weights=arr, minlength=nd)
            S = {k: sums(arr) for k, arr in
                 (("n", np.ones_like(xx)), ("x", xx), ("y", yy),
                  ("xx", xx * xx), ("yy", yy * yy), ("xy", xx * yy))}

            def corr(w):
                n = w @ S["n"]
                sx, sy = w @ S["x"], w @ S["y"]
                cov = w @ S["xy"] / n - (sx / n) * (sy / n)
                vx = w @ S["xx"] / n - (sx / n) ** 2
                vy = w @ S["yy"] / n - (sy / n) ** 2
                return cov / np.sqrt(np.maximum(vx * vy, 1e-300))

            r0 = float(corr(np.ones((1, nd)))[0])
            rb = corr(W.astype(float))
            srows.append({"vendor": v, "feature": f, "pearson_r": r0,
                          "n_windows": int(ok.sum()), "n_drives": int(nd),
                          "n_boot": N_SIGN_BOOT,
                          "frac_boot_positive": float((rb > 0).mean())})
    return diag, pl.DataFrame(srows)


# -----------------------------------------------------------------

def _experiment_config():
    from notebooks import (fast_repr_sweep as FR, paper_results as PR,
                           validate_wear as VW, within_vendor_wear as WV)
    from src.config import CFG
    return {
        "dataset": "Alibaba SSD SMART (dcbrain ssd_smart_logs), 2018-2019",
        "sampler": "all failed drives + healthy drives sampled uniformly "
                   "without replacement from the (model, disk_id)-sorted "
                   "list with polars DataFrame.sample(seed)",
        "labels": {"horizon_days": CFG.horizon_days},
        "main_tables": {"stride": PR.STRIDE, "max_train_windows": PR.MAX_TRAIN,
                        "n_trees": PR.N_TREES, "normalize": PR.NORMALIZE,
                        "features": PR.FLATTEN, "n_stages": PR.N_STAGES,
                        "alpha": PR.MAIN_ALPHA, "alphas_swept": list(PR.ALPHAS),
                        "bootstrap_resamples": 1000, "seed": PR.SEED},
        "within_vendor_and_model": {"stride": WV.STRIDE,
                                    "max_train_windows": WV.MAX_TRAIN,
                                    "n_trees": WV.N_TREES,
                                    "normalize": WV.NORMALIZE,
                                    "features": WV.FLATTEN, "seed": WV.SEED},
        "robustness": {"stride": VW.STRIDE, "seed": SEED},
        "representation_sweep": {"stride": FR.STRIDE,
                                 "max_train_windows": FR.MAX_TRAIN,
                                 "n_trees": FR.N_TREES, "seed": FR.SEED},
        "held_out_vendor_conformal": {"stride": 30, "features": "last",
                                      "normalize": "rank",
                                      "feature_selection": "WEFR, per fold",
                                      "methods": ["split", "mondrian_class",
                                                  "weighted"], "seed": SEED},
        "seed_analysis": [s for t, _, s in SAMPLES if t.startswith("seed")]
                         + [42],
    }


def _bundle():
    zp = f"{OUT}/manifests.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(MAN)):
            z.write(f"{MAN}/{f}", arcname=f"manifests/{f}")
    return zp


def run(load_slim, rel_dir: str = REL, tags=None, extras: bool = True):
    """
    tags   : sample tags to process (default: all, largest sample first).
    extras : also window the whole 60k sample for the stage hold-out
             manifest and the transfer diagnostics (the slow part).

    Everything is written as soon as it exists -- samples.json,
    regeneration_check.csv and manifests.zip are refreshed after every
    sample -- so a run that stops early still leaves usable output.
    """
    t0 = time.time()
    os.makedirs(MAN, exist_ok=True)
    infos, checks = [], []
    order = sorted(SAMPLES, key=lambda s: -s[1] if s[0] in ("r1", "r2p7", "r5")
                   else 0)
    order = [s for s in order if s[0] in ("r5", "r2p7", "r1")] + \
            [s for s in order if s[0] not in ("r5", "r2p7", "r1")]
    if tags is not None:
        order = [s for s in order if s[0] in tags]
    with open(f"{MAN}/experiments.json", "w") as fh:
        json.dump(_experiment_config(), fh, indent=2)

    for tag, n, seed in order:
        labelled, man, info = sample_manifest(load_slim, tag, n, seed)
        if tag == "r2p7":
            man = main_extra_splits(labelled, man)
            if extras:
                from notebooks.paper_results import STRIDE
                from src.data.windowing import make_windows
                ws = make_windows(labelled, stride=STRIDE)
                man, edges = stage_holdout_manifest(ws, man)
                info["holdout_stage_edges"] = edges
                diag, sign = transfer_diagnostics(labelled, ws)
                diag.write_csv(f"{MAN}/transfer_diagnostics.csv")
                sign.write_csv(f"{MAN}/transfer_sign_bootstrap.csv")
                with pl.Config(tbl_rows=40, tbl_cols=12):
                    print(diag)
                    print(sign)
                del ws
                gc.collect()
            checks += verify_within_vendor(labelled, rel_dir)
        if tag in ("r1", "r2p7", "r5"):
            checks += verify_dumps(labelled, tag, seed, rel_dir)
        man.write_csv(f"{MAN}/drives_{tag}.csv")
        infos.append(info)
        with open(f"{MAN}/samples.json", "w") as fh:
            json.dump(infos, fh, indent=2)
        if checks:
            pl.DataFrame(checks).write_csv(f"{MAN}/regeneration_check.csv")
        _bundle()
        print(f"  [{tag}] done at {time.time() - t0:.0f}s", flush=True)
        del labelled, man
        gc.collect()

    zp = _bundle()
    bad = [c for c in checks if c.get("status") != "match"]
    print(f"\nregeneration check: {len(checks) - len(bad)}/{len(checks)} match")
    for c in bad:
        print("  ", c)
    print(f"wrote {zp} ({os.path.getsize(zp) / 1e6:.1f} MB) "
          f"in {(time.time() - t0) / 60:.0f} min")
