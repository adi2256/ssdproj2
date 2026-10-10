# SSD Failure Prediction under Manufacturer Distribution Shift

Conformal prediction sets for SSD failure prediction on the Alibaba
production SMART dataset, audited by wear stage and stress-tested by
holding out a manufacturer.

**Status (October 2026):** experiments finished; manuscript under
revision for a Springer LNCS/LNEE conference. All reported numbers come
from one batch run (`notebooks/revision_batch.py`, seed 42, deterministic
drive sampler) whose outputs are in `paper_results/final/`. Test suite:
357 passed, 5 skipped.

---

## What this is, and what is claimed

Existing SSD failure predictors output a score with no defined
probability meaning and are evaluated only by fleet-level metrics. This
project wraps such a classifier in split and Mondrian conformal
prediction and measures where the resulting coverage holds and where it
does not.

| | |
|---|---|
| **Guaranteed** | Split conformal's marginal coverage holds for an exchangeable unit. Here that unit is the **drive**, so the guarantee applies to one-window-per-drive protocols (`notebooks/drive_level_check.py`). |
| **Measured, not guaranteed** | Coverage over windows, per wear stage, per vendor and on a held-out vendor. These are empirical estimates with drive-clustered bootstrap intervals. |
| **Claimed** | Marginal coverage is met while failure-class coverage collapses on the oldest wear stage; class x stage Mondrian conformal repairs this on the same fleet; under a held-out vendor no tested representation gives a classifier that transfers. |
| **Not claimed** | Better accuracy than WEFR; novel sequence modelling; that conformal prediction is new to storage (Vishwakarma et al., COPA 2023); any guarantee for an unseen vendor; any population claim from three vendors (they are three case studies). |

---

## Setup

```bash
git clone https://github.com/adi2256/ssdproj2
cd ssdproj2
pip install -r requirements.txt
python3 -m pytest -q            # 357 passed, 5 skipped
```

Unit tests need no data: every module is tested against
`src/data/synthetic.py`, a fixture with known ground truth calibrated to
the real fleet's vendor proportions, failure-rate ordering, censoring rate
and per-vendor attribute availability.

---

## Repository

```
src/
  config.py            every tunable number
  schema.py            105 columns, per-vendor availability, COMMON_IDS
  data/                synthetic fixture, labels, splits, windowing
  features/wefr.py     ensemble ranking (Xu et al., DSN 2021)
  models/              Random Forest baseline, GRU sequence model
  conformal/           split, Mondrian (class/group), weighted
  evaluation/          metrics, LOMO orchestration
scripts/               preprocessing entry points, run in numeric order
notebooks/             Kaggle batch runs, table generation, audits
paper/                 LaTeX table/figure helpers, number audit
paper_results/final/   released results, predictions and manifests
reports/               profiling and diagnostics output
tests/
```

---

## Reproducing the paper

### Regenerating tables and figures

```bash
# every table, figure and \R{...} number macro of the manuscript
python -m notebooks.paper_tables paper_results/final OUT_DIR

# coverage at the exchangeable unit (one window per drive, 200 draws)
python -m notebooks.drive_level_check paper_results/final

# trace every number in the built PDF back to a released file
python3 -I paper/number_audit.py paper.pdf paper_results/final OUT_DIR reports
```

These need only the files in `paper_results/final/`; no access to the raw
dataset is required.

### Full rerun

`notebooks/revision_batch.py` regenerates `paper_results/final/` from the
public dataset (Kaggle CPU, background "Save Version" run; see
`notebooks/kaggle_setup.py`). Every step uses the 60,000-drive seed-42
sample (`r2p7`) unless stated. The sampler sorts the drive list by
`(model, disk_id)` before `DataFrame.sample(seed)` and prints a SHA1
fingerprint of the sampled keys; downstream ordering (drive table, split
assignment, window construction, training-window cap, Random Forest) is
seeded. `notebooks/export_manifests.py` writes the manifests below.

### Artifact index

Status: **P** = recomputed from the released per-window predictions alone,
with the released scripts. **R** = a released result file; regenerating it
needs a full rerun. Full reruns have been checked at the level of samples
and split assignments (fingerprints, manifests, cross-session comparison,
consistency with the released predictions), not re-executed end to end
and compared number by number.

Paths are relative to `paper_results/final/`; manifests are in
`manifests/`.

| Paper result | Released file(s) | Manifest (file: column) | Seed | Status |
|---|---|---|---|---|
| Table 2 (main, by wear stage), Table 3 (set composition), Fig. 4 (coverage vs set size) | `preds_r2p7_{cal,test}.parquet`, `meta_r2p7.json` (stage edges), `boot_pooled.csv` | `drives_r2p7.csv.gz`: `part_standard`; `drive_index_r2p7_{cal,test}.csv.gz` | 42 | P |
| Table 4 (sampling ratio) | `preds_{r1,r2p7,r5}_{cal,test}.parquet`, `ratio_summary.csv` | `drives_{r1,r2p7,r5}.csv.gz`: `part_standard`; `drive_index_*` | 42 | P |
| Table 5 (vendor x stage failure coverage) | `preds_r2p7_{cal,test}.parquet` | as Table 2 | 42 | P |
| Table 6, Fig. 3 (within-vendor audit) | `preds_wv{A,B,C}_{cal,test}.parquet`, `meta_wv.json`, `boot_wv.csv` | `drives_r2p7.csv.gz`: `part_within_vendor`; `drive_index_wv*` | 42 | P |
| Within-model audit (Sect. 5.3) | `preds_wm{MA1..MC2}_{cal,test}.parquet`, `meta_wm.json` | `drives_r2p7.csv.gz`: `part_within_model` (no drive-index map) | 42 | P |
| Table 7 (recalibration) | `preds_r2p7_{train,cal,test}.parquet`, `platt_*_r2p7.csv` | as Table 2 | 42 | P |
| Drive-level coverage (Sect. 5.1) | `preds_r2p7_{cal,test}.parquet` | as Table 2 | 42 (draws) | P |
| Fig. 2 (power-on hours by vendor) | `age_overlap.csv` | `drives_r2p7.csv.gz`: `part_standard` | 42 | R |
| Ten-seed spread (Sect. 5.2) | `seeds.csv` | `drives_seed{0..8}.csv.gz`, `drives_r2p7.csv.gz`: `part_standard` | 0-8, 42 | R |
| Wear-stage hold-out (Sect. 5.2) | `v3.csv` | `drives_r2p7.csv.gz`: `modal_stage`, `part_holdout_stage_{0..4}` | 42 | R |
| Stage binning (Sect. 5.2) | `v2.csv` | `drives_r2p7.csv.gz`: `part_standard` | 42 | R |
| Classifier swap, GBM / logistic (Sect. 5.2) | `v6.csv` | `drives_r2p7.csv.gz`: `part_standard` | 42 | R |
| Prevalence-matched random strata (Sect. 5.2) | `v5.csv` | `drives_r2p7.csv.gz`: `part_standard`; strata drawn from the seed, not listed | 42 | R |
| Held-out-vendor conformal (Sect. 5.5) | `shift_rank.csv`, `shift_cells/` | `drives_r2p7.csv.gz`: `part_holdout_vendor_{A,B,C}` | 42 | R |
| Representation sweep (Sect. 5.5) | `repr_sweep.csv`, `lomm_sweep.csv` | `drives_r2p7.csv.gz` | 42 | R |
| Transfer diagnostics (Sect. 5.5) | `manifests/transfer_diagnostics.csv`, `manifests/transfer_sign_bootstrap.csv` | `drives_r2p7.csv.gz` | 42 | R |
| Dataset size, vendor profile (Sect. 3) | `reports/vendor_profile.csv` (repo root) | full dataset | - | R |

Table and figure numbers refer to the Springer manuscript. Run
configurations (stride, training-window cap, trees, normalisation,
features, alpha) are in `manifests/experiments.json`; per-sample sizes,
labelling counts and fingerprints in `manifests/samples.json`.
`pip_freeze.txt` records package versions of the batch run.

### Known limitations of the release

* The within-model prediction files have split assignments but no
  drive-index map, so they are outside `manifests/consistency_check.csv`.
* The training-window subsample, the random strata of the prevalence
  control and the bootstrap draws are not listed; they are derived from
  the recorded seed by the released code.
* `v5.csv` reports only split and class-Mondrian rows. An earlier version
  also had stage-Mondrian rows that were invalid by construction
  (calibrated on wear stages, applied to random strata); the code is
  fixed (`notebooks/validate_wear.py`, `_report(methods=...)`) and those
  rows were removed. No paper result used them.
* `paper_results/revision/` holds pre-fix predictions from the earlier,
  non-deterministic sampler and is kept only for the record.

---

## Data preparation


**Alibaba SSD SMART logs**, the dataset released with the base paper.
https://github.com/alibaba-edu/dcbrain/tree/master/ssd_smart_logs

| | |
|---|---|
| Drives | 475,056 |
| Rows | 273,112,284 |
| Span | 2018-01-01 to 2019-12-31 |
| Vendors | 3 (A, B, C), 6 drive models |
| Failed drives | 16,305 |

Failure rates differ 3.7x across vendors (A 1.42%, B 2.59%, C 5.21%),
so LOMO breaks exchangeability on the label axis as well as the
covariate axis.

### Pipeline

```bash
python3 scripts/01_preprocess.py        # daily CSVs -> parquet (9.78 GB)
python3 scripts/02_verify_profile.py    # -> reports/vendor_profile.csv
python3 scripts/03_label_diagnostics.py # -> reports/label_diagnostics.txt
python3 scripts/04_export_slim.py       # -> 203 MB slim export
python3 scripts/04_export_slim.py --check   # must print PASS
```

The slim export keeps the 16 common SMART columns as float32 with ZSTD.
It drops **no rows** — row filtering is `build_labels()`'s job. 49x
smaller, small enough to move between machines.

---

## Feature policy: the common 16

Of 102 SMART columns, 36 are empty for every vendor and only **16 are
populated by all three**:

```
COMMON_IDS = [5, 9, 12, 183, 184, 187, 197, 199]   # n_i and r_i each
```

Under LOMO, columns exclusive to the training vendors are all-NaN at
test time, and columns exclusive to the held-out vendor were never
trained on. The feature space itself differs by manufacturer, so the
shift is structural, not merely distributional.

This is a locked decision and a reportable finding.

---

## Invariants the code enforces

These are asserted, not remembered.

| Rule | Where |
|---|---|
| Drive key is `(model, disk_id)` — 119,213 disk_ids appear under more than one model | `config.DRIVE_KEY` |
| Splits are drive-level; no drive in two parts | `splits.validate_split` |
| Held-out vendor absent from train AND cal — its presence would restore exchangeability and void the experiment | `validate_split`, re-checked in `prepare_fold` |
| WEFR selection runs inside each fold, on training vendors only | `prepare_fold` |
| Normalisation statistics from training windows only | `prepare_fold` |
| Decision threshold chosen on training scores only | `run_cell` |
| Every failed drive contributes at least one positive window | `prepare_fold` |

---

## Reporting rules

1. **Set size and singleton rate in every table.** A coverage target can
   be met by widening sets until they carry no information.
2. **Class-conditional coverage alongside marginal.** At roughly 1.5%
   window prevalence, marginal coverage is dominated by healthy windows
   and can be met while almost every failure is missed.
3. **Failed-drive counts, not positive-window counts.** Windows from one
   drive are dependent; window counts overstate statistical power.
4. **Per-vendor results are three case studies, not a population
   estimate.**
5. **Window-level coverage is empirical.** The finite-sample guarantee
   is for the drive; report drive-level checks beside window-level
   numbers.

---

## Base paper

Fan Xu, Shujie Han, Patrick P. C. Lee, Yi Liu, Cheng He, Jiongzhou Liu.
"General Feature Selection for Failure Prediction in Large-scale SSD
Deployment." IEEE/IFIP DSN 2021, pp. 263-270.

Sequence modelling and multi-horizon prediction are prior art (Koh et
al., IEEE Access 2024; Zhang et al., FAST '23) and are treated as
implementation, not contribution.

---

## Contributing

```bash
python3 -c "import ast; ast.parse(open('src/evaluation/lomo.py').read())"
python3 -m pytest -q
```

Both must pass before any push touching `src/`. A previous push landed
a file that did not parse, which cost a debugging session.
