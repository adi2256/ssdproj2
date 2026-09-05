# SSD Failure Prediction under Manufacturer Distribution Shift

Conformal prediction sets with per-drive coverage guarantees for SSD
failure prediction, evaluated under leave-one-manufacturer-out (LOMO)
distribution shift on the Alibaba production SSD SMART dataset.

**Status:** pipeline complete and tested (321 tests). Diagnosing why
models fail to transfer across manufacturers before the final run.

---

## What this is

Existing SSD failure predictors output a score. That score has no
defined meaning, is never checked for calibration, and comes with only
fleet-level accuracy figures — nothing about any individual drive.

This project replaces the score with a **prediction set** carrying a
formal, distribution-free, finite-sample coverage guarantee, then asks
whether that guarantee survives the situation it actually meets in
production: a manufacturer the system has never seen.

**Not claimed:** better accuracy, novel sequence modelling, or that
conformal prediction is new to storage (Vishwakarma et al., COPA 2023,
applied Mondrian conformal to disk health).

**Claimed:** a measurement of what happens to conformal validity under
manufacturer shift, and of whether shift-robust variants repair it.

---

## Setup

```bash
git clone <this repo>
cd SSD-Failure-Prediction
pip install -r requirements.txt
python3 -m pytest -q            # 321 passed
```

No data is required. Every module is developed and tested against
`src/data/synthetic.py`, a 400-drive fixture calibrated to the real
fleet's vendor proportions, failure-rate ordering, censoring rate and
per-vendor attribute availability.

---

## Repository

```
src/
  config.py            every tunable number; nothing hardcodes settings
  schema.py            105 columns, per-vendor availability, COMMON_IDS
  data/
    synthetic.py       fixture generator with known ground truth
    labels.py          0 < days_to_failure <= 30, censoring rules
    splits.py          standard + LOMO folds, leakage assertions
    windowing.py       trajectory tensors, z-score / rank scaling
  features/wefr.py     ensemble ranking (Xu et al., DSN 2021)
  models/              Random Forest baseline, GRU sequence model
  conformal/           split, Mondrian (class/group), weighted
  evaluation/          metrics, LOMO orchestration

scripts/               entry points; run in numeric order
tests/                 321 tests
notebooks/             Kaggle setup helpers
reports/               provenance: profiling and diagnostics output
```

---

## Data

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

## Running experiments

```python
from src.data.labels import build_labels
from src.evaluation import RunConfig, run_experiment, coverage_gap_table

labelled, stats = build_labels(df)

cfg = RunConfig(
    stride=30,
    normalize="rank",                    # or "zscore"
    drop_constant_test_features=True,
    models=("random_forest", "gru"),
    methods=("split", "mondrian_class", "mondrian_group", "weighted"),
    results_dir="results",
)
res = run_experiment(labelled, cfg, verbose=True)
print(coverage_gap_table(res))
```

`run_experiment` writes one JSON per (fold, model, method) and skips
completed cells, so a run cut short by a session timeout resumes rather
than restarts.

On Kaggle, `notebooks/kaggle_setup.py` handles data location, upload
verification, and a stale-module check.

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

1. **Set size and singleton rate in every table.** A guarantee can be
   met by widening sets until they carry no information; coverage alone
   hides this entirely.
2. **Class-conditional coverage alongside marginal.** At 1.4%
   prevalence, marginal coverage is dominated by healthy drives —
   measured 0.896 marginal against 0.049 failure-class coverage on the
   standard split.
3. **Failed-drive counts, not positive-window counts.** Thirty windows
   from one drive are thirty views of the same failure; window counts
   overstate statistical power by roughly 30x.
4. **Per-fold results are three case studies, not a population
   estimate.** n=3 cannot support a distributional claim.

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
