# CLAUDE.md

Context for AI agents working on this repository. Read fully before editing code.

---

## 1. What this project is

A publication-targeted research project on **SSD failure prediction under manufacturer
distribution shift**. Target venues: IEEE TrustCom, DSN, ACM SYSTOR, IEEE Access.

The paper's contribution is **not** better accuracy. It is a change in output type plus a
stress test:

1. Replace point failure scores with **conformal prediction sets** carrying a formal,
   distribution-free, finite-sample per-drive coverage guarantee.
2. Test whether that guarantee survives **Leave-One-Manufacturer-Out (LOMO)** evaluation,
   which deliberately breaks conformal's exchangeability precondition in the way it breaks
   in production.

Target abstract shape:

> Conformal prediction gives distribution-free per-drive coverage guarantees for SSD
> failure prediction. We show these hold under standard splits but degrade to X% under
> leave-one-manufacturer-out evaluation on 500K production drives, and that [variant]
> restores validity at a cost of Y% wider sets.

---

## 2. Base paper

**WEFR** — Xu, Han, Lee, Liu, He, Liu. "General Feature Selection for Failure Prediction
in Large-scale SSD Deployment." IEEE/IFIP DSN 2021, pp. 263-270.

WEFR pipeline (retained as our front end):
1. Robust ensemble feature ranking with Kendall Tau outlier removal
2. Automated feature count selection via data complexity measures
3. Wear-out updating via Bayesian change point detection
4. Random Forest classifier, 30-day horizon

We keep steps 1-3, replace step 4.

---

## 3. Novelty claim — do not drift from this

**IN scope as novelty:**
- Conformal prediction sets with per-drive coverage guarantee (new output type)
- LOMO evaluation of whether that guarantee survives manufacturer shift
- Evaluation of shift-robust conformal variants and their set-size cost

**OUT of scope as novelty (implementation details only):**
- Sequence modeling over SMART trajectories — covered by Koh et al. (IEEE Access 2024)
  and MVTRF (FAST 2023)
- Multi-horizon prediction — same reason
- Any accuracy improvement over WEFR

**Do not write code, comments, or docs that frame the sequence model as the contribution.**

### Precise framing of the gap

Prior work **does** predict per-drive. The gap is not the prediction unit. It is that prior
work produces per-drive scores with:
- no defined cardinal meaning (RF vote fractions are not probabilities, never calibrated)
- no per-drive validity guarantee (only fleet-level F0.5 / AUC)
- no way to express "I don't know"

Never write "prior work only did fleet-level prediction" — that mischaracterises WEFR and a
reviewer will catch it.

---

## 4. Dataset

**Alibaba SSD SMART logs** (the WEFR dataset).

- Source: https://github.com/alibaba-edu/dcbrain/tree/master/ssd_smart_logs
- Download: https://tianchi.aliyun.com/dataset/dataDetail?dataId=95044
- Files: `smartlog2018ssd.zip` (4.33GB), `smartlog2019ssd.zip` (4.37GB),
  `ssd_failure_label.csv.zip` (140KB)
- Span: 2018-01-01 to 2019-12-31, daily
- ~500K drives, 6 models: MA1, MA2, MB1, MB2, MC1, MC2 (letter = vendor A/B/C)

**SMART log schema** (105 columns): `disk_id` (int), `ds` (date), `model` (str),
`n_i` (normalized SMART value for ID i), `r_i` (raw SMART value for ID i).

**Failure label schema** (3 columns): `model`, `disk_id`, `failure_time`.
Only failed drives appear. Healthy drives are defined by absence.

### Do not use `ssd_open_data/`
The sibling FAST'21 dataset in the same repo is single-day snapshots. No time series, so no
windowing, so no sequence model. Useless here.

### Access note
Tianchi has been inaccessible from India. Fallback plan if access fails permanently:
Backblaze HDD drive stats. The novelty claim transfers intact (it was never SSD-specific)
and gives more folds, but WEFR reproduction becomes an adaptation rather than a replication.
**Do not switch datasets without an explicit instruction.**

---

## 5. Repository structure

```
.
├── CLAUDE.md                     # this file
├── README.md
├── requirements.txt              # Phase 1 deps; later phases commented out
├── .gitignore                    # excludes data/raw, data/processed, results/
├── src/
│   ├── inspect_dataset.py        # DONE — 1000-row sampler, exploratory only
│   ├── preprocess_data.py        # DONE — daily CSVs -> single parquet + failure join
│   ├── verify_and_profile.py     # WRITTEN, NOT RUN — diagnostics + profiling
│   ├── config.py                 # TODO
│   ├── synthetic.py              # TODO
│   ├── labels.py                 # TODO
│   ├── splits.py                 # TODO
│   ├── windowing.py              # TODO
│   ├── wefr.py                   # TODO
│   ├── conformal.py              # TODO
│   ├── models.py                 # TODO
│   ├── metrics.py                # TODO
│   └── lomo.py                   # TODO
├── tests/                        # TODO
├── data/
│   ├── raw/                      # gitignored
│   │   ├── smartlog2018ssd/      # yyyymmdd.csv per day
│   │   ├── smartlog2019ssd/
│   │   └── ssd_failure_label.csv/
│   └── processed/                # gitignored
│       └── alibaba_ssd_processed.parquet
├── reports/                      # profiling output, committed
└── results/                      # gitignored
```

Note: `label2.py` at repo root is `verify_and_profile.py` under an old name. It does no
labelling. Rename it; do not add labelling logic to it.

---

## 6. Current state

**Done:**
- `preprocess_data.py` has run. Produced `data/processed/alibaba_ssd_processed.parquet`:
  all daily SMART rows consolidated, failure table left-joined on `(model, disk_id)`,
  plus derived `days_to_failure` and `failure_event`.

**Not done:**
- Verification/profiling has not been run. **No per-vendor failure counts exist yet.**
- The 30-day prediction label does not exist.
- No splits, no LOMO folds, no models, no conformal layer.

**Known issue:** `failure_event = (days_to_failure <= 0)` is a "has already failed" flag,
not the prediction target. It is 1 on roughly one row per failed drive. The real label is
`0 < days_to_failure <= 30`. Do not use `failure_event` as a training target.

---

## 7. Open risks in the existing parquet

These are checked by `verify_and_profile.py`. Resolve before building on the parquet.

1. **Positional column cast.** `preprocess_data.py` used
   `table.cast(writer.schema, safe=False)`, which aligns fields **by position, not name**.
   If any of the ~730 daily CSVs has a different column order, its values were written
   under wrong column names with no error. If the check fails, re-ingest with name-based
   alignment.
2. **Merge fan-out.** Duplicate `(model, disk_id)` in the failure table silently duplicates
   rows on the left join.
3. **Post-failure rows.** Rows with `days_to_failure < 0` need an explicit drop/keep
   decision.
4. **Right censoring.** Drives whose telemetry stops before 2019-12-31 are censored, not
   confirmed healthy. Needs an explicit rule, stated in the paper.
5. **NaN vs null.** `pd.to_numeric(errors="coerce")` produces NaN. Polars treats NaN and
   null as distinct. **Any missingness check must test both** or empty columns will report
   as fully populated. This matters most for the attribute availability matrix.

---

## 8. Decisions already made

| Decision | Rationale |
|---|---|
| Drive key is `(model, disk_id)`, never `disk_id` alone | `disk_id` is only unique within a model |
| Vendor derived as 2nd char of model code (`MA1` -> `A`) | Fail loudly if format differs |
| WEFR feature selection re-runs **inside each LOMO fold** | Running once on pooled data leaks held-out vendor labels. Most likely reviewer objection. |
| Conformal calibration set drawn **only from training vendors** | Including held-out vendor restores exchangeability and destroys the experiment |
| Splits grouped at drive level | Row-level splits leak across train/test |
| Per-fold coverage reported as **3 case studies**, not a population estimate | n=3 folds cannot support a population parameter claim |
| Conformal validated on standard split **before** LOMO | Otherwise undercoverage cannot be distinguished from a bug |
| Sequence model gets one ablation, stays out of the abstract | Not a novelty claim |
| Primary shift axis = vendor (3 folds); model-level (6 folds) as backup | Switch to leave-one-model-out if any vendor has too few failures |
| Backblaze HDD deferred to optional Step 7 | Second dataset roughly doubles Step 1 work; risks not finishing |
| Dataset adapter interface in the loader | Keeps Backblaze fallback cheap without committing to it now |

---

## 9. Constraints

- Must run on a laptop or free Colab GPU tier
- Target ~1,500 lines of Python total
- Data processing: **Polars** (not pandas, except in legacy `src/preprocess_data.py`)
- Modeling: **PyTorch**
- Utilities: **scikit-learn**
- Plotting: **Matplotlib**
- PDF generation: **ReportLab** (Platypus)
- Never load the full parquet into memory; use `scan_parquet` + streaming collect
- Streaming collect must work on both Polars 0.20 and 1.x — use the `collect()` helper
  pattern in `verify_and_profile.py`

---

## 10. Build order

Blocked on data (only these three):
- Running the profiling script
- Training any model
- Final coverage numbers

Everything else is built and tested against synthetic data first.

1. `synthetic.py` — fixture generator matching the real schema, ~200 drives, 3 vendors,
   per-vendor attribute availability differences, injected failure signal, known ground truth
2. `config.py` — horizon (30d), window length, stride, alpha, seeds, paths
3. `conformal.py` — split conformal + coverage validation on synthetic exchangeable data.
   **Priority item.** Must hit ~90% coverage at alpha=0.1 before proceeding.
4. `labels.py` + `splits.py` — with leakage assertions as tests
5. `windowing.py`
6. `wefr.py` — as `select_features(X_train, y_train) -> list[str]`, callable per fold
7. `lomo.py` — orchestration, end-to-end on synthetic with a dummy model
8. Swap in real data

---

## 11. Working style

- Direct, structured output. Tables over prose. No storytelling.
- State explicitly what is and is not being claimed.
- Flag over-broad framing without being asked.
- Prefer correcting an imprecise claim over agreeing with it.
- Before writing code that touches the parquet, check whether it can be written against
  synthetic data instead.
