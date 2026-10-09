# Saved predictions behind the paper's Tables I–VII and Figs. 2–5

Produced by `notebooks/revision_runs.py` and `notebooks/revision_analysis.py`
(seed 42; 1,000 drive-clustered bootstrap resamples) on the Alibaba SSD release.

Regenerate every table, figure and in-text number:

    python -m notebooks.paper_tables paper_results/revision OUT_DIR

| file | contents |
|---|---|
| `preds_{r1,r2p7,r5}_{train,cal,test}.parquet` | per-window score `p`, label `y`, drive, vendor, wear stage, and conformal set membership per method (test) |
| `preds_wv{A,B,C}_{cal,test}.parquet` | within-vendor runs |
| `meta_*.json` | counts, AUC, config |
| `boot_*.csv`, `ratio_summary.csv` | bootstrap point estimates and 95% intervals |
| `platt_*.csv`, `reliability_*.csv` | calibration baseline |
| `age_overlap.csv` | power-on-hour percentiles by vendor (test split) |

These runs were made before the healthy-drive sampler was made order-independent
(commit eea4c39). Re-training with the current sampler draws a different healthy
subset; the files here reproduce the reported numbers exactly.
