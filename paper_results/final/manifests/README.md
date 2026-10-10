# Per-run manifests

Written by `notebooks/export_manifests.py` on Kaggle from the public Alibaba
SSD SMART release, with the deterministic sampler (drive list sorted by
(model, disk_id) before `DataFrame.sample(seed)`).

| File | Content |
|---|---|
| `samples.json` | per sample: tag, requested size, seed, drives sampled, failed drives, drives left after labelling, fingerprint |
| `experiments.json` | configuration of every experiment (stride, training cap, trees, normalisation, features, alpha, seeds) |
| `drives_<tag>.csv.gz` | every sampled drive: model, disk_id, failed, vendor, drive_label, in_labelled, `part_standard` (train/cal/test) |
| `drives_r2p7.csv.gz` | additionally: `part_within_vendor`, `part_within_model`, `part_holdout_vendor_{A,B,C}`, `modal_stage`, `part_holdout_stage_{0..4}` |
| `drive_index_<dump>_<part>.csv.gz` | `drive_idx` -> (model, disk_id) for each released prediction file |
| `transfer_diagnostics.csv`, `transfer_sign_bootstrap.csv` | Sect. 5.5 range, scale and correlation-sign diagnostics on the 60k seed-42 sample |
| `consistency_check.csv` | released predictions vs manifests (see below) |
| `cross_session_check.csv` | the same manifests produced in two independent Kaggle sessions, compared file by file |

Tags: `r2p7` = 60,000 drives (main), `r1` = 32,610, `r5` = 97,830 (all seed 42);
`seed0`..`seed8` = 60,000 drives at seeds 0-8.

Checks performed:

* Fingerprints: `r2p7` is `ee3ce82fe326`, the value printed by the run that
  produced the released predictions.
* Two independent Kaggle sessions produced identical manifests and drive
  indices for every file they share (`cross_session_check.csv`).
* `consistency_check.csv`: for each of the 12 calibration/test prediction
  files of the ratio (r1, r2p7, r5) and within-vendor runs, the
  drive index has exactly the drives present in the file, every drive is in
  the recorded split part, every window maps to a drive of the recorded
  vendor, and every positive window maps to a failed drive.
* The wear-stage hold-out reproduces the original per-stage drive counts
  (19,347 / 5,181 / 11,980 / 9,541 / 11,726).

The exporter's own window-by-window comparison could not run on Kaggle in
these sessions because the prediction files were not yet in the repository
(they were excluded by `.gitignore`); they are committed now, so
`run(load_slim, extras=False)` performs that comparison directly.

The within-drive-model prediction files (`preds_wm*`) have split assignments
here (`part_within_model`) but no drive-index map, so they are not covered by
the consistency check. The training-window subsample, the random strata of
the prevalence control and the bootstrap draws are not listed; they are
derived from the recorded seed by the released code.
