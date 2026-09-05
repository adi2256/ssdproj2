"""
Central configuration. Every downstream module reads from here rather
than hardcoding values, so horizon/window/alpha stay consistent across
label construction, windowing, conformal calibration and evaluation.

If a number matters to the experiment, it belongs in this file. The
first full run hardcoded its subset size inside a script instead, and
that single number -- 20,000 drives -- silently reduced each LOMO test
fold to about 95 failed drives and produced ROC-AUC 0.499.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
RESULTS_DIR = ROOT / "results"

# The full 108-column parquet produced by scripts/01_preprocess.py.
# 9.78 GB. Read by scripts/04_export_slim.py, and by any
# union-plus-imputation ablation that needs the columns the slim
# export drops. Not the file experiments should run on.
PARQUET = PROCESSED_DIR / "alibaba_ssd_processed.parquet"

# The slim export (scripts/04_export_slim.py): the 16 common SMART
# columns as float32, ZSTD, one file per drive model. Identical row
# count to the full parquet (273,112,284) -- only columns were dropped,
# never rows, because row filtering is build_labels()'s job.
#
# 9.78 GB -> 203 MB, a 49x reduction, because the file is sorted by
# (model, disk_id, ds) and SMART values change slowly, so consecutive
# rows compress hard.
#
# EVERYTHING DOWNSTREAM READS SLIM_GLOB.
#
# Note: 203 MB is the compressed size. Fully materialised it is roughly
# 17 GB, so keep using scan_parquet with a streaming collect rather than
# read_parquet.
SLIM_DIR = PROCESSED_DIR / "alibaba_ssd_slim"
SLIM_GLOB = str(SLIM_DIR / "*.parquet")

SMART_DIRS = [RAW_DIR / "smartlog2018ssd", RAW_DIR / "smartlog2019ssd"]
FAILURE_GLOB = str(RAW_DIR / "ssd_failure_label.csv" / "**" / "*.csv")


# ---------------------------------------------------------------
# Schema
# ---------------------------------------------------------------

META_COLS = ["disk_id", "ds", "model"]
DERIVED_COLS = ["failure_time", "days_to_failure", "failure_event"]
NON_SMART_COLS = set(META_COLS) | set(DERIVED_COLS)

# Never disk_id alone: 119,213 disk_ids appear under more than one
# drive model, so keying on it would merge unrelated drives.
DRIVE_KEY = ["model", "disk_id"]

# Vendor is the 2nd character of the model code: MA1 -> A.
# verify_and_profile.py raises if any model code breaks this.
VENDOR_SLICE = (1, 1)

DRIVE_MODELS = ["MA1", "MA2", "MB1", "MB2", "MC1", "MC2"]
VENDORS = ["A", "B", "C"]

# Fleet totals, from reports/vendor_profile.csv. Kept here so scripts
# can sanity-check what they loaded against what should be there.
FLEET_DRIVES = 475_056
FLEET_FAILED = 16_305
FLEET_ROWS = 273_112_284


@dataclass(frozen=True)
class Config:
    # -- Label construction -------------------------------------
    horizon_days: int = 30          # WEFR's prediction horizon
    drop_post_failure: bool = True  # rows with days_to_failure <= 0

    # The last `horizon_days` of any drive that did not fail. For a
    # drive last seen at T, the interval (T, T + horizon] is never
    # observed, so "does it fail within 30 days" has no answer.
    # Labelling those rows 0 asserts survival nobody witnessed.
    # Applies identically to right-censored drives and to drives
    # surviving to 2019-12-31; nothing special-cases censoring.
    #
    # Diagnostics: 120,531 healthy drives end early, median 82 days
    # before the global end, so this is not a corner case.
    drop_unobservable: bool = True

    # -- Windowing ----------------------------------------------
    window_len: int = 30            # daily observations per sequence
    stride: int = 1
    min_valid_frac: float = 0.5     # min non-missing fraction to keep

    # Windows whose label is 1 are sampled at this stride. None means
    # "same as stride", i.e. no oversampling, which is the default and
    # should stay that way for headline results: the central finding is
    # that failure-class coverage collapses at low prevalence, and
    # inflating prevalence here would mask exactly that effect.
    positive_stride: int | None = None

    # -- Conformal ----------------------------------------------
    alpha: float = 0.10             # target miscoverage -> 90% coverage
    cal_frac: float = 0.20          # of the non-test drive pool

    # -- Splits -------------------------------------------------
    test_frac: float = 0.20         # standard (non-LOMO) split only
    seed: int = 42

    # -- Subsampling --------------------------------------------
    # The first full run sampled 20,000 of 475,056 drives inside a
    # script. That left ~686 failed drives fleet-wide and ~95 in each
    # LOMO test fold. Windows from one drive are 30 overlapping views
    # of the same failure, so the effective sample size is the number
    # of failed DRIVES, not windows -- too few to learn a cross-vendor
    # signal from or to calibrate a 90% guarantee against. ROC-AUC came
    # out at 0.499 and 0.502 on LOMO-A and LOMO-B.
    #
    # The slim export is 203 MB, so there is no longer any reason to
    # subsample. Leave subset_n_drives at None to use the whole fleet.
    #
    # If memory forces a subset, keep_all_failed retains every one of
    # the 16,305 failed drives and thins only the healthy ones. That
    # raises drive-level prevalence above the fleet value, so the
    # healthy sampling ratio must be reported alongside any result.
    subset_n_drives: int | None = None    # None = whole fleet
    keep_all_failed: bool = True

    # Report failed-drive counts, not just positive-window counts, in
    # every results table. Positive windows overstate statistical power
    # by roughly 27x.
    report_drive_counts: bool = True

    # -- WEFR ---------------------------------------------------
    n_rankers: int = 10
    kendall_outlier_z: float = 2.0

    # Feature selection re-runs INSIDE each LOMO fold, on training
    # vendors only. Selecting once on pooled data lets the held-out
    # vendor's labels influence the feature set, which is the most
    # likely objection a reviewer will raise.
    feature_selection: bool = True

    # -- Synthetic fixtures -------------------------------------
    # Calibrated to reports/vendor_profile.csv. Per-model failure rates
    # live in data/synthetic.py; failure_scale multiplies them so small
    # fixtures carry enough positives while preserving the real A<B<C
    # ordering. censor_rate matches the real 120,531/475,056 = 25%.
    synth_n_drives: int = 400
    synth_days: int = 180
    synth_failure_scale: float = 4.0
    synth_censor_rate: float = 0.28
    synth_seed: int = 42

    # -- LOMO ---------------------------------------------------
    lomo_axis: str = "vendor"       # "vendor" (3 folds) | "model" (6)
    folds: tuple = field(default_factory=lambda: tuple(VENDORS))

    # -- Results ------------------------------------------------
    # run_experiment writes one JSON per (fold, model, method) here and
    # skips cells already on disk, so a run cut short by a cloud session
    # timeout resumes instead of restarting.
    results_dir: str = "results"


CFG = Config()