"""
Evaluation layer.

metrics.py  fleet metrics (precision, recall, F0.5, AUC) for comparison
            with the base paper. Set metrics live in src/conformal.
lomo.py     experiment orchestration: prepare_fold, run_cell,
            run_experiment, and the two result tables.
"""

from .lomo import (RunConfig, coverage_gap_table, prepare_fold,
                   results_table, run_cell, run_experiment)
from .metrics import (auc, average_precision, drive_level_metrics, f_beta,
                      fleet_metrics, pick_threshold)

__all__ = ["RunConfig", "prepare_fold", "run_cell", "run_experiment",
           "results_table", "coverage_gap_table",
           "auc", "average_precision", "f_beta", "fleet_metrics",
           "drive_level_metrics", "pick_threshold"]
