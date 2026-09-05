"""
Base-model fix: does removing vendor-specific LEVEL restore transfer?

Run as a Kaggle cell after notebooks/kaggle_setup.setup().

WHAT THIS TESTS

Every experiment so far used flatten(ws, "last") -- the final timestep's
raw level. Levels are vendor conventions: different scales, different
baselines, at least one attribute whose correlation with failure
reverses sign between vendors. A Random Forest fitted on levels learns
vendor-specific split points, which is consistent with what we measured:

    within-fleet AUC   0.73
    LOMO-A             0.42   (inverted)
    LOMO-B             0.50   (chance)
    LOMO-C             0.65

Two representations remove level:

    normalize="per_drive"   scale each window against its own first days
    flatten_how="dynamics"  slope, delta, volatility, change rate

The hypothesis: a drive whose reallocated-sector count is RISING is
deteriorating whatever units its vendor reports in. Level is a vendor
convention; trend is closer to physics.

WHAT COUNTS AS SUCCESS

    LOMO-A AUC >= 0.60 on any (normalize, flatten) pair.

That is the gate. Below it, the shared attributes do not carry
transferable signal in any form we have tried, and the project pivots to
a negative result. At or above it, the conformal experiment can finally
be run on a model that works, and leave-one-MODEL-out onboarding becomes
the target paper.

Note the synthetic fixture shows large gains from dynamics, but its
injected signal IS a ramp, so that confirms the implementation rather
than predicting real-data behaviour. This run is the real test.
"""

import polars as pl

from src.data.labels import build_labels
from src.evaluation import RunConfig, run_experiment

# Assumes `load_slim` from notebooks.kaggle_setup is in scope.
df = load_slim(n_drives=60_000)          # noqa: F821
labelled, stats = build_labels(df)
print(stats)

FOLDS = ["standard", "lomo_A", "lomo_B", "lomo_C"]
rows = []

for method in ("zscore", "rank", "per_drive"):
    for how in ("last", "dynamics", "dynamics_only"):
        cfg = RunConfig(
            stride=30,
            seed=42,
            normalize=method,
            flatten_how=how,
            # Off deliberately. With only five universally non-constant
            # attributes, WEFR selecting 5-7 per fold is a confound; we
            # are testing the representation, not the selector.
            feature_selection=False,
            drop_constant_test_features=True,
            models=("random_forest",),
            methods=("split",),
            results_dir=f"/kaggle/working/fix_{method}_{how}",
        )
        r = run_experiment(labelled, cfg, verbose=False)
        d = {row["split"]: row for row in r.iter_rows(named=True)}
        rows.append({
            "normalize": method,
            "flatten": how,
            **{f"auc_{k.replace('lomo_', '')}": d[k]["win_auc"]
               for k in FOLDS if k in d},
            **{f"cov_{k.replace('lomo_', '')}": d[k]["empirical_coverage"]
               for k in FOLDS if k in d},
            **{f"fail_{k.replace('lomo_', '')}": d[k]["coverage_failure"]
               for k in FOLDS if k in d},
            "n_feat": d["standard"]["n_features"],
        })
        print(f"{method:<10}{how:<15}" + "".join(
            f"{d[k]['win_auc']:>9.3f}" for k in FOLDS if k in d))

res = pl.DataFrame(rows)

print("\n=== AUC ===")
print(res.select(["normalize", "flatten", "n_feat",
                  "auc_standard", "auc_A", "auc_B", "auc_C"]))

print("\n=== marginal coverage (target 0.90) ===")
print(res.select(["normalize", "flatten",
                  "cov_standard", "cov_A", "cov_B", "cov_C"]))

print("\n=== failure-class coverage ===")
print(res.select(["normalize", "flatten",
                  "fail_standard", "fail_A", "fail_B", "fail_C"]))

# -- the gate ---------------------------------------------------
best = res.sort("auc_A", descending=True).row(0, named=True)
print(f"\nbest LOMO-A: {best['auc_A']:.3f} "
      f"({best['normalize']}, {best['flatten']})")

if best["auc_A"] >= 0.60:
    print("\nPASS -- level was the problem. Next: leave-one-MODEL-out "
          "with this representation, then the onboarding experiment.")
elif best["auc_A"] > 0.55:
    print("\nPARTIAL -- representation helps but does not clear the gate. "
          "Try per-vendor rank on dynamics features, and check whether "
          "leave-one-MODEL-out (same vendor, consistent semantics) "
          "transfers even if leave-one-VENDOR-out does not.")
else:
    print("\nFAIL -- the shared attributes do not carry transferable "
          "signal in any representation tried. Check leave-one-MODEL-out "
          "before concluding; if that also fails, the honest output is a "
          "negative result.")

res.write_csv("/kaggle/working/base_model_fix.csv")
