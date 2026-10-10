# Final results (fixed sampler)

Output of `notebooks/revision_batch.py` run once on Kaggle (CPU, background
version) with the deterministic drive sampler (seed 42; ten seeds in
`seeds.csv`). Every table, figure and in-text number of the Springer
manuscript is generated from this folder:

    python -m notebooks.paper_tables paper_results/final OUT_DIR

Stage edges are read from `meta_r2p7.json`. `pip_freeze.txt` records the
exact package versions of the run. `paper_results/revision/` holds the
earlier pre-fix predictions and is kept only for the record.

Known issue: in `v5.csv` the `mondrian_stage` / `mondrian_both` rows for
random stratum 4 show marginal coverage near 0.44, unlike every other cell;
the manuscript uses only the split-conformal rows of that file.
