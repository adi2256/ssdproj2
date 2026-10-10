# Final results (fixed sampler)

Output of `notebooks/revision_batch.py` run once on Kaggle (CPU, background
version) with the deterministic drive sampler (seed 42; ten seeds in
`seeds.csv`). Every table, figure and in-text number of the Springer
manuscript is generated from this folder:

    python -m notebooks.paper_tables paper_results/final OUT_DIR

Stage edges are read from `meta_r2p7.json`. `pip_freeze.txt` records the
exact package versions of the run. `paper_results/revision/` holds the
earlier pre-fix predictions and is kept only for the record.

`v5.csv` (prevalence-matched random strata) holds split and class-Mondrian
rows only. The batch run also wrote stage-Mondrian and class x stage rows,
which were invalid by construction: their calibration groups were wear
stages while their test groups were random strata, so a stage threshold
was applied to an unrelated group (marginal coverage near 0.44 in stratum
4 was the symptom). `notebooks/validate_wear.py` now restricts that
experiment to methods whose calibration and test groups share one grouping
(`_report(methods=...)`), and those rows were removed from the file. No
manuscript result used them.

The artifact index in the top-level `README.md` maps each table, figure
and in-text result to its file here, its manifest column and seed.
