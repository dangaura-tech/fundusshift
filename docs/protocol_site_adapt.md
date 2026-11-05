# Label-free site standardization: protocol

Written on 5 November 2026, before the run. Code: `experiments/site_adapt.py`.

Idea: the token features of a site are standardized with that site's own mean and standard deviation, computed from its images without labels (the frozen-feature analogue of AdaBN). Text scores are unchanged.

Variants
- `lotus`: the published LOTUS (training mean and std for every row).
- `test_site`: LOTUS heads unchanged; each unseen site (test source, Chákṣu, external cohort) is standardized with its own statistics.
- `all_site` (primary): every source, development sources included, is standardized with its own statistics; heads refit.

Evaluation
- Primary: the 50 random 7/7 rotations, seed 0 (the seed of the saved rotation logits), with the benchmark's own metrics (`revision_runs.evaluate`).
- Adoption rule for each variant, fixed now: mean rotation bAcc above LOTUS, higher on at least 30 of 50 rotations, and mean ECE after TS no more than 0.005 above LOTUS. Otherwise it is reported as a negative result.
- Messidor-2 and G1020 (published split, seeds 0-2) are **exploratory only**: their results were seen before this idea was written down, and nothing is tuned on them.
- Known risk: a site holding a single class (HYGD, PALM) or a skewed class mix shifts its own mean, which this method treats as site effect.
