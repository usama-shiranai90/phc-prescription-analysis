"""Feature engineering and feature-importance analysis for PHC-RxGen.

`build_features` turns the encounter/order tables into a single wide, one-row-
per-encounter design matrix (`data/processed/features.parquet`) with an
explicit feature -> family map; `importance` fits interpretable gradient-boosted
models per prescription component and reports permutation + mutual-information
importance at both the individual-feature and family level.
"""
