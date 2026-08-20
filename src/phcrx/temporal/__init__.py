"""Forward-in-time (temporal-split) re-measurement of the deployed recommender.

The project's deployability headline -- "micro-F1 0.208 -> 0.066, a -66% drop"
-- was measured at **brand** level (719 labels) with the **autoregressive**
encoder-decoder. Both choices have since been superseded:

  * `data/processed/drug_normalization.parquet` maps 97.6% of orders onto 46
    pharmacological classes, and the cross-era *strict* unseen-order rate is
    3.4% for classes against 12.2% for brands;
  * `docs/neural_mlc.md` measured +0.109 micro-F1 from replacing the decoder
    with a multi-label head, and the deployed system is now the linear
    multi-label recommender in `src/phcrx/recommend/`.

So the number that is published as *the* deployability figure is measured on a
target and an estimator that the project no longer uses. This package
re-measures it on the target and the estimator that are actually deployed.

    python -m src.phcrx.temporal.class_temporal          # class46, both arms
    python -m src.phcrx.temporal.class_temporal --targets brand717 --c-grid 4.0

Nothing here refits or rewrites a shared artefact. In particular
`src.phcrx.preprocess --split temporal` is **not** run: it rewrites
`data/processed/rxgen_*.parquet` in place and reshuffles the vocabularies,
which silently invalidates every downstream artefact fitted against them. The
temporal split is derived in memory from the `year` column that
`rxgen_encounters.parquet` already carries.
"""
from __future__ import annotations

from ..config import RESULTS

OUT = RESULTS / "temporal"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 0

# Train <= 2015, validate on 2016, test >= 2017 -- the boundaries used by the
# published brand-level temporal result (`DataConfig.temporal_train_end` /
# `temporal_val_end`), reproduced here so the comparison keeps the era
# definition fixed and changes only the target and the estimator.
TRAIN_END = 2015
VAL_END = 2016
