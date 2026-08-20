"""Production-shaped prescription-component recommender.

One scikit-learn Pipeline per prescription component, each with its own
feature set chosen on VALIDATION, its own calibrator and its own operating
point. Nothing here re-derives a result the project has already measured; the
modules import the corpus, the text pipeline, the drug normalisation and the
engineered feature matrix that other workstreams produced.

    python -m src.phcrx.recommend.pipeline     # fit + select + calibrate
    python -m src.phcrx.recommend.evaluate     # held-out test, bootstrap CIs
    python -m src.phcrx.recommend.recommend --demo
"""
from __future__ import annotations

from ..config import RESULTS

OUT = RESULTS / "recommend"
MODELS_DIR = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
