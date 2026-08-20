"""Text preprocessing for the symptom free-text field.

The symptom note is the dominant predictive signal in this cohort (shuffling
it across patients at inference costs -0.126 micro-F1, a 66% relative drop,
versus -0.025 for vitals). The production tokeniser in `preprocess.py` is a
bare regex over the raw string, so the encoder sees `D/M`, `H/O`, `LBP` and
`diabetis` as opaque, unrelated types.

This package builds and *evaluates* a replacement:

    lexicon.py     medical + corpus vocabulary; bounded-edit-distance corrector
    normalize.py   mojibake -> glossary -> orthography -> typo correction
    segment.py     multi-complaint notes -> per-complaint spans
    concepts.py    corpus-grounded concept vocabulary (frequency + SapBERT)
    diagnostics.py measures what the current tokenisation loses
    evaluate.py    ablation on a proxy task (drug-class multi-label)
    build_features.py  writes data/processed/text_features.parquet

Nothing here modifies the production pipeline; adoption is a separate step
that the ablation table in docs/text_preprocessing.md is meant to justify.
"""
