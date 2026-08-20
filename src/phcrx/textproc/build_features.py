"""Materialise the text features for every encounter.

    python -m src.phcrx.textproc.build_features

Writes `data/processed/text_features.parquet`, keyed by prescription_id:

    symptom_text      the raw note, for provenance
    normalized_text   mojibake -> glossary -> orthography -> typo correction
    complaint_spans   list[str], one entry per complaint
    n_spans, no_complaint
    cpt_NNN_<slug>    multi-hot over the induced concept vocabulary

The normaliser, corrector and concept vocabulary are all fitted on the TRAIN
split only, then applied to every row.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from ..config import PROCESSED, RESULTS
from ..nlp.icd_index import SapBertEncoder
from .concepts import ConceptVocab
from .normalize import Normalizer
from .segment import segment

OUT = RESULTS / "textproc"
OUT.mkdir(parents=True, exist_ok=True)


def _slug(s: str, n: int = 28) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")
    return (s[:n].rstrip("_") or "concept")


def main() -> None:
    enc = pd.read_parquet(PROCESSED / "rxgen_encounters.parquet")
    enc["symptom_text"] = enc["symptom_text"].fillna("").astype(str)
    train_texts = enc.loc[enc.split == "train", "symptom_text"].tolist()

    nz = Normalizer().fit(train_texts)
    norm = nz.transform(enc["symptom_text"].tolist())
    spans = [segment(s) for s in norm]

    encoder = SapBertEncoder()
    try:
        cv = ConceptVocab.load()
    except Exception:
        cv = ConceptVocab.induce(nz.transform(train_texts), encoder)
        cv.save()
    M, flag = cv.assign(norm, encoder)

    out = pd.DataFrame({
        "prescription_id": enc["prescription_id"],
        "split": enc["split"],
        "symptom_text": enc["symptom_text"],
        "normalized_text": norm,
        "complaint_spans": spans,
        "n_spans": [len(s) for s in spans],
        "no_complaint": flag.astype(np.int8),
    })
    used = set()
    cols = []
    for i, lab in enumerate(cv.labels):
        name = f"cpt_{i:03d}_{_slug(lab)}"
        while name in used:
            name += "_x"
        used.add(name)
        cols.append(name)
        out[name] = M[:, i]

    path = PROCESSED / "text_features.parquet"
    out.to_parquet(path, index=False)

    with_text = out["symptom_text"].str.strip() != ""
    print(f"rows={len(out)}  with_text={int(with_text.sum())}")
    print(f"concepts={len(cv)}  "
          f"coverage={float((M[with_text.to_numpy()].sum(1) > 0).mean()):.1%} of "
          f"notes with text carry >=1 concept")
    print(f"mean spans/note (text only)={out.loc[with_text, 'n_spans'].mean():.2f}  "
          f"no_complaint={int(out['no_complaint'].sum())}")
    print(f"corrector applied {len(nz.corrector.mapping)} distinct token corrections")
    print("wrote", path)

    (OUT / "corrections.json").write_text(json.dumps(
        dict(sorted(nz.corrector.mapping.items())), indent=1, ensure_ascii=False),
        encoding="utf-8")
    (OUT / "concept_columns.json").write_text(json.dumps(
        {c: l for c, l in zip(cols, cv.labels)}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print("wrote", OUT / "corrections.json")


if __name__ == "__main__":
    main()
