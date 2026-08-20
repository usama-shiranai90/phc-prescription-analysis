"""Measure what the production tokenisation loses.

    python -m src.phcrx.textproc.diagnostics

Reports, on the frozen train/test split:
  * out-of-vocabulary rate of the 1,420-type word vocabulary,
  * how much of that OOV mass is a *spelling variant* of a term the vocabulary
    already contains (i.e. recoverable, not genuinely novel),
  * how vocabulary size and OOV rate move as each normalisation stage is added,
  * abbreviation load, and how much of it the verified glossary covers,
  * what expanding an abbreviation costs in discriminativeness -- the IDF of
    the abbreviation type against the IDF of the words it expands to, which is
    the mechanism behind the negative result for glossary expansion,
  * how many notes carry more than one complaint.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter

import pandas as pd

from ..config import PROCESSED, RESULTS, DataConfig
from ..nlp.glossary import GLOSSARY
from ..preprocess import tokenize
from .lexicon import TypoCorrector, edit_budget
from .normalize import Normalizer
from .segment import is_no_complaint, segment

OUT = RESULTS / "textproc"
OUT.mkdir(parents=True, exist_ok=True)


def load_encounters() -> pd.DataFrame:
    enc = pd.read_parquet(PROCESSED / "rxgen_encounters.parquet")
    enc["symptom_text"] = enc["symptom_text"].fillna("").astype(str)
    return enc


def _vocab(token_lists, min_freq: int) -> tuple[set, Counter]:
    c = Counter(t for toks in token_lists for t in toks)
    return {w for w, f in c.items() if f >= min_freq}, c


def _oov(token_lists, vocab: set) -> dict:
    toks = [t for row in token_lists for t in row]
    oov = [t for t in toks if t not in vocab]
    return {
        "n_tokens": len(toks), "n_types": len(set(toks)),
        "oov_tokens": len(oov), "oov_types": len(set(oov)),
        "oov_token_rate": round(len(oov) / max(len(toks), 1), 4),
        "oov_type_rate": round(len(set(oov)) / max(len(set(toks)), 1), 4),
    }


def stage_table(enc: pd.DataFrame, min_freq: int) -> pd.DataFrame:
    """Vocabulary size and test OOV under cumulative normalisation stages."""
    tr_txt = enc.loc[enc.split == "train", "symptom_text"].tolist()
    te_txt = enc.loc[enc.split == "test", "symptom_text"].tolist()

    configs = [
        ("current (regex only)", dict(glossary=False, orthography=False, typo=False)),
        ("+ glossary", dict(glossary=True, orthography=False, typo=False)),
        ("+ orthography", dict(glossary=True, orthography=True, typo=False)),
        ("+ typo correction", dict(glossary=True, orthography=True, typo=True)),
    ]
    rows = []
    for name, kw in configs:
        nz = Normalizer(**kw).fit(tr_txt)
        tr_tok = [tokenize(s) for s in nz.transform(tr_txt)]
        te_tok = [tokenize(s) for s in nz.transform(te_txt)]
        vocab, _ = _vocab(tr_tok, min_freq)
        stats = _oov(te_tok, vocab)
        rows.append({"stage": name, "train_vocab": len(vocab), **stats})
    return pd.DataFrame(rows)


def recoverable_oov(enc: pd.DataFrame, min_freq: int) -> dict:
    """How much test OOV mass is a spelling variant of an in-vocabulary term.

    Uses the same trusted vocabulary and edit budget as the corrector, so this
    is a direct estimate of the headroom typo correction can address -- not a
    generic 'nearest string' count.
    """
    tr_txt = enc.loc[enc.split == "train", "symptom_text"].tolist()
    te_txt = enc.loc[enc.split == "test", "symptom_text"].tolist()
    base = Normalizer(glossary=False, orthography=False, typo=False)
    tr_tok = [tokenize(s) for s in base.transform(tr_txt)]
    te_tok = [tokenize(s) for s in base.transform(te_txt)]
    vocab, tr_counts = _vocab(tr_tok, min_freq)

    corr = TypoCorrector(min_corpus_freq=min_freq).fit(base.transform(tr_txt))
    oov = Counter(t for row in te_tok for t in row if t not in vocab)

    fixable, examples = Counter(), []
    for tok, n in oov.items():
        if not tok.isalpha() or len(tok) < 4:
            continue
        hit = corr._nearest(tok, corr.trusted, edit_budget(tok))
        if hit is not None and hit != tok:
            fixable[tok] = n
            if len(examples) < 40:
                examples.append((tok, hit, n))
    total_oov = sum(oov.values())
    return {
        "test_oov_tokens": total_oov,
        "test_oov_types": len(oov),
        "recoverable_tokens": sum(fixable.values()),
        "recoverable_types": len(fixable),
        "recoverable_token_share": round(sum(fixable.values()) / max(total_oov, 1), 3),
        "examples": [{"oov": a, "mapped_to": b, "n": n} for a, b, n in
                     sorted(examples, key=lambda x: -x[2])],
    }


def abbreviation_load(enc: pd.DataFrame) -> dict:
    """Share of tokens that are glossary abbreviations, and their reach."""
    tok_lists = [tokenize(s) for s in enc["symptom_text"]]
    toks = [t for row in tok_lists for t in row]
    c = Counter(toks)
    covered = {a: c.get(a, 0) for a in GLOSSARY}
    hit = {a: n for a, n in covered.items() if n}
    n_notes = sum(1 for row in tok_lists if any(t in GLOSSARY for t in row))
    return {
        "total_tokens": len(toks),
        "abbrev_tokens": sum(hit.values()),
        "abbrev_token_share": round(sum(hit.values()) / max(len(toks), 1), 4),
        "glossary_entries_seen": len(hit),
        "glossary_entries_total": len(GLOSSARY),
        "notes_with_abbrev": n_notes,
        "notes_total": int((enc["symptom_text"].str.strip() != "").sum()),
        "top": sorted(hit.items(), key=lambda kv: -kv[1])[:20],
    }


def glossary_dispersion(enc: pd.DataFrame, min_df: int = 10) -> dict:
    """What expanding an abbreviation costs a bag-of-words model.

    An abbreviation is a rare, highly specific type: `lbp` occurs in 333 train
    notes out of 9,900, so its IDF is high and a linear model can give it a
    large weight without over-firing. Expanding it to `low back pain` deletes
    that type and adds its mass to `low`, `back` and `pain`, each of which
    already occurs in thousands of notes for unrelated reasons. The expansion
    is more *readable* and less *discriminative*, and this table measures the
    second half of that trade directly.

    Also counts how often the corpus writes the expansion out in full, which
    is the opposite argument: those notes and the abbreviated ones are the
    same concept and a bag-of-words model has no way to know it.
    """
    tr = enc.loc[enc.split == "train", "symptom_text"]
    n_docs = len(tr)
    doc_tokens = [set(tokenize(s)) for s in tr]
    df = Counter()
    for toks in doc_tokens:
        df.update(toks)
    lowered = tr.str.lower()

    rows = []
    for abbr, value in GLOSSARY.items():
        d_abbr = df.get(abbr, 0)
        if d_abbr < min_df:
            continue
        words = [w for w in re.findall(r"[a-z]+", value.lower()) if len(w) >= 3]
        if not words:
            continue
        # Notes that already spell the expansion out, with no abbreviation.
        phrase = r"(?<![a-z0-9])" + r"[^a-z0-9]+".join(
            re.escape(w) for w in re.findall(r"[a-z0-9]+", value.lower())) \
            + r"(?![a-z0-9])"
        spelled = int(lowered.str.contains(phrase, regex=True).sum())
        comps = [{"word": w, "df_train": df.get(w, 0),
                  "df_after_expansion": df.get(w, 0) + d_abbr} for w in words]
        idf_abbr = math.log(n_docs / max(d_abbr, 1))
        idf_after = sum(math.log(n_docs / max(c["df_after_expansion"], 1))
                        for c in comps) / len(comps)
        rows.append({
            "abbrev": abbr, "expansion": value,
            "df_train": d_abbr, "spelled_out_train_notes": spelled,
            "idf_abbrev": round(idf_abbr, 2),
            "mean_idf_of_expansion": round(idf_after, 2),
            "idf_lost": round(idf_abbr - idf_after, 2),
            "components": comps,
        })
    rows.sort(key=lambda r: -r["df_train"])
    multi = [r for r in rows if len(r["components"]) > 1]
    return {
        "n_train_notes": n_docs,
        "n_abbrevs_measured": len(rows),
        "mean_idf_abbrev": round(sum(r["idf_abbrev"] for r in multi)
                                 / max(len(multi), 1), 2),
        "mean_idf_of_expansion": round(sum(r["mean_idf_of_expansion"] for r in multi)
                                       / max(len(multi), 1), 2),
        "mean_idf_lost": round(sum(r["idf_lost"] for r in multi)
                               / max(len(multi), 1), 2),
        "abbrev_notes_total": sum(r["df_train"] for r in rows),
        "spelled_out_notes_total": sum(r["spelled_out_train_notes"] for r in rows),
        "rows": rows,
    }


def complaint_structure(enc: pd.DataFrame) -> dict:
    txt = enc.loc[enc["symptom_text"].str.strip() != "", "symptom_text"]
    spans = [segment(t) for t in txt]
    n = [len(s) for s in spans]
    nc = sum(1 for t in txt if is_no_complaint(t))
    return {
        "notes_with_text": len(txt),
        "no_complaint_notes": nc,
        "no_complaint_share": round(nc / max(len(txt), 1), 3),
        "mean_spans": round(sum(n) / max(len(n), 1), 2),
        "notes_multi_complaint": sum(1 for k in n if k >= 2),
        "multi_complaint_share": round(sum(1 for k in n if k >= 2) / max(len(n), 1), 3),
        "span_count_hist": dict(sorted(Counter(n).items())),
    }


def main() -> None:
    cfg = DataConfig()
    enc = load_encounters()
    print(f"encounters={len(enc)}  with_text="
          f"{int((enc['symptom_text'].str.strip() != '').sum())}  "
          f"unique={enc['symptom_text'].nunique()}")

    print("\n=== 1. abbreviation load (whole corpus, current tokenisation) ===")
    ab = abbreviation_load(enc)
    print(f"  {ab['abbrev_tokens']}/{ab['total_tokens']} tokens "
          f"({ab['abbrev_token_share']:.2%}) are glossary abbreviations")
    print(f"  {ab['notes_with_abbrev']}/{ab['notes_total']} notes with text "
          f"({ab['notes_with_abbrev']/max(ab['notes_total'],1):.1%}) contain at least one")
    print(f"  {ab['glossary_entries_seen']}/{ab['glossary_entries_total']} "
          f"glossary entries actually occur")
    print("  top: " + ", ".join(f"{a}({n})" for a, n in ab["top"][:12]))

    print("\n=== 2. vocabulary / OOV under cumulative normalisation ===")
    tbl = stage_table(enc, cfg.min_word_freq)
    print(tbl.to_string(index=False))

    print("\n=== 3. how much test OOV is a recoverable spelling variant ===")
    rec = recoverable_oov(enc, cfg.min_word_freq)
    print(f"  test OOV: {rec['test_oov_tokens']} tokens / {rec['test_oov_types']} types")
    print(f"  of which within edit budget of a trusted term: "
          f"{rec['recoverable_tokens']} tokens ({rec['recoverable_token_share']:.1%}), "
          f"{rec['recoverable_types']} types")
    print("  examples: " + ", ".join(
        f"{e['oov']}->{e['mapped_to']}({e['n']})" for e in rec["examples"][:14]))

    print("\n=== 4. what expansion costs in discriminativeness (TRAIN) ===")
    gd = glossary_dispersion(enc)
    print(f"  {gd['n_abbrevs_measured']} abbreviations occur in >=10 train notes")
    print(f"  mean IDF of the abbreviation type: {gd['mean_idf_abbrev']}")
    print(f"  mean IDF of its expansion words:   {gd['mean_idf_of_expansion']}  "
          f"(lost {gd['mean_idf_lost']} nats per expansion)")
    print(f"  {gd['abbrev_notes_total']} abbreviated mentions vs "
          f"{gd['spelled_out_notes_total']} spelled-out ones in train")
    print(f"  {'abbrev':7s} {'df':>5s} {'spelled':>8s} {'idf':>5s} {'idf_exp':>8s}  expansion")
    for r in gd["rows"][:10]:
        print(f"  {r['abbrev']:7s} {r['df_train']:5d} {r['spelled_out_train_notes']:8d} "
              f"{r['idf_abbrev']:5.2f} {r['mean_idf_of_expansion']:8.2f}  {r['expansion']}")

    print("\n=== 5. complaint structure ===")
    cs = complaint_structure(enc)
    print(f"  no-complaint notes: {cs['no_complaint_notes']} "
          f"({cs['no_complaint_share']:.1%})")
    print(f"  multi-complaint notes: {cs['notes_multi_complaint']} "
          f"({cs['multi_complaint_share']:.1%}), mean spans={cs['mean_spans']}")
    print(f"  span-count histogram: {cs['span_count_hist']}")

    (OUT / "diagnostics.json").write_text(json.dumps({
        "abbreviation_load": ab,
        "stage_table": tbl.to_dict("records"),
        "recoverable_oov": rec,
        "glossary_dispersion": gd,
        "complaint_structure": cs,
    }, indent=2), encoding="utf-8")
    print("\nwrote", OUT / "diagnostics.json")


if __name__ == "__main__":
    main()
