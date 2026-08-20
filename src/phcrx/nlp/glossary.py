"""Corpus-specific clinical abbreviation glossary.

Motivation, measured rather than assumed: asked to expand the note
"D/M, HTN for several years", medgemma returned *"Dementia / Mild Cognitive
Impairment"* and coded it F02 (dementia). In this corpus D/M is **Diabetes
Mellitus** -- the same encounters carry blood-glucose readings and Z131
("screening examination for diabetes mellitus") codes.

A general-purpose medical LLM does not know a site's local shorthand. This
module mines the corpus for candidate abbreviations, resolves them against
corpus evidence (co-occurring vitals and drugs), and emits a glossary that is
injected into every downstream prompt.

    python -m src.phcrx.nlp.glossary
"""
from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd

from ..config import PROCESSED, RESULTS

OUT = RESULTS / "nlp"
OUT.mkdir(parents=True, exist_ok=True)

# Curated from corpus inspection. Each entry is asserted against evidence in
# `verify_glossary()` below, so a wrong mapping shows up as a low support count
# rather than silently corrupting every downstream prompt.
GLOSSARY: dict[str, str] = {
    "d/m": "Diabetes Mellitus",
    "dm": "Diabetes Mellitus",
    "htn": "Hypertension",
    "h/o": "History of",
    "c/o": "Complains of",
    "b/p": "Blood pressure",
    "bp": "Blood pressure",
    "p/v": "Per vaginal",
    "p/a": "Per abdomen",
    "s/b": "Seen by",
    "f/u": "Follow up",
    "o/e": "On examination",
    "n/v": "Nausea and vomiting",
    "sob": "Shortness of breath",
    "loc": "Loss of consciousness",
    "uti": "Urinary tract infection",
    "urti": "Upper respiratory tract infection",
    "ihd": "Ischaemic heart disease",
    "copd": "Chronic obstructive pulmonary disease",
    "ckd": "Chronic kidney disease",
    "cva": "Cerebrovascular accident",
    "tah": "Total abdominal hysterectomy",
    "blso": "Bilateral salpingo-oophorectomy",
    "pud": "Peptic ulcer disease",
    "gerd": "Gastro-oesophageal reflux disease",
    "ibs": "Irritable bowel syndrome",
    "osa": "Osteoarthritis",
    "ra": "Rheumatoid arthritis",
    "plid": "Prolapsed lumbar intervertebral disc",
    "lbp": "Low back pain",
    "igt": "Impaired glucose tolerance",
    "rbs": "Random blood sugar",
    "fbs": "Fasting blood sugar",
    "pbs": "Post-prandial blood sugar",
    "bmi": "Body mass index",
    "anc": "Antenatal care",
    "lmp": "Last menstrual period",
    "pid": "Pelvic inflammatory disease",
    "dub": "Dysfunctional uterine bleeding",
    "cxr": "Chest X-ray",
    "ecg": "Electrocardiogram",
    "hb": "Haemoglobin",
    "rt": "right",
    "lt": "left",
    "bil": "bilateral",
    "wt": "Weight",
    "gt": "Glucose tolerance",
}

# Frequent misreadings by general medical LLMs on this corpus, stated
# explicitly so the model cannot fall back on its prior.
DISAMBIGUATION = [
    "D/M means Diabetes Mellitus in this corpus, NOT dementia.",
    "RA means Rheumatoid Arthritis, not right atrium.",
    "PID means Pelvic Inflammatory Disease, not prolapsed intervertebral disc "
    "(that is PLID here).",
    "This is a rural primary-care / telemedicine screening cohort in "
    "Bangladesh; the dominant conditions are hypertension, diabetes, "
    "gastritis/peptic disease, musculoskeletal pain and anaemia.",
]

# Anchored on word boundaries: an unanchored {2,6} quantifier slices long words
# into fragments ('generalized' -> 'genera' + 'lized') and floods the candidate
# list with noise.
_TOK = re.compile(r"\b[a-z]+(?:[/.][a-z]+)+\b|\b[a-z]{2,6}\b")

_STOP = {
    "the", "and", "for", "of", "in", "on", "with", "no", "he", "she", "his",
    "her", "was", "has", "had", "have", "is", "are", "be", "to", "at", "as",
    "but", "not", "all", "any", "few", "per", "day", "days", "week", "weeks",
    "month", "year", "years", "since", "ago", "last", "left", "right", "both",
    "pain", "back", "body", "known", "case", "from", "that", "this", "also",
    "very", "mild", "same", "time", "times", "after", "over", "some", "one",
    "two", "three", "up", "down", "out", "off", "old", "new", "now", "yes",
}


def mine_candidates(texts: pd.Series, top: int = 60) -> list[tuple[str, int]]:
    """Slashed or short all-caps-style tokens are abbreviation candidates.

    Slashed forms (d/m, h/o) are always kept; bare short tokens are kept only
    when they are not ordinary English, to keep the curation backlog usable.
    """
    cnt = Counter()
    for t in texts.fillna(""):
        for tok in _TOK.findall(str(t).lower()):
            if "/" in tok or "." in tok:
                cnt[tok] += 1
            elif 2 <= len(tok) <= 5 and tok.isalpha() and tok not in _STOP:
                cnt[tok] += 1
    return cnt.most_common(top)


def verify_glossary(enc: pd.DataFrame) -> dict[str, dict]:
    """Check key expansions against independent corpus evidence.

    If 'd/m' really means diabetes, encounters mentioning it should show
    markedly higher blood glucose than those that do not. This turns the
    glossary from an assertion into a checked claim.
    """
    txt = enc["symptom_text"].fillna("").str.lower()
    out = {}

    def contrast(term: str, col: str) -> dict:
        has = txt.str.contains(term, regex=False)
        a = pd.to_numeric(enc.loc[has, col], errors="coerce").dropna()
        b = pd.to_numeric(enc.loc[~has, col], errors="coerce").dropna()
        return {"n_mentions": int(has.sum()),
                "mean_with": round(float(a.mean()), 1) if len(a) else None,
                "mean_without": round(float(b.mean()), 1) if len(b) else None,
                "delta": (round(float(a.mean() - b.mean()), 1)
                          if len(a) and len(b) else None)}

    out["d/m -> Diabetes Mellitus"] = {
        "evidence": "blood_glucose (mg/dL)", **contrast("d/m", "blood_glucose")}
    out["dm -> Diabetes Mellitus"] = {
        "evidence": "blood_glucose (mg/dL)", **contrast("dm", "blood_glucose")}
    out["htn -> Hypertension"] = {
        "evidence": "bp_sys (mmHg)", **contrast("htn", "bp_sys")}
    return out


def expand(text: str) -> str:
    """Rewrite a note with abbreviations expanded (longest match first)."""
    s = f" {str(text).lower()} "
    for abbr in sorted(GLOSSARY, key=len, reverse=True):
        s = re.sub(rf"(?<![a-z0-9]){re.escape(abbr)}(?![a-z0-9])",
                   GLOSSARY[abbr], s)
    return re.sub(r"\s+", " ", s).strip()


def prompt_block() -> str:
    """Glossary text injected into downstream LLM prompts."""
    lines = ["Local clinical abbreviations used at this site:"]
    lines += [f"  {a.upper()} = {v}" for a, v in sorted(GLOSSARY.items())]
    lines.append("Important disambiguations:")
    lines += [f"  - {d}" for d in DISAMBIGUATION]
    return "\n".join(lines)


def main() -> None:
    enc = pd.read_parquet(PROCESSED / "rxgen_encounters.parquet")
    cands = mine_candidates(enc["symptom_text"])
    uncovered = [(t, c) for t, c in cands if t not in GLOSSARY][:25]
    checks = verify_glossary(enc)

    (OUT / "glossary.json").write_text(json.dumps(
        {"glossary": GLOSSARY, "disambiguation": DISAMBIGUATION,
         "verification": checks,
         "top_candidates": cands[:60],
         "uncovered_candidates": uncovered}, indent=2), encoding="utf-8")

    print("=" * 68)
    print("GLOSSARY VERIFICATION (corpus evidence, not assertion)")
    for k, v in checks.items():
        print(f"  {k}")
        print(f"    n={v['n_mentions']:5d}  {v['evidence']}: "
              f"with={v['mean_with']}  without={v['mean_without']}  "
              f"delta={v['delta']:+}" if v["delta"] is not None else "    (no data)")
    print("\nTop uncovered abbreviation candidates (curation backlog):")
    print("  " + ", ".join(f"{t}({c})" for t, c in uncovered[:18]))
    print("\nExample expansions:")
    for s in ["D/M, HTN for several years.", "H/O DM, C/O LBP",
              "Known case of D/M, HTN"]:
        print(f"  {s!r}\n    -> {expand(s)!r}")
    print("=" * 68)
    print("wrote", OUT / "glossary.json")


if __name__ == "__main__":
    main()
