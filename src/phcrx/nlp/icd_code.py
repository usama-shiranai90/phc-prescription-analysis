"""Assign ICD-10 codes to encounters, and validate the assignment.

Two independent coders over the same candidate pool:
  * retrieval  - hybrid SapBERT + char-TFIDF top-1 (cheap, runs on all 14,074)
  * LLM        - medgemma via local Ollama, choosing from the top-k candidates
                 with the corpus glossary injected (expensive, run on a sample)

Agreement between them is the validation signal. The corpus's own ICD labels
cannot serve as ground truth: there are only 195 of them and they are mostly
external-cause/screening codes, not diagnoses.

The LLM is constrained to the retrieved candidate list plus an explicit
"none of these" option, so it can never invent a code that does not exist.

    python -m src.phcrx.nlp.icd_code --sample 200
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
import pandas as pd

import re

from ..config import PROCESSED, RESULTS
from .glossary import prompt_block
from .ollama_client import Ollama, OllamaError

# Nearest-neighbour retrieval always returns *something*. On notes that record
# no complaint it returned confident nonsense: "NO" -> I96 Gangrene (n=192),
# "No complaint" -> T81 Complications of procedures (n=243). These notes are
# screening visits, not diagnoses, and must be excluded before coding.
NO_COMPLAINT = re.compile(
    r"^\s*(?:\d+[\.\)]\s*)?(?:"
    r"no|nil|none|n/?a|no\s+complaints?|no\s+complaint\s+right\s+now|"
    r"no\s+specific\s+complaints?|nothing|nothing\s+specific|"
    r"general\s+check\s*-?\s*up|check\s*-?\s*up|routine\s+check\s*-?\s*up|"
    r"for\s+check\s*-?\s*up|healthy|well|normal|good|ok"
    r")\s*[\.\,]?\s*$", re.I)

# Below this hybrid score the top-1 candidate is not trustworthy. Chosen from
# the observed score distribution (median 0.443) and inspection of what falls
# either side of it.
MIN_SCORE = 0.50


def is_no_complaint(text: str) -> bool:
    return bool(NO_COMPLAINT.match(str(text or "").strip()))

CACHE = PROCESSED / "icd_index"
OUT = RESULTS / "nlp"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM = (
    "You are a careful clinical coder working with primary-care records from a "
    "rural telemedicine programme in Bangladesh. You assign ICD-10 codes "
    "conservatively: choose only what the note supports, and prefer a symptom "
    "code over a disease code when the note records a symptom rather than a "
    "confirmed diagnosis."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "codes": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    },
    "required": ["codes", "confidence"],
}


def build_prompt(note: str, expanded: str, candidates: list[dict],
                 vitals: str = "") -> str:
    cand = "\n".join(f"  {c['icd']}  {c['descr']}" for c in candidates)
    return f"""{prompt_block()}

Clinical note (verbatim): "{note}"
Note with abbreviations expanded: "{expanded}"
{f'Recorded vitals: {vitals}' if vitals else ''}

Candidate ICD-10 categories retrieved for this note:
{cand}

Choose the 1-3 candidate codes best supported by the note.
Rules:
- Use ONLY codes from the candidate list above. Never invent a code.
- If none of the candidates fit, return an empty list.
- If the note records no clinical complaint (e.g. "no complaints",
  "general check up"), return an empty list.
Return JSON: {{"codes": [...], "confidence": "high|medium|low", "reason": "..."}}"""


def vitals_str(row) -> str:
    bits = []
    for col, lab, fmt in (("bp_sys", "BP-sys", "{:.0f}"), ("bp_dia", "BP-dia", "{:.0f}"),
                          ("blood_glucose", "glucose", "{:.0f} mg/dL"),
                          ("bmi", "BMI", "{:.1f}"),
                          ("blood_hemoglobin", "Hb", "{:.1f} g/dL")):
        v = row.get(col)
        if pd.notna(v):
            bits.append(f"{lab}={fmt.format(float(v))}")
    return ", ".join(bits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200,
                    help="encounters to adjudicate with the LLM; -1 = whole "
                         "confident tier (LLM becomes the authoritative coder)")
    ap.add_argument("--model", default="medgemma:latest")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cands = pd.read_parquet(CACHE / "icd_candidates.parquet")
    enc = pd.read_parquet(PROCESSED / "rxgen_encounters.parquet")
    df = cands.merge(enc, on="prescription_id", how="left", suffixes=("", "_e"))
    df = df[df["symptom_text"].fillna("").str.strip() != ""].reset_index(drop=True)

    # --- retrieval coder over the whole corpus ---------------------------
    df["raw_top1"] = df["candidates"].map(lambda c: c[0]["icd"] if len(c) else None)
    df["retrieval_score"] = df["candidates"].map(
        lambda c: c[0]["score"] if len(c) else 0.0)
    df["no_complaint"] = df["symptom_text"].map(is_no_complaint)

    # Three tiers, so downstream users can trade coverage against precision
    # instead of inheriting a single silent choice.
    df["tier"] = np.where(
        df["no_complaint"], "no_complaint",
        np.where(df["retrieval_score"] >= MIN_SCORE, "confident", "low_confidence"))
    df["retrieval_top1"] = np.where(df["tier"] == "confident", df["raw_top1"], None)

    n_nc = int(df["no_complaint"].sum())
    n_conf = int((df["tier"] == "confident").sum())
    n_low = int((df["tier"] == "low_confidence").sum())
    print(f"\nencounters with symptom text : {len(df)}")
    print(f"  no complaint recorded      : {n_nc:6d} ({n_nc/len(df):5.1%})  -> no code")
    print(f"  confident (score >= {MIN_SCORE:.2f})   : {n_conf:6d} ({n_conf/len(df):5.1%})  -> coded")
    print(f"  low confidence             : {n_low:6d} ({n_low/len(df):5.1%})  -> withheld")

    keep = ["prescription_id", "symptom_text", "expanded", "tier",
            "raw_top1", "retrieval_top1", "retrieval_score"]
    df[keep].to_parquet(PROCESSED / "icd_autocoded.parquet", index=False)

    coded = df[df["tier"] == "confident"]
    ch = Counter(c[0] for c in coded["retrieval_top1"].dropna())
    print("\nICD chapter distribution (confident tier):")
    for c, n in sorted(ch.items(), key=lambda kv: -kv[1])[:10]:
        print(f"   {c}  {n:6d}  ({100*n/max(len(coded),1):5.1f}%)")
    descr = {c["icd"]: c["descr"] for row in df["candidates"] for c in row}
    print("\nTop 15 assigned categories (confident tier):")
    for code, n in Counter(coded["retrieval_top1"].dropna()).most_common(15):
        print(f"   {code:5s} {n:5d}  {descr.get(code, '')[:52]}")

    dropped = Counter(df.loc[df["tier"] != "confident", "raw_top1"].dropna())
    print("\nSpurious codes removed by the gate (top 8):")
    for code, n in dropped.most_common(8):
        print(f"   {code:5s} {n:5d}  {descr.get(code, '')[:52]}")

    # --- LLM adjudication -------------------------------------------------
    # sample == 0 skips the LLM; sample < 0 means "adjudicate the whole tier".
    if args.sample == 0:
        return
    if not Ollama.available():
        print("\nOllama unreachable — skipping LLM adjudication.")
        return

    # Sample from the tier that actually ships, so the agreement figure
    # describes the assignments a user would receive.
    pool = df[df["tier"] == "confident"]
    samp = (pool if args.sample < 0
            else pool.sample(min(args.sample, len(pool)), random_state=args.seed))
    llm = Ollama(model=args.model, num_predict=220, timeout=180)
    rows, t0 = [], time.time()
    for i, (_, r) in enumerate(samp.iterrows(), 1):
        prompt = build_prompt(r["symptom_text"], r["expanded"],
                              r["candidates"][:args.top_k], vitals_str(r))
        try:
            res = llm.generate_json(prompt, SCHEMA, system=SYSTEM) or {}
        except OllamaError as e:
            res = {"codes": [], "confidence": "low", "reason": f"error: {e}"}
        valid = {c["icd"] for c in r["candidates"][:args.top_k]}
        codes = [c for c in (res.get("codes") or []) if c in valid]
        rows.append({
            "prescription_id": int(r["prescription_id"]),
            "symptom_text": r["symptom_text"],
            "retrieval_top1": r["retrieval_top1"],
            "llm_codes": codes,
            "llm_primary": codes[0] if codes else None,
            "llm_confidence": res.get("confidence"),
            "llm_reason": (res.get("reason") or "")[:200],
            "hallucinated": [c for c in (res.get("codes") or []) if c not in valid],
        })
        if i % 25 == 0:
            print(f"   … {i}/{len(samp)}  ({(time.time()-t0)/i:.1f}s/case)", flush=True)

    adj = pd.DataFrame(rows)
    adj.to_json(OUT / "icd_llm_adjudication.json", orient="records", indent=1)

    # When the whole confident tier was adjudicated, the LLM choice supersedes
    # retrieval top-1: it is demonstrably the better coder on the disagreements
    # (I10 over I1A, B90 over R61, R50 over A78).
    if args.sample < 0:
        cols = ["prescription_id", "symptom_text", "expanded", "tier",
                "raw_top1", "retrieval_top1", "retrieval_score", "no_complaint"]
        final = df[cols].merge(
            adj[["prescription_id", "llm_primary", "llm_codes", "llm_confidence"]],
            on="prescription_id", how="left")
        final["icd_final"] = np.where(final["tier"] == "confident",
                                      final["llm_primary"], None)
        final.to_parquet(PROCESSED / "icd_autocoded.parquet", index=False)
        n_final = int(final["icd_final"].notna().sum())
        print(f"\nauthoritative coder = LLM; {n_final} encounters coded "
              f"({n_final/len(final):.1%} of notes)")
        print("top final categories:")
        for code, n in Counter(final["icd_final"].dropna()).most_common(12):
            print(f"   {code:5s} {n:5d}")

    both = adj[adj["llm_primary"].notna()]
    exact = (both["llm_primary"] == both["retrieval_top1"]).mean() if len(both) else 0.0
    inlist = both["retrieval_top1"].isin(
        [c for cs in both["llm_codes"] for c in cs]).mean() if len(both) else 0.0
    chap = ((both["llm_primary"].str[0] == both["retrieval_top1"].str[0]).mean()
            if len(both) else 0.0)
    halluc = adj["hallucinated"].map(len).sum()
    empty = adj["llm_primary"].isna().mean()

    print(f"\n{'='*66}\nLLM ADJUDICATION vs RETRIEVAL  (n={len(adj)}, {args.model})")
    print(f"  LLM returned no code            : {empty:.1%}")
    print(f"  exact agreement on primary code : {exact:.1%}")
    print(f"  same ICD chapter                : {chap:.1%}")
    print(f"  invented codes (constraint viol): {halluc}")
    print(f"  LLM confidence: "
          f"{dict(Counter(adj['llm_confidence'].dropna()))}")
    print("\nDisagreement examples:")
    dis = both[both["llm_primary"] != both["retrieval_top1"]].head(6)
    for _, r in dis.iterrows():
        print(f"   {r['symptom_text'][:52]!r}\n"
              f"      retrieval={r['retrieval_top1']}  llm={r['llm_primary']}  "
              f"({r['llm_reason'][:70]})")
    print("=" * 66)
    print("wrote", OUT / "icd_llm_adjudication.json")


if __name__ == "__main__":
    main()
