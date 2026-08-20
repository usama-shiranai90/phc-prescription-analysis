"""ICD-10 retrieval index over the local WHO code list.

Fills the corpus's largest gap. Only 195 of 14,074 encounters carry an ICD
code, and those that do are mostly *not diagnoses* -- the single most frequent
is Y440 "Iron preparations and other anti-hypochromic-anaemia preparations"
(128 of 216 rows), an external-cause/drug code. Meanwhile 86% of encounters
have a free-text symptom note. So the diagnosis signal exists, just not as
codes.

The code list itself needs no WHO API call: `tb_data_icd` in the database is a
complete ICD-10 reference (5,127 codes with descriptions), extracted offline.

**The database's own ICD table is unusable as a reference.** `tb_data_icd`
holds 5,127 rows but covers only chapters A, P, Q, R, S, T and V-Z: B through O
are entirely absent. Every code this cohort actually needs -- E11 diabetes,
I10 hypertension, K21 reflux, M54 back pain, N39 UTI, J06 URTI -- is missing.
That also explains the existing labels: clinicians coded R030 "elevated
blood-pressure reading" and Z131 "screening for diabetes" because I10 and E11
were **not selectable in the application**. The reference is therefore taken
from the CMS ICD-10-CM order file distributed on Hugging Face.

Retrieval is hybrid: SapBERT (cambridgeltl/SapBERT-from-PubMedBERT-fulltext,
trained for biomedical entity linking) supplies semantics, and a character
n-gram TF-IDF supplies lexical anchoring. Semantics alone failed badly here --
"Angina Pectoris" retrieved "Chest pain, unspecified" rather than the exact
I20 "Angina pectoris" match.

Everything runs locally on the GPU; no patient text leaves the machine.

    python -m src.phcrx.nlp.icd_index --build
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from ..config import INTERIM, PROCESSED, RESULTS
from .glossary import expand

OUT = RESULTS / "nlp"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = PROCESSED / "icd_index"
CACHE.mkdir(parents=True, exist_ok=True)

MODEL_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# ICD-10 chapters that are diagnoses. V-Y are external causes of morbidity and
# Z are factors influencing health status; neither is a clinical diagnosis, and
# letting them into the candidate pool is what produced the Y440 labels in the
# existing data.
EXCLUDE_PREFIX = tuple("VWXY")


HF_ICD_REPO = ("TachyHealth/International_Classification_Diseases_"
               "Clinical_Modification_icd10cm_order_April_2024")
HF_ICD_FILE = "icd10cm-order-April-2024.csv"

# ICD-10-CM-only additions that do not exist in WHO ICD-10. They must not
# compete for a WHO-coded cohort. I1A ("Other hypertension", added 2022) was
# outranking I10 ("Essential (primary) hypertension") on 1,450 encounters --
# the largest category in the corpus -- purely because its shorter description
# scores higher on character n-gram overlap with the token "hypertension".
#
# Removing I1A did NOT fix the bias: I15 "Secondary hypertension" simply took
# its place (517 encounters), which is equally wrong. The bias is structural in
# lexical matching, not a property of any particular code, which is why LLM
# adjudication -- not a longer exclusion list -- is the actual fix.
CM_ONLY = {"I1A", "E08", "E09", "D3A", "M1A", "C7A", "C7B", "Z3A"}


def load_icd(include_z: bool = True, level: int = 3) -> pd.DataFrame:
    """ICD-10-CM reference from Hugging Face, cut to `level`-character categories.

    Three-character categories (E11, I10, K21) are the right granularity for
    primary-care epidemiology; the full CM file carries 97k billable codes with
    laterality and encounter suffixes that this corpus cannot support.
    """
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HF_ICD_REPO, HF_ICD_FILE, repo_type="dataset")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    df = df.rename(columns={"Code": "id", "Long Description": "descr"})
    df = df.dropna(subset=["id", "descr"])
    df["id"] = df["id"].str.strip().str.upper()
    df = df[df["id"].str.len() == level]
    df["descr"] = df["descr"].str.strip()
    df = df[~df["id"].str.startswith(EXCLUDE_PREFIX)]
    df = df[~df["id"].isin(CM_ONLY)]
    if not include_z:
        df = df[~df["id"].str.startswith("Z")]
    df["chapter"] = df["id"].str[0]
    return df[["id", "descr", "chapter"]].drop_duplicates("id").reset_index(drop=True)


def load_db_icd() -> pd.DataFrame:
    """The application's own (incomplete) ICD table, kept for provenance."""
    df = pd.read_csv(INTERIM / "icd_reference.csv", dtype=str,
                     keep_default_na=False, na_values=[""])
    df["id"] = df["id"].str.strip().str.upper()
    return df.dropna(subset=["id", "descr"])


class SapBertEncoder:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 128,
               max_len: int = 48) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = [t if t else "unknown" for t in texts[i:i + batch_size]]
            enc = self.tok(chunk, padding=True, truncation=True,
                           max_length=max_len, return_tensors="pt").to(self.device)
            # SapBERT is trained with [CLS] as the concept representation.
            h = self.model(**enc).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def build(include_z: bool = True) -> None:
    icd = load_icd(include_z=include_z)
    enc = SapBertEncoder()
    print(f"encoding {len(icd)} ICD-10 descriptions on {enc.device} …")
    emb = enc.encode(icd["descr"].tolist())
    np.save(CACHE / "icd_emb.npy", emb)
    icd.to_parquet(CACHE / "icd_ref.parquet", index=False)
    print(f"  index: {emb.shape}")

    e = pd.read_parquet(PROCESSED / "rxgen_encounters.parquet")
    txt = e["symptom_text"].fillna("").astype(str)
    # Expand local abbreviations first: SapBERT has never seen 'D/M' but
    # matches 'Diabetes Mellitus' well.
    expanded = [expand(t) if t.strip() else "" for t in txt]
    print(f"encoding {len(expanded)} symptom notes …")
    semb = enc.encode(expanded)
    np.save(CACHE / "symptom_emb.npy", semb)
    pd.DataFrame({"prescription_id": e["prescription_id"],
                  "symptom_text": txt, "expanded": expanded}
                 ).to_parquet(CACHE / "symptoms.parquet", index=False)
    print("  built ->", CACHE)


def retrieve(top_k: int = 5, alpha: float = 0.5):
    """Top-k ICD candidates by hybrid semantic + lexical similarity.

    `alpha` weights SapBERT cosine against character n-gram TF-IDF cosine.
    Pure semantics missed exact term matches (Angina Pectoris -> I20); pure
    lexical matching misses paraphrase. The blend handles both.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    icd = pd.read_parquet(CACHE / "icd_ref.parquet")
    iemb = np.load(CACHE / "icd_emb.npy")
    sym = pd.read_parquet(CACHE / "symptoms.parquet")
    semb = np.load(CACHE / "symptom_emb.npy")

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                          sublinear_tf=True)
    I_lex = vec.fit_transform(icd["descr"].str.lower())
    S_lex = vec.transform(sym["expanded"].fillna("").str.lower())
    from sklearn.preprocessing import normalize
    I_lex, S_lex = normalize(I_lex), normalize(S_lex)

    rows = []
    B = 256
    for i in range(0, len(semb), B):
        sem = semb[i:i + B] @ iemb.T
        lex = (S_lex[i:i + B] @ I_lex.T).toarray()
        sims = alpha * sem + (1 - alpha) * lex
        k = min(top_k, sims.shape[1] - 1)
        idx = np.argpartition(-sims, k, axis=1)[:, :top_k]
        for r, cand in enumerate(idx):
            order = cand[np.argsort(-sims[r, cand])]
            rec = sym.iloc[i + r]
            rows.append({
                "prescription_id": rec["prescription_id"],
                "symptom_text": rec["symptom_text"],
                "expanded": rec["expanded"],
                "candidates": [
                    {"icd": icd.iloc[j]["id"], "descr": icd.iloc[j]["descr"],
                     "score": round(float(sims[r, j]), 4),
                     "sem": round(float(sem[r, j]), 4),
                     "lex": round(float(lex[r, j]), 4)} for j in order],
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--include-z", action="store_true",
                    help="keep Z-codes (screening / encounter reasons)")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    if args.build or not (CACHE / "icd_emb.npy").exists():
        build(include_z=args.include_z)

    res = retrieve(args.top_k)
    res.to_parquet(CACHE / "icd_candidates.parquet", index=False)

    has = res[res["symptom_text"].str.strip() != ""]
    print(f"\nretrieved candidates for {len(has)} encounters with symptom text")
    print("\nExamples:")
    for _, r in has.head(8).iterrows():
        print(f"\n  note     : {r['symptom_text'][:80]!r}")
        print(f"  expanded : {r['expanded'][:80]!r}")
        for c in r["candidates"][:3]:
            print(f"    {c['score']:.3f}  {c['icd']:5s} {c['descr'][:60]}")
    print("\nwrote", CACHE / "icd_candidates.parquet")


if __name__ == "__main__":
    main()
