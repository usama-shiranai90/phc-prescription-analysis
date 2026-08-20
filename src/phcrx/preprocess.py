"""Clean the frozen CSV extract into model-ready Parquet + vocabularies.

Handles three data defects found during profiling:
  1. `drug_duration` is triple-encoded: Bengali digits, ASCII digits, and
     mojibake (UTF-8 bytes decoded as Latin-1, e.g. 'à§§' == '১').
  2. Structured Rx attributes are absent for a large minority of orders
     (instruction 25%, size 29%, duration-unit 11%) -> explicit NA class.
  3. Vitals carry unit drift (glucose in mmol/L, temperature in C) and
     physiologically impossible outliers.

Splits are computed here once and reused by every model so comparisons are
strictly like-for-like.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from .config import (
    INTERIM, PROCESSED, VITAL_COLS, VITAL_RANGES,
    DataConfig, SPECIALS, PAD, UNK,
)

# --- 1. Text normalisation -------------------------------------------------

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def fix_mojibake(s: str) -> str:
    """Repair text whose UTF-8 bytes were decoded as Latin-1/CP1252.

    'à§§' -> '১'. Applied only when it actually yields Bengali/Devanagari,
    so correctly-encoded strings are left untouched.
    """
    if not s or not any(c in s for c in "àáâãäåÃÂ"):
        return s
    for enc in ("latin-1", "cp1252"):
        try:
            repaired = s.encode(enc, errors="strict").decode("utf-8", errors="strict")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if any("ঀ" <= c <= "৿" for c in repaired):
            return repaired
    return s


def norm_text(s) -> str:
    if not isinstance(s, str):
        return ""
    s = fix_mojibake(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(BN_DIGITS)
    s = re.sub(r"<[^>]+>", " ", s)            # stray HTML (2 rows)
    s = s.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()


# Keep medical abbreviations intact: 'D/M', 'H/O', 'B.P.' must not be split.
_TOKEN_RE = re.compile(r"[a-z]+(?:[/.][a-z]+)+|[a-z]+|\d+(?:\.\d+)?")


def tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(norm_text(s).lower())


def parse_duration_qty(raw) -> float | None:
    """Extract the numeric duration from the mixed-encoding free-text field."""
    t = norm_text(raw)
    if not t:
        return None
    m = re.search(r"\d+(?:\.\d+)?", t)
    if not m:
        return None
    v = float(m.group())
    # Beyond a year of continuous therapy the field is almost certainly a typo.
    return v if 0 < v <= 365 else None


_DOSE_SEP = re.compile(r"\s*[+\-]\s*")


def canon_dose(s) -> str:
    """Canonicalise dose strings so '1+0+1', '1-0-1', '1 + 0 + 1' collapse."""
    t = norm_text(s).lower()
    if not t:
        return ""
    if re.fullmatch(r"[\d/\s+\-]+", t):
        return "+".join(p for p in _DOSE_SEP.split(t.strip()) if p)
    return t


# --- 2. Vitals harmonisation ----------------------------------------------

def harmonise_vitals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Glucose: a minority of rows are mmol/L (<= 20). Convert to mg/dL.
    g = pd.to_numeric(df.get("blood_glucose"), errors="coerce")
    df["blood_glucose"] = np.where(g.notna() & (g <= 20), g * 18.0, g)
    # Temperature: values in the 30-45 band are Celsius. Convert to Fahrenheit.
    t = pd.to_numeric(df.get("temperature"), errors="coerce")
    df["temperature"] = np.where(t.notna() & (t.between(30, 45)), t * 9 / 5 + 32, t)
    # Clip the rest to plausible ranges; out-of-range becomes missing.
    for col, (lo, hi) in VITAL_RANGES.items():
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            df[col] = v.where(v.between(lo, hi))
    return df


# --- 3. Load + assemble ----------------------------------------------------

def load_raw() -> dict[str, pd.DataFrame]:
    read = lambda n: pd.read_csv(INTERIM / n, dtype=str, keep_default_na=False, na_values=[""])
    return {
        "enc": read("encounters.csv"),
        "orders": read("rx_orders.csv"),
        "advice": read("rx_advice.csv"),
        "tests": read("rx_tests.csv"),
        "cc": read("rx_cc.csv"),
        "hist": read("patient_history.csv"),
    }


def build_encounters(enc: pd.DataFrame) -> pd.DataFrame:
    df = enc.copy()
    for c in ("prescription_id", "checkup_id", "user_id", "site_id"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["checkup_date"] = pd.to_datetime(df["checkup_date"], errors="coerce", utc=True)
    df["dateofbirth"] = pd.to_datetime(df["dateofbirth"], errors="coerce", utc=True)

    for c in VITAL_COLS + ["waist", "hip", "urinary_ph"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = harmonise_vitals(df)

    # Age is year-precision only: many DOBs are Jan-1 placeholders.
    df["age"] = (df["checkup_date"].dt.year - df["dateofbirth"].dt.year).astype("float")
    df.loc[~df["age"].between(0, 110), "age"] = np.nan

    df["sex"] = df["sex"].str.upper().str[:1].where(lambda s: s.isin(["M", "F"]))
    df["symptom_text"] = df["extra_symptom"].map(norm_text)
    df["symptom_tokens"] = df["symptom_text"].map(tokenize)

    df["smoker_flag"] = (df["smoker"].fillna("").str.strip().str.lower() == "yes").astype(int)
    df["glucose_type"] = df["blood_glucose_type"].fillna("NA").str.upper().str.strip()
    df["year"] = df["checkup_date"].dt.year
    df["site_district"] = pd.to_numeric(df["site_district"], errors="coerce").astype("Int64")
    return df


def build_orders(orders: pd.DataFrame) -> pd.DataFrame:
    df = orders.copy()
    for c in ("order_id", "prescription_id", "drug_id", "type_id",
              "doze_id", "drug_instruction_id", "drug_duration_id", "drug_size_id"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["dose_canon"] = df["doze"].map(canon_dose)
    df["duration_qty"] = df["duration_qty_raw"].map(parse_duration_qty)
    df["duration_unit"] = df["duration_unit"].fillna("NA").str.strip()
    df["instruction"] = df["drug_instruction"].fillna("NA").str.strip()
    df["type_name"] = df["type_name"].fillna("NA").str.strip()
    df = df[df["drug_id"].notna()].copy()

    # Duration in days, so 'Weeks'/'Month' are comparable; then bucketed.
    mult = {"Days": 1.0, "Weeks": 7.0, "Month": 30.0}
    df["duration_days"] = df["duration_qty"] * df["duration_unit"].map(mult).fillna(np.nan)
    df.loc[df["duration_unit"] == "Cont.", "duration_days"] = 999.0
    return df


DURATION_BINS = [0, 3, 7, 14, 30, 90, 1000]
DURATION_LABELS = ["<=3d", "4-7d", "8-14d", "15-30d", "31-90d", ">90d/cont"]


def bucket_duration(days: pd.Series) -> pd.Series:
    b = pd.cut(days, bins=DURATION_BINS, labels=DURATION_LABELS, right=True)
    return b.astype(object).fillna("NA")


# --- 4. Vocabularies -------------------------------------------------------

def build_vocab(counter: Counter, min_freq: int = 1, specials=("<na>",)) -> dict[str, int]:
    v = {s: i for i, s in enumerate(specials)}
    for tok, c in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        if c >= min_freq and tok not in v:
            v[tok] = len(v)
    return v


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["patient", "temporal"], default=None,
                    help="patient-level (default) or temporal distribution-shift split")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = DataConfig()
    if args.split:
        cfg.split = args.split
    if args.seed is not None:
        cfg.seed = args.seed
    print(f"split={cfg.split} seed={cfg.seed}")
    rng = np.random.default_rng(cfg.seed)
    raw = load_raw()

    enc = build_encounters(raw["enc"])
    orders = build_orders(raw["orders"])
    orders["duration_bucket"] = bucket_duration(orders["duration_days"])

    print(f"encounters={len(enc)}  orders={len(orders)}  "
          f"rx_with_drugs={orders['prescription_id'].nunique()}")

    # --- splits (computed once, shared by every model) ---------------------
    if cfg.split == "patient":
        patients = np.asarray(enc["user_id"].dropna().unique(), dtype=np.int64)
        rng.shuffle(patients)
        n = len(patients)
        n_test = int(n * cfg.test_frac)
        n_val = int(n * cfg.val_frac)
        assign = {}
        for i, p in enumerate(patients):
            assign[p] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
        enc["split"] = enc["user_id"].map(assign)
    else:
        enc["split"] = np.where(
            enc["year"] <= cfg.temporal_train_end, "train",
            np.where(enc["year"] <= cfg.temporal_val_end, "val", "test"))

    print(enc["split"].value_counts().to_string())
    # Leakage guard: a patient must never straddle two splits.
    straddle = enc.groupby("user_id")["split"].nunique()
    n_straddle = int((straddle > 1).sum())
    if cfg.split == "patient":
        assert n_straddle == 0, f"{n_straddle} patients straddle splits"
    print(f"patients straddling splits: {n_straddle}")

    train_ids = set(enc.loc[enc["split"] == "train", "prescription_id"].dropna())

    # --- vocabularies built on TRAIN ONLY (prevents lexical leakage) -------
    tr_tokens = enc.loc[enc["split"] == "train", "symptom_tokens"]
    word_counter = Counter(t for toks in tr_tokens for t in toks)
    char_counter = Counter(c for toks in tr_tokens for t in toks for c in t)

    word_vocab = {s: i for i, s in enumerate(SPECIALS)}
    for w, c in sorted(word_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if c >= cfg.min_word_freq:
            word_vocab[w] = len(word_vocab)
    char_vocab = {"<pad>": PAD, "<unk>": UNK}
    for ch, c in sorted(char_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if c >= cfg.min_char_freq:
            char_vocab[ch] = len(char_vocab)

    # Drug vocabulary: full label space, frequency computed on train only so we
    # can report head/mid/tail strata honestly rather than truncating the tail.
    tr_orders = orders[orders["prescription_id"].isin(train_ids)]
    drug_freq = Counter(tr_orders["drug_id"].dropna().astype(int))
    drug_vocab = {s: i for i, s in enumerate(SPECIALS)}
    for d in sorted(orders["drug_id"].dropna().astype(int).unique()):
        drug_vocab[int(d)] = len(drug_vocab)

    attr_vocabs = {
        "type": build_vocab(Counter(tr_orders["type_name"])),
        "dose": build_vocab(Counter(tr_orders["dose_canon"].replace("", "<na>"))),
        "duration": build_vocab(Counter(tr_orders["duration_bucket"])),
        "instruction": build_vocab(Counter(tr_orders["instruction"])),
    }

    # --- drug -> pharmacological category ---------------------------------
    # 606/717 prescribed brands map to 90 categories (PPI spans 16 brands,
    # Paracetamol 14). Brand choice is largely formulary, not clinical, so the
    # category level is the clinically meaningful unit of evaluation and is
    # supervised as an auxiliary head.
    vd = pd.read_csv(INTERIM / "vocab_drug.csv", dtype=str, keep_default_na=False,
                     na_values=[""])
    vm = pd.read_csv(INTERIM / "vocab_misc.csv", dtype=str, keep_default_na=False,
                     na_values=[""])
    cat_names = dict(zip(vm.loc[vm["kind"] == "category", "id"],
                         vm.loc[vm["kind"] == "category", "val"]))
    drug2cat_raw = {}
    for _, r in vd.iterrows():
        did = pd.to_numeric(r["drug_id"], errors="coerce")
        cid = pd.to_numeric(r["cat_id"], errors="coerce")
        if pd.notna(did):
            drug2cat_raw[int(did)] = (int(cid) if pd.notna(cid) and int(cid) != 0
                                      else None)
    used = sorted(orders["drug_id"].dropna().astype(int).unique())
    cat_vocab = {"<na>": 0}
    for cid in sorted({c for d, c in drug2cat_raw.items() if c is not None and d in set(used)}):
        cat_vocab[str(cid)] = len(cat_vocab)
    drug2cat = {int(d): cat_vocab.get(str(drug2cat_raw.get(int(d)) or ""), 0) for d in used}
    n_mapped = sum(1 for d in used if drug2cat[d] != 0)
    print(f"drug->category: {n_mapped}/{len(used)} brands mapped to "
          f"{len(cat_vocab) - 1} categories")

    advice_vocab = build_vocab(Counter(
        pd.to_numeric(raw["advice"]["advice_id"], errors="coerce").dropna().astype(int)), specials=())
    test_vocab = build_vocab(Counter(raw["tests"]["test_id"].dropna()), specials=())
    cc_vocab = build_vocab(Counter(
        pd.to_numeric(raw["cc"]["cc_id"], errors="coerce").dropna().astype(int)), specials=())
    district_vocab = build_vocab(Counter(enc["site_district"].dropna().astype(int)))
    glucose_type_vocab = build_vocab(Counter(enc["glucose_type"]))

    print(f"vocab: words={len(word_vocab)} chars={len(char_vocab)} drugs={len(drug_vocab)} "
          f"advice={len(advice_vocab)} tests={len(test_vocab)} cc={len(cc_vocab)}")
    print("attr sizes: " + ", ".join(f"{k}={len(v)}" for k, v in attr_vocabs.items()))

    # --- head/mid/tail strata for stratified reporting --------------------
    strata = {}
    for d, f in drug_freq.items():
        strata[int(d)] = "head" if f >= 100 else ("mid" if f >= 10 else "tail")
    for d in orders["drug_id"].dropna().astype(int).unique():
        strata.setdefault(int(d), "unseen")

    # --- persist ----------------------------------------------------------
    keep = ["prescription_id", "checkup_id", "user_id", "site_id", "site_district",
            "checkup_date", "year", "split", "age", "sex", "smoker_flag",
            "glucose_type", "symptom_text", "prescriber_id"] + VITAL_COLS
    enc_out = enc[keep].copy()
    enc_out["symptom_tokens"] = enc["symptom_tokens"]
    enc_out.to_parquet(PROCESSED / "rxgen_encounters.parquet", index=False)

    order_cols = ["prescription_id", "order_id", "drug_id", "drug_name", "type_name",
                  "dose_canon", "duration_bucket", "duration_days", "instruction"]
    orders[order_cols].to_parquet(PROCESSED / "rxgen_orders.parquet", index=False)

    for name, df in (("advice", raw["advice"]), ("tests", raw["tests"]), ("cc", raw["cc"])):
        d = df.copy()
        d["prescription_id"] = pd.to_numeric(d["prescription_id"], errors="coerce").astype("Int64")
        d.to_parquet(PROCESSED / f"rxgen_{name}.parquet", index=False)

    vocabs = {
        "word": word_vocab,
        "char": char_vocab,
        "drug": {str(k): v for k, v in drug_vocab.items()},
        "attr": {k: {str(a): b for a, b in v.items()} for k, v in attr_vocabs.items()},
        "advice": {str(k): v for k, v in advice_vocab.items()},
        "test": {str(k): v for k, v in test_vocab.items()},
        "cc": {str(k): v for k, v in cc_vocab.items()},
        "district": {str(k): v for k, v in district_vocab.items()},
        "glucose_type": {str(k): v for k, v in glucose_type_vocab.items()},
        "drug_stratum": {str(k): v for k, v in strata.items()},
        "category": cat_vocab,
        "category_names": cat_names,
        "drug2cat": {str(k): v for k, v in drug2cat.items()},
        "drug_train_freq": {str(k): int(v) for k, v in drug_freq.items()},
        "vital_cols": VITAL_COLS,
        "duration_labels": DURATION_LABELS,
    }
    (PROCESSED / "rxgen_vocab.json").write_text(json.dumps(vocabs, indent=1, ensure_ascii=False))

    # Standardisation stats from TRAIN ONLY.
    tr = enc_out[enc_out["split"] == "train"]
    stats = {c: {"mean": float(tr[c].mean()), "std": float(tr[c].std() or 1.0),
                 "missing_rate": float(tr[c].isna().mean())} for c in VITAL_COLS}
    stats["age"] = {"mean": float(tr["age"].mean()), "std": float(tr["age"].std() or 1.0),
                    "missing_rate": float(tr["age"].isna().mean())}
    (PROCESSED / "rxgen_norm_stats.json").write_text(json.dumps(stats, indent=1))

    print("\nwrote:", PROCESSED)
    print("duration bucket distribution:")
    print(orders["duration_bucket"].value_counts().to_string())


if __name__ == "__main__":
    main()
