"""Build the encounter-level feature matrix for prescription-component analysis.

    python -m src.phcrx.features.build_features

One row per encounter (14,074), written to `data/processed/features.parquet`
together with `data/processed/feature_families.json`, which maps every column
to one of ten families:

    demog  vitals  derived  ncd  icd  text  temporal  history  site  prescriber

Design rules that the downstream importance analysis depends on:

* **Everything that is fitted is fitted on the TRAIN split only** -- TF-IDF
  vocabulary, the char-n-gram SVD basis, category codes, and the site /
  prescriber prescribing profiles. For train rows the profiles are computed
  leave-one-out so a row never sees its own label.
* **Missingness is a feature, not a nuisance.** Which vitals were measured
  reflects what the operator suspected, so every vital carries an explicit
  `_miss` bit and the raw value keeps its NaN (HistGradientBoosting learns a
  split direction for NaN natively).
* **A withheld ICD tier is "unknown", never "no disease".** Low-confidence and
  no-complaint encounters get their own indicator columns; they are not folded
  into the zero level of the chapter one-hots.
* **`prescriber_*` and `site_*` are confounders**, kept in their own families so
  the analysis can refit with them removed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import PROCESSED, VITAL_COLS
from ..nlp.glossary import expand

OUT_PARQUET = PROCESSED / "features.parquet"
OUT_FAMILIES = PROCESSED / "feature_families.json"

SEED = 0
SMOOTH_M = 20.0          # empirical-Bayes prior weight for target encodings
N_PROFILE_CLASSES = 10   # drug classes in the site/prescriber profile
N_TOP_CLASSES = 20       # classes exported as modelling targets
TFIDF_WORD_FEATURES = 100
CHAR_SVD_COMPONENTS = 16

_FUNCTION_WORDS = {
    "of", "for", "and", "in", "the", "with", "on", "to", "at", "an",
    "is", "are", "was", "were", "from", "by", "as", "that", "this", "it",
    "his", "her", "he", "she", "they", "them", "or", "be", "been", "has",
    "have", "had", "but", "also", "per", "into", "than", "then", "there",
}

# ---------------------------------------------------------------------------
# Clinical concept lexicon.
#
# Mined from the frequency profile of the TRAIN split *after* glossary
# expansion (so "D/M" already reads "Diabetes Mellitus" and "HTN" reads
# "Hypertension"). Each pattern is a coarse presence flag, not a diagnosis.
# ---------------------------------------------------------------------------
CONCEPTS: dict[str, str] = {
    "hypertension":   r"hypertens|high blood pressure|raised bp",
    "diabetes":       r"diabet|hyperglyc|glucose tolerance|impaired glucose",
    "back_pain":      r"back pain|lumbar|lumbo|spondyl",
    "neck_pain":      r"neck pain|cervical",
    "joint_pain":     r"joint|arthrit|osteoarth|rheumat|knee",
    "limb_pain":      r"\bleg\b|limb|\bfoot\b|\bfeet\b|\bhand\b|shoulder|ankle|heel",
    "chest_pain":     r"chest pain|chest discomfort|retrosternal",
    "abdominal_pain": r"abdominal pain|pain in (the )?abdomen|epigastric|lower abdominal",
    "headache":       r"headache|head ache|migrain",
    "pain_any":       r"\bpain|\bache|painful",
    "weakness":       r"weakness|fatigue|lethargy|tired|malaise",
    "fever":          r"fever|pyrexia|febrile",
    "cough_cold":     r"cough|common cold|\bcold\b|sneez|runny nose|sore throat",
    "breathless":     r"breathless|shortness of breath|dyspn|respiratory distress",
    "asthma_copd":    r"asthma|chronic obstructive|wheez|bronch",
    "gastric":        r"acidity|gastric|gastritis|heart ?burn|reflux|dyspep|\bsour|belch|peptic|ulcer|gastro-oesophageal",
    "constipation":   r"constipat|hard stool",
    "diarrhoea":      r"diarrh|loose motion|loose stool|dysent",
    "nausea_vomit":   r"nausea|vomit",
    "anorexia":       r"anorexia|appetite",
    "burning_sens":   r"burning sensation|burning",
    "micturition":    r"micturi|micturation|dysuria|urinary|urination|frequency of urine",
    "vertigo":        r"vertigo|dizz|giddi|light ?head",
    "palpitation":    r"palpitat",
    "insomnia":       r"insomnia|sleepless|sleep disturb|disturbed sleep|\bsleep\b",
    # \btension: an unanchored "tension" also matches inside "hypertension",
    # which put 15% of encounters in this bucket on the first pass.
    "anxiety_depr":   r"anxiet|depress|\btension|\bstress|panic|irritab",
    "itching_skin":   r"itch|pruritus|rash|skin|eczema|fungal|scabies",
    "swelling":       r"swell|oedema|edema|puffi",
    "numb_tingle":    r"numb|tingl|paraesthes|parasthes|pins and needles",
    "vision":         r"vision|\beye\b|blurr|cataract|visual",
    "ear_hearing":    r"\bear\b|hearing|tinnitus|deaf",
    "menstrual":      r"menstr|menorrha|per vaginal|vaginal|leucorrh|amenorrh|dysfunctional uterine|pelvic inflammatory",
    "pregnancy":      r"pregnan|antenatal|last menstrual period|gravid|lactat",
    "weight_change":  r"weight loss|loss of weight|weight gain|\bweight\b|obes",
    "hyperlipid":     r"lipid|cholesterol|hyperlipid|dyslipid",
    "thyroid":        r"thyroid|goit",
    "anaemia_text":   r"anaemi|anemi|\bpale\b|haemoglobin|hemoglobin",
    "cardiac":        r"ischaemic heart|ischemic heart|cardiac|heart disease|angina|\bihd\b",
    "stroke_neuro":   r"cerebrovascular|stroke|paralys|paresis|seizure|epilep|convuls",
    "kidney":         r"kidney|renal|creatinine|nephro",
    "liver":          r"liver|hepat|jaundice",
    "allergy":        r"allerg|urticar",
    "worm_infect":    r"\bworm|helminth|infest",
    "chronic_marker": r"history of|known case|for years|for year|for months|long standing|old case",
    "acute_marker":   r"for days|for day|few days|for weeks|for week|since yesterday",
    "on_treatment":   r"taking (drug|medicine|tablet)|on medication|on treatment|regular treatment|on regular|taking drugs",
    "uncontrolled":   r"uncontrol|not controlled|poorly controlled|irregular treatment",
    "routine_check":  r"check ?up|routine|screening|general check|follow up",
    "no_complaint":   r"no complaint|no complaints|no history|no specific|\bnil\b|nothing",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s or "x"


def _loo_target_encode(keys: pd.Series, y: np.ndarray, is_train: np.ndarray,
                       m: float = SMOOTH_M) -> np.ndarray:
    """Empirical-Bayes group mean of `y`, fitted on TRAIN rows only.

    Train rows get a leave-one-out estimate (their own label is subtracted), so
    the feature is never a copy of the row's own target. Val/test rows get the
    full-train estimate. Groups unseen in train fall back to the train prior.
    """
    y = np.asarray(y, dtype=float)
    prior = float(np.nanmean(y[is_train])) if is_train.any() else 0.0
    contrib = np.where(is_train, np.nan_to_num(y), 0.0)
    frame = pd.DataFrame({"k": keys.values, "y": contrib, "n": is_train.astype(float)})
    agg = frame.groupby("k")[["y", "n"]].sum()
    s = keys.map(agg["y"]).to_numpy(dtype=float)
    n = keys.map(agg["n"]).to_numpy(dtype=float)
    s = np.where(is_train, s - np.nan_to_num(y), s)
    n = np.where(is_train, n - 1.0, n)
    n = np.clip(n, 0.0, None)
    return (s + m * prior) / (n + m)


def _codes(series: pd.Series, is_train: np.ndarray) -> tuple[np.ndarray, int]:
    """Integer codes for a categorical, levels learned on TRAIN only.

    Levels unseen in train collapse to a single trailing "OTHER" code so that
    HistGradientBoosting's native categorical support (which requires
    non-negative ints) never sees a -1.
    """
    levels = sorted(series[is_train].dropna().unique().tolist())
    lut = {v: i for i, v in enumerate(levels)}
    other = len(levels)
    return series.map(lambda v: lut.get(v, other)).astype("int32").to_numpy(), other + 1


ICD_CHAPTERS = [
    ("A00", "B99", "infectious"), ("C00", "D48", "neoplasm"),
    ("D50", "D89", "blood"), ("E00", "E90", "endocrine"),
    ("F00", "F99", "mental"), ("G00", "G99", "nervous"),
    ("H00", "H59", "eye"), ("H60", "H95", "ear"),
    ("I00", "I99", "circulatory"), ("J00", "J99", "respiratory"),
    ("K00", "K93", "digestive"), ("L00", "L99", "skin"),
    ("M00", "M99", "musculoskeletal"), ("N00", "N99", "genitourinary"),
    ("O00", "O99", "pregnancy"), ("P00", "P96", "perinatal"),
    ("Q00", "Q99", "congenital"), ("R00", "R99", "symptoms"),
    ("S00", "T98", "injury"), ("Z00", "Z99", "health_status"),
]


def _icd_chapter(code) -> str | None:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    c = str(code).strip().upper()
    if len(c) < 3 or not c[0].isalpha() or not c[1:3].isdigit():
        return None
    for lo, hi, name in ICD_CHAPTERS:
        if lo <= c[:3] <= hi:
            return name
    return "other"


# ---------------------------------------------------------------------------
# feature blocks
# ---------------------------------------------------------------------------
def _demographics(enc: pd.DataFrame, F: dict, fam: dict) -> None:
    age = enc["age"].astype(float)
    F["age"] = age.to_numpy()
    F["age_missing"] = age.isna().astype("int8").to_numpy()
    F["age_band"] = pd.cut(age, [-1, 17, 29, 39, 49, 59, 69, 200],
                           labels=False).astype("float32").to_numpy()
    F["sex_is_male"] = (enc["sex"].astype(str).str.upper() == "M").astype("int8").to_numpy()
    F["smoker_flag"] = enc["smoker_flag"].astype("int8").to_numpy()
    for k in ("age", "age_missing", "age_band", "sex_is_male", "smoker_flag"):
        fam[k] = "demog"


def _vitals(enc: pd.DataFrame, F: dict, fam: dict) -> None:
    for c in VITAL_COLS:
        v = enc[c].astype(float)
        F[c] = v.to_numpy()
        F[f"{c}_miss"] = v.isna().astype("int8").to_numpy()
        fam[c] = "vitals"
        fam[f"{c}_miss"] = "vitals"


def _derived_clinical(enc: pd.DataFrame, F: dict, fam: dict) -> None:
    sys_bp = enc["bp_sys"].astype(float)
    dia_bp = enc["bp_dia"].astype(float)
    bmi = enc["bmi"].astype(float)
    hb = enc["blood_hemoglobin"].astype(float)
    glu = enc["blood_glucose"].astype(float)
    gtype = enc["glucose_type"].astype(str).str.upper()
    male = (enc["sex"].astype(str).str.upper() == "M").to_numpy()

    # --- blood pressure -----------------------------------------------------
    F["map_bp"] = (dia_bp + (sys_bp - dia_bp) / 3.0).to_numpy()
    F["pulse_pressure"] = (sys_bp - dia_bp).to_numpy()
    # ACC/AHA 2017 staging, taking the worse of the two readings.
    ok = (sys_bp.notna() & dia_bp.notna()).to_numpy()
    s, d = sys_bp.to_numpy(), dia_bp.to_numpy()
    bp_cat = np.full(len(enc), np.nan)
    with np.errstate(invalid="ignore"):
        bp_cat = np.where(ok, 0.0, bp_cat)
        bp_cat = np.where(ok & (s >= 120) & (s < 130) & (d < 80), 1.0, bp_cat)
        bp_cat = np.where(ok & (((s >= 130) & (s < 140)) | ((d >= 80) & (d < 90))), 2.0, bp_cat)
        bp_cat = np.where(ok & ((s >= 140) | (d >= 90)), 3.0, bp_cat)
    F["bp_category"] = bp_cat
    F["bp_hypertensive"] = np.where(ok, (bp_cat >= 3).astype(float), np.nan)

    # --- anthropometry (ASIAN BMI cut-points, not WHO) ----------------------
    b = bmi.to_numpy()
    okb = bmi.notna().to_numpy()
    bmi_cat = np.full(len(enc), np.nan)
    with np.errstate(invalid="ignore"):
        bmi_cat = np.where(okb & (b < 18.5), 0.0, bmi_cat)
        bmi_cat = np.where(okb & (b >= 18.5) & (b < 23.0), 1.0, bmi_cat)
        bmi_cat = np.where(okb & (b >= 23.0) & (b < 27.5), 2.0, bmi_cat)
        bmi_cat = np.where(okb & (b >= 27.5), 3.0, bmi_cat)
    F["bmi_category_asian"] = bmi_cat
    F["bsa_mosteller"] = np.sqrt(
        (enc["height"].astype(float) * enc["weight"].astype(float) / 3600.0).to_numpy())
    whr = enc["waist_hip_ratio"].astype(float)
    whr_thr = np.where(male, 0.90, 0.85)
    with np.errstate(invalid="ignore"):
        F["whr_risk"] = np.where(whr.notna().to_numpy(),
                                 (whr.to_numpy() > whr_thr).astype(float), np.nan)

    # --- glycaemia, conditioned on the assay actually run -------------------
    is_fbs = (gtype == "FBS").to_numpy()
    F["glucose_is_fbs"] = is_fbs.astype("int8")
    dm_thr = np.where(is_fbs, 126.0, 200.0)     # FBS vs post-prandial/random
    pre_thr = np.where(is_fbs, 100.0, 140.0)
    g = glu.to_numpy()
    okg = glu.notna().to_numpy()
    glu_cat = np.full(len(enc), np.nan)
    with np.errstate(invalid="ignore"):
        glu_cat = np.where(okg, 0.0, glu_cat)
        glu_cat = np.where(okg & (g >= pre_thr), 1.0, glu_cat)
        glu_cat = np.where(okg & (g >= dm_thr), 2.0, glu_cat)
    F["glucose_category"] = glu_cat
    F["glucose_over_dm_threshold"] = np.where(okg, g / dm_thr, np.nan)

    # --- haematology / other ------------------------------------------------
    hb_thr = np.where(male, 13.0, 12.0)         # WHO, non-pregnant adults
    okh = hb.notna().to_numpy()
    with np.errstate(invalid="ignore"):
        F["anaemia_sex_specific"] = np.where(okh, (hb.to_numpy() < hb_thr).astype(float), np.nan)
    F["hb_deficit"] = np.where(okh, hb_thr - hb.to_numpy(), np.nan)

    def _flag(col: str, fn) -> np.ndarray:
        v = enc[col].astype(float)
        okv = v.notna().to_numpy()
        with np.errstate(invalid="ignore"):
            return np.where(okv, fn(v.to_numpy()).astype(float), np.nan)

    F["fever_flag"] = _flag("temperature", lambda x: x >= 100.4)
    F["hypoxaemia_flag"] = _flag("oxygen_of_blood", lambda x: x < 94.0)
    F["tachycardia_flag"] = _flag("pulse_rate", lambda x: x > 100.0)
    F["bradycardia_flag"] = _flag("pulse_rate", lambda x: x < 60.0)
    F["cholesterol_high"] = _flag("cholesterol", lambda x: x >= 200.0)
    ua = enc["uric_acid"].astype(float)
    ua_thr = np.where(male, 7.0, 6.0)
    with np.errstate(invalid="ignore"):
        F["uric_acid_high"] = np.where(ua.notna().to_numpy(),
                                       (ua.to_numpy() > ua_thr).astype(float), np.nan)

    # --- how much of the panel was actually run (clinical suspicion proxy) ---
    n_meas = enc[VITAL_COLS].notna().sum(axis=1).to_numpy()
    F["n_vitals_measured"] = n_meas.astype("float32")
    F["frac_vitals_measured"] = (n_meas / len(VITAL_COLS)).astype("float32")

    for k in ("map_bp", "pulse_pressure", "bp_category", "bp_hypertensive",
              "bmi_category_asian", "bsa_mosteller", "whr_risk",
              "glucose_is_fbs", "glucose_category", "glucose_over_dm_threshold",
              "anaemia_sex_specific", "hb_deficit", "fever_flag",
              "hypoxaemia_flag", "tachycardia_flag", "bradycardia_flag",
              "cholesterol_high", "uric_acid_high", "n_vitals_measured",
              "frac_vitals_measured"):
        fam[k] = "derived"


def _ncd(enc: pd.DataFrame, F: dict, fam: dict) -> None:
    ncd = pd.read_parquet(PROCESSED / "ncd_flags.parquet")
    keep = [c for c in ncd.columns
            if c not in {"prescription_id", "user_id", "age", "sex", "year"}]
    ncd = ncd[["prescription_id"] + keep].drop_duplicates("prescription_id")
    m = enc[["prescription_id"]].merge(ncd, on="prescription_id", how="left")
    for c in keep:
        name = f"ncd_{c}"
        F[name] = m[c].astype(float).to_numpy()
        fam[name] = "ncd"


def _icd(enc: pd.DataFrame, F: dict, fam: dict, is_train: np.ndarray) -> None:
    """ICD block.

    `icd_autocoded` withholds a code whenever the retrieval/LLM pipeline was not
    confident, and some encounters are not in the table at all. Those states are
    carried as their own indicators; a withheld tier must never read as "this
    patient has no disease".
    """
    icd = pd.read_parquet(PROCESSED / "icd_autocoded.parquet")
    icd = icd[["prescription_id", "tier", "icd_final", "llm_confidence",
               "retrieval_score"]].drop_duplicates("prescription_id")
    m = enc[["prescription_id"]].merge(icd, on="prescription_id", how="left")

    tier = m["tier"].fillna("absent_from_table").astype(str)
    for t in ["confident", "low_confidence", "no_complaint", "absent_from_table"]:
        name = f"icd_tier_{t}"
        F[name] = (tier == t).astype("int8").to_numpy()
        fam[name] = "icd"

    conf_map = {"high": 3.0, "medium": 2.0, "low": 1.0}
    F["icd_llm_confidence"] = m["llm_confidence"].map(conf_map).astype(float).to_numpy()
    F["icd_retrieval_score"] = m["retrieval_score"].astype(float).to_numpy()
    fam["icd_llm_confidence"] = "icd"
    fam["icd_retrieval_score"] = "icd"

    code = m["icd_final"]
    has_code = code.notna()
    F["icd_has_code"] = has_code.astype("int8").to_numpy()
    fam["icd_has_code"] = "icd"

    # Chapter one-hots. "unknown" is an explicit level, kept for every encounter
    # whose code was withheld -- it is not the zero level of the other columns.
    chapter = code.map(_icd_chapter).where(has_code, other=None).fillna("unknown")
    tr_counts = chapter[is_train].value_counts()
    keep_ch = sorted(c for c in tr_counts.index if tr_counts[c] >= 25 and c != "unknown")
    for ch in keep_ch:
        name = f"icd_ch_{ch}"
        F[name] = (chapter == ch).astype("int8").to_numpy()
        fam[name] = "icd"
    F["icd_ch_other_rare"] = (~chapter.isin(keep_ch + ["unknown"])).astype("int8").to_numpy()
    F["icd_ch_unknown"] = (chapter == "unknown").astype("int8").to_numpy()
    fam["icd_ch_other_rare"] = "icd"
    fam["icd_ch_unknown"] = "icd"

    # A handful of individual 3-character codes carry most of the coded mass.
    for c in code[is_train].value_counts().head(12).index.tolist():
        name = f"icd_code_{_slug(c)}"
        F[name] = (code == c).astype("int8").to_numpy()
        fam[name] = "icd"


def _text(enc: pd.DataFrame, F: dict, fam: dict, is_train: np.ndarray) -> dict:
    """Glossary expansion FIRST, then everything else reads the expanded text."""
    raw = enc["symptom_text"].fillna("").astype(str)
    expanded = raw.map(expand)          # D/M -> Diabetes Mellitus, HTN -> Hypertension
    low = expanded.str.lower()

    F["txt_has_text"] = (raw.str.strip().str.len() > 0).astype("int8").to_numpy()
    F["txt_n_chars"] = expanded.str.len().astype("float32").to_numpy()
    F["txt_n_words"] = expanded.str.split().map(len).astype("float32").to_numpy()
    F["txt_n_tokens"] = enc["symptom_tokens"].map(
        lambda t: len(t) if t is not None else 0).astype("float32").to_numpy()
    F["txt_n_items"] = (low.str.count(r"[,;]") + 1).astype("float32").to_numpy()
    for k in ("txt_has_text", "txt_n_chars", "txt_n_words", "txt_n_tokens", "txt_n_items"):
        fam[k] = "text"

    for name, pat in CONCEPTS.items():
        col = f"cpt_{name}"
        F[col] = low.str.contains(pat, regex=True, na=False).astype("int8").to_numpy()
        fam[col] = "text"

    # Word TF-IDF, vocabulary fitted on TRAIN only, kept small and named so the
    # individual importances are readable as actual clinical words.
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=25, sublinear_tf=True,
                         max_features=TFIDF_WORD_FEATURES,
                         token_pattern=r"(?u)\b[a-z]{2,}\b",
                         stop_words=sorted(_FUNCTION_WORDS))
    wv.fit(low[is_train])
    W = wv.transform(low).toarray().astype("float32")
    for j, term in enumerate(wv.get_feature_names_out()):
        col = f"tfw_{_slug(term)}"
        F[col] = W[:, j]
        fam[col] = "text"

    # Char n-grams absorb the corpus's heavy misspelling ("micturation",
    # "bodyache"); compressed by SVD because individual char n-grams are not
    # interpretable anyway.
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=20,
                         sublinear_tf=True, max_features=4000)
    cv.fit(low[is_train])
    C = cv.transform(low)
    svd = TruncatedSVD(n_components=CHAR_SVD_COMPONENTS, random_state=SEED)
    svd.fit(C[is_train])
    S = svd.transform(C).astype("float32")
    for j in range(S.shape[1]):
        col = f"tfc_svd{j:02d}"
        F[col] = S[:, j]
        fam[col] = "text"

    return {"word_vocab": int(len(wv.get_feature_names_out())),
            "word_terms": [str(t) for t in wv.get_feature_names_out()],
            "char_vocab": int(len(cv.get_feature_names_out())),
            "char_svd_explained_var": float(svd.explained_variance_ratio_.sum())}


def _temporal_history(enc: pd.DataFrame, F: dict, fam: dict,
                      is_train: np.ndarray, any_drug: np.ndarray,
                      n_drugs: np.ndarray) -> None:
    dt = pd.to_datetime(enc["checkup_date"], utc=True)
    F["year"] = enc["year"].astype("float32").to_numpy()
    month = dt.dt.month
    F["month"] = month.astype("float32").to_numpy()
    F["month_sin"] = np.sin(2 * np.pi * month.to_numpy() / 12).astype("float32")
    F["month_cos"] = np.cos(2 * np.pi * month.to_numpy() / 12).astype("float32")
    # Bangladesh seasons: 0 winter, 1 pre-monsoon, 2 monsoon, 3 post-monsoon.
    F["season"] = month.map(lambda m: 0 if m in (12, 1, 2) else
                            1 if m in (3, 4, 5) else
                            2 if m in (6, 7, 8, 9) else 3).astype("int8").to_numpy()
    F["day_of_week"] = dt.dt.dayofweek.astype("int8").to_numpy()
    t0 = dt[is_train].min()
    F["days_since_cohort_start"] = (
        (dt - t0).dt.total_seconds() / 86400.0).astype("float32").to_numpy()
    for k in ("year", "month", "month_sin", "month_cos", "season",
              "day_of_week", "days_since_cohort_start"):
        fam[k] = "temporal"

    # Patient history. Ordered by date within patient; the previous encounter is
    # strictly earlier in time, so its outcome is legitimately observable.
    order = np.lexsort((dt.to_numpy(), enc["user_id"].to_numpy()))
    inv = np.empty(len(enc), dtype=int)
    inv[order] = np.arange(len(enc))
    tmp = pd.DataFrame({
        "user_id": enc["user_id"].to_numpy(), "dt": dt.to_numpy(),
        "any_drug": any_drug, "n_drugs": n_drugs,
    }).iloc[order].reset_index(drop=True)
    g = tmp.groupby("user_id", sort=False)
    visit_index = g.cumcount().to_numpy().astype("float32")
    days_prev = ((tmp["dt"] - g["dt"].shift(1)).dt.total_seconds() / 86400.0).to_numpy()
    prev_any = g["any_drug"].shift(1).to_numpy()
    prev_n = g["n_drugs"].shift(1).to_numpy()

    F["visit_index"] = visit_index[inv]
    F["is_first_visit"] = (visit_index[inv] == 0).astype("int8")
    F["days_since_prev_visit"] = days_prev[inv].astype("float32")
    F["prev_visit_any_drug"] = prev_any[inv].astype("float32")
    F["prev_visit_n_drugs"] = prev_n[inv].astype("float32")
    for k in ("visit_index", "is_first_visit", "days_since_prev_visit",
              "prev_visit_any_drug", "prev_visit_n_drugs"):
        fam[k] = "history"


def _site_prescriber(enc: pd.DataFrame, F: dict, fam: dict, is_train: np.ndarray,
                     any_drug: np.ndarray, n_drugs: np.ndarray,
                     class_mat: pd.DataFrame, profile_classes: list[str],
                     any_advice: np.ndarray, any_test: np.ndarray,
                     cat_meta: dict) -> None:
    for key, family in (("site_id", "site"), ("site_district", "site"),
                        ("prescriber_id", "prescriber")):
        codes, n_lev = _codes(enc[key], is_train)
        name = f"{key}_code"
        F[name] = codes
        fam[name] = family
        cat_meta[name] = n_lev

    for key, family in (("site_id", "site"), ("prescriber_id", "prescriber")):
        tag = "site" if family == "site" else "prescriber"
        k = enc[key]
        F[f"{tag}_train_volume"] = k.map(k[is_train].value_counts()).fillna(
            0.0).to_numpy().astype("float32")
        fam[f"{tag}_train_volume"] = family
        # Prescribing profile: leave-one-out train-only rates. These are target
        # encodings by construction -- they quantify practice variation and are
        # therefore confounders, which is exactly why they sit in their own
        # families and get dropped in the ablated model variants.
        for label, y in (("any_drug", any_drug), ("n_drugs", n_drugs),
                         ("any_advice", any_advice), ("any_test", any_test)):
            col = f"{tag}_rate_{label}"
            F[col] = _loo_target_encode(k, y, is_train).astype("float32")
            fam[col] = family
        for c in profile_classes:
            col = f"{tag}_rate_{_slug(c)}"
            F[col] = _loo_target_encode(k, class_mat[c].to_numpy(), is_train).astype("float32")
            fam[col] = family


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
def _build_targets(enc: pd.DataFrame, is_train: np.ndarray):
    orders = pd.read_parquet(PROCESSED / "rxgen_orders.parquet")
    advice = pd.read_parquet(PROCESSED / "rxgen_advice.parquet")
    tests = pd.read_parquet(PROCESSED / "rxgen_tests.parquet")
    vocab = json.loads((PROCESSED / "rxgen_vocab.json").read_text(encoding="utf-8"))
    drug2cat = vocab["drug2cat"]
    catnames = vocab["category_names"]

    pid = enc["prescription_id"]
    T = pd.DataFrame({"prescription_id": pid.to_numpy()})

    n = orders.groupby("prescription_id").size()
    T["y_n_drugs"] = pid.map(n).fillna(0).astype(int).to_numpy()
    T["y_any_drug"] = (T["y_n_drugs"] > 0).astype("int8")

    orders = orders.copy()
    orders["cat"] = orders["drug_id"].astype(str).map(drug2cat)

    # Category ids 0 and 31 carry no entry in `category_names` and, on
    # inspection, pool pharmacologically unrelated brands (laxative, statin,
    # muscle relaxant, antihistamine, cephalosporin, iron, ORS...). They are
    # unclassified residue, not a pharmacological class, so they are held out
    # of the per-class targets and reported as a single "unclassified" bucket.
    named = orders["cat"].map(lambda c: pd.notna(c) and str(int(c)) in catnames)
    orders["cat_label"] = np.where(
        named, orders["cat"].map(
            lambda c: f"{_slug(catnames.get(str(int(c)), 'cat'))}_{int(c)}"
            if pd.notna(c) else None), None)

    pres_cat = orders.dropna(subset=["cat_label"]).groupby("cat_label")["prescription_id"].apply(set)
    tr_pids = set(pid[is_train].tolist())
    freq = {c: len(s & tr_pids) for c, s in pres_cat.items()}
    top_classes = [c for c, _ in sorted(freq.items(), key=lambda kv: -kv[1])][:N_TOP_CLASSES]
    for c in top_classes:
        T[f"y_class_{c}"] = pid.isin(pres_cat[c]).astype("int8").to_numpy()
    unclassified = set(orders.loc[~named.to_numpy(), "prescription_id"])
    T["y_class_unclassified"] = pid.isin(unclassified).astype("int8").to_numpy()

    # Structured attributes: the modal value across the orders of a prescription
    # (defined only where at least one drug was prescribed).
    o = orders.set_index("prescription_id")
    for src, dst, topn in (("dose_canon", "y_dose_mode", 6),
                           ("duration_bucket", "y_duration_mode", 7),
                           ("instruction", "y_instruction_mode", 5),
                           ("type_name", "y_type_mode", 4)):
        s = o[src].replace("", np.nan).dropna()
        modes = s.groupby(level=0).agg(lambda x: x.value_counts().idxmax())
        vals = pid.map(modes)
        tr_top = vals[is_train].value_counts().head(topn).index.tolist()
        keep = vals.where(vals.isin(tr_top), other="OTHER")
        T[dst] = keep.where(vals.notna(), other=None).to_numpy()

    T["y_chronic_therapy"] = pid.isin(
        set(orders.loc[orders["duration_bucket"] == ">90d/cont", "prescription_id"])
    ).astype("int8").to_numpy()

    T["y_n_advice"] = pid.map(advice.groupby("prescription_id").size()).fillna(
        0).astype(int).to_numpy()
    T["y_any_advice"] = (T["y_n_advice"] > 0).astype("int8")
    adv_sets = advice.groupby("advice_id")["prescription_id"].apply(set)
    adv_freq = {a: len(s & tr_pids) for a, s in adv_sets.items()}
    top_adv = [a for a, _ in sorted(adv_freq.items(), key=lambda kv: -kv[1])][:10]
    adv_lbl = advice.drop_duplicates("advice_id").set_index("advice_id")["advice_en"].to_dict()
    for a in top_adv:
        T[f"y_advice_{_slug(a)}"] = pid.isin(adv_sets[a]).astype("int8").to_numpy()

    T["y_any_test"] = pid.isin(set(tests["prescription_id"])).astype("int8").to_numpy()
    T["y_n_tests"] = pid.map(tests.groupby("prescription_id").size()).fillna(
        0).astype(int).to_numpy()
    tst_sets = tests.groupby("test_id")["prescription_id"].apply(set)
    tst_freq = {t: len(s & tr_pids) for t, s in tst_sets.items()}
    top_tst = [t for t, _ in sorted(tst_freq.items(), key=lambda kv: -kv[1])][:8]
    tst_lbl = tests.drop_duplicates("test_id").set_index("test_id")["test_name"].to_dict()
    for t in top_tst:
        T[f"y_test_{_slug(t)}"] = pid.isin(tst_sets[t]).astype("int8").to_numpy()

    class_mat = pd.DataFrame({c: T[f"y_class_{c}"].to_numpy() for c in top_classes})
    meta = {
        "top_classes": top_classes,
        "class_prevalence": {c: float(class_mat[c].mean()) for c in top_classes},
        "profile_classes": top_classes[:N_PROFILE_CLASSES],
        "unclassified_prevalence": float(T["y_class_unclassified"].mean()),
        "top_advice": {f"y_advice_{_slug(a)}": str(adv_lbl.get(a, a)).strip()[:110]
                       for a in top_adv},
        "top_tests": {f"y_test_{_slug(t)}": str(tst_lbl.get(t, t)).strip()
                      for t in top_tst},
    }
    return T, meta, class_mat


# ---------------------------------------------------------------------------
# Temporal split boundaries, matching src/phcrx/temporal/class_temporal.py so a
# feature matrix built here is directly comparable with that evaluation.
TEMPORAL_TRAIN_END, TEMPORAL_VAL_END = 2015, 2016


def temporal_split(enc: pd.DataFrame) -> pd.Series:
    """train <=2015 / val 2016 / test >=2017, derived from the year column."""
    yr = pd.to_numeric(enc["year"], errors="coerce")
    return pd.Series(
        np.where(yr <= TEMPORAL_TRAIN_END, "train",
                 np.where(yr <= TEMPORAL_VAL_END, "val", "test")),
        index=enc.index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["patient", "temporal"], default="patient",
                    help="which split defines the fitting mask. Every transform "
                         "in this module is fitted on the TRAIN rows of this "
                         "split, so a temporal evaluation needs --split temporal "
                         "or the features have already seen later-era rows.")
    ap.add_argument("--out", default=None,
                    help="output parquet; defaults to features.parquet for "
                         "patient and features_temporal.parquet otherwise")
    args = ap.parse_args()

    enc = pd.read_parquet(PROCESSED / "rxgen_encounters.parquet").reset_index(drop=True)
    if args.split == "temporal":
        enc = enc.copy()
        enc["split"] = temporal_split(enc).values
        yr = pd.to_numeric(enc["year"], errors="coerce")
        # Assert the eras really are disjoint before anything is fitted.
        for a, b in (("train", "val"), ("val", "test"), ("train", "test")):
            ya, yb = yr[enc["split"] == a], yr[enc["split"] == b]
            if len(ya) and len(yb):
                assert ya.max() < yb.min(), f"{a}/{b} eras overlap"
    is_train = (enc["split"] == "train").to_numpy()
    out_path = (pathlib.Path(args.out) if args.out
                else (OUT_PARQUET if args.split == "patient"
                      else OUT_PARQUET.with_name("features_temporal.parquet")))
    print(f"split={args.split}  encounters={len(enc)}  train={is_train.sum()}  "
          f"val={(enc['split'] == 'val').sum()}  test={(enc['split'] == 'test').sum()}")
    if args.split == "temporal":
        yr = pd.to_numeric(enc["year"], errors="coerce")
        print("   era bounds: " + "  ".join(
            f"{s}=[{int(yr[enc['split']==s].min())},{int(yr[enc['split']==s].max())}]"
            for s in ("train", "val", "test") if (enc["split"] == s).any()))

    T, tmeta, class_mat = _build_targets(enc, is_train)
    print(f"targets built: {sum(c.startswith('y_') for c in T.columns)} columns")

    F: dict[str, np.ndarray] = {}
    fam: dict[str, str] = {}
    cat_meta: dict[str, int] = {}

    _demographics(enc, F, fam)
    _vitals(enc, F, fam)
    _derived_clinical(enc, F, fam)
    _ncd(enc, F, fam)
    _icd(enc, F, fam, is_train)
    tmeta["text"] = _text(enc, F, fam, is_train)
    _temporal_history(enc, F, fam, is_train,
                      T["y_any_drug"].to_numpy().astype(float),
                      T["y_n_drugs"].to_numpy().astype(float))
    _site_prescriber(enc, F, fam, is_train,
                     T["y_any_drug"].to_numpy().astype(float),
                     T["y_n_drugs"].to_numpy().astype(float),
                     class_mat, tmeta["profile_classes"],
                     T["y_any_advice"].to_numpy().astype(float),
                     T["y_any_test"].to_numpy().astype(float),
                     cat_meta)

    X = pd.DataFrame(F)
    out = pd.concat([
        enc[["prescription_id", "user_id", "split", "checkup_date"]].reset_index(drop=True),
        X, T.drop(columns=["prescription_id"]),
    ], axis=1)
    out.to_parquet(out_path, index=False)

    counts = pd.Series(fam).value_counts().to_dict()
    fam_path = (OUT_FAMILIES if args.split == "patient"
                else OUT_FAMILIES.with_name("feature_families_temporal.json"))
    fam_path.write_text(json.dumps({
        "families": fam,
        "family_counts": counts,
        "categorical_features": cat_meta,
        "feature_columns": list(X.columns),
        "target_columns": [c for c in out.columns if c.startswith("y_")],
        "meta": tmeta,
        "notes": {
            "confounders": ["site", "prescriber"],
            "fitted_on": "train split only (TF-IDF vocab, char SVD, category "
                         "levels, site/prescriber target encodings; train rows "
                         "leave-one-out)",
        },
    }, indent=2), encoding="utf-8")

    print(f"\nfeatures: {X.shape[1]} columns")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<12} {v:4d}")
    print(f"\nwrote {out_path}  shape={out.shape}")
    print(f"wrote {fam_path}")
    print("\ntop drug classes (targets):")
    for c in tmeta["top_classes"]:
        print(f"  {c:<34} prevalence={tmeta['class_prevalence'][c]:.3f}")


if __name__ == "__main__":
    main()
