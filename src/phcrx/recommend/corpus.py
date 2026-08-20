"""Corpus, targets and label spaces for the recommender.

One row per encounter (14,074), the patient-level `split` column taken as
given from `data/processed/rxgen_encounters.parquet` -- no patient appears in
two splits and no split is re-derived here.

Six components are modelled. Two are deliberately absent:

  * **dose** and **duration**. `docs/feature_importance.md` measures them at
    accuracy 0.506 / 0.503 against a strong majority class, and *worse* when
    prescriber and site are removed (0.474 / 0.438). They are predictable from
    who wrote the prescription, not from the patient. A suggestion list that
    proposed a dose on that evidence would be presenting prescriber habit as
    clinical advice.

Drug classes are the **normalised pharmacological classes** from
`data/processed/drug_normalization.parquet` (46 classes, 97.6% of orders
mapped), not the 719 brands and not the 89 legacy `rx_category` values. The
drug-normalisation workstream measured the cross-era strict unseen-order rate
at 12.2% for brands versus 3.4% for classes; the brand is procurement, the
class is the clinical decision. The legacy 89-category space is still built on
request (`label_space="cat89"`) because that is the space the published neural
numbers live in, and the head-to-head has to be run there to be honest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import PROCESSED

# An advice item or diagnostic test needs this many TRAIN encounters before it
# enters the label space. Below it a per-label F1 is an accident of two or
# three rows and a "suggestion" is not supportable.
MIN_LABEL_SUPPORT = 30

RAW_VITALS = ["height", "weight", "bmi", "waist_hip_ratio", "temperature",
              "oxygen_of_blood", "bp_sys", "bp_dia", "blood_glucose",
              "blood_hemoglobin", "pulse_rate", "cholesterol", "uric_acid"]
RAW_NUMERIC = ["age", "smoker_flag"]
RAW_CATEGORICAL = ["sex", "glucose_type"]

# Families excluded by the `clinical` arms. prescriber+site reproduce the
# feature-importance workstream's `clinical_only` variant; `temporal` is added
# because it carries the largest permutation importance of any family (0.115)
# and is cohort drift rather than patient signal.
NON_CLINICAL_FAMILIES = ("prescriber", "site", "temporal")

DRUG_FORM_KEEP = ("Tab", "Cap", "Syp")
DRUG_FORM_OTHER = "other"


# ---------------------------------------------------------------------------
# frame
# ---------------------------------------------------------------------------
def load_frame() -> tuple[pd.DataFrame, dict]:
    """Encounters joined to the engineered feature matrix, plus family map.

    The engineered columns come from `data/processed/features.parquet`, which
    the feature workstream fitted on TRAIN only (TF-IDF vocabulary, char-SVD
    basis, categorical levels, and leave-one-out site/prescriber target
    encodings). They are consumed here, never refitted.
    """
    enc = pd.read_parquet(PROCESSED / "rxgen_encounters.parquet")
    enc["symptom_text"] = enc["symptom_text"].fillna("").astype(str)
    enc["prescription_id"] = enc["prescription_id"].astype(int)

    feat = pd.read_parquet(PROCESSED / "features.parquet")
    feat["prescription_id"] = feat["prescription_id"].astype(int)
    FF = json.loads((PROCESSED / "feature_families.json").read_text())
    eng_cols = [c for c in FF["feature_columns"] if c in feat.columns]

    feat = feat.set_index("prescription_id").reindex(enc["prescription_id"])
    for c in eng_cols:
        enc[c] = feat[c].to_numpy()

    FF["engineered_columns"] = eng_cols
    return enc.reset_index(drop=True), FF


def split_index(df: pd.DataFrame, mask: np.ndarray | None = None) -> dict:
    split = df["split"].to_numpy()
    keep = np.ones(len(df), dtype=bool) if mask is None else np.asarray(mask)
    return {s: np.where(keep & (split == s))[0] for s in ("train", "val", "test")}


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
def _orders() -> pd.DataFrame:
    o = pd.read_parquet(PROCESSED / "rxgen_orders.parquet")
    return o.dropna(subset=["prescription_id"]).copy()


def _multi_hot(df: pd.DataFrame, pids: np.ndarray, key: str,
               labels: list) -> np.ndarray:
    pid2row = {int(p): i for i, p in enumerate(pids)}
    lab2col = {v: j for j, v in enumerate(labels)}
    Y = np.zeros((len(pids), len(labels)), dtype=bool)
    for p, v in zip(df["prescription_id"], df[key]):
        r, j = pid2row.get(int(p)), lab2col.get(v)
        if r is not None and j is not None:
            Y[r, j] = True
    return Y


def drug_class_map() -> dict[int, str]:
    dn = pd.read_parquet(PROCESSED / "drug_normalization.parquet")
    dn = dn[dn["drug_class"].notna()]
    return dict(zip(dn["drug_id"].astype(int), dn["drug_class"].astype(str)))


def category_map() -> tuple[dict[int, int], dict[int, str]]:
    """Legacy 89-way `rx_category` space, for the neural head-to-head only."""
    V = json.loads((PROCESSED / "rxgen_vocab.json").read_text(encoding="utf-8"))
    drug2cat = {int(k): int(v) for k, v in V["drug2cat"].items()}
    inv = {int(v): k for k, v in V["category"].items()}
    names = {c: V["category_names"].get(inv.get(c, ""), f"cat{c}")
             for c in set(drug2cat.values())}
    return drug2cat, names


def drug_class_labels(df: pd.DataFrame, label_space: str = "class46"):
    """(Y, label names). `class46` = normalised classes; `cat89` = legacy."""
    pids = df["prescription_id"].to_numpy()
    o = _orders().dropna(subset=["drug_id"])
    o["drug_id"] = o["drug_id"].astype(int)
    if label_space == "class46":
        o["lab"] = o["drug_id"].map(drug_class_map())
        o = o[o["lab"].notna()]
        labels = sorted(o["lab"].unique().tolist())
        names = list(labels)
    elif label_space == "cat89":
        drug2cat, cat_names = category_map()
        o["lab"] = o["drug_id"].map(drug2cat).fillna(0).astype(int)
        o = o[o["lab"] != 0]
        labels = sorted(o["lab"].unique().tolist())
        # The numeric category id is kept in the label string: the cached
        # neural predictions are lists of category *ids*, so the head-to-head
        # needs to map them back to columns.
        names = [f"{c}|{cat_names.get(c, c)}" for c in labels]
    else:
        raise ValueError(label_space)
    return _multi_hot(o, pids, "lab", labels), names


def _support_filtered(df, tab, key, pids, min_support):
    tab = tab.drop_duplicates(["prescription_id", key])
    train_pids = set(df.loc[df.split == "train", "prescription_id"].astype(int))
    freq = tab.loc[tab["prescription_id"].astype(int).isin(train_pids),
                   key].value_counts()
    labels = [v for v, n in freq.items() if n >= min_support]
    labels = sorted(labels, key=lambda v: (-int(freq[v]), str(v)))
    return _multi_hot(tab, pids, key, labels), labels


def advice_labels(df, min_support: int = MIN_LABEL_SUPPORT):
    adv = pd.read_parquet(PROCESSED / "rxgen_advice.parquet")
    Y, labels = _support_filtered(df, adv, "advice_id",
                                  df["prescription_id"].to_numpy(), min_support)
    text = (adv.drop_duplicates("advice_id").set_index("advice_id")["advice_en"]
            .astype(str).str.replace(r"\s+", " ", regex=True).str.strip())
    return Y, [str(a) for a in labels], [text.get(a, str(a))[:110] for a in labels]


def test_labels(df, min_support: int = MIN_LABEL_SUPPORT):
    tst = pd.read_parquet(PROCESSED / "rxgen_tests.parquet")
    Y, labels = _support_filtered(df, tst, "test_id",
                                  df["prescription_id"].to_numpy(), min_support)
    text = (tst.drop_duplicates("test_id").set_index("test_id")["test_name"]
            .astype(str).str.strip())
    return Y, [str(t) for t in labels], [text.get(t, str(t)) for t in labels]


def count_labels(df) -> np.ndarray:
    n = _orders().groupby("prescription_id").size()
    return df["prescription_id"].map(n).fillna(0).to_numpy(dtype=float)


def form_labels(df) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Modal dispensing form per encounter, over encounters with >=1 order.

    Forms outside the three that clear a usable support floor are folded into
    `other` rather than dropped, so the row set is exactly "every encounter
    that received pharmacotherapy".
    """
    o = _orders().dropna(subset=["type_name"])
    mode = (o.groupby("prescription_id")["type_name"]
            .agg(lambda s: s.value_counts().idxmax()))
    v = df["prescription_id"].map(mode)
    mask = v.notna().to_numpy()
    y = np.where(v.isin(DRUG_FORM_KEEP), v.astype(object), DRUG_FORM_OTHER)
    y = np.where(mask, y, DRUG_FORM_OTHER)
    classes = list(DRUG_FORM_KEEP) + [DRUG_FORM_OTHER]
    return y.astype(object), mask, classes


# ---------------------------------------------------------------------------
# component registry
# ---------------------------------------------------------------------------
@dataclass
class Component:
    name: str
    kind: str                      # binary | count | multilabel | multiclass
    headline: str                  # val metric used to choose the feature set
    Y: object = None
    mask: np.ndarray = None
    labels: list = field(default_factory=list)
    label_text: list = field(default_factory=list)

    @property
    def n_labels(self) -> int:
        return len(self.labels)


def build_components(df: pd.DataFrame,
                     label_space: str = "class46",
                     min_support: int = MIN_LABEL_SUPPORT) -> dict:
    n = len(df)
    allrows = np.ones(n, dtype=bool)
    comps: dict[str, Component] = {}

    n_drugs = count_labels(df)
    comps["any_drug"] = Component(
        "any_drug", "binary", "auroc",
        Y=(n_drugs > 0).astype(np.int8), mask=allrows,
        labels=["any_drug"], label_text=["any pharmacotherapy at all"])
    comps["n_drugs"] = Component(
        "n_drugs", "count", "mae", Y=n_drugs, mask=allrows,
        labels=["n_drugs"], label_text=["number of drugs on the prescription"])

    Yc, cnames = drug_class_labels(df, label_space)
    comps["drug_classes"] = Component(
        "drug_classes", "multilabel", "micro_f1", Y=Yc, mask=allrows,
        labels=cnames, label_text=cnames)

    Ya, alab, atxt = advice_labels(df, min_support)
    comps["advice"] = Component("advice", "multilabel", "micro_f1",
                                Y=Ya, mask=allrows, labels=alab, label_text=atxt)

    Yt, tlab, ttxt = test_labels(df, min_support)
    comps["tests"] = Component("tests", "multilabel", "micro_f1",
                               Y=Yt, mask=allrows, labels=tlab, label_text=ttxt)

    yf, fmask, fclasses = form_labels(df)
    comps["drug_form"] = Component("drug_form", "multiclass", "macro_f1",
                                   Y=yf, mask=fmask, labels=fclasses,
                                   label_text=fclasses)
    return comps


# Components measured and deliberately not modelled, with the measurement that
# rules them out. Printed by `evaluate` and quoted in docs/recommender.md.
EXCLUDED = {
    "dose": ("accuracy 0.506 against a strong majority class, and 0.474 once "
             "prescriber and site are removed -- predictable from the "
             "prescriber, not the patient (docs/feature_importance.md)"),
    "duration": ("accuracy 0.503, falling to 0.438 without prescriber/site; "
                 "same diagnosis as dose (docs/feature_importance.md)"),
    "instruction": ("accuracy 0.725 -> 0.636 without prescriber/site; excluded "
                    "for the same reason, and it is a dispensing convention "
                    "rather than a clinical decision"),
}
