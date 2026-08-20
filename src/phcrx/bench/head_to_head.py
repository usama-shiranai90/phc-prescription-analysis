"""Matched-protocol head-to-head: PHC-RxGen vs. linear baselines.

    python -m src.phcrx.bench.head_to_head

Why this module exists
----------------------
Two numbers were floating around that looked comparable and are not:

  * PHC-RxGen (3.5-5.1M params) scores category-level micro-F1 ~0.335 on the
    test split (`results/rx_generation/rxgen_ablations.json`, field
    `test.category_level.cat_micro_f1`).
  * A TF-IDF + one-vs-rest logistic regression proxy scores 0.4204
    (`results/rx_generation/textproc/ablation.csv`, arm a).

They disagree on three axes at once:

  1. **rows** - the proxy keeps only encounters that have a symptom note *and*
     at least one in-vocabulary drug class (1,923 of 2,780 test encounters);
     the neural evaluation scores every test encounter, including the ~21% with
     an empty prescription and the ones with no free text at all.
  2. **label space** - the proxy keeps the 47 classes with >=10 training
     orders; the neural evaluation maps generated brands through `drug2cat`
     into the full 89-way category vocabulary with no frequency floor.
  3. **decision rule** - the proxy thresholds 47 independent probabilities at a
     single cut-off tuned on VAL; the neural model emits a *set*
     autoregressively with no tunable operating point.

None of that is a defect of either evaluation; they were built for different
questions. But it means the 0.07 gap is uninterpretable as stated. This module
fixes ONE protocol at a time - same rows, same label set, same metric - and
runs every system through it.

Protocols
---------
``full89``
    All 2,780 test encounters. Label space = every non-``<na>`` category that
    occurs anywhere in the corpus, no frequency floor. Encounters with an empty
    prescription are kept and an empty prediction is the correct answer for
    them. This is the protocol the neural evaluation already used, so the
    neural number here should reproduce the published `cat_micro_f1`.

``restricted47``
    The proxy's protocol, reproduced exactly: rows with a non-empty symptom
    note *and* >=1 label inside the 47-class space (classes with >=10 train
    orders). Gold *and* predictions from every system - the neural model
    included - are intersected with those 47 classes, so nobody is charged for
    emitting a class the protocol does not score.

Systems
-------
    empty                 predict nothing (floor)
    prior_topk            always the k most frequent train classes, k tuned on VAL
    lr_text               TF-IDF (uni+bigram) over the symptom note
    lr_tab_physio         vitals + derived + NCD flags + demographics + visit
                          index: strictly non-text, no confounders
    lr_tab_clinical       engineered columns minus {prescriber, site, temporal}
    lr_tab_all            all 309 engineered columns. NB 170 of them are
                          text-derived and 30 are ICD codes read off the note,
                          so this arm is *not* text-free
    lr_text_tab_physio    TF-IDF + physiology/demographics
    lr_text_tab_clinical  TF-IDF + clinical families
    lr_text_tab_all       TF-IDF + all engineered columns
    neural_rxgen          the trained checkpoint, generated set -> categories

Every linear arm is one-vs-rest L2 logistic regression, identical `C`, with a
single global threshold tuned on VAL by micro-F1 - the same estimator the
textproc ablation used, so the architecture is the only thing that varies.
Vectoriser, imputer, scaler and one-hot levels are fitted on TRAIN ONLY.

The ``clinical_only`` family drop is motivated by a measured result from the
feature-importance workstream: removing prescriber+site raised held-out AUROC
for `y_any_drug` from 0.707 to 0.863 and cut `y_n_drugs` MAE from 1.185 to
0.977 (`results/rx_generation/features/variant_comparison.csv`). Here the drop
also includes `temporal`, which carries the largest permutation importance of
any family in the `full` variant (0.115) and is pure cohort drift.

Operating point
---------------
The neural decoder has no threshold, so a threshold tuned for micro-F1 gives
the linear arms a free parameter. Each linear arm is therefore *also* reported
at a "size-matched" threshold, chosen on VAL so its mean predicted set size
matches the neural model's mean predicted set size on VAL. If the linear arm
still wins there, the win is not an artefact of operating-point tuning.

Uncertainty
-----------
Paired bootstrap over test encounters (default 5,000 resamples) on the micro-F1
*difference* from the neural model. Paired, because every system is scored on
the same rows and an unpaired interval would badly overstate the uncertainty of
a comparison. Micro-F1 under resampling is computed exactly from cached
per-row (tp, |pred|, |gold|) counts, since
``micro_f1 = 2*sum(tp) / (sum(|pred|) + sum(|gold|))``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..config import PROCESSED, MODELS, DataConfig
from ..preprocess import tokenize

HERE = Path(__file__).resolve().parent
SEED = 0
C_LR = 4.0                       # matches textproc.evaluate.fit_predict
THR_GRID = np.arange(0.05, 0.71, 0.01)
# Families dropped by the `clinical_only` variant. prescriber+site reproduce the
# feature-importance workstream's definition; `temporal` is added because it
# carries the largest permutation importance of any family in the `full` variant
# (0.115) and is cohort drift rather than patient signal.
NON_CLINICAL_FAMILIES = ("prescriber", "site", "temporal")
# Strictly non-text clinical signal. `text` (170 cols) and `icd` (30 cols, coded
# from the note) are both derived from the symptom note, so excluding them
# isolates the tabular channel the way the neural `tabular_only` ablation does.
PHYSIO_FAMILIES = ("demog", "vitals", "derived", "ncd", "history")


# ---------------------------------------------------------------------------
# corpus / labels
# ---------------------------------------------------------------------------
def load_tables():
    enc = pd.read_parquet(PROCESSED / "rxgen_encounters.parquet")
    enc["symptom_text"] = enc["symptom_text"].fillna("").astype(str)
    orders = pd.read_parquet(PROCESSED / "rxgen_orders.parquet")
    V = json.loads((PROCESSED / "rxgen_vocab.json").read_text(encoding="utf-8"))
    return enc, orders, V


def encounter_categories(enc, orders, V):
    """prescription_id -> set of non-zero drug-category ids, plus order counts.

    Mirrors `train.evaluate`: brands are mapped through `drug2cat` and the
    unmapped class (0) is *dropped*, not collapsed, so unmapped brands cannot
    inflate agreement.
    """
    drug2cat = {int(k): int(v) for k, v in V["drug2cat"].items()}
    o = orders.dropna(subset=["drug_id", "prescription_id"]).copy()
    o["cat"] = o["drug_id"].astype(int).map(drug2cat).fillna(0).astype(int)
    o = o[o["cat"] != 0]
    o["prescription_id"] = o["prescription_id"].astype(int)
    return o


def label_space(o, enc, min_label_freq: int) -> list[int]:
    """Categories kept by the protocol.

    `min_label_freq` counts *orders* in the train split (reproducing
    `textproc.evaluate.load_task`); 0 means "every category in the corpus".
    """
    if min_label_freq <= 0:
        return sorted(o["cat"].unique().tolist())
    tr = set(enc.loc[enc.split == "train", "prescription_id"].astype(int))
    freq = o.loc[o["prescription_id"].isin(tr), "cat"].value_counts()
    return sorted(int(c) for c, n in freq.items() if n >= min_label_freq)


def gold_matrix(enc, o, labels) -> np.ndarray:
    pid2row = {int(p): i for i, p in enumerate(enc["prescription_id"])}
    cat2col = {c: j for j, c in enumerate(labels)}
    Y = np.zeros((len(enc), len(labels)), dtype=bool)
    for pid, cat in zip(o["prescription_id"], o["cat"]):
        r, j = pid2row.get(int(pid)), cat2col.get(int(cat))
        if r is not None and j is not None:
            Y[r, j] = True
    return Y


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def row_counts(Y: np.ndarray, P: np.ndarray):
    """Per-row (tp, |pred|, |gold|) - everything micro-F1 needs, bootstrap included."""
    tp = (Y & P).sum(1).astype(np.float64)
    return tp, P.sum(1).astype(np.float64), Y.sum(1).astype(np.float64)


def micro_f1_from_counts(tp, np_, ng) -> float:
    denom = np_.sum() + ng.sum()
    return float(2 * tp.sum() / denom) if denom else 0.0


def set_metrics(Y: np.ndarray, P: np.ndarray) -> dict:
    """Identical maths to `metrics.set_metrics`, on binary indicator matrices."""
    tp, np_, ng = row_counts(Y, P)
    union = np_ + ng - tp
    jac = np.where(union == 0, 1.0, tp / np.maximum(union, 1e-9))
    exact = (Y == P).all(1)
    TP, FP, FN = tp.sum(), (np_ - tp).sum(), (ng - tp).sum()
    pr = TP / (TP + FP) if TP + FP else 0.0
    rc = TP / (TP + FN) if TP + FN else 0.0
    present = Y.any(0)                       # labels with >=1 gold row here
    ltp = (Y & P).sum(0).astype(float)
    lf1 = 2 * ltp / np.maximum(Y.sum(0) + P.sum(0), 1e-9)
    return {
        "micro_f1": micro_f1_from_counts(tp, np_, ng),
        "micro_precision": float(pr),
        "micro_recall": float(rc),
        "macro_f1": float(lf1[present].mean()) if present.any() else 0.0,
        "macro_f1_all_labels": float(lf1.mean()),
        "jaccard": float(jac.mean()),
        "exact_match": float(exact.mean()),
        "mean_pred_size": float(np_.mean()),
        "mean_gold_size": float(ng.mean()),
        "empty_pred_rate": float((np_ == 0).mean()),
    }


def bootstrap_micro(tp, np_, ng, n: int, seed: int):
    rng = np.random.default_rng(seed)
    m = len(tp)
    vals = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, m, m)
        vals[k] = micro_f1_from_counts(tp[idx], np_[idx], ng[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap(ref, arm, n: int, seed: int):
    """95% CI on micro_f1(arm) - micro_f1(ref), resampling test encounters."""
    rng = np.random.default_rng(seed)
    (tp_r, np_r, ng), (tp_a, np_a, _) = ref, arm
    m = len(ng)
    d = np.empty(n)
    for k in range(n):
        i = rng.integers(0, m, m)
        d[k] = (micro_f1_from_counts(tp_a[i], np_a[i], ng[i])
                - micro_f1_from_counts(tp_r[i], np_r[i], ng[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta_mean": float(d.mean()), "delta_lo95": float(lo),
            "delta_hi95": float(hi), "p_better": float((d > 0).mean())}


# ---------------------------------------------------------------------------
# neural predictions (generate once, cache)
# ---------------------------------------------------------------------------
def neural_category_sets(ckpt: Path, batch_size: int = 64, use_cache: bool = True):
    """prescription_id -> predicted category set, for val and test.

    Runs `model.generate` exactly as `train.evaluate` does (greedy, no-repeat,
    max_len = cfg.data.max_rx_len) and maps vocabulary ids through
    `corpus.vid2cat`, dropping category 0.
    """
    cache = HERE / "neural_preds_cache.json"
    key = f"{ckpt.name}:{int(ckpt.stat().st_mtime)}"
    if use_cache and cache.exists():
        blob = json.loads(cache.read_text())
        if blob.get("key") == key:
            return {s: {int(k): v for k, v in d.items()}
                    for s, d in blob["preds"].items()}

    import torch
    from torch.utils.data import DataLoader
    from ..data import RxCorpus, RxDataset
    from ..nlp.predict_helper import load_rxgen

    corpus = RxCorpus(DataConfig())
    model, device = load_rxgen(ckpt, corpus)      # asserts vocab compatibility
    preds = {}
    with torch.no_grad():
        for split in ("val", "test"):
            ds = RxDataset(corpus, split)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
            out = {}
            for batch in dl:
                b = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
                gen = model.generate(b, max_len=corpus.cfg.max_rx_len)
                for i, pid in enumerate(b["pid"].tolist()):
                    cats = {corpus.vid2cat.get(d, 0) for d in gen["drugs"][i]}
                    out[int(pid)] = sorted(cats - {0})
            preds[split] = out
    cache.write_text(json.dumps({"key": key, "preds": preds}))
    return preds


# ---------------------------------------------------------------------------
# feature builders (TRAIN-ONLY fitting throughout)
# ---------------------------------------------------------------------------
def text_docs(texts) -> list[list[str]]:
    """Unigrams + bigrams, i.e. textproc arm (a) 'raw text'.

    `Normalizer(glossary=False, orthography=False, typo=False)` reduces to
    `norm_text(t).lower()`, which `preprocess.tokenize` already applies, so
    calling `tokenize` directly reproduces that arm exactly.
    """
    out = []
    for t in texts:
        toks = tokenize(t)
        out.append(toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])])
    return out


def build_text_blocks(text, idx, min_df: int = 2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    d = {s: text_docs(text[idx[s]]) for s in ("train", "val", "test")}
    vec = TfidfVectorizer(analyzer=lambda x: x, min_df=min_df, sublinear_tf=True)
    Xtr = vec.fit_transform(d["train"])
    return (Xtr, vec.transform(d["val"]), vec.transform(d["test"]))


def build_tabular_blocks(feat: pd.DataFrame, cols, onehot_cols, idx):
    """Median-impute + standardise numeric columns, one-hot the code columns."""
    tr = idx["train"]
    blocks = {s: [] for s in ("train", "val", "test")}

    num_cols = [c for c in cols if c not in onehot_cols]
    A = feat[num_cols].to_numpy(dtype=np.float64)
    med = np.nanmedian(A[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    A = np.where(np.isnan(A), med, A)
    mu, sd = A[tr].mean(0), A[tr].std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    A = (A - mu) / sd
    for s in ("train", "val", "test"):
        blocks[s].append(sp.csr_matrix(A[idx[s]]))

    for c in onehot_cols:
        if c not in cols:
            continue
        levels = sorted(pd.unique(feat[c].iloc[tr].dropna()))
        lut = {v: j for j, v in enumerate(levels)}
        for s in ("train", "val", "test"):
            v = feat[c].iloc[idx[s]].to_numpy()
            r = [i for i, x in enumerate(v) if x in lut]
            cc = [lut[v[i]] for i in r]
            blocks[s].append(sp.csr_matrix(
                (np.ones(len(r)), (r, cc)), shape=(len(v), len(levels))))

    return tuple(sp.hstack(blocks[s]).tocsr() for s in ("train", "val", "test"))


# ---------------------------------------------------------------------------
# systems
# ---------------------------------------------------------------------------
def fit_ovr(Xtr, Ytr, Xva, Xte, C: float = C_LR):
    """One-vs-rest L2 logistic regression; columns with no train positive are 0."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    keep = np.where(Ytr.sum(0) > 0)[0]
    # n_jobs=1: liblinear needs a writable buffer, and joblib memory-maps
    # worker arrays read-only. Serial fitting costs seconds here.
    clf = OneVsRestClassifier(
        LogisticRegression(C=C, max_iter=3000, solver="liblinear"), n_jobs=1)
    clf.fit(Xtr, np.ascontiguousarray(Ytr[:, keep].astype(np.int8)))
    out = []
    for X in (Xva, Xte):
        P = np.zeros((X.shape[0], Ytr.shape[1]))
        P[:, keep] = clf.predict_proba(X)
        out.append(P)
    return out


def tune_threshold(Yva, Pva) -> float:
    best, bf = float(THR_GRID[0]), -1.0
    for t in THR_GRID:
        tp, np_, ng = row_counts(Yva, Pva >= t)
        f = micro_f1_from_counts(tp, np_, ng)
        if f > bf:
            best, bf = float(t), f
    return best


def size_matched_threshold(Pva, target_size: float) -> float:
    grid = np.arange(0.01, 0.96, 0.005)
    sizes = np.array([(Pva >= t).sum(1).mean() for t in grid])
    return float(grid[int(np.argmin(np.abs(sizes - target_size)))])


def prior_topk(Ytr, Yva, kmax: int = 8):
    order = np.argsort(-Ytr.mean(0))
    best_k, bf = 1, -1.0
    for k in range(1, kmax + 1):
        P = np.zeros_like(Yva)
        P[:, order[:k]] = True
        tp, np_, ng = row_counts(Yva, P)
        f = micro_f1_from_counts(tp, np_, ng)
        if f > bf:
            best_k, bf = k, f
    return best_k, order[:best_k]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_protocol(name: str, cfg: dict, ctx: dict, args) -> dict:
    enc, o, feat, FF, neural = (ctx[k] for k in
                                ("enc", "orders_cat", "feat", "families", "neural"))
    labels = label_space(o, enc, cfg["min_label_freq"])
    Yall = gold_matrix(enc, o, labels)
    cat2col = {c: j for j, c in enumerate(labels)}

    split = enc["split"].to_numpy()
    mask = np.ones(len(enc), dtype=bool)
    if cfg["require_text"]:
        mask &= (enc["symptom_text"].str.strip() != "").to_numpy()
    if cfg["require_label"]:
        mask &= Yall.any(1)
    idx = {s: np.where(mask & (split == s))[0] for s in ("train", "val", "test")}
    Y = {s: Yall[idx[s]] for s in ("train", "val", "test")}

    pids = {s: enc["prescription_id"].to_numpy()[idx[s]].astype(int)
            for s in ("val", "test")}
    text = enc["symptom_text"].to_numpy()

    # neural predictions -> the protocol's label space
    def neural_P(s):
        P = np.zeros((len(pids[s]), len(labels)), dtype=bool)
        for i, pid in enumerate(pids[s]):
            for c in neural[s].get(int(pid), []):
                j = cat2col.get(int(c))
                if j is not None:
                    P[i, j] = True
        return P

    Pn = {s: neural_P(s) for s in ("val", "test")}
    neural_val_size = float(Pn["val"].sum(1).mean())

    print(f"\n### protocol {name}: labels={len(labels)} "
          f"rows train={len(idx['train'])} val={len(idx['val'])} test={len(idx['test'])}")
    print(f"    mean gold classes/encounter: train={Y['train'].sum(1).mean():.3f} "
          f"test={Y['test'].sum(1).mean():.3f}; "
          f"empty-gold test rows={float((Y['test'].sum(1) == 0).mean()):.3f}")
    print(f"    neural mean predicted set size: val={neural_val_size:.3f}")

    # --- feature blocks ----------------------------------------------------
    fam = FF["families"]
    all_cols = [c for c in FF["feature_columns"] if c in feat.columns]
    clin_cols = [c for c in all_cols if fam.get(c) not in NON_CLINICAL_FAMILIES]
    physio_cols = [c for c in all_cols if fam.get(c) in PHYSIO_FAMILIES]
    onehot = list(FF["categorical_features"])
    blocks = {}
    blocks["text"] = build_text_blocks(text, idx)
    blocks["tab_all"] = build_tabular_blocks(feat, all_cols, onehot, idx)
    blocks["tab_clinical"] = build_tabular_blocks(feat, clin_cols, onehot, idx)
    blocks["tab_physio"] = build_tabular_blocks(feat, physio_cols, onehot, idx)
    print(f"    features: text={blocks['text'][0].shape[1]} "
          f"tab_all={blocks['tab_all'][0].shape[1]} ({len(all_cols)} cols) "
          f"tab_clinical={blocks['tab_clinical'][0].shape[1]} ({len(clin_cols)} cols) "
          f"tab_physio={blocks['tab_physio'][0].shape[1]} ({len(physio_cols)} cols)")

    arms = {
        "lr_text": ["text"],
        "lr_tab_physio": ["tab_physio"],
        "lr_tab_clinical": ["tab_clinical"],
        "lr_tab_all": ["tab_all"],
        "lr_text_tab_physio": ["text", "tab_physio"],
        "lr_text_tab_clinical": ["text", "tab_clinical"],
        "lr_text_tab_all": ["text", "tab_all"],
    }

    results, test_pred = {}, {}

    # --- floors ------------------------------------------------------------
    test_pred["empty"] = np.zeros_like(Y["test"])
    results["empty"] = {"threshold": None, "val_micro_f1": 0.0,
                        **set_metrics(Y["test"], test_pred["empty"])}

    k, top = prior_topk(Y["train"], Y["val"])
    Pv = np.zeros_like(Y["val"]); Pv[:, top] = True
    Pt = np.zeros_like(Y["test"]); Pt[:, top] = True
    test_pred[f"prior_top{k}"] = Pt
    results[f"prior_top{k}"] = {
        "threshold": None,
        "val_micro_f1": micro_f1_from_counts(*row_counts(Y["val"], Pv)),
        **set_metrics(Y["test"], Pt)}

    # --- linear arms -------------------------------------------------------
    for arm, parts in arms.items():
        t0 = time.time()
        X = [sp.hstack([blocks[p][i] for p in parts]).tocsr() if len(parts) > 1
             else blocks[parts[0]][i] for i in range(3)]
        Pva, Pte = fit_ovr(X[0], Y["train"], X[1], X[2])
        thr = tune_threshold(Y["val"], Pva)
        test_pred[arm] = Pte >= thr
        results[arm] = {
            "threshold": thr,
            "val_micro_f1": micro_f1_from_counts(*row_counts(Y["val"], Pva >= thr)),
            **set_metrics(Y["test"], test_pred[arm])}
        # operating point matched to the neural decoder's mean set size on VAL
        sthr = size_matched_threshold(Pva, neural_val_size)
        nm = f"{arm}@sizematch"
        test_pred[nm] = Pte >= sthr
        results[nm] = {
            "threshold": sthr,
            "val_micro_f1": micro_f1_from_counts(*row_counts(Y["val"], Pva >= sthr)),
            **set_metrics(Y["test"], test_pred[nm])}
        print(f"    {arm:24s} thr={thr:.2f} test_micro={results[arm]['micro_f1']:.4f}"
              f"  | @sizematch thr={sthr:.3f} "
              f"test_micro={results[nm]['micro_f1']:.4f}  ({time.time()-t0:.0f}s)")

    # --- neural ------------------------------------------------------------
    test_pred["neural_rxgen"] = Pn["test"]
    results["neural_rxgen"] = {
        "threshold": None,
        "val_micro_f1": micro_f1_from_counts(*row_counts(Y["val"], Pn["val"])),
        **set_metrics(Y["test"], Pn["test"])}
    print(f"    {'neural_rxgen':24s} test_micro="
          f"{results['neural_rxgen']['micro_f1']:.4f}")

    # --- uncertainty -------------------------------------------------------
    counts = {k_: row_counts(Y["test"], P) for k_, P in test_pred.items()}
    ref = counts["neural_rxgen"]
    for k_, r in results.items():
        tp, np_, ng = counts[k_]
        lo, hi = bootstrap_micro(tp, np_, ng, args.bootstrap, SEED)
        r["micro_f1_lo95"], r["micro_f1_hi95"] = lo, hi
        if k_ != "neural_rxgen":
            r.update({f"vs_neural_{a}": b for a, b in
                      paired_bootstrap(ref, counts[k_], args.bootstrap, SEED).items()})

    return {
        "n_labels": len(labels),
        "n_rows": {s: int(len(idx[s])) for s in ("train", "val", "test")},
        "neural_val_mean_set_size": neural_val_size,
        "gold_mean_size_test": float(Y["test"].sum(1).mean()),
        "empty_gold_rate_test": float((Y["test"].sum(1) == 0).mean()),
        "results": results,
    }


PROTOCOLS = {
    "full89": {"min_label_freq": 0, "require_text": False, "require_label": False},
    "restricted47": {"min_label_freq": 10, "require_text": True, "require_label": True},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(MODELS / "rxgen_full_patient_seed0.pt"))
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()

    t0 = time.time()
    enc, orders, V = load_tables()
    o = encounter_categories(enc, orders, V)
    feat = pd.read_parquet(PROCESSED / "features.parquet")
    feat = feat.set_index("prescription_id").reindex(
        enc["prescription_id"].astype(int)).reset_index()
    FF = json.loads((PROCESSED / "feature_families.json").read_text())

    print("generating neural predictions ...", flush=True)
    neural = neural_category_sets(Path(args.ckpt), args.batch_size,
                                  use_cache=not args.no_cache)
    ctx = {"enc": enc, "orders_cat": o, "feat": feat, "families": FF,
           "neural": neural}

    out = {"checkpoint": Path(args.ckpt).name, "bootstrap": args.bootstrap,
           "protocols": {}}
    for p in args.protocols:
        out["protocols"][p] = run_protocol(p, PROTOCOLS[p], ctx, args)

    rows = []
    for p, blob in out["protocols"].items():
        for sysname, r in blob["results"].items():
            rows.append({"protocol": p, "system": sysname, **r})
    df = pd.DataFrame(rows)
    cols = ["protocol", "system", "threshold", "val_micro_f1", "micro_f1",
            "micro_f1_lo95", "micro_f1_hi95", "macro_f1", "jaccard",
            "exact_match", "micro_precision", "micro_recall", "mean_pred_size",
            "mean_gold_size", "empty_pred_rate", "vs_neural_delta_mean",
            "vs_neural_delta_lo95", "vs_neural_delta_hi95", "vs_neural_p_better"]
    df = df.reindex(columns=cols)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "head_to_head.json").write_text(json.dumps(out, indent=2, default=float))
    df.to_csv(outdir / "head_to_head.csv", index=False)

    print("\n" + "=" * 118)
    print("MATCHED-PROTOCOL HEAD-TO-HEAD (drug-class sets; identical rows, labels, metric)")
    print("=" * 118)
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(df.round(4).to_string(index=False))
    print(f"\nwrote {outdir/'head_to_head.csv'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
