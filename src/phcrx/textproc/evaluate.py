"""Ablation over text representations, on a fast proxy for the real task.

Retraining the sequence model for every representation would cost hours per
arm and confound the text change with decoder capacity. Instead every arm
feeds the *same* classifier -- one-vs-rest L2 logistic regression -- on the
*same* rows and the *same* labels, so the only thing that varies is how the
symptom note is represented.

Proxy task: predict the set of pharmacological classes prescribed at the
encounter (`drug2cat` from rxgen_vocab.json). Class, not brand: brand choice
inside a class is formulary, not clinical, so brand-level scoring would
mostly measure procurement.

Protocol, identical for every arm:
  * vectoriser, normaliser, typo corrector and concept vocabulary are fitted
    on TRAIN ONLY;
  * one global decision threshold is tuned on VAL by micro-F1;
  * the number reported is TEST micro-F1 / macro-F1 at that threshold;
  * a paired bootstrap over test rows gives the CI on the *difference* from a
    reference arm, because the arms are correlated and an unpaired CI would
    overstate the uncertainty of the comparison. The same 1,000 resample
    index vectors are reused for every arm, so the comparisons are mutually
    consistent as well as internally paired.

Four metrics are bootstrapped, not one:
  micro_f1          the headline; dominated by the frequent classes
  macro_f1          unweighted mean over all classes -- moves when the rare
                    classes move, which is exactly where the production model
                    is weakest (tail recall 0.006)
  macro_f1_present  macro over classes with at least one positive in the
                    resample, so the CI is not damped by classes that a
                    resample happens to drop entirely
  tail_macro_f1     macro over the half of the classes with the fewest
                    training positives

    python -m src.phcrx.textproc.evaluate
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..config import PROCESSED, RESULTS
from .normalize import Normalizer
from .segment import segment

OUT = RESULTS / "textproc"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 0

BASE = "a. raw text (current tokenisation)"
SPACED = "b. + glossary expansion (spaced)"
FULL = "c. + typo correction (spaced)"
SEG = "d. + complaint segmentation (spaced)"

# The glossary surface form, crossed with the rest of the stack. Both levels
# are run for every form rather than picking a form on VAL and running only
# that one: the five forms are within 0.003 val micro-F1 of each other, which
# is far inside the noise, so a VAL pick would be a coin toss reported as a
# decision.
GLOSS_ARMS: dict[str, tuple[str, str]] = {
    # mode:                (glossary-only arm,            full-stack arm)
    "spaced": (SPACED, FULL),
    "joined": ("b3. glossary as joined token",
               "c3. joined glossary + orthography + typo"),
    "augmented": ("b4. glossary as abbrev + joined token",
                  "c4. abbrev+joined glossary + orthography + typo"),
    "canonical": ("b5. glossary as canonical concept token",
                  "c5. canonical glossary + orthography + typo"),
    "canonical_augmented": ("b6. glossary as surface + canonical token",
                            "c6. surface+canonical glossary + orthography + typo"),
}
GLOSS_ONLY = {v[0]: k for k, v in GLOSS_ARMS.items()}
GLOSS_FULL = {v[1]: k for k, v in GLOSS_ARMS.items()}

# Bigrams are joined with '|' rather than '_' so that a joined glossary token
# ('diabetes_mellitus') can never collide with the bigram of the same words
# written out. Arms a-d are unaffected: the production tokeniser never emits
# an underscore, so no collision existed for them under either separator.
BIGRAM_SEP = "|"

METRICS = ("micro_f1", "macro_f1", "macro_f1_present", "tail_macro_f1")


# --- data ------------------------------------------------------------------

def load_task(min_label_freq: int = 10):
    """Encounters with a symptom note, labelled by prescribed drug classes."""
    enc = pd.read_parquet(PROCESSED / "rxgen_encounters.parquet")
    enc["symptom_text"] = enc["symptom_text"].fillna("").astype(str)
    orders = pd.read_parquet(PROCESSED / "rxgen_orders.parquet")
    V = json.loads((PROCESSED / "rxgen_vocab.json").read_text(encoding="utf-8"))
    drug2cat = {int(k): int(v) for k, v in V["drug2cat"].items()}
    cat_names = V["category_names"]
    inv_cat = {v: k for k, v in V["category"].items()}     # index -> cat id str

    o = orders.dropna(subset=["drug_id", "prescription_id"]).copy()
    o["cat"] = o["drug_id"].astype(int).map(drug2cat).fillna(0).astype(int)
    o = o[o["cat"] != 0]

    pid2row = {int(p): i for i, p in enumerate(enc["prescription_id"])}
    tr_pids = set(enc.loc[enc.split == "train", "prescription_id"].dropna().astype(int))
    freq = o.loc[o["prescription_id"].astype(int).isin(tr_pids), "cat"].value_counts()
    keep = sorted(int(c) for c, n in freq.items() if n >= min_label_freq)
    cat2col = {c: j for j, c in enumerate(keep)}

    rows, cols = [], []
    for pid, cat in zip(o["prescription_id"].astype(int), o["cat"]):
        r, j = pid2row.get(pid), cat2col.get(int(cat))
        if r is not None and j is not None:
            rows.append(r)
            cols.append(j)
    Y = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                      shape=(len(enc), len(keep)))
    Y.data[:] = 1
    Y = (Y > 0).astype(np.int8)

    has_text = enc["symptom_text"].str.strip() != ""
    has_label = np.asarray(Y.sum(1)).ravel() > 0
    mask = (has_text & has_label).to_numpy()
    label_names = [cat_names.get(inv_cat.get(c, ""), f"cat{c}") for c in keep]
    return enc, Y, mask, label_names


# --- feature builders ------------------------------------------------------

def flat_features(texts, nz: Normalizer) -> list[list[str]]:
    """Unigrams + bigrams over the whole note (arms a-c)."""
    out = []
    for t in texts:
        toks = nz.tokenize(nz(t))
        out.append(toks + [f"{a}{BIGRAM_SEP}{b}" for a, b in zip(toks, toks[1:])])
    return out


def segmented_features(texts, nz: Normalizer) -> list[list[str]]:
    """Unigrams over the note + bigrams *within* complaint spans (arm d).

    Differs from `flat_features` by exactly one thing: a bigram may not cross
    a complaint boundary. Plus two structural markers that only exist once the
    note has been segmented.
    """
    out = []
    for t in texts:
        s = nz(t)
        toks = nz.tokenize(s)
        spans = segment(s)
        feats = list(toks)
        for sp_ in spans:
            st = nz.tokenize(sp_)
            feats += [f"{a}{BIGRAM_SEP}{b}" for a, b in zip(st, st[1:])]
        feats.append(f"__nspans{min(len(spans), 4)}__")
        if not spans and s.strip():
            feats.append("__nocomplaint__")
        out.append(feats)
    return out


def tfidf(docs_tr, docs_va, docs_te, min_df: int = 2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer=lambda d: d, min_df=min_df, sublinear_tf=True)
    return vec.fit_transform(docs_tr), vec.transform(docs_va), vec.transform(docs_te), vec


# --- model / metrics -------------------------------------------------------

def fit_predict(Xtr, Ytr, Xva, Xte, C: float = 4.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    # n_jobs=1: joblib memory-maps large arrays to worker processes read-only,
    # and liblinear needs a writable buffer -> "WRITEBACKIFCOPY base is
    # read-only". Serial fitting is a few seconds here and avoids the issue.
    clf = OneVsRestClassifier(
        LogisticRegression(C=C, max_iter=3000, solver="liblinear"), n_jobs=1)
    Ytr = np.ascontiguousarray(Ytr)
    clf.fit(Xtr, Ytr)
    return clf.predict_proba(Xva), clf.predict_proba(Xte)


def micro_f1(Y, P) -> float:
    tp = float((Y & P).sum())
    return 2 * tp / max(float(Y.sum() + P.sum()), 1e-9)


def per_class_f1(Y, P) -> np.ndarray:
    tp = (Y & P).sum(0).astype(float)
    return 2 * tp / np.maximum(Y.sum(0) + P.sum(0), 1e-9)


def macro_f1(Y, P) -> float:
    return float(per_class_f1(Y, P).mean())


def macro_f1_present(Y, P) -> float:
    """Macro over classes with a positive in this sample.

    A bootstrap resample can drop a rare class entirely; scoring it 0 for both
    arms shrinks the measured difference toward zero rather than widening it,
    so this variant is reported alongside as the less conservative reading.
    """
    keep = Y.sum(0) > 0
    if not keep.any():
        return 0.0
    return float(per_class_f1(Y[:, keep], P[:, keep]).mean())


def tune_threshold(Yva, Pva) -> float:
    grid = np.arange(0.05, 0.71, 0.01)
    return float(max(grid, key=lambda t: micro_f1(Yva, (Pva >= t).astype(np.int8))))


def all_metrics(Y, P, tail) -> dict[str, float]:
    return {"micro_f1": micro_f1(Y, P),
            "macro_f1": macro_f1(Y, P),
            "macro_f1_present": macro_f1_present(Y, P),
            "tail_macro_f1": macro_f1(Y[:, tail], P[:, tail])}


def evaluate_arm(Xtr, Ytr, Xva, Yva, Xte, Yte, tail, C: float = 4.0):
    Pva, Pte = fit_predict(Xtr, Ytr, Xva, Xte, C=C)
    thr = tune_threshold(Yva, Pva)
    Bte = (Pte >= thr).astype(np.int8)
    m = all_metrics(Yte, Bte, tail)
    return {"threshold": round(thr, 3),
            "val_micro_f1": round(micro_f1(Yva, (Pva >= thr).astype(np.int8)), 4),
            **{k: round(v, 4) for k, v in m.items()}}, Bte


def paired_bootstrap(Yte, B_ref, B_arm, tail, boot_idx) -> dict:
    """95% CI on the difference in each metric, resampling test encounters."""
    d = {k: np.empty(len(boot_idx)) for k in METRICS}
    for k, idx in enumerate(boot_idx):
        Y, A, R = Yte[idx], B_arm[idx], B_ref[idx]
        ma, mr = all_metrics(Y, A, tail), all_metrics(Y, R, tail)
        for name in METRICS:
            d[name][k] = ma[name] - mr[name]
    out = {}
    for name, v in d.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        short = {"micro_f1": "micro", "macro_f1": "macro",
                 "macro_f1_present": "macro_present",
                 "tail_macro_f1": "tail_macro"}[name]
        out[f"{short}_delta"] = round(float(v.mean()), 4)
        out[f"{short}_lo95"] = round(float(lo), 4)
        out[f"{short}_hi95"] = round(float(hi), 4)
        out[f"{short}_p_better"] = round(float((v > 0).mean()), 3)
    return out


# --- driver ----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-label-freq", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--no-sapbert", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    enc, Y, mask, label_names = load_task(args.min_label_freq)
    split = enc["split"].to_numpy()
    idx = {s: np.where(mask & (split == s))[0] for s in ("train", "val", "test")}
    text = enc["symptom_text"].to_numpy()
    Yd = np.asarray(Y.todense(), dtype=np.int8)
    Ytr, Yva, Yte = (Yd[idx[s]] for s in ("train", "val", "test"))

    # Tail = the half of the classes with the fewest training positives. Fixed
    # before any arm is fitted, so it cannot be chosen to flatter a result.
    train_support = Ytr.sum(0)
    order_sup = np.argsort(train_support, kind="stable")
    tail = np.sort(order_sup[: len(order_sup) // 2])
    head = np.setdiff1d(np.arange(Ytr.shape[1]), tail)

    print(f"labels={len(label_names)} (classes with >={args.min_label_freq} train orders)")
    print(f"rows: train={len(idx['train'])} val={len(idx['val'])} test={len(idx['test'])}")
    print(f"mean classes/encounter: train={Ytr.sum(1).mean():.2f} test={Yte.sum(1).mean():.2f}")
    print(f"tail = {len(tail)} classes, train support {int(train_support[tail].min())}"
          f"-{int(train_support[tail].max())}, "
          f"{float(Yte[:, tail].sum() / Yte.sum()):.1%} of test positives")

    train_texts_all = enc.loc[enc.split == "train", "symptom_text"].tolist()

    # Normalisers, each fitted on TRAIN text only.
    nz_raw = Normalizer(glossary=False, orthography=False, typo=False)
    nz_orth = Normalizer(glossary=True, orthography=True, typo=False)
    nz_gloss, nz_stack = {}, {}
    for mode, (b_arm, c_arm) in GLOSS_ARMS.items():
        nz_gloss[mode] = Normalizer(glossary=True, orthography=False, typo=False,
                                    glossary_mode=mode)
        nz_stack[mode] = Normalizer(glossary=True, orthography=True, typo=True,
                                    glossary_mode=mode).fit(train_texts_all)

    def blocks(fn, nz):
        return [fn(text[idx[s]], nz) for s in ("train", "val", "test")]

    def add(name, fn, nz, store):
        dtr, dva, dte = blocks(fn, nz)
        Xtr, Xva, Xte, _ = tfidf(dtr, dva, dte)
        store[name] = (Xtr, Xva, Xte)
        print(f"  {name:50s} features={Xtr.shape[1]}")

    feature_sets: dict[str, tuple] = {}
    add(BASE, flat_features, nz_raw, feature_sets)
    for mode, (b_arm, _) in GLOSS_ARMS.items():
        add(b_arm, flat_features, nz_gloss[mode], feature_sets)
    add("b2. + orthography (spaced)", flat_features, nz_orth, feature_sets)
    for mode, (_, c_arm) in GLOSS_ARMS.items():
        add(c_arm, flat_features, nz_stack[mode], feature_sets)
    add(SEG, segmented_features, nz_stack["spaced"], feature_sets)

    boot_idx = np.random.default_rng(SEED).integers(
        0, Yte.shape[0], (args.bootstrap, Yte.shape[0]))
    results, preds = {}, {}

    def run(name):
        Xtr, Xva, Xte = feature_sets[name]
        r, B = evaluate_arm(Xtr, Ytr, Xva, Yva, Xte, Yte, tail)
        results[name], preds[name] = r, B
        print(f"    {name:50s} val={r['val_micro_f1']:.4f} test micro={r['micro_f1']:.4f} "
              f"macro={r['macro_f1']:.4f} tail={r['tail_macro_f1']:.4f}")
        return r

    for name in list(feature_sets):
        run(name)

    # Segmentation is orthogonal to the glossary form, but check it once on
    # top of whichever full stack VAL prefers, so the recommendation for it
    # does not rest on the spaced stack alone.
    best_stack = max(GLOSS_FULL, key=lambda k: results[k]["val_micro_f1"])
    print(f"\n  best full stack on VAL: {best_stack}")
    if best_stack != FULL:
        seg2 = f"d2. + complaint segmentation ({GLOSS_FULL[best_stack]})"
        add(seg2, segmented_features, nz_stack[GLOSS_FULL[best_stack]], feature_sets)
        run(seg2)
    else:
        seg2 = None

    text_arms = [k for k in feature_sets]

    # --- concept vectors ---------------------------------------------------
    if not args.no_sapbert:
        from ..nlp.icd_index import SapBertEncoder
        from .concepts import ConceptVocab
        encoder = SapBertEncoder()
        try:
            cv = ConceptVocab.load()
        except Exception:
            cv = ConceptVocab.induce(nz_stack["spaced"].transform(train_texts_all), encoder)
            cv.save()
        norm_all = nz_stack["spaced"].transform(list(text))
        M, flag = cv.assign(norm_all, encoder)
        Mfull = np.hstack([M, flag[:, None]]).astype(np.float32)
        feature_sets["e. concept vectors only"] = tuple(
            sp.csr_matrix(Mfull[idx[s]]) for s in ("train", "val", "test"))
        print(f"  e. concept vectors only                      features={Mfull.shape[1]} "
              f"(concepts={len(cv)})")

        # SapBERT sentence embeddings of the glossary-expanded note, matching
        # how the encoder is used for ICD retrieval. Spaced expansion, always:
        # SapBERT is a language model and 'diabetes_mellitus' is not a word.
        gexp = [nz_gloss["spaced"](t) for t in text]
        E = encoder.encode(gexp)
        feature_sets["f. SapBERT embeddings only"] = tuple(
            sp.csr_matrix(E[idx[s]]) for s in ("train", "val", "test"))
        print(f"  f. SapBERT embeddings only                   features={E.shape[1]}")
        run("e. concept vectors only")
        run("f. SapBERT embeddings only")

    # --- combinations: chosen on VAL, reported on TEST ---------------------
    best_text = max(text_arms, key=lambda k: results[k]["val_micro_f1"])
    best_combo = None
    if not args.no_sapbert:
        print(f"\n  best text arm on VAL: {best_text}")
        combos = {
            "g1. best text + concepts": [best_text, "e. concept vectors only"],
            "g2. best text + SapBERT": [best_text, "f. SapBERT embeddings only"],
            "g3. best text + concepts + SapBERT":
                [best_text, "e. concept vectors only", "f. SapBERT embeddings only"],
        }
        for name, parts in combos.items():
            feature_sets[name] = tuple(
                sp.hstack([feature_sets[p][i] for p in parts]).tocsr() for i in range(3))
            run(name)
        best_combo = max(combos, key=lambda k: results[k]["val_micro_f1"])
        print(f"  best combination on VAL: {best_combo}")

    # --- prior baseline for context ---------------------------------------
    prior = Ytr.mean(0)
    order = np.argsort(-prior)
    best_k, best_f = 1, -1.0
    for k in range(1, 8):
        P = np.zeros_like(Yva)
        P[:, order[:k]] = 1
        f = micro_f1(Yva, P)
        if f > best_f:
            best_k, best_f = k, f
    P = np.zeros_like(Yte)
    P[:, order[:best_k]] = 1
    results[f"0. prior baseline (top-{best_k} classes always)"] = {
        "threshold": None, "val_micro_f1": round(best_f, 4),
        **{k: round(v, 4) for k, v in all_metrics(Yte, P, tail).items()}}

    # --- paired bootstrap vs the current tokenisation ----------------------
    print(f"\n  bootstrapping {args.bootstrap} resamples x {len(preds)} arms …")
    boots = {name: paired_bootstrap(Yte, preds[BASE], B, tail, boot_idx)
             for name, B in preds.items() if name != BASE}

    # Comparisons that are not against the baseline, because the question they
    # answer is 'did changing this one thing help', not 'is this better than
    # doing nothing'.
    # Every alternative glossary surface form against the spaced expansion it
    # is meant to replace -- this is the direct test of the fragmentation
    # hypothesis, and it is not the same question as 'better than raw text'.
    pairs = [(b, SPACED) for b, _ in GLOSS_ARMS.values() if b != SPACED]
    pairs += [(c, FULL) for _, c in GLOSS_ARMS.values() if c != FULL]
    pairs += [(SEG, FULL)]
    if seg2:
        pairs += [(seg2, best_stack)]
    if best_combo:
        pairs += [(best_combo, best_text),
                  ("g3. best text + concepts + SapBERT", "g2. best text + SapBERT")]
    contrasts = {}
    for arm, ref in pairs:
        if arm in preds and ref in preds and arm != ref:
            contrasts[f"{arm}  vs  {ref}"] = paired_bootstrap(
                Yte, preds[ref], preds[arm], tail, boot_idx)

    # --- per-class detail for the head/tail story --------------------------
    per_class = {}
    for name in preds:
        f = per_class_f1(Yte, preds[name])
        per_class[name] = {
            "head_mean_f1": round(float(f[head].mean()), 4),
            "tail_mean_f1": round(float(f[tail].mean()), 4),
            "n_tail_classes_nonzero": int((f[tail] > 0).sum()),
            "by_class": {label_names[j]: round(float(f[j]), 4)
                         for j in range(len(label_names))},
        }

    table = []
    for name in sorted(results):
        row = {"arm": name, **results[name]}
        row.update(boots.get(name, {}))
        table.append(row)
    df = pd.DataFrame(table)

    cols = ["arm", "val_micro_f1", "micro_f1", "macro_f1", "tail_macro_f1",
            "micro_delta", "micro_lo95", "micro_hi95",
            "macro_delta", "macro_lo95", "macro_hi95"]
    print("\n" + "=" * 118)
    print("ABLATION (proxy task: multi-label drug-class prediction, "
          "identical rows / labels / classifier)")
    print("=" * 118)
    print(df.reindex(columns=cols).to_string(index=False))

    print("\n" + "=" * 118)
    print("PAIRED CONTRASTS (not vs baseline)")
    print("=" * 118)
    for k, v in contrasts.items():
        print(f"  {k}")
        print(f"      micro {v['micro_delta']:+.4f} [{v['micro_lo95']:+.4f}, "
              f"{v['micro_hi95']:+.4f}]   macro {v['macro_delta']:+.4f} "
              f"[{v['macro_lo95']:+.4f}, {v['macro_hi95']:+.4f}]")

    (OUT / "ablation.json").write_text(json.dumps({
        "n_labels": len(label_names),
        "n_train": int(len(idx["train"])), "n_val": int(len(idx["val"])),
        "n_test": int(len(idx["test"])),
        "label_names": label_names,
        "train_support": {label_names[j]: int(train_support[j])
                          for j in range(len(label_names))},
        "tail_classes": [label_names[j] for j in tail],
        "best_full_stack_on_val": best_stack,
        "best_text_arm": best_text, "best_combination": best_combo,
        "bootstrap_n": int(args.bootstrap),
        "results": results, "bootstrap_vs_baseline": boots,
        "paired_contrasts": contrasts,
        "per_class": per_class,
    }, indent=2), encoding="utf-8")
    df.to_csv(OUT / "ablation.csv", index=False)
    print(f"\nwrote {OUT/'ablation.csv'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
