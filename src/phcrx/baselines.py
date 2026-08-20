"""Non-neural reference points for the prescription-generation task.

These exist to calibrate the deep models, not to compete with them. Without
them a micro-F1 of ~0.20 on a 719-way label space is uninterpretable: the
reader cannot tell whether the network learned clinical structure or merely
memorised the prescribing frequency prior.

  1. GlobalPrior     - always emit the k most frequent training drugs.
  2. PrescriberPrior - emit the top-k drugs of the attending prescriber
                       (67 prescribers; a strong confounder worth quantifying).
  3. TfidfKNN        - retrieve the nearest training encounter by symptom text
                       and copy its prescription.

Run:  python -m src.phcrx.baselines
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from .config import RESULTS, DataConfig
from .data import RxCorpus, RxDataset
from . import metrics as M


def _gold_sets(corpus: RxCorpus, split: str):
    ds = RxDataset(corpus, split)
    rows, gold = [], []
    for _, r in ds.rows.iterrows():
        pid = int(r["prescription_id"])
        g = corpus.rx_orders.get(pid)
        ids = ([corpus.drug_vocab.get(int(d), 3)
                for d in g["drug_id"].dropna().astype(int)][: corpus.cfg.max_rx_len]
               if g is not None else [])
        gold.append(ids)
        rows.append(r)
    return rows, gold


def global_prior(corpus: RxCorpus, k: int):
    _, gtr = _gold_sets(corpus, "train")
    cnt = Counter(d for g in gtr for d in g)
    top = [d for d, _ in cnt.most_common(k)]
    _, gte = _gold_sets(corpus, "test")
    return [list(top) for _ in gte], gte


def prescriber_prior(corpus: RxCorpus, k: int):
    rtr, gtr = _gold_sets(corpus, "train")
    per = defaultdict(Counter)
    glob = Counter()
    for r, g in zip(rtr, gtr):
        per[r["prescriber_id"]].update(g)
        glob.update(g)
    gtop = [d for d, _ in glob.most_common(k)]
    rte, gte = _gold_sets(corpus, "test")
    pred = []
    for r in rte:
        c = per.get(r["prescriber_id"])
        pred.append([d for d, _ in c.most_common(k)] if c else list(gtop))
    return pred, gte


def tfidf_knn(corpus: RxCorpus, n_neighbors: int = 5, min_votes: int = 2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    rtr, gtr = _gold_sets(corpus, "train")
    rte, gte = _gold_sets(corpus, "test")
    txt_tr = [str(r["symptom_text"] or "") for r in rtr]
    txt_te = [str(r["symptom_text"] or "") for r in rte]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                          max_features=50000)
    Xtr = vec.fit_transform(txt_tr)
    Xte = vec.transform(txt_te)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, Xtr.shape[0]),
                          metric="cosine").fit(Xtr)
    _, idx = nn.kneighbors(Xte)

    pred = []
    for nbrs in idx:
        votes = Counter(d for j in nbrs for d in set(gtr[j]))
        pred.append([d for d, c in votes.items() if c >= min_votes])
    return pred, gte


def _score(name: str, pred, gold, corpus: RxCorpus) -> dict:
    res = M.set_metrics(pred, gold)
    res.update(M.empty_rx_metrics(pred, gold))
    v2c = corpus.vid2cat
    to_cat = lambda seqs: [[v2c.get(d, 0) for d in s if v2c.get(d, 0) != 0] for s in seqs]
    res["category_level"] = {f"cat_{k}": v
                             for k, v in M.set_metrics(to_cat(pred), to_cat(gold)).items()}
    strata = {corpus.drug_vocab.get(int(k), -1): v
              for k, v in corpus.vocab["drug_stratum"].items()}
    res["by_stratum"] = M.stratified_metrics(pred, gold, strata)
    res["macro_f1"] = M.macro_f1_by_label(pred, gold, len(corpus.drug_vocab))
    return {"model": name, "test": res}


def main() -> None:
    corpus = RxCorpus(DataConfig())
    runs = []
    for k in (1, 3, 5):
        p, g = global_prior(corpus, k)
        runs.append(_score(f"global_prior@{k}", p, g, corpus))
    for k in (1, 3, 5):
        p, g = prescriber_prior(corpus, k)
        runs.append(_score(f"prescriber_prior@{k}", p, g, corpus))
    p, g = tfidf_knn(corpus)
    runs.append(_score("tfidf_knn", p, g, corpus))

    out = RESULTS / "baselines.json"
    out.write_text(json.dumps(runs, indent=2, default=float))

    hdr = f"{'baseline':22s} {'microF1':>8s} {'jaccard':>8s} {'macroF1':>8s} {'exact':>8s} {'catF1':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        t = r["test"]
        print(f"{r['model']:22s} {t['micro_f1']:8.4f} {t['jaccard']:8.4f} "
              f"{t['macro_f1']:8.4f} {t['exact_match']:8.4f} "
              f"{t['category_level']['cat_micro_f1']:8.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
