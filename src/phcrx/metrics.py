"""Evaluation metrics for prescription generation.

Set-level metrics dominate because a prescription is fundamentally a *set* of
drug orders; sequence metrics are reported alongside for the decoder. Metrics
are additionally stratified by drug frequency band (head/mid/tail), since a
719-way label space over 11k prescriptions is heavily long-tailed and a single
micro-average would hide total failure on the tail.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import PAD, BOS, EOS


def _sets(pred: list[list[int]], gold: list[list[int]]):
    special = {PAD, BOS, EOS}
    P = [set(p) - special for p in pred]
    G = [set(g) - special for g in gold]
    return P, G


def set_metrics(pred: list[list[int]], gold: list[list[int]]) -> dict[str, float]:
    P, G = _sets(pred, gold)
    jac, f1s, exact = [], [], []
    tp = fp = fn = 0
    for p, g in zip(P, G):
        inter, union = len(p & g), len(p | g)
        jac.append(1.0 if union == 0 else inter / union)
        exact.append(float(p == g))
        prec = inter / len(p) if p else (1.0 if not g else 0.0)
        rec = inter / len(g) if g else (1.0 if not p else 0.0)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        tp += inter
        fp += len(p - g)
        fn += len(g - p)
    micro_p = tp / (tp + fp) if tp + fp else 0.0
    micro_r = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 0.0 if micro_p + micro_r == 0 else 2 * micro_p * micro_r / (micro_p + micro_r)
    return {
        "jaccard": float(np.mean(jac)),
        "sample_f1": float(np.mean(f1s)),
        "exact_match": float(np.mean(exact)),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "mean_pred_size": float(np.mean([len(p) for p in P])),
        "mean_gold_size": float(np.mean([len(g) for g in G])),
    }


def macro_f1_by_label(pred, gold, n_labels: int) -> float:
    P, G = _sets(pred, gold)
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for p, g in zip(P, G):
        for d in p & g: tp[d] += 1
        for d in p - g: fp[d] += 1
        for d in g - p: fn[d] += 1
    f1s = []
    for d in set(list(tp) + list(fp) + list(fn)):
        pr = tp[d] / (tp[d] + fp[d]) if tp[d] + fp[d] else 0.0
        rc = tp[d] / (tp[d] + fn[d]) if tp[d] + fn[d] else 0.0
        f1s.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return float(np.mean(f1s)) if f1s else 0.0


def stratified_metrics(pred, gold, stratum_of: dict[int, str]) -> dict[str, dict]:
    """Recall/precision computed within each frequency band."""
    P, G = _sets(pred, gold)
    agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for p, g in zip(P, G):
        for d in p & g: agg[stratum_of.get(d, "unseen")]["tp"] += 1
        for d in p - g: agg[stratum_of.get(d, "unseen")]["fp"] += 1
        for d in g - p: agg[stratum_of.get(d, "unseen")]["fn"] += 1
    out = {}
    for band, c in agg.items():
        pr = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        rc = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        out[band] = {
            "precision": pr, "recall": rc,
            "f1": 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc),
            "support": c["tp"] + c["fn"],
        }
    return out


def ranking_metrics(scores: np.ndarray, gold: list[list[int]], ks=(1, 3, 5, 10)) -> dict:
    """Precision@k / Recall@k from a per-drug score matrix (N, V)."""
    special = {PAD, BOS, EOS}
    out = {}
    order = np.argsort(-scores, axis=1)
    for k in ks:
        ps, rs = [], []
        for i, g in enumerate(gold):
            gs = set(g) - special
            if not gs:
                continue
            topk = [d for d in order[i] if d not in special][:k]
            hit = len(gs & set(topk))
            ps.append(hit / k)
            rs.append(hit / len(gs))
        out[f"precision@{k}"] = float(np.mean(ps)) if ps else 0.0
        out[f"recall@{k}"] = float(np.mean(rs)) if rs else 0.0
    return out


def empty_rx_metrics(pred, gold) -> dict[str, float]:
    """Can the model correctly withhold pharmacotherapy? (21.5% of the corpus)"""
    P, G = _sets(pred, gold)
    pe = np.array([len(p) == 0 for p in P])
    ge = np.array([len(g) == 0 for g in G])
    tp = int((pe & ge).sum()); fp = int((pe & ~ge).sum()); fn = int((~pe & ge).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {
        "empty_precision": pr, "empty_recall": rc,
        "empty_f1": 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc),
        "empty_rate_pred": float(pe.mean()), "empty_rate_gold": float(ge.mean()),
    }


def multilabel_metrics(prob: np.ndarray, gold: np.ndarray, thr: float = 0.5) -> dict:
    pred = prob >= thr
    tp = float((pred & (gold > 0.5)).sum())
    fp = float((pred & (gold < 0.5)).sum())
    fn = float((~pred & (gold > 0.5)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {
        "micro_precision": pr, "micro_recall": rc,
        "micro_f1": 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc),
    }


def expected_calibration_error(prob: np.ndarray, label: np.ndarray, n_bins: int = 15) -> float:
    prob = prob.ravel(); label = label.ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(prob)
    for i in range(n_bins):
        m = (prob > edges[i]) & (prob <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(label[m].mean() - prob[m].mean())
    return float(ece)


def attribute_accuracy(pred_attrs, gold_attrs, pred_drugs, gold_drugs) -> dict:
    """Attribute accuracy over positions where the drug itself was correct.

    Scoring attributes on wrongly-predicted drugs would be meaningless, so we
    align on the matched drug and compare its structured fields.
    """
    out = {}
    for k in pred_attrs:
        correct = total = 0
        for i, (pd_, gd_) in enumerate(zip(pred_drugs, gold_drugs)):
            gold_pos = {d: j for j, d in enumerate(gd_)}
            for j, d in enumerate(pd_):
                if d in gold_pos and j < len(pred_attrs[k][i]):
                    gj = gold_pos[d]
                    if gj < len(gold_attrs[k][i]):
                        total += 1
                        correct += int(pred_attrs[k][i][j] == gold_attrs[k][i][gj])
        out[f"attr_{k}_acc"] = correct / total if total else 0.0
        out[f"attr_{k}_n"] = total
    return out
