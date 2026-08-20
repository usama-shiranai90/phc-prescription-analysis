"""Held-out test evaluation of the fitted recommender.

    python -m src.phcrx.recommend.evaluate                 # all components
    python -m src.phcrx.recommend.evaluate --matched       # + neural head-to-head

Nothing is fitted here. `pipeline.py` cached the validation and test
probabilities of the selected model; this module opens the test labels for the
first time and scores them.

Every component is scored against its own floor, because "better than nothing"
means something different for each:

    multi-label   always-empty (the modal prescription *is* the empty set) and
                  a frequency prior at the top-k classes, k tuned on VAL
    binary        the constant train prevalence -- AUROC 0.5 by construction,
                  average precision = the positive rate
    count         the constant train median and the constant train mean
    multiclass    the majority class

Uncertainty is a paired bootstrap over test encounters: one fixed set of
resample indices is shared by every system inside a component, so a difference
interval is computed on the same rows for both systems. Set metrics are
recomputed exactly under each resample from per-row-per-label counts rather
than approximated.

The neural comparison (`--matched`) cannot be run in the normalised 46-class
space, because the published PHC-RxGen numbers live in the legacy 89-way
`rx_category` space. It therefore re-scores a model fitted in *that* space
against the cached neural predictions, on identical rows and labels. Comparing
a 46-class micro-F1 to an 89-class micro-F1 would not be a comparison.
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from . import OUT, SEED
from .corpus import EXCLUDED
from .metrics import (auroc, avg_precision, brier, ci, delta_stats, ece,
                      ece_macro, precision_recall_at_k, Resampler, set_metrics,
                      tune_threshold)

warnings.filterwarnings("ignore")

PRED_DIR = OUT / "preds"
NEURAL_CACHE = (OUT.parents[2] / "src" / "phcrx" / "bench"
                / "neural_preds_cache.json")

# Published PHC-RxGen numbers, quoted verbatim from docs/model_comparison.md.
# They are a *reference row*, not a re-measurement: the encoder-decoder is not
# re-run here.
NEURAL_PUBLISHED = {
    "full89": {"micro_f1": 0.3315, "micro_f1_lo95": 0.318,
               "micro_f1_hi95": 0.344, "macro_f1": 0.1464, "jaccard": 0.3561,
               "exact_match": 0.2335, "mean_pred_size": 1.33},
    "restricted47": {"micro_f1": 0.3563, "micro_f1_lo95": 0.342,
                     "micro_f1_hi95": 0.370, "macro_f1": 0.2256},
}


# ---------------------------------------------------------------------------
def _load(tag: str):
    blob = json.loads((OUT / f"pipeline{tag}.json").read_text())
    preds = {}
    for name in blob["components"]:
        f = PRED_DIR / f"{name}{tag}.npz"
        if f.exists():
            preds[name] = dict(np.load(f, allow_pickle=False))
    return blob, preds


def _prior_topk(Ytr, Yva, kmax: int = 10):
    order = np.argsort(-Ytr.mean(0))
    best_k, bf = 1, -1.0
    for k in range(1, kmax + 1):
        P = np.zeros_like(Yva, dtype=bool)
        P[:, order[:k]] = True
        tp = float((Yva.astype(bool) & P).sum())
        f = 2 * tp / max(float(P.sum() + Yva.sum()), 1e-9)
        if f > bf:
            best_k, bf = k, f
    return best_k, order[:best_k]


def _fmt(v, n=4):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v:.{n}f}"


# ---------------------------------------------------------------------------
# per-kind evaluation
# ---------------------------------------------------------------------------
def eval_multilabel(name, meta, d, n_boot, ranked=True):
    Ytr, Yva, Yte = d["y_train"].astype(bool), d["y_val"].astype(bool), \
        d["y_test"].astype(bool)
    tail = d["tail"]
    R = Resampler(len(Yte), n_boot, SEED)

    systems: dict[str, np.ndarray] = {}
    systems["always_empty"] = np.zeros_like(Yte)
    k, top = _prior_topk(Ytr, Yva)
    P = np.zeros_like(Yte)
    P[:, top] = True
    systems[f"prior_top{k}"] = P

    thr = meta["threshold"]
    systems["recommender"] = d["test_cal"] >= thr
    if "test_cal_dep" in d:
        systems["recommender_deployable"] = d["test_cal_dep"] >= meta["threshold_dep"]

    rows, dists = [], {}
    for sysname, B in systems.items():
        point = set_metrics(Yte, B, tail)
        dist = R.set_metrics(Yte, B, tail)
        dists[sysname] = dist
        row = {"component": name, "system": sysname}
        for m, v in point.items():
            row[m] = v
            if m in dist:
                lo, hi = ci(dist[m])
                row[f"{m}_lo95"], row[f"{m}_hi95"] = lo, hi
        rows.append(row)

    ref = dists["always_empty"]
    for r in rows:
        if r["system"] == "always_empty":
            continue
        for m in ("micro_f1", "macro_f1", "tail_macro_f1", "jaccard",
                  "exact_match"):
            s = delta_stats(dists[r["system"]][m], ref[m])
            r[f"d_empty_{m}"] = s["delta"]
            r[f"d_empty_{m}_lo95"], r[f"d_empty_{m}_hi95"] = \
                s["delta_lo95"], s["delta_hi95"]
            r[f"d_empty_{m}_p"] = s["p_better"]
    pk = next(s for s in systems if s.startswith("prior_top"))
    for r in rows:
        if r["system"] in ("always_empty", pk):
            continue
        for m in ("micro_f1", "macro_f1", "tail_macro_f1"):
            s = delta_stats(dists[r["system"]][m], dists[pk][m])
            r[f"d_prior_{m}"] = s["delta"]
            r[f"d_prior_{m}_lo95"], r[f"d_prior_{m}_hi95"] = \
                s["delta_lo95"], s["delta_hi95"]
            r[f"d_prior_{m}_p"] = s["p_better"]

    for r in rows:
        if r["system"] == "recommender" and ranked:
            r.update(precision_recall_at_k(Yte, d["test_cal"]))
    return rows


def eval_binary(name, meta, d, n_boot):
    yte = d["y_test"].ravel().astype(int)
    p_raw, p_cal = d["test"].ravel(), d["test_cal"].ravel()
    ytr = d["y_train"].ravel().astype(int)
    R = Resampler(len(yte), n_boot, SEED)

    rows = []
    prev = float(ytr.mean())
    rows.append({"component": name, "system": "train_prevalence",
                 "auroc": 0.5, "average_precision": float(yte.mean()),
                 "ece": ece(np.full_like(p_cal, prev), yte),
                 "brier": brier(np.full_like(p_cal, prev), yte)})
    d_au = R.curve_metric(yte, p_cal, lambda y, p: auroc(y, p))
    d_ap = R.curve_metric(yte, p_cal, lambda y, p: avg_precision(y, p))
    lo_au, hi_au = ci(d_au)
    lo_ap, hi_ap = ci(d_ap)
    thr = meta["threshold"]
    B = (p_cal >= thr).astype(int)
    tp = float(((B == 1) & (yte == 1)).sum())
    prec = tp / max(B.sum(), 1)
    rec = tp / max(yte.sum(), 1)
    rows.append({"component": name, "system": "recommender",
                 "auroc": auroc(yte, p_cal), "auroc_lo95": lo_au,
                 "auroc_hi95": hi_au, "average_precision": avg_precision(yte, p_cal),
                 "average_precision_lo95": lo_ap, "average_precision_hi95": hi_ap,
                 "ece": ece(p_cal, yte), "ece_uncalibrated": ece(p_raw, yte),
                 "brier": brier(p_cal, yte), "threshold": thr,
                 "accuracy": float((B == yte).mean()),
                 "precision": float(prec), "recall": float(rec),
                 "f1": float(2 * prec * rec / max(prec + rec, 1e-9))})
    if "test_cal_dep" in d:
        pd_ = d["test_cal_dep"].ravel()
        rows.append({"component": name, "system": "recommender_deployable",
                     "auroc": auroc(yte, pd_),
                     "average_precision": avg_precision(yte, pd_),
                     "ece": ece(pd_, yte), "brier": brier(pd_, yte)})
    return rows


def eval_count(name, meta, d, n_boot):
    yte, p = d["y_test"].ravel(), d["test"].ravel()
    ytr = d["y_train"].ravel()
    R = Resampler(len(yte), n_boot, SEED)
    rows = []
    for sysname, pred in (("train_median", np.full_like(p, np.median(ytr))),
                          ("train_mean", np.full_like(p, ytr.mean())),
                          ("recommender", p),
                          ("recommender_rounded", np.round(p))):
        err = np.abs(yte - pred)
        dist = R.row_mean(err)
        lo, hi = ci(dist)
        ss = float(((yte - pred) ** 2).sum())
        st = float(((yte - yte.mean()) ** 2).sum())
        rows.append({"component": name, "system": sysname,
                     "mae": float(err.mean()), "mae_lo95": lo, "mae_hi95": hi,
                     "rmse": float(np.sqrt((err ** 2).mean())),
                     "r2": 1 - ss / st})
    base = R.row_mean(np.abs(yte - np.median(ytr)))
    for r in rows:
        if r["system"].startswith("recommender"):
            pred = p if r["system"] == "recommender" else np.round(p)
            s = delta_stats(base, R.row_mean(np.abs(yte - pred)))  # lower is better
            r["d_median_mae"] = -s["delta"]
            r["d_median_mae_lo95"], r["d_median_mae_hi95"] = \
                -s["delta_hi95"], -s["delta_lo95"]
            r["d_median_mae_p"] = s["p_better"]
    if "test_dep" in d:
        pdp = d["test_dep"].ravel()
        e = np.abs(yte - pdp)
        lo, hi = ci(R.row_mean(e))
        rows.append({"component": name, "system": "recommender_deployable",
                     "mae": float(e.mean()), "mae_lo95": lo, "mae_hi95": hi})
    return rows


def eval_multiclass(name, meta, d, n_boot):
    from sklearn.metrics import f1_score
    yte = d["y_test"].ravel().astype(int)
    ytr = d["y_train"].ravel().astype(int)
    P_raw, P_cal = d["test"], d["test_cal"]
    K = P_cal.shape[1]
    R = Resampler(len(yte), n_boot, SEED)
    Yoh = np.eye(K, dtype=np.int8)[yte]
    maj = int(np.bincount(ytr, minlength=K).argmax())

    rows = []
    for sysname, pred, P in (("majority_class", np.full_like(yte, maj), None),
                             ("recommender", P_cal.argmax(1), P_cal)):
        acc = (pred == yte).astype(float)
        lo, hi = ci(R.row_mean(acc))
        f1m = R.curve_metric(
            yte, pred, lambda y, q: f1_score(y, q, average="macro"))
        flo, fhi = ci(f1m)
        r = {"component": name, "system": sysname,
             "accuracy": float(acc.mean()), "accuracy_lo95": lo,
             "accuracy_hi95": hi,
             "macro_f1": float(f1_score(yte, pred, average="macro")),
             "macro_f1_lo95": flo, "macro_f1_hi95": fhi}
        if P is not None:
            r["ece"] = ece(P, Yoh)
            r["ece_uncalibrated"] = ece(P_raw, Yoh)
            r["ece_macro"] = ece_macro(P, Yoh)
            r["brier"] = brier(P, Yoh)
            s = delta_stats(R.row_mean(acc),
                            R.row_mean((np.full_like(yte, maj) == yte).astype(float)))
            r["d_majority_accuracy"] = s["delta"]
            r["d_majority_accuracy_lo95"] = s["delta_lo95"]
            r["d_majority_accuracy_hi95"] = s["delta_hi95"]
            r["d_majority_accuracy_p"] = s["p_better"]
        rows.append(r)
    if "test_cal_dep" in d:
        pdp = d["test_cal_dep"].argmax(1)
        rows.append({"component": name, "system": "recommender_deployable",
                     "accuracy": float((pdp == yte).mean()),
                     "macro_f1": float(f1_score(yte, pdp, average="macro"))})
    return rows


# ---------------------------------------------------------------------------
# calibration report
# ---------------------------------------------------------------------------
def calibration_rows(blob, preds):
    out = []
    for name, meta in blob["components"].items():
        if name not in preds or meta["kind"] == "count":
            continue
        d = preds[name]
        Y = d["y_test"]
        if meta["kind"] == "multiclass":
            Y = np.eye(d["test"].shape[1], dtype=np.int8)[Y.ravel().astype(int)]
        Y = Y.reshape(d["test"].shape)
        out.append({
            "component": name, "method": meta["calibration"]["method"],
            "val_ece_raw": meta["calibration"]["val_ece_raw"],
            "val_ece_cal": meta["calibration"]["val_ece_cal"],
            "test_ece_raw": ece(d["test"], Y),
            "test_ece_cal": ece(d["test_cal"], Y),
            "test_ece_macro_raw": ece_macro(d["test"], Y),
            "test_ece_macro_cal": ece_macro(d["test_cal"], Y),
            "test_brier_raw": brier(d["test"], Y),
            "test_brier_cal": brier(d["test_cal"], Y),
            "cv_ece_none": meta["calibration"]["cv_ece"].get("none"),
            "cv_ece_sigmoid": meta["calibration"]["cv_ece"].get("sigmoid"),
            "cv_ece_isotonic": meta["calibration"]["cv_ece"].get("isotonic"),
        })
    return out


# ---------------------------------------------------------------------------
# neural head-to-head, legacy 89-category space
# ---------------------------------------------------------------------------
def matched_neural(n_boot: int):
    tag = "_cat89"
    try:
        blob, preds = _load(tag)
    except FileNotFoundError:
        print("  (no cat89 run found -- "
              "`python -m src.phcrx.recommend.pipeline --components drug_classes "
              "--label-space cat89 --tag _cat89` first)")
        return []
    if not NEURAL_CACHE.exists():
        print(f"  (no cached neural predictions at {NEURAL_CACHE})")
        return []

    from .corpus import load_frame
    df, _ = load_frame()
    meta = blob["components"]["drug_classes"]
    d = preds["drug_classes"]

    # Labels are stored as "<category id>|<name>", so the cached neural
    # category-id lists map straight onto columns.
    col_of = {int(str(lab).split("|", 1)[0]): j
              for j, lab in enumerate(meta["labels"])}

    cache = json.loads(NEURAL_CACHE.read_text())
    npred = {int(k): v for k, v in cache["preds"]["test"].items()}
    pids = df["prescription_id"].to_numpy()[d["idx_test"]]
    Yte = d["y_test"].astype(bool)
    Pn = np.zeros_like(Yte)
    hit = 0
    for i, pid in enumerate(pids):
        got = npred.get(int(pid))
        if got is None:
            continue
        hit += 1
        for c in got:
            j = col_of.get(int(c))
            if j is not None:
                Pn[i, j] = True
    print(f"  neural predictions matched for {hit}/{len(pids)} test encounters")

    tail = d["tail"]
    R = Resampler(len(Yte), n_boot, SEED)
    B = d["test_cal"] >= meta["threshold"]
    out = []
    dists = {}
    for sysname, Pm in (("neural_rxgen", Pn), ("recommender", B),
                        ("always_empty", np.zeros_like(Yte))):
        pt = set_metrics(Yte, Pm, tail)
        dists[sysname] = R.set_metrics(Yte, Pm, tail)
        row = {"protocol": "cat89_full", "system": sysname}
        for m, v in pt.items():
            row[m] = v
            if m in dists[sysname]:
                lo, hi = ci(dists[sysname][m])
                row[f"{m}_lo95"], row[f"{m}_hi95"] = lo, hi
        out.append(row)
    for r in out:
        if r["system"] == "neural_rxgen":
            continue
        for m in ("micro_f1", "macro_f1", "tail_macro_f1", "jaccard",
                  "exact_match"):
            s = delta_stats(dists[r["system"]][m], dists["neural_rxgen"][m])
            r[f"vs_neural_{m}"] = s["delta"]
            r[f"vs_neural_{m}_lo95"], r[f"vs_neural_{m}_hi95"] = \
                s["delta_lo95"], s["delta_hi95"]
            r[f"vs_neural_{m}_p"] = s["p_better"]
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--matched", action="store_true")
    args = ap.parse_args()

    blob, preds = _load(args.tag)
    handlers = {"multilabel": eval_multilabel, "binary": eval_binary,
                "count": eval_count, "multiclass": eval_multiclass}

    all_rows = []
    for name, meta in blob["components"].items():
        if name not in preds:
            continue
        print(f"[{name}] {meta['kind']}  feature_set={meta['feature_set']}  "
              f"labels={meta['n_labels']}  test rows={meta['rows']['test']}",
              flush=True)
        rows = handlers[meta["kind"]](name, meta, preds[name], args.bootstrap)
        for r in rows:
            r["feature_set"] = meta["feature_set"]
            r["best_deployable_set"] = meta["best_deployable"]
        all_rows += rows
        for r in rows:
            head = [f"{k}={_fmt(r.get(k))}" for k in
                    ("micro_f1", "macro_f1", "tail_macro_f1", "jaccard",
                     "exact_match", "auroc", "average_precision", "mae",
                     "accuracy") if r.get(k) is not None]
            print(f"    {r['system']:26s} " + "  ".join(head))
        print()

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / f"test_results{args.tag}.csv", index=False)

    cal = pd.DataFrame(calibration_rows(blob, preds))
    cal.to_csv(OUT / f"calibration{args.tag}.csv", index=False)
    print("=" * 100)
    print("CALIBRATION (ECE, 10 equal-width bins; calibrator fitted on VAL only)")
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    matched = []
    if args.matched:
        print("\n" + "=" * 100)
        print("MATCHED PROTOCOL vs PHC-RxGen (legacy 89-category space, "
              "all test encounters)")
        matched = matched_neural(args.bootstrap)
        if matched:
            m = pd.DataFrame(matched)
            m.to_csv(OUT / "neural_head_to_head.csv", index=False)
            cols = [c for c in ("system", "micro_f1", "micro_f1_lo95",
                                "micro_f1_hi95", "macro_f1", "tail_macro_f1",
                                "jaccard", "exact_match", "mean_pred_size",
                                "vs_neural_micro_f1", "vs_neural_micro_f1_lo95",
                                "vs_neural_micro_f1_hi95",
                                "vs_neural_micro_f1_p") if c in m.columns]
            print(m[cols].to_string(index=False,
                                    float_format=lambda v: f"{v:.4f}"))
            print(f"\n  published PHC-RxGen full89 micro-F1 = "
                  f"{NEURAL_PUBLISHED['full89']['micro_f1']:.4f} "
                  f"[{NEURAL_PUBLISHED['full89']['micro_f1_lo95']}, "
                  f"{NEURAL_PUBLISHED['full89']['micro_f1_hi95']}] "
                  f"(docs/model_comparison.md)")

    (OUT / f"test_results{args.tag}.json").write_text(json.dumps(
        {"bootstrap": args.bootstrap, "rows": all_rows,
         "calibration": cal.to_dict("records"), "matched_neural": matched,
         "neural_published": NEURAL_PUBLISHED,
         "excluded_components": EXCLUDED}, indent=2, default=float))
    print(f"\nwrote {OUT/f'test_results{args.tag}.csv'}")


if __name__ == "__main__":
    main()
