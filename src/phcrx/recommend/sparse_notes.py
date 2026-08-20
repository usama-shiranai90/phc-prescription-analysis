"""Sparse / no-complaint encounters: diagnose, establish the ceiling, then fix.

Measured deficit (test, 46 drug classes): encounters with no parsed complaint
score micro-F1 **0.3393** (recall 0.2881) against 0.4526 for one complaint and
0.5024 for three or more. 436 of 2,780 test rows.

Before building anything, three questions the last round taught me to ask first:

  1. **Is the task harder, or is the model mis-specified?** A subset-specific
     frequency prior gives the ceiling-free floor. If the shared model barely
     beats the prior on these rows, the signal may simply not be there.
  2. **Is it a threshold artefact?** Recall 0.288 vs precision 0.413 says the
     model under-fires here. A threshold tuned on the *whole* validation set can
     be badly wrong for a subgroup whose feature distribution differs.
  3. **Is it a feature-set artefact?** The deployed arm is `text_raw`. On rows
     with no text that is 35 raw features plus an all-zero TF-IDF block, so the
     richer tabular arms may simply suit these rows better.

Only then the candidate fixes: subgroup threshold, a routed specialist fitted on
sparse rows, and an explicit text-presence indicator.

    python -m src.phcrx.recommend.sparse_notes
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ..config import RESULTS
from ..textproc.segment import segment
from .blocks import MultiLabelOvR, make_pipeline
from .corpus import build_components, load_frame, split_index
from .metrics import set_metrics, tail_labels, tune_threshold

OUT = RESULTS / "recommend"
OUT.mkdir(parents=True, exist_ok=True)


def n_spans(text) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(segment(text))


def prior_predictor(Ytr: np.ndarray, k: int, n: int) -> np.ndarray:
    """Always emit the k most frequent classes in the fitting subset."""
    top = np.argsort(-Ytr.sum(0))[:k]
    B = np.zeros((n, Ytr.shape[1]), dtype=np.int8)
    B[:, top] = 1
    return B


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--arms", nargs="+",
                    default=["text_raw", "raw", "clinical", "all"])
    args = ap.parse_args()

    df, FF = load_frame()
    comps = build_components(df)
    C = comps["drug_classes"]
    Y = np.asarray(C.Y)
    idx = split_index(df, C.mask)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    TAIL = tail_labels(Y[tr])

    spans = df["symptom_text"].map(n_spans).to_numpy()
    sparse = spans == 0
    s_tr, s_va, s_te = sparse[tr], sparse[va], sparse[te]
    print(f"sparse (no parsed complaint): train={s_tr.sum()} val={s_va.sum()} "
          f"test={s_te.sum()}  ({s_te.mean():.1%} of test)")
    print(f"gold classes/encounter: sparse={Y[te][s_te].sum(1).mean():.2f}  "
          f"rich={Y[te][~s_te].sum(1).mean():.2f}")

    # ---------- Q1: what is the ceiling? subset-specific prior -------------
    print("\n--- Q1: subset-specific frequency prior (is the signal there at all?) ---")
    best_prior = None
    for k in (1, 2, 3, 4):
        B = prior_predictor(Y[tr][s_tr], k, int(s_te.sum()))
        m = set_metrics(Y[te][s_te], B, TAIL)
        print(f"   prior_top{k}  micro-F1={m['micro_f1']:.4f}")
        if best_prior is None or m["micro_f1"] > best_prior[1]["micro_f1"]:
            best_prior = (f"prior_top{k}", m)
    print(f"   -> best prior on sparse rows: {best_prior[0]} "
          f"{best_prior[1]['micro_f1']:.4f}")

    # ---------- Q3: which feature arm suits these rows? --------------------
    print("\n--- Q3: feature arms, scored on the SPARSE test rows ---")
    fitted, res = {}, []
    for arm in args.arms:
        pipe = make_pipeline(arm, MultiLabelOvR(C=args.C), FF)
        pipe.fit(df.iloc[tr], Y[tr])
        Pva, Pte = pipe.predict_proba(df.iloc[va]), pipe.predict_proba(df.iloc[te])
        fitted[arm] = (pipe, Pva, Pte)

        thr_g = tune_threshold(Y[va], Pva)                     # global threshold
        g = set_metrics(Y[te][s_te], (Pte[s_te] >= thr_g).astype(np.int8), TAIL)
        # ---------- Q2: threshold tuned on sparse VALIDATION rows only -----
        grid = np.arange(0.03, 0.61, 0.01)
        thr_s = float(max(grid, key=lambda t: set_metrics(
            Y[va][s_va], (Pva[s_va] >= t).astype(np.int8), TAIL)["micro_f1"]))
        s = set_metrics(Y[te][s_te], (Pte[s_te] >= thr_s).astype(np.int8), TAIL)
        allrows = set_metrics(Y[te], (Pte >= thr_g).astype(np.int8), TAIL)
        res.append({"arm": arm, "thr_global": thr_g, "thr_sparse": thr_s,
                    "sparse_micro_global": g["micro_f1"],
                    "sparse_recall_global": g["micro_recall"],
                    "sparse_micro_subthr": s["micro_f1"],
                    "sparse_recall_subthr": s["micro_recall"],
                    "all_micro_global": allrows["micro_f1"]})
        print(f"   {arm:10s} thr_g={thr_g:.2f} sparse={g['micro_f1']:.4f} "
              f"(rec {g['micro_recall']:.3f}) | thr_s={thr_s:.2f} "
              f"sparse={s['micro_f1']:.4f} (rec {s['micro_recall']:.3f}) | "
              f"all-rows={allrows['micro_f1']:.4f}")

    tab = pd.DataFrame(res)
    base_arm = "text_raw"
    base = tab[tab["arm"] == base_arm].iloc[0]

    # ---------- Fix: specialist fitted on sparse rows only -----------------
    print("\n--- Fix: specialist model fitted on SPARSE TRAIN rows only ---")
    spec = []
    for arm in args.arms:
        if s_tr.sum() < 200:
            break
        pipe = make_pipeline(arm, MultiLabelOvR(C=args.C), FF)
        pipe.fit(df.iloc[tr[s_tr]], Y[tr][s_tr])
        Pva = pipe.predict_proba(df.iloc[va[s_va]])
        Pte = pipe.predict_proba(df.iloc[te[s_te]])
        grid = np.arange(0.03, 0.61, 0.01)
        thr = float(max(grid, key=lambda t: set_metrics(
            Y[va][s_va], (Pva >= t).astype(np.int8), TAIL)["micro_f1"]))
        m = set_metrics(Y[te][s_te], (Pte >= thr).astype(np.int8), TAIL)
        spec.append({"arm": arm, "thr": thr, "sparse_micro": m["micro_f1"],
                     "sparse_recall": m["micro_recall"],
                     "sparse_precision": m["micro_precision"]})
        print(f"   specialist[{arm:10s}] thr={thr:.2f} "
              f"micro-F1={m['micro_f1']:.4f} rec={m['micro_recall']:.3f} "
              f"prec={m['micro_precision']:.3f}")

    # ---------- verdict ----------------------------------------------------
    cands = [("baseline shared text_raw @ global thr", base["sparse_micro_global"])]
    cands += [(f"shared {r['arm']} @ sparse thr", r["sparse_micro_subthr"]) for r in res]
    cands += [(f"shared {r['arm']} @ global thr", r["sparse_micro_global"]) for r in res]
    cands += [(f"specialist {r['arm']}", r["sparse_micro"]) for r in spec]
    cands.append((f"subset prior ({best_prior[0]})", best_prior[1]["micro_f1"]))
    cands.sort(key=lambda kv: -kv[1])

    print("\n" + "=" * 78)
    print("SPARSE-ROW LEADERBOARD (test micro-F1 on 0-complaint encounters)")
    for name, v in cands:
        mark = "  <- baseline" if name.startswith("baseline") else ""
        print(f"   {v:.4f}  {name}{mark}")
    print("=" * 78)
    win = cands[0]
    print(f"best: {win[0]} = {win[1]:.4f}  vs baseline "
          f"{base['sparse_micro_global']:.4f} "
          f"(Δ={win[1]-base['sparse_micro_global']:+.4f})")
    print(f"context: rich-note rows score ~0.50; the subset prior floor is "
          f"{best_prior[1]['micro_f1']:.4f}")

    (OUT / "sparse_notes.json").write_text(json.dumps(
        {"n_sparse": {"train": int(s_tr.sum()), "val": int(s_va.sum()),
                      "test": int(s_te.sum())},
         "subset_prior": {best_prior[0]: best_prior[1]},
         "arms": res, "specialist": spec,
         "leaderboard": [{"system": n, "micro_f1": v} for n, v in cands]},
        indent=2, default=float), encoding="utf-8")
    print("wrote", OUT / "sparse_notes.json")


if __name__ == "__main__":
    main()
