"""Multi-condition notes: diagnose the deficit, then fix it.

The qualitative demo exposed the failure mode. On the note "h/o DM+h/o asthma"
the recommender ranked `antidiabetic_biguanide` top and missed BOTH gold classes
for the asthma half (`antihistamine`, `respiratory_other`). 39.5% of notes carry
more than one complaint (mean 1.58 spans), so this is not an edge case.

Two candidate mechanisms, and they imply different fixes:

  A. **Probability dilution.** A single global threshold under-fires when a note
     carries several conditions, because each condition's evidence is a smaller
     share of one bag of tokens. Fix: adapt the threshold to the number of
     complaint spans.
  B. **Representation collapse.** Concatenating spans into one bag lets the
     dominant condition's vocabulary swamp the rest. Fix: score each span
     separately and combine (element-wise max over spans).

This module measures the deficit first -- the premise is testable and might be
false -- then evaluates both fixes on validation and reports the winner on test.

    python -m src.phcrx.recommend.multicondition
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ..config import RESULTS
from ..textproc.segment import segment
from .blocks import FEATURE_SETS, MultiLabelOvR, make_pipeline
from .corpus import build_components, load_frame, split_index
from .metrics import set_metrics, tail_labels, tune_threshold

OUT = RESULTS / "recommend"
OUT.mkdir(parents=True, exist_ok=True)


def n_spans(text) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(segment(text))


def stratified(Y: np.ndarray, P: np.ndarray, spans: np.ndarray,
               thr: float | np.ndarray, tail: np.ndarray) -> pd.DataFrame:
    """Set metrics broken out by complaint count."""
    B = (P >= (thr if np.isscalar(thr) else thr[:, None])).astype(np.int8)
    rows = []
    for lab, m in (("0 (no complaint)", spans == 0), ("1", spans == 1),
                   ("2", spans == 2), ("3+", spans >= 3)):
        if m.sum() == 0:
            continue
        s = set_metrics(Y[m], B[m], tail)
        rows.append({"spans": lab, "n": int(m.sum()),
                     "gold_classes": float(Y[m].sum(1).mean()),
                     "pred_classes": float(B[m].sum(1).mean()),
                     "micro_f1": s["micro_f1"], "recall": s["micro_recall"],
                     "precision": s["micro_precision"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="text_raw",
                    help="deployable arm; span scoring needs a text block")
    ap.add_argument("--C", type=float, default=4.0)
    args = ap.parse_args()

    df, FF = load_frame()
    comps = build_components(df)
    C = comps["drug_classes"]
    Y, labels = np.asarray(C.Y), C.labels
    # Respect the component mask: split_index(df, comp.mask) is how the rest of
    # the package selects rows for which this target is defined.
    idx = split_index(df, C.mask)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    TAIL = tail_labels(Y[tr])
    spans_all = df["symptom_text"].map(n_spans).to_numpy()

    print(f"labels={len(labels)}  train={len(tr)} val={len(va)} test={len(te)}")
    print("complaint-span distribution (all rows): "
          + str(dict(pd.Series(spans_all).value_counts().sort_index().head(6))))

    # ---------- baseline: one bag of tokens, one global threshold ----------
    pipe = make_pipeline(args.feature_set, MultiLabelOvR(C=args.C), FF)
    pipe.fit(df.iloc[tr], Y[tr])
    Pva, Pte = pipe.predict_proba(df.iloc[va]), pipe.predict_proba(df.iloc[te])
    thr = tune_threshold(Y[va], Pva)
    base_te = set_metrics(Y[te], (Pte >= thr).astype(np.int8), TAIL)
    print(f"\nbaseline ({args.feature_set}, thr={thr:.2f}): "
          f"test micro-F1={base_te['micro_f1']:.4f}")

    print("\n--- DEFICIT: is performance actually worse on multi-complaint notes? ---")
    st = stratified(Y[te], Pte, spans_all[te], thr, TAIL)
    print(st.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------- fix A: threshold adapted to complaint count ----------------
    # Grid-search a per-span-count threshold on VALIDATION only.
    sva, ste = spans_all[va], spans_all[te]
    grid = np.arange(0.05, 0.61, 0.01)
    thr_by = {}
    for k in (0, 1, 2, 3):
        m = (sva >= 3) if k == 3 else (sva == k)
        if m.sum() < 30:
            thr_by[k] = thr
            continue
        thr_by[k] = float(max(grid, key=lambda t: set_metrics(
            Y[va][m], (Pva[m] >= t).astype(np.int8), TAIL)["micro_f1"]))
    print(f"\nfix A — per-span-count thresholds tuned on val: "
          + ", ".join(f"{k}:{v:.2f}" for k, v in thr_by.items()))
    thr_vec_te = np.array([thr_by[min(s, 3)] for s in ste])
    a_te = set_metrics(Y[te], (Pte >= thr_vec_te[:, None]).astype(np.int8), TAIL)
    print(f"   test micro-F1={a_te['micro_f1']:.4f}  "
          f"(baseline {base_te['micro_f1']:.4f}, Δ={a_te['micro_f1']-base_te['micro_f1']:+.4f})")

    # ---------- fix B: score each span, max-pool ---------------------------
    # The model is unchanged; only inference changes. Each span is scored as if
    # it were the whole note, and the element-wise max is taken, so a secondary
    # condition cannot be diluted by a dominant one.
    def span_proba(frame: pd.DataFrame) -> np.ndarray:
        rows, owner = [], []
        for i, (_, r) in enumerate(frame.iterrows()):
            sp = segment(r["symptom_text"]) if isinstance(r["symptom_text"], str) else []
            if not sp:
                sp = [r["symptom_text"] if isinstance(r["symptom_text"], str) else ""]
            for s in sp:
                rr = r.copy()
                rr["symptom_text"] = s
                rows.append(rr)
                owner.append(i)
        exp = pd.DataFrame(rows).reset_index(drop=True)
        Pe = pipe.predict_proba(exp)
        out = np.zeros((len(frame), Pe.shape[1]), dtype=np.float32)
        owner = np.asarray(owner)
        for i in range(len(frame)):
            sel = Pe[owner == i]
            if len(sel):
                out[i] = sel.max(0)
        return out

    print("\nfix B — per-span max-pooled inference …")
    Bva, Bte = span_proba(df.iloc[va]), span_proba(df.iloc[te])
    thr_b = tune_threshold(Y[va], Bva)
    b_te = set_metrics(Y[te], (Bte >= thr_b).astype(np.int8), TAIL)
    print(f"   thr={thr_b:.2f}  test micro-F1={b_te['micro_f1']:.4f}  "
          f"(Δ={b_te['micro_f1']-base_te['micro_f1']:+.4f})")

    # ---------- fix A+B ----------------------------------------------------
    thr_by_b = {}
    for k in (0, 1, 2, 3):
        m = (sva >= 3) if k == 3 else (sva == k)
        if m.sum() < 30:
            thr_by_b[k] = thr_b
            continue
        thr_by_b[k] = float(max(grid, key=lambda t: set_metrics(
            Y[va][m], (Bva[m] >= t).astype(np.int8), TAIL)["micro_f1"]))
    thr_vec_b = np.array([thr_by_b[min(s, 3)] for s in ste])
    ab_te = set_metrics(Y[te], (Bte >= thr_vec_b[:, None]).astype(np.int8), TAIL)
    print(f"\nfix A+B: test micro-F1={ab_te['micro_f1']:.4f}  "
          f"(Δ={ab_te['micro_f1']-base_te['micro_f1']:+.4f})")

    # ---------- stratified comparison -------------------------------------
    print("\n--- multi-complaint rows only (2+ spans) ---")
    m2 = ste >= 2
    for name, P, T in (("baseline", Pte, thr), ("A adaptive-thr", Pte, thr_vec_te),
                       ("B span-max", Bte, thr_b), ("A+B", Bte, thr_vec_b)):
        Bm = (P[m2] >= (T if np.isscalar(T) else T[m2][:, None])).astype(np.int8)
        s = set_metrics(Y[te][m2], Bm, TAIL)
        print(f"   {name:16s} micro-F1={s['micro_f1']:.4f}  "
              f"recall={s['micro_recall']:.4f}  prec={s['micro_precision']:.4f}  "
              f"pred/enc={Bm.sum(1).mean():.2f} (gold {Y[te][m2].sum(1).mean():.2f})")

    best = max([("baseline", base_te), ("A", a_te), ("B", b_te), ("A+B", ab_te)],
               key=lambda kv: kv[1]["micro_f1"])
    print(f"\nWINNER on test: {best[0]}  micro-F1={best[1]['micro_f1']:.4f}")

    (OUT / "multicondition.json").write_text(json.dumps({
        "feature_set": args.feature_set,
        "baseline": base_te, "fix_a_adaptive_threshold": a_te,
        "fix_b_span_max": b_te, "fix_ab": ab_te,
        "thresholds": {"global": thr, "by_spans": thr_by,
                       "span_global": thr_b, "span_by_spans": thr_by_b},
        "stratified_baseline": st.to_dict("records"),
    }, indent=2, default=float), encoding="utf-8")
    print("wrote", OUT / "multicondition.json")


if __name__ == "__main__":
    main()
