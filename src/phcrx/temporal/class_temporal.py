"""Patient-split vs temporal-split evaluation of the deployed recommender.

    python -m src.phcrx.temporal.class_temporal
    python -m src.phcrx.temporal.class_temporal --targets brand717 --c-grid 4.0

One code path, two splits, three possible targets. For every (target, feature
set, split) the module fits the *same* estimator the deployed recommender uses
-- `recommend.blocks.MultiLabelOvR` inside `recommend.blocks.make_pipeline` --
selects C on VALIDATION, calibrates on VALIDATION, sets the operating point on
VALIDATION, and reads TEST once.

Why the split is built in memory
--------------------------------
`python -m src.phcrx.preprocess --split temporal` **rewrites**
`data/processed/rxgen_*.parquet` in place and reshuffles the drug/word
vocabularies. Every downstream artefact -- `features.parquet`, the
`drug_normalization.parquet` keys, the cached neural predictions, the fitted
recommender -- is aligned to the current corpus, so regenerating it would
silently invalidate them, and a previous session did exactly that. Nothing
about the temporal split requires it: `rxgen_encounters.parquet` already
carries a `year` column, so the split is a boolean over rows.

    train  year <= 2015      val  year == 2016      test  year >= 2017

`recommend.corpus.split_index` reads `df["split"]`, so the whole change is one
overwritten column on a copy of the frame. The row set, the label matrices and
the feature blocks are otherwise identical between the two arms of the
comparison, which is the point: the *only* thing that varies is which rows are
train and which are test.

What is fitted where
--------------------
Everything that learns anything is inside `Pipeline.fit`, which only ever sees
the train slice of whichever split is being run: the TF-IDF vocabulary and its
IDF weights, the typo corrector, the imputation medians, the standardisation
constants and the one-hot levels. `blocks._MEMO` keys on a digest of the exact
training texts, so the patient-train and temporal-train fits cannot collide.
`_leak_report` asserts the era boundaries and the disjointness of the slices,
and the fitted TF-IDF vocabulary size is recorded per split so the refit is
evidenced rather than asserted in prose.

The one exception is documented and *not* fixable from here: the `clinical`,
`text_clinical`, `all` and `all_sem` feature sets read engineered columns from
`data/processed/features.parquet`, which `features/build_features.py` fitted
with `is_train = (split == "train")` -- the **patient** split. Under the
temporal split those columns (the TF-IDF/SVD text basis, ICD chapter levels,
categorical levels, and leave-one-out site/prescriber target encodings) saw
later-era rows during their offline fit. Rebuilding that artefact is out of
scope here, so `all` under the temporal split is reported as an **optimistic
upper bound**. That biases the comparison in `all`'s favour, which only
strengthens the finding if `all` still degrades more than the leak-free
`text_raw`.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from . import OUT, SEED, TRAIN_END, VAL_END
from ..recommend import corpus as _corpus
from ..recommend.blocks import (DEPLOYABLE, FEATURE_SETS, MultiLabelOvR,
                                make_pipeline)
from ..recommend.corpus import drug_class_labels, load_frame, split_index
from ..recommend.evaluate import _prior_topk
from ..recommend.metrics import (Resampler, ci, delta_stats, select_calibrator,
                                 set_metrics, tail_labels, tune_threshold)

warnings.filterwarnings("ignore")

SPLIT_NAMES = ("train", "val", "test")
HEADLINE = ("micro_f1", "macro_f1", "tail_macro_f1", "jaccard", "exact_match")
REPORTED = HEADLINE + ("micro_precision", "micro_recall", "mean_pred_size",
                       "mean_gold_size")
C_GRID = (1.0, 2.0, 4.0, 8.0)

# Feature sets whose engineered columns come from `features.parquet`, which was
# fitted on the PATIENT train split. Flagged in every output row.
PATIENT_FITTED_SETS = {"clinical", "text_clinical", "all", "all_sem"}


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def temporal_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of `df` with `split` overwritten by the era rule. Nothing on disk."""
    year = pd.to_numeric(df["year"], errors="coerce").to_numpy()
    if not np.isfinite(year).all():
        raise ValueError("rxgen_encounters.parquet has non-numeric years")
    out = df.copy()
    out["split"] = np.where(year <= TRAIN_END, "train",
                            np.where(year <= VAL_END, "val", "test"))
    return out


def frame_for(df: pd.DataFrame, split_kind: str) -> pd.DataFrame:
    if split_kind == "patient":
        return df
    if split_kind == "temporal":
        return temporal_frame(df)
    raise ValueError(split_kind)


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
def brand_labels(df: pd.DataFrame):
    """Brand target: one label per `drug_id` that appears in any order.

    This is the space the published -66% was measured in. The published count
    is 719 because the neural drug vocabulary carries `<pad>`/`<eos>`; the
    orders table contains 717 real drug ids, and a special token is not a
    prediction, so the two extra columns are not created here.
    """
    o = _corpus._orders().dropna(subset=["drug_id"]).copy()
    o["drug_id"] = o["drug_id"].astype(int)
    labels = sorted(o["drug_id"].unique().tolist())
    name = (o.drop_duplicates("drug_id").set_index("drug_id")["drug_name"]
            .astype(str))
    Y = _corpus._multi_hot(o, df["prescription_id"].to_numpy(), "drug_id",
                           labels)
    return Y, [f"{d}|{name.get(d, d)}" for d in labels]


def build_target(df: pd.DataFrame, target: str):
    """(Y, label names). Label spaces are split-independent by construction.

    None of the three depends on which rows are train, so the two splits score
    the identical label matrix on the identical rows -- only the row *roles*
    change. (`corpus.advice_labels`/`test_labels` do apply a train-support
    filter, which is one reason only the drug target is run here.)
    """
    if target in ("class46", "cat89"):
        return drug_class_labels(df, target)
    if target == "brand717":
        return brand_labels(df)
    raise ValueError(target)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def _tfidf_vocab_size(pipe):
    union = pipe.named_steps["features"]
    for name, t in union.transformer_list:
        if name == "text" and hasattr(t, "vec_"):
            return int(len(t.vec_.vocabulary_))
    return None


def _leak_report(df: pd.DataFrame, idx: dict, split_kind: str) -> dict:
    """Hard checks on the split itself, run before anything is fitted."""
    pid = df["prescription_id"].to_numpy()
    year = pd.to_numeric(df["year"], errors="coerce").to_numpy()
    sets = {s: set(pid[idx[s]].tolist()) for s in SPLIT_NAMES}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if sets[a] & sets[b]:
            raise AssertionError(f"{split_kind}: {a}/{b} share encounters")
    rep = {"split": split_kind,
           "rows": {s: int(len(idx[s])) for s in SPLIT_NAMES},
           "year_range": {s: [int(year[idx[s]].min()), int(year[idx[s]].max())]
                          for s in SPLIT_NAMES}}
    if split_kind == "temporal":
        if rep["year_range"]["train"][1] > TRAIN_END:
            raise AssertionError("temporal train leaks past 2015")
        if rep["year_range"]["test"][0] <= VAL_END:
            raise AssertionError("temporal test reaches back into the val era")
        if rep["year_range"]["val"] != [VAL_END, VAL_END]:
            raise AssertionError("temporal val is not exactly 2016")
    if split_kind == "patient":
        uid = df["user_id"].to_numpy()
        u = {s: set(uid[idx[s]].tolist()) for s in SPLIT_NAMES}
        rep["patients_straddling"] = int(len(
            (u["train"] & u["val"]) | (u["train"] & u["test"])
            | (u["val"] & u["test"])))
    return rep


def label_shift(Y: np.ndarray, idx: dict, topn: int = 10) -> dict:
    """How much of the *label space* moves between train and test.

    Reproduces, on these exact rows, the quantities the drug-normalisation
    workstream reports for brands vs classes: vocabulary Jaccard, the share of
    test gold positives whose label was never seen in train, and how many of
    the train top-10 survive into the test top-10.
    """
    tr, te = Y[idx["train"]], Y[idx["test"]]
    tr_pos, te_pos = tr.sum(0) > 0, te.sum(0) > 0
    union = int((tr_pos | te_pos).sum())
    gold = float(te.sum())
    tr_top = set(np.argsort(-tr.sum(0))[:topn].tolist())
    te_top = set(np.argsort(-te.sum(0))[:topn].tolist())
    return {
        "n_labels": int(Y.shape[1]),
        "labels_in_train": int(tr_pos.sum()),
        "labels_in_test": int(te_pos.sum()),
        "vocab_jaccard": float((tr_pos & te_pos).sum() / max(union, 1)),
        "unseen_gold_positive_rate": float(te[:, ~tr_pos].sum()
                                           / max(gold, 1.0)),
        "rows_with_unseen_label": float((te[:, ~tr_pos].sum(1) > 0).mean()),
        "top%d_shared" % topn: int(len(tr_top & te_top)),
        "gold_positives_test": int(gold),
    }


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
def fit_arm(df: pd.DataFrame, FF: dict, Y: np.ndarray, idx: dict, arm: str,
            c_grid, verbose: bool = True) -> dict:
    """Fit / select C / calibrate / threshold -- val only -- then score test.

    Mirrors `recommend.pipeline.fit_component` for a multi-label component,
    minus the feature-set search: the feature set is pinned by the caller
    because the whole question here is how a *given* arm behaves under shift.
    """
    Xs = {s: df.iloc[idx[s]] for s in SPLIT_NAMES}
    Ys = {s: Y[idx[s]] for s in SPLIT_NAMES}
    tail = tail_labels(Ys["train"])

    best = {"val_micro_f1": -np.inf}
    hp = []
    for c in c_grid:
        t0 = time.time()
        pipe = make_pipeline(arm, MultiLabelOvR(C=c), FF)
        pipe.fit(Xs["train"], Ys["train"])
        Pva = pipe.predict_proba(Xs["val"])
        thr = tune_threshold(Ys["val"], Pva)
        s = set_metrics(Ys["val"], Pva >= thr, tail)["micro_f1"]
        hp.append({"C": float(c), "val_micro_f1": float(s),
                   "seconds": round(time.time() - t0, 1)})
        if verbose:
            print("      C=%-5g val micro-F1 %.4f  (%.0fs)"
                  % (c, s, time.time() - t0), flush=True)
        if s > best["val_micro_f1"]:
            best = {"C": float(c), "val_micro_f1": float(s), "pipe": pipe,
                    "Pva": Pva}

    # Calibrate on VAL, choosing the method by K-fold CV *inside* val, then set
    # the operating point on the calibrated val probabilities. The line below
    # is the first and only read of the test rows' features; test labels are
    # opened in `score_systems`.
    method, calib, cv_ece = select_calibrator(best["Pva"], Ys["val"], seed=SEED)
    Pva_cal = calib.transform(best["Pva"])
    thr = tune_threshold(Ys["val"], Pva_cal)
    Pte = calib.transform(best["pipe"].predict_proba(Xs["test"]))

    return {
        "arm": arm, "C": best["C"], "val_micro_f1": best["val_micro_f1"],
        "calibration": method, "cv_ece": cv_ece, "threshold": float(thr),
        "tfidf_vocab": _tfidf_vocab_size(best["pipe"]),
        "n_features": int(best["pipe"].named_steps["features"]
                          .transform(Xs["val"][:2]).shape[1]),
        "hp_search": hp,
        "P_test": Pte, "Y": Ys, "tail": tail,
    }


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def score_systems(fits: dict, Y: np.ndarray, idx: dict, resampler: Resampler,
                  tail: np.ndarray):
    """Point estimates + paired bootstrap for every system on one split.

    All systems inside a split are scored on the same rows with the same
    resample indices, so the within-split deltas (recommender vs floor, arm vs
    arm) are paired. The frequency prior is `recommend.evaluate._prior_topk`:
    label order from this split's TRAIN, k chosen on this split's VAL.
    """
    Ytr, Yva, Yte = (Y[idx[s]].astype(bool) for s in SPLIT_NAMES)
    systems = {"always_empty": np.zeros_like(Yte)}
    k, top = _prior_topk(Ytr, Yva)
    P = np.zeros_like(Yte)
    P[:, top] = True
    prior_name = "prior_top%d" % k
    systems[prior_name] = P
    for arm, f in fits.items():
        systems["recommender[%s]" % arm] = f["P_test"] >= f["threshold"]

    rows, dists = [], {}
    for name, B in systems.items():
        point = set_metrics(Yte, B, tail)
        dist = resampler.set_metrics(Yte, B, tail)
        dists[name] = {m: np.asarray(v) for m, v in dist.items()}
        r = {"system": name}
        for m in REPORTED:
            r[m] = point[m]
            if m in dist:
                lo, hi = ci(dist[m])
                r["%s_lo95" % m], r["%s_hi95" % m] = lo, hi
        r["empty_pred_rate"] = point["empty_pred_rate"]
        rows.append(r)

    for r in rows:
        for ref_tag, ref in (("empty", "always_empty"), ("prior", prior_name)):
            if r["system"] == ref:
                continue
            for m in HEADLINE:
                s = delta_stats(dists[r["system"]][m], dists[ref][m])
                r["d_%s_%s" % (ref_tag, m)] = s["delta"]
                r["d_%s_%s_lo95" % (ref_tag, m)] = s["delta_lo95"]
                r["d_%s_%s_hi95" % (ref_tag, m)] = s["delta_hi95"]
                r["d_%s_%s_p" % (ref_tag, m)] = s["p_better"]
    return rows, dists


def cross_split_delta(pat: dict, tem: dict) -> list:
    """Temporal minus patient, as an **unpaired** bootstrap.

    The two splits do not share a test set -- 2,780 patient-test encounters
    against 2,880 temporal-test encounters, overlapping only partially -- so
    the paired machinery used *within* a split does not apply. Each split gets
    its own independent resampler (different seeds) and the difference
    distribution is taken across independent draws. Where the two test sets do
    overlap the estimates are positively correlated, so treating them as
    independent *overstates* the width of the difference interval: the
    intervals reported are conservative.
    """
    out = []
    for name in sorted(set(pat) & set(tem)):
        for m in HEADLINE:
            a, b = np.asarray(tem[name][m]), np.asarray(pat[name][m])
            d = a - b
            lo, hi = ci(d)
            row = {"system": name, "metric": m,
                   "patient": float(b.mean()), "temporal": float(a.mean()),
                   "delta": float(d.mean()), "delta_lo95": lo,
                   "delta_hi95": hi,
                   "p_temporal_worse": float((d < 0).mean())}
            ok = np.abs(b) > 1e-9
            if ok.mean() > 0.99:
                rel = 100.0 * d[ok] / b[ok]
                rlo, rhi = ci(rel)
                row.update({"rel_pct": float(rel.mean()),
                            "rel_pct_lo95": rlo, "rel_pct_hi95": rhi})
            out.append(row)
    return out


# ---------------------------------------------------------------------------
def run_target(df: pd.DataFrame, FF: dict, target: str, arms, splits,
               c_grid, n_boot: int):
    Y, labels = build_target(df, target)
    print("\n=== target %s: %d labels, %d gold positives over %d encounters ==="
          % (target, Y.shape[1], int(Y.sum()), len(df)), flush=True)

    rows, deltas = [], []
    meta = {"n_labels": int(Y.shape[1]), "splits": {}}
    dists_by_split = {}
    for si, kind in enumerate(splits):
        d2 = frame_for(df, kind)
        idx = split_index(d2)
        leak = _leak_report(d2, idx, kind)
        shift = label_shift(Y, idx)
        print("\n  [%s] train/val/test = %d/%d/%d   years %s"
              % (kind, leak["rows"]["train"], leak["rows"]["val"],
                 leak["rows"]["test"], leak["year_range"]))
        print("  [%s] label-space shift: Jaccard %.3f  unseen gold positives "
              "%.1f%%  top10 shared %d/10"
              % (kind, shift["vocab_jaccard"],
                 100 * shift["unseen_gold_positive_rate"],
                 shift["top10_shared"]), flush=True)

        fits = {}
        for arm in arms:
            print("    arm=%s" % arm, flush=True)
            fits[arm] = fit_arm(d2, FF, Y, idx, arm, c_grid)
            f = fits[arm]
            print("      -> C=%g calib=%s thr=%.2f feats=%d tfidf_vocab=%s"
                  % (f["C"], f["calibration"], f["threshold"],
                     f["n_features"], f["tfidf_vocab"]), flush=True)

        # Independent resamplers: the two splits have different test rows.
        R = Resampler(len(idx["test"]), n_boot, SEED + si)
        tail = (fits[arms[0]]["tail"] if arms
                else tail_labels(Y[idx["train"]]))
        srows, dists = score_systems(fits, Y, idx, R, tail)
        dists_by_split[kind] = dists
        for r in srows:
            arm = (r["system"].split("[", 1)[1][:-1]
                   if "[" in r["system"] else None)
            r.update({"target": target, "split": kind, "feature_set": arm,
                      "deployable": None if arm is None else arm in DEPLOYABLE,
                      "features_parquet_fitted_on_patient_split":
                          bool(arm in PATIENT_FITTED_SETS)})
            if arm is not None:
                r.update({"C": fits[arm]["C"],
                          "threshold": fits[arm]["threshold"],
                          "calibration": fits[arm]["calibration"],
                          "val_micro_f1": fits[arm]["val_micro_f1"],
                          "n_features": fits[arm]["n_features"],
                          "tfidf_vocab": fits[arm]["tfidf_vocab"]})
            rows.append(r)
        meta["splits"][kind] = {
            "leak_report": leak, "label_shift": shift,
            "arms": {a: {k: v for k, v in f.items()
                         if k not in ("P_test", "Y", "tail")}
                     for a, f in fits.items()}}

        for r in srows:
            print("    %-26s micro-F1 %.4f [%.4f, %.4f]  macro %.4f  "
                  "tail %.4f  jac %.4f  exact %.4f"
                  % (r["system"], r["micro_f1"], r.get("micro_f1_lo95", np.nan),
                     r.get("micro_f1_hi95", np.nan), r["macro_f1"],
                     r["tail_macro_f1"], r["jaccard"], r["exact_match"]),
                  flush=True)

    if "patient" in dists_by_split and "temporal" in dists_by_split:
        for r in cross_split_delta(dists_by_split["patient"],
                                   dists_by_split["temporal"]):
            r["target"] = target
            deltas.append(r)
    return rows, deltas, meta


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["class46"],
                    choices=("class46", "cat89", "brand717"))
    ap.add_argument("--arms", nargs="+", default=["text_raw", "all"],
                    choices=list(FEATURE_SETS))
    ap.add_argument("--splits", nargs="+", default=["patient", "temporal"],
                    choices=("patient", "temporal"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--c-grid", nargs="+", type=float, default=list(C_GRID))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    df, FF = load_frame()
    print("corpus: %d encounters, years %d-%d"
          % (len(df), int(df["year"].min()), int(df["year"].max())))
    print("targets=%s  arms=%s  splits=%s  C grid=%s  bootstrap=%d"
          % (args.targets, args.arms, args.splits, args.c_grid,
             args.bootstrap))

    # Test-row overlap between the two splits, for the unpaired-bootstrap note.
    pid = df["prescription_id"].to_numpy()
    pat_te = set(pid[split_index(df)["test"]].tolist())
    tem_te = set(pid[split_index(temporal_frame(df))["test"]].tolist())
    overlap = {"patient_test": len(pat_te), "temporal_test": len(tem_te),
               "shared": len(pat_te & tem_te)}
    print("test rows: patient %d, temporal %d, shared %d"
          % (overlap["patient_test"], overlap["temporal_test"],
             overlap["shared"]))

    rows, deltas, meta = [], [], {}
    for target in args.targets:
        r, d, m = run_target(df, FF, target, args.arms, args.splits,
                             tuple(args.c_grid), args.bootstrap)
        rows += r
        deltas += d
        meta[target] = m

    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    lead = ["target", "split", "system", "feature_set", "deployable"]
    res = res[lead + [c for c in res.columns if c not in lead]]
    res.to_csv(OUT / ("class_temporal%s.csv" % args.tag), index=False)
    dl = pd.DataFrame(deltas)
    if len(dl):
        dl.to_csv(OUT / ("split_deltas%s.csv" % args.tag), index=False)
    (OUT / ("class_temporal%s.json" % args.tag)).write_text(json.dumps({
        "config": vars(args), "train_end": TRAIN_END, "val_end": VAL_END,
        "test_row_overlap": overlap,
        "preprocess_rerun": False,
        "note": ("temporal split derived in memory from the `year` column; "
                 "src.phcrx.preprocess was NOT run -- it rewrites "
                 "data/processed/rxgen_*.parquet in place and reshuffles the "
                 "vocabularies"),
        "features_parquet_caveat": (
            "features.parquet was fitted with is_train = (patient split == "
            "'train'); the clinical/all feature sets therefore leak later-era "
            "rows into their offline fit under the temporal split and are an "
            "optimistic upper bound there"),
        "targets": meta}, indent=2, default=str))

    print("\n" + "=" * 108)
    print("%-10s %-9s %-26s %22s %8s %8s %8s %8s"
          % ("target", "split", "system", "micro-F1", "macro", "tail",
             "jaccard", "exact"))
    print("-" * 108)
    for r in rows:
        band = ("%.4f [%.4f,%.4f]"
                % (r["micro_f1"], r.get("micro_f1_lo95", np.nan),
                   r.get("micro_f1_hi95", np.nan)))
        print("%-10s %-9s %-26s %22s %8.4f %8.4f %8.4f %8.4f"
              % (r["target"], r["split"], r["system"], band, r["macro_f1"],
                 r["tail_macro_f1"], r["jaccard"], r["exact_match"]))

    if len(dl):
        print("\n" + "=" * 108)
        print("TEMPORAL minus PATIENT (unpaired bootstrap, conservative)")
        print("-" * 108)
        for r in deltas:
            if r["metric"] != "micro_f1":
                continue
            rel = ("%+7.1f%% [%+.1f,%+.1f]"
                   % (r["rel_pct"], r["rel_pct_lo95"], r["rel_pct_hi95"])
                   if "rel_pct" in r else "-")
            print("%-10s %-26s %.4f -> %.4f   delta %+.4f [%+.4f,%+.4f]   %s"
                  % (r["target"], r["system"], r["patient"], r["temporal"],
                     r["delta"], r["delta_lo95"], r["delta_hi95"], rel))

    print("\nwrote %s  (%.0fs)" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
