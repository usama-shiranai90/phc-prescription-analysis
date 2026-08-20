"""Fit one scikit-learn Pipeline per prescription component.

    python -m src.phcrx.recommend.pipeline

For every component the module does four things, in this order and no other:

  1. **fit on TRAIN** -- one `Pipeline` per candidate feature set, every
     vectoriser / imputer / scaler / one-hot level learned inside
     `Pipeline.fit` on the train rows alone;
  2. **choose the feature set on VALIDATION** -- this is the target-dependent
     question the feature-importance workstream left open. Dropping temporal,
     prescriber and site *raised* held-out AUROC for drug-presence targets
     (0.707 -> 0.863) but the multi-label class-set task preferred the full
     matrix (0.4104 vs 0.3689), so the answer is measured per component rather
     than assumed once;
  3. **calibrate on VALIDATION** -- isotonic or Platt, the method chosen by
     K-fold CV *inside* val so the choice is not made on the same rows the
     calibrator is fitted to;
  4. **set the operating point on VALIDATION** -- one global threshold per
     multi-label component, tuned for micro-F1 after calibration.

TEST is not read by this module at all beyond materialising the feature rows;
its labels are opened for the first time in `evaluate.py`.

Outputs land in `results/rx_generation/recommend/`:

    models/<component>.joblib   fitted pipeline + calibrator + threshold
    preds/<component>.npz       cached val/test probabilities
    selection.csv               every feature set x component val score
    pipeline.json               what was chosen, and why
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, SGDRegressor

from . import MODELS_DIR, OUT, SEED
from .blocks import (C_DEFAULT, DEPLOYABLE, FEATURE_SETS, MulticlassLR,
                     MultiLabelOvR, make_pipeline)
from .corpus import EXCLUDED, build_components, load_frame, split_index
from .metrics import (auroc, ece, select_calibrator, set_metrics, tail_labels,
                      tune_threshold)

warnings.filterwarnings("ignore")

PRED_DIR = OUT / "preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)

C_GRID = (1.0, 2.0, 4.0, 8.0)
COUNT_MODELS = {
    "ridge_a1": lambda: Ridge(alpha=1.0, random_state=SEED),
    "ridge_a10": lambda: Ridge(alpha=10.0, random_state=SEED),
    "ridge_a100": lambda: Ridge(alpha=100.0, random_state=SEED),
    # epsilon-insensitive with epsilon=0 is absolute loss, i.e. it optimises
    # the metric the count component is scored on instead of squared error.
    "l1_sgd": lambda: SGDRegressor(loss="epsilon_insensitive", epsilon=0.0,
                                   alpha=1e-4, max_iter=4000, tol=1e-4,
                                   random_state=SEED),
}


# ---------------------------------------------------------------------------
def _estimator(kind: str, C: float = C_DEFAULT, count_model: str = "ridge_a10",
               n_classes: int = 2):
    if kind in ("binary", "multilabel"):
        return MultiLabelOvR(C=C)
    if kind == "multiclass":
        return MulticlassLR(n_classes=n_classes, C=C)
    if kind == "count":
        return COUNT_MODELS[count_model]()
    raise ValueError(kind)


def _fmt(v, n=4):
    return "-" if v is None else f"{v:.{n}f}"


def _targets(comp, rows):
    Y = comp.Y
    if comp.kind == "binary":
        return np.asarray(Y)[rows].reshape(-1, 1)
    if comp.kind in ("count", "multiclass"):
        return np.asarray(Y)[rows]
    return np.asarray(Y)[rows]


def _proba(pipe, X, kind):
    if kind == "count":
        return np.clip(pipe.predict(X), 0.0, None)
    return pipe.predict_proba(X)


def _val_score(kind, Yva, Pva, tail):
    """The headline validation metric for this component kind."""
    if kind == "binary":
        return auroc(Yva.ravel(), Pva.ravel()), None
    if kind == "count":
        return -float(np.mean(np.abs(Yva - Pva))), None
    if kind == "multiclass":
        from sklearn.metrics import f1_score
        pred = Pva.argmax(1)
        return float(f1_score(Yva, pred, average="macro")), None
    thr = tune_threshold(Yva, Pva)
    return set_metrics(Yva, Pva >= thr, tail)["micro_f1"], thr


# ---------------------------------------------------------------------------
def fit_component(comp, df, FF, arms, verbose=True) -> dict:
    idx = split_index(df, comp.mask)
    Xs = {s: df.iloc[idx[s]] for s in ("train", "val", "test")}
    Ys = {s: _targets(comp, idx[s]) for s in ("train", "val", "test")}

    if comp.kind == "multiclass":
        classes = list(comp.labels)
        c2i = {c: i for i, c in enumerate(classes)}
        Ys = {s: np.array([c2i[v] for v in Ys[s]]) for s in Ys}
    tail = (tail_labels(Ys["train"]) if comp.kind == "multilabel"
            else np.array([0]))

    def mk_est(**kw):
        return _estimator(comp.kind, n_classes=comp.n_labels, **kw)

    rows = []
    fitted = {}
    for arm in arms:
        t0 = time.time()
        pipe = make_pipeline(arm, mk_est(), FF)
        pipe.fit(Xs["train"], Ys["train"])
        Pva = _proba(pipe, Xs["val"], comp.kind)
        score, thr = _val_score(comp.kind, Ys["val"], Pva, tail)
        n_feat = pipe.named_steps["features"].transform(Xs["val"][:2]).shape[1]
        rows.append({"component": comp.name, "feature_set": arm,
                     "deployable": arm in DEPLOYABLE, "n_features": n_feat,
                     "val_score": score, "val_threshold": thr,
                     "seconds": round(time.time() - t0, 1)})
        fitted[arm] = pipe
        if verbose:
            print(f"    {arm:16s} feat={n_feat:6d} val={score:+.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    tab = pd.DataFrame(rows)
    best = tab.loc[tab["val_score"].idxmax(), "feature_set"]
    dep = tab[tab["deployable"]]
    best_dep = dep.loc[dep["val_score"].idxmax(), "feature_set"]
    if verbose:
        print(f"    -> winner: {best}   (best deployable: {best_dep}, "
              f"val {dep['val_score'].max():+.4f} vs {tab['val_score'].max():+.4f})")

    # --- hyper-parameter, on VAL, for the winning feature set only ---------
    hp_rows = []
    if comp.kind == "count":
        grid = [("count_model", k) for k in COUNT_MODELS]
    else:
        grid = [("C", c) for c in C_GRID]
    best_hp, best_hp_score, best_pipe, best_thr = None, -np.inf, None, None
    for key, v in grid:
        if key == "C" and v == C_DEFAULT:
            pipe = fitted[best]
        else:
            pipe = make_pipeline(best, mk_est(**{key: v}), FF)
            pipe.fit(Xs["train"], Ys["train"])
        Pva = _proba(pipe, Xs["val"], comp.kind)
        s, thr = _val_score(comp.kind, Ys["val"], Pva, tail)
        hp_rows.append({"param": key, "value": v, "val_score": s})
        if s > best_hp_score:
            best_hp, best_hp_score, best_pipe, best_thr = v, s, pipe, thr
    if verbose:
        print(f"    -> {grid[0][0]}={best_hp} val={best_hp_score:+.4f}")

    # --- calibrate on VAL, then set the operating point on VAL -------------
    def finalize(pipe, suffix=""):
        P = {s: _proba(pipe, Xs[s], comp.kind) for s in ("val", "test")}
        calib = {"method": "none", "cv_ece": {}, "val_ece_raw": None,
                 "val_ece_cal": None}
        calibrator = None
        if comp.kind in ("binary", "multilabel", "multiclass"):
            Yv = Ys["val"]
            if comp.kind == "multiclass":
                Yv = np.eye(len(comp.labels), dtype=np.int8)[Yv]
            method, calibrator, scores = select_calibrator(P["val"], Yv,
                                                           seed=SEED)
            Pcal = calibrator.transform(P["val"])
            calib = {"method": method, "cv_ece": scores,
                     "val_ece_raw": ece(P["val"], Yv),
                     "val_ece_cal": ece(Pcal, Yv)}
            P["val_cal"], P["test_cal"] = Pcal, calibrator.transform(P["test"])
        else:
            P["val_cal"], P["test_cal"] = P["val"], P["test"]
        threshold = (tune_threshold(Ys["val"], P["val_cal"])
                     if comp.kind in ("binary", "multilabel") else None)
        if verbose:
            extra = "" if threshold is None else f"  threshold={threshold:.2f}"
            print(f"    -> {suffix or 'selected'}: calibration="
                  f"{calib['method']}  val ECE "
                  f"{_fmt(calib['val_ece_raw'])} -> {_fmt(calib['val_ece_cal'])}"
                  f"{extra}")
        return {"probs": {k + suffix: v for k, v in P.items()},
                "calibration": calib, "threshold": threshold,
                "calibrator": calibrator, "pipeline": pipe}

    win = finalize(best_pipe)
    probs = dict(win["probs"])
    dep_res = None
    if best_dep != best:
        dep_res = finalize(fitted[best_dep], suffix="_dep")
        probs.update(dep_res["probs"])

    return {
        "component": comp.name, "kind": comp.kind,
        "feature_set": best, "best_deployable": best_dep,
        "hyperparam": {grid[0][0]: best_hp},
        "threshold": win["threshold"], "calibration": win["calibration"],
        "threshold_dep": None if dep_res is None else dep_res["threshold"],
        "calibration_dep": None if dep_res is None else dep_res["calibration"],
        "n_labels": comp.n_labels,
        "labels": list(comp.labels), "label_text": list(comp.label_text),
        "tail_labels": tail.tolist(),
        "rows": {s: int(len(idx[s])) for s in ("train", "val", "test")},
        "selection": tab.to_dict("records"), "hp_search": hp_rows,
        "_pipe": best_pipe, "_calibrator": win["calibrator"],
        "_pipe_dep": None if dep_res is None else dep_res["pipeline"],
        "_calibrator_dep": None if dep_res is None else dep_res["calibrator"],
        "_probs": probs, "_idx": idx, "_Y": Ys, "_tail": tail,
    }


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", nargs="+", default=None)
    ap.add_argument("--arms", nargs="+", default=list(FEATURE_SETS))
    ap.add_argument("--label-space", default="class46",
                    choices=("class46", "cat89"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    import joblib

    df, FF = load_frame()
    comps = build_components(df, label_space=args.label_space)
    names = args.components or list(comps)
    print(f"corpus: {len(df)} encounters  "
          f"train/val/test = "
          f"{(df.split=='train').sum()}/{(df.split=='val').sum()}/{(df.split=='test').sum()}")
    print(f"feature sets: {', '.join(args.arms)}")
    print(f"excluded components: {', '.join(EXCLUDED)}\n")

    manifest, sel_rows = {}, []
    for name in names:
        comp = comps[name]
        print(f"[{name}] kind={comp.kind} labels={comp.n_labels} "
              f"headline={comp.headline}", flush=True)
        res = fit_component(comp, df, FF, args.arms)
        sel_rows += res["selection"]

        tag = f"{name}{args.tag}"
        joblib.dump({"pipeline": res["_pipe"], "calibrator": res["_calibrator"],
                     "threshold": res["threshold"], "kind": comp.kind,
                     "labels": list(comp.labels),
                     "label_text": list(comp.label_text),
                     "feature_set": res["feature_set"],
                     "pipeline_deployable": res["_pipe_dep"],
                     "calibrator_deployable": res["_calibrator_dep"],
                     "threshold_deployable": res["threshold_dep"],
                     "feature_set_deployable": res["best_deployable"],
                     "label_space": args.label_space},
                    MODELS_DIR / f"{tag}.joblib", compress=3)
        np.savez_compressed(
            PRED_DIR / f"{tag}.npz",
            **res["_probs"],
            y_val=res["_Y"]["val"], y_test=res["_Y"]["test"],
            y_train=res["_Y"]["train"], tail=res["_tail"],
            idx_val=res["_idx"]["val"], idx_test=res["_idx"]["test"],
            idx_train=res["_idx"]["train"])
        manifest[name] = {k: v for k, v in res.items() if not k.startswith("_")}
        print()

    pd.DataFrame(sel_rows).to_csv(OUT / f"selection{args.tag}.csv", index=False)
    (OUT / f"pipeline{args.tag}.json").write_text(
        json.dumps({"label_space": args.label_space,
                    "feature_sets": {k: list(v[0]) for k, v in FEATURE_SETS.items()},
                    "deployable": sorted(DEPLOYABLE),
                    "excluded_components": EXCLUDED,
                    "components": manifest}, indent=2, default=str))

    print("=" * 96)
    print(f"{'component':14s} {'winner':16s} {'best deployable':16s} "
          f"{'calib':9s} {'thr':>5s}")
    print("-" * 96)
    for n, m in manifest.items():
        print(f"{n:14s} {m['feature_set']:16s} {m['best_deployable']:16s} "
              f"{m['calibration']['method']:9s} "
              f"{'-' if m['threshold'] is None else format(m['threshold'], '.2f'):>5s}")
    print(f"\nwrote {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
