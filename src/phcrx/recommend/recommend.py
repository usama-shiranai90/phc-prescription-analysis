"""Inference entry point: one encounter in, a ranked suggestion list out.

    python -m src.phcrx.recommend.recommend --demo
    python -m src.phcrx.recommend.recommend --pid 41207
    python -m src.phcrx.recommend.recommend --text "d/m, htn, burning sensation" \
        --age 54 --sex F --bp-sys 150 --bp-dia 95 --glucose 11.2
    python -m src.phcrx.recommend.recommend --verify-policy

Two input paths, and the difference between them is deliberate:

* **`--pid`** scores an encounter that already exists in the corpus, so the
  engineered feature matrix from `features/build_features.py` is available and
  whichever feature set won on validation can be used.
* **`--text/--age/...` (or `--json`)** is a *walk-in*: a symptom note, vitals
  and demographics, and nothing else. The engineered matrix cannot be
  reproduced for a row that was not part of the offline feature build -- it
  contains a train-fitted TF-IDF/SVD basis, retrieved ICD codes and
  leave-one-out site/prescriber target encodings. This path therefore uses the
  best **deployable** model per component, the one selected on validation
  among the feature sets that need only the raw record. `evaluate.py` reports
  what that restriction costs on the test split, per component, so the number
  is measured rather than hand-waved.

Probabilities printed are the **calibrated** ones. Every component prints its
own operating point, and components that `docs/recommender.md` rules unfit to
surface are printed under a separate heading so they cannot be mistaken for
clinical suggestions.

**Abstention.** `--policy` selects the calibrated selective-prediction rule
measured by `abstain.py` and frozen into
`results/rx_generation/recommend/abstention_policy.json`:

    none              recommend on every encounter (the pre-abstention system)
    global80          cover the most confident 80% of encounters
    sparse_targeted   cover every note with a parsed complaint; on a note with
                      none, cover only the most confident 40%   [default]

The decision is per component and per encounter. Confidence is `expected_f1`,
the expected micro-F1 of the set the model would emit, and the cut-point it is
compared against is an **absolute number fixed on validation** -- not a
quantile of whatever batch happens to be in hand, which a deployed recommender
scoring one walk-in patient could not compute anyway.

An abstained component prints `INSUFFICIENT INFORMATION` in place of its
suggestion set. It still prints the ranked probabilities underneath, marked as
audit-only, because withholding a suggestion is not a reason to hide what the
model actually thought.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import MODELS_DIR, OUT
from .abstain import POLICIES, POLICY_JSON, expected_f1, n_spans
from .abstain import decide as policy_decide
from .corpus import RAW_VITALS, build_components, load_frame, split_index

COMPONENT_ORDER = ["any_drug", "n_drugs", "drug_classes", "drug_form",
                   "advice", "tests"]

# The components an abstention can apply to: those that emit a *set*.
SET_COMPONENTS = ("drug_classes", "advice", "tests")
DEFAULT_POLICY = "sparse_targeted"

# What the encounter is told it is not getting.
WITHHELD_NOUN = {"drug_classes": "drug-class", "advice": "advice",
                 "tests": "diagnostic-test"}


# ---------------------------------------------------------------------------
class AbstentionPolicy:
    """The measured decision rule, loaded from the artefact `abstain.py` writes.

    Everything needed to judge a single encounter is in the artefact: the
    confidence-signal definition, the per-component operating threshold the
    signal is defined against, and the coverage cut-points as absolute
    confidence values. Nothing here consults another encounter.
    """

    def __init__(self, name: str = DEFAULT_POLICY, mode: str = "best",
                 path=POLICY_JSON):
        if name not in POLICIES:
            raise SystemExit(f"unknown --policy {name!r}; expected one of "
                             f"{', '.join(sorted(POLICIES))}")
        self.name, self.mode, self.warnings = name, mode, []
        self.meta, self.entries = None, {}
        if name == "none":
            return
        if not path.exists():
            self.warnings.append(
                f"no abstention policy at {path}; run `python -m "
                "src.phcrx.recommend.abstain --policy` to write it. "
                "Falling back to --policy none.")
            self.name = "none"
            return
        self.meta = json.loads(path.read_text(encoding="utf-8"))
        for comp, blk in self.meta.get("components", {}).items():
            e = blk["modes"].get(mode) or blk["modes"].get("best")
            if e is not None and "alias" in e:          # deployable == winner
                e = blk["modes"][e["alias"]]
            self.entries[comp] = e

    # -- provenance guard ---------------------------------------------------
    def check_threshold(self, comp: str, threshold: float) -> None:
        """Refuse to act on a cut-point derived at a different operating point.

        `expected_f1` is defined *relative to* the threshold that decides the
        emitted set, so a cut-point measured at one threshold means nothing at
        another. Covering everything is the safe direction on a stale artefact:
        it degrades to the pre-abstention system rather than withholding on a
        number that no longer refers to anything.
        """
        e = self.entries.get(comp)
        if e is None or abs(float(e["threshold"]) - float(threshold)) <= 1e-9:
            return
        self.warnings.append(
            f"policy artefact for '{comp}' was derived at threshold "
            f"{e['threshold']:.4f} but the loaded model operates at "
            f"{threshold:.4f}; the artefact is stale. Abstention disabled for "
            f"this component -- re-run `abstain.py --policy`.")
        self.entries[comp] = None

    # -- the decision -------------------------------------------------------
    def decide(self, comp: str, P_row: np.ndarray, threshold: float,
               sparse: bool) -> dict:
        conf = float(expected_f1(np.asarray(P_row).reshape(1, -1),
                                 threshold)[0])
        return policy_decide(self.name, self.entries.get(comp), conf, sparse)

    def describe(self) -> str:
        spec = POLICIES[self.name]
        return f"{self.name}: {spec['rule']}"


# ---------------------------------------------------------------------------
class Recommender:
    """The six fitted component pipelines, loaded for inference."""

    def __init__(self, tag: str = "", mode: str = "best",
                 policy: str = DEFAULT_POLICY):
        import joblib
        self.mode = mode
        self.models = {}
        for name in COMPONENT_ORDER:
            f = MODELS_DIR / f"{name}{tag}.joblib"
            if f.exists():
                self.models[name] = joblib.load(f)
        self.policy = AbstentionPolicy(policy, mode=mode)
        for name in SET_COMPONENTS:
            if name in self.models:
                self.policy.check_threshold(name, self._pick(self.models[name])[2])
        for w in self.policy.warnings:
            print(f"  !! {w}")

    def _pick(self, m):
        """(pipeline, calibrator, threshold, feature_set) for the active mode."""
        if self.mode == "deployable" and m.get("pipeline_deployable") is not None:
            return (m["pipeline_deployable"], m["calibrator_deployable"],
                    m["threshold_deployable"], m["feature_set_deployable"])
        return m["pipeline"], m["calibrator"], m["threshold"], m["feature_set"]

    def score(self, X: pd.DataFrame) -> dict:
        out = {}
        for name, m in self.models.items():
            pipe, cal, thr, fset = self._pick(m)
            try:
                if m["kind"] == "count":
                    P = np.clip(pipe.predict(X), 0.0, None).reshape(-1, 1)
                else:
                    P = pipe.predict_proba(X)
                    if cal is not None:
                        P = cal.transform(P)
            except ValueError as e:
                if "columns are missing" not in str(e):
                    raise
                raise ValueError(
                    f"the '{name}' model was selected on the '{fset}' feature "
                    "set, which needs the engineered matrix from "
                    "features/build_features.py. Score an existing encounter "
                    "with --pid, or use Recommender(mode='deployable') for a "
                    "walk-in record.") from e
            out[name] = {"proba": np.atleast_2d(P), "threshold": thr,
                         "labels": m["labels"], "label_text": m["label_text"],
                         "kind": m["kind"], "feature_set": fset}
        return out

    def recommend(self, X: pd.DataFrame, top_k: int = 8) -> list[dict]:
        s = self.score(X)
        # Sparseness is a property of the note, so it is computed once per
        # encounter and shared by every component's decision.
        sparse = [n_spans(t) == 0
                  for t in X["symptom_text"].fillna("").astype(str)]
        recs = []
        for i in range(len(X)):
            r = {"_sparse": bool(sparse[i]), "_policy": self.policy.name}
            if "any_drug" in s:
                p = float(s["any_drug"]["proba"][i, 0])
                r["any_drug"] = {"p": p, "suggest": p >= s["any_drug"]["threshold"],
                                 "threshold": s["any_drug"]["threshold"]}
            if "n_drugs" in s:
                r["n_drugs"] = {"expected": float(s["n_drugs"]["proba"][i, 0])}
            if "drug_form" in s:
                P = s["drug_form"]["proba"][i]
                j = int(P.argmax())
                r["drug_form"] = {"form": s["drug_form"]["labels"][j],
                                  "p": float(P[j])}
            for comp in SET_COMPONENTS:
                if comp not in s:
                    continue
                P = s[comp]["proba"][i]
                thr = s[comp]["threshold"]
                order = np.argsort(-P)[:top_k]
                d = self.policy.decide(comp, P, thr, sparse[i])
                above = [s[comp]["labels"][j] for j in np.where(P >= thr)[0]]
                r[comp] = {
                    "threshold": thr,
                    "abstention": d,
                    "ranked": [{"label": s[comp]["labels"][j],
                                "text": s[comp]["label_text"][j],
                                "p": float(P[j]), "suggest": bool(P[j] >= thr)}
                               for j in order],
                    # `suggested` is what is actually surfaced. On an
                    # abstention it is empty, and what would have been shown
                    # is kept in `withheld` for audit.
                    "suggested": above if d["cover"] else [],
                    "withheld": [] if d["cover"] else above,
                }
            r["_feature_sets"] = {c: s[c]["feature_set"] for c in s}
            recs.append(r)
        return recs


# ---------------------------------------------------------------------------
# raw-encounter input
# ---------------------------------------------------------------------------
def raw_encounter(**kw) -> pd.DataFrame:
    """One-row frame from the fields a clinician has at the consultation."""
    row = {"prescription_id": -1, "symptom_text": kw.get("text", "") or "",
           "age": kw.get("age", np.nan), "sex": kw.get("sex", "NA"),
           "smoker_flag": kw.get("smoker", 0),
           "glucose_type": kw.get("glucose_type", "PBS")}
    for v in RAW_VITALS:
        row[v] = kw.get(v, np.nan)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------
def _bar(p: float, w: int = 12) -> str:
    n = int(round(p * w))
    return "#" * n + "." * (w - n)


def format_recommendation(rec: dict, gold: dict | None = None,
                          top_k: int = 6) -> str:
    L = []
    a = rec.get("any_drug")
    if a:
        g = "" if gold is None else f"   gold: {'yes' if gold['any_drug'] else 'no'}"
        L.append(f"  any pharmacotherapy   p={a['p']:.2f} {_bar(a['p'])}  "
                 f"-> {'PRESCRIBE' if a['suggest'] else 'no drug'}"
                 f" (thr {a['threshold']:.2f}){g}")
    n = rec.get("n_drugs")
    if n:
        g = "" if gold is None else f"   gold: {gold['n_drugs']:.0f}"
        L.append(f"  expected # drugs      {n['expected']:.1f}{g}")
    f = rec.get("drug_form")
    if f:
        g = "" if gold is None else f"   gold: {gold['drug_form']}"
        L.append(f"  dispensing form       {f['form']} (p={f['p']:.2f}){g}")

    titles = {"drug_classes": "DRUG CLASSES", "advice": "ADVICE",
              "tests": "DIAGNOSTIC TESTS"}
    for comp, title in titles.items():
        if comp not in rec:
            continue
        ab = rec[comp].get("abstention") or {}
        declined = not ab.get("cover", True)
        if declined:
            L.append(f"  {title}")
            L.append(f"    INSUFFICIENT INFORMATION -- no "
                     f"{WITHHELD_NOUN[comp]} suggestion "
                     f"(confidence {ab['confidence']:.2f} < "
                     f"{ab['cutpoint']:.2f})")
            L.append(f"      policy '{ab['policy']}': {ab['reason']}")
            L.append("      ranked probabilities below are audit only -- "
                     "nothing here was surfaced:")
        else:
            L.append(f"  {title}  (suggest at p >= {rec[comp]['threshold']:.2f})")
        goldset = set(gold[comp]) if gold else set()
        for e in rec[comp]["ranked"][:top_k]:
            mark = "~" if declined else ("*" if e["suggest"] else " ")
            hit = "  <- in gold" if e["label"] in goldset else ""
            txt = e["text"] if comp != "drug_classes" else e["label"]
            L.append(f"    {mark} {e['p']:.2f} {_bar(e['p'])}  {txt[:78]}{hit}")
        if gold is not None:
            miss = goldset - {e["label"] for e in rec[comp]["ranked"]}
            L.append(f"      gold: {', '.join(sorted(goldset)) or '(none)'}")
            if miss:
                L.append(f"      missed outside top-{len(rec[comp]['ranked'])}: "
                         f"{', '.join(sorted(miss))}")
    return "\n".join(L)


def _gold_for(df, comps, rows, adv_text):
    out = []
    for i in rows:
        g = {"any_drug": bool(comps["any_drug"].Y[i]),
             "n_drugs": float(comps["n_drugs"].Y[i]),
             "drug_form": str(comps["drug_form"].Y[i])
             if comps["drug_form"].mask[i] else "(no drugs)"}
        for comp in SET_COMPONENTS:
            c = comps[comp]
            g[comp] = [c.labels[j] for j in np.where(c.Y[i])[0]]
        out.append(g)
    return out


# ---------------------------------------------------------------------------
def demo(n: int = 10, tag: str = "", seed: int = 0, top_k: int = 6,
         policy: str = DEFAULT_POLICY) -> None:
    df, _ = load_frame()
    comps = build_components(df)
    idx = split_index(df)
    rec = Recommender(tag=tag, policy=policy)

    test = idx["test"]
    rng = np.random.default_rng(seed)

    # Sampling rule, fixed before any prediction is inspected:
    #   * a candidate pool of 400 test encounters drawn at random;
    #   * 2 encounters the policy DECLINES and 2 it COVERS on a sparse note,
    #     so both halves of the behaviour under test are on the page;
    #   * 2 whose gold prescription contains no pharmacotherapy at all, so the
    #     demo also shows what the model does when "prescribe nothing" is
    #     right;
    #   * the remainder text-bearing.
    # Stratifying on the abstention decision is the point of the demo -- it is
    # not a search for a flattering example. Every encounter in each stratum
    # is equally likely to be drawn.
    pool = rng.choice(test, size=min(400, len(test)), replace=False)
    pool_recs = rec.recommend(df.iloc[pool], top_k=8)
    declined = np.array([not r["drug_classes"]["abstention"]["cover"]
                         for r in pool_recs])
    sparse = np.array([r["_sparse"] for r in pool_recs])
    has_text = df["symptom_text"].to_numpy()[pool] != ""
    empty_rx = comps["any_drug"].Y[pool] == 0

    def take(mask, k, chosen):
        cand = pool[mask & ~np.isin(pool, chosen)]
        k = min(k, len(cand))
        return rng.choice(cand, size=k, replace=False) if k else cand[:0]

    pick = np.array([], dtype=test.dtype)
    for mask, k in ((declined & sparse, 2), (~declined & sparse, 2),
                    (empty_rx, 2), (has_text & ~declined, n)):
        pick = np.concatenate([pick, take(mask, k, pick)])
        if len(pick) >= n:
            break
    pick = np.unique(pick)[:n]

    X = df.iloc[pick]
    recs = rec.recommend(X, top_k=max(top_k, 8))
    golds = _gold_for(df, comps, pick, None)
    n_declined = sum(not r["drug_classes"]["abstention"]["cover"] for r in recs)

    print("=" * 100)
    print(f"WORKED TEST-SET EXAMPLES  ({len(pick)} encounters, "
          f"sampled with seed {seed} from the held-out test split)")
    print(f"feature set in use per component: "
          f"{json.dumps(recs[0]['_feature_sets'])}")
    print(f"abstention policy: {rec.policy.describe()}")
    if rec.policy.name == "none":
        print("  abstention is off, so nothing here is declined; the sample "
              "is 2 sparse-note,\n  2 no-pharmacotherapy and the rest "
              "text-bearing encounters.")
    else:
        print(f"  {n_declined} of {len(pick)} encounters are DECLINED for "
              "drug classes. The sample is stratified on the abstention "
              "decision\n  (2 declined, 2 covered-sparse, 2 "
              "no-pharmacotherapy, rest text-bearing) so both behaviours are "
              "visible; within\n  each stratum the draw is uniform.")
    print("  '*' = above the validation-tuned operating point, i.e. actually "
          "surfaced to the clinician")
    if rec.policy.name != "none":
        print("  '~' = the component abstained; the row is audit output, not "
              "a suggestion")
    print("=" * 100)
    for k, (i, r, g) in enumerate(zip(pick, recs, golds), 1):
        row = df.iloc[i]
        vit = []
        if not pd.isna(row["bp_sys"]):
            vit.append(f"BP {row['bp_sys']:.0f}/{row['bp_dia']:.0f}")
        if not pd.isna(row["blood_glucose"]):
            vit.append(f"glucose {row['blood_glucose']:.1f} ({row['glucose_type']})")
        if not pd.isna(row["bmi"]):
            vit.append(f"BMI {row['bmi']:.1f}")
        age = "?" if pd.isna(row["age"]) else f"{row['age']:.0f}"
        print(f"\n[{k}] pid={row['prescription_id']}  {age}y {row['sex']}  "
              f"{'  '.join(vit)}"
              f"{'   [sparse note]' if r['_sparse'] else ''}")
        note = row["symptom_text"].strip() or "(no symptom note recorded)"
        print(f"    note: {note[:150]}")
        print(format_recommendation(r, gold=g, top_k=top_k))
    print("\n" + "=" * 100)
    print("NOT SURFACED: dose and duration. Measured at accuracy 0.506 / 0.503 "
          "against a strong\nmajority class and worse without prescriber "
          "identity -- they are prescriber habit, not\nclinical signal "
          "(docs/feature_importance.md, docs/recommender.md).")


# ---------------------------------------------------------------------------
# does the single-encounter path reproduce the batch analysis?
# ---------------------------------------------------------------------------
def verify_policy(tag: str = "", mode: str = "best", top_k: int = 8) -> dict:
    """Run every policy over the whole test split *through inference*.

    The point is not to re-measure the risk-coverage curve -- `abstain.py`
    did that -- but to check that the deployed rule, which judges one
    encounter at a time against a frozen cut-point, lands where the batch
    analysis said it would.
    """
    from .metrics import set_metrics, tail_labels

    df, _ = load_frame()
    comps = build_components(df)
    idx = split_index(df)
    te, tr = idx["test"], idx["train"]
    Xte = df.iloc[te]

    spans = np.array([n_spans(t) for t in
                      Xte["symptom_text"].fillna("").astype(str)])

    report = {"mode": mode, "n_test": int(len(te)), "policies": {}}
    for pol in ("none", "global80", "sparse_targeted"):
        rec = Recommender(tag=tag, mode=mode, policy=pol)
        recs = rec.recommend(Xte, top_k=top_k)
        sparse = np.array([r["_sparse"] for r in recs])
        block = {"sparse_test_rows": int(sparse.sum()), "components": {}}
        for comp in SET_COMPONENTS:
            if comp not in rec.models:
                continue
            c = comps[comp]
            Y = np.asarray(c.Y)[te]
            tail = tail_labels(np.asarray(c.Y)[tr])
            lab2col = {v: j for j, v in enumerate(c.labels)}
            # Reconstruct the *surfaced* set exactly as a caller would see it.
            Pred = np.zeros_like(Y, dtype=np.int8)
            for i, r in enumerate(recs):
                for lab in r[comp]["suggested"]:
                    Pred[i, lab2col[lab]] = 1
            cover = np.array([r[comp]["abstention"]["cover"] for r in recs])
            conf = np.array([r[comp]["abstention"]["confidence"] for r in recs])

            def m(sel):
                return (set_metrics(Y[sel], Pred[sel], tail)["micro_f1"]
                        if sel.sum() else float("nan"))

            # The exact wiring test. `abstain.py` selects rows by *ranking*
            # the batch and taking the top fraction; inference compares each
            # row on its own against a frozen number. Over the population the
            # cut-point actually ranks, those two must pick the identical set
            # of rows -- if they do, the only thing separating the deployed
            # policy from the published curve is *where on the curve* it
            # lands, not how it got there. A mismatch means the per-encounter
            # path is not the analysis.
            #
            # `sparse_targeted` ranks only the sparse subgroup; it covers rich
            # notes unconditionally, so a global top-k comparison there is
            # meaningless by construction rather than a failure.
            scope = "all" if pol == "global80" else "sparse"
            rank_eq, n_diff = {}, {}
            for sc, sel in (("all", np.ones(len(te), bool)),
                            ("sparse", sparse)):
                k = int(cover[sel].sum())
                top = np.zeros(int(sel.sum()), bool)
                top[np.argsort(-conf[sel], kind="stable")[:k]] = True
                rank_eq[sc] = bool((top == cover[sel]).all())
                n_diff[sc] = int((top != cover[sel]).sum())

            block["components"][comp] = {
                "rank_check_scope": scope,
                "rank_equivalent_to_batch_topk": rank_eq,
                "rows_differing_from_batch_topk": n_diff,
                "coverage": float(cover.mean()),
                "n_covered": int(cover.sum()),
                "micro_f1_on_covered": m(cover),
                "micro_f1_all_rows_as_surfaced": m(np.ones(len(te), bool)),
                "sparse": {
                    "coverage": float(cover[sparse].mean()),
                    "n_covered": int(cover[sparse].sum()),
                    "micro_f1_on_covered": m(cover & sparse),
                },
                "rich": {
                    "coverage": float(cover[~sparse].mean()),
                    "n_covered": int(cover[~sparse].sum()),
                    "micro_f1_on_covered": m(cover & ~sparse),
                },
                "cutpoint": next((r[comp]["abstention"]["cutpoint"]
                                  for r in recs
                                  if r[comp]["abstention"]["cutpoint"]
                                  is not None), None),
                "mean_confidence": float(conf.mean()),
                # Nothing tells the policy how many complaints a note has;
                # this is what it does with that information withheld.
                "abstained_by_complaint_count": {
                    lab: {"n": int(msk.sum()),
                          "abstained": float(1 - cover[msk].mean())}
                    for lab, msk in (("0", spans == 0), ("1", spans == 1),
                                     ("2", spans == 2), ("3+", spans >= 3))
                    if msk.sum()},
            }
        report["policies"][pol] = block

    # --- what the analysis said, for a side-by-side ------------------------
    # Two references, and both are worth printing:
    #   abstention.json         the published section-8 curve, measured on an
    #                           arm abstain.py refits for itself (C=4.0);
    #   abstention_served.json  the same analysis run on the hyper-parameter
    #                           the served model actually uses (C=1.0), which
    #                           is the like-for-like comparison.
    refs = {}
    for key, fname, note in (
            ("published", "abstention.json",
             "docs/remaining_work.md section 8. Measured on an arm abstain.py "
             "refits for itself (MultiLabelOvR C=4.0, threshold 0.21); the "
             "served drug_classes model is the pipeline.py winner (C=1.0, "
             "threshold 0.24). Absolute micro-F1 must differ -- it is a "
             "different model -- so the lift is the comparable quantity."),
            ("served_model", "abstention_served.json",
             "the same batch analysis at the served model's hyper-parameter "
             "(abstain.py --analysis --C 1.0 --out-tag _served). Like for "
             "like: any residual gap is the val-frozen cut-point landing at a "
             "slightly different coverage than the test quantile the batch "
             "analysis takes by construction.")):
        p = OUT / fname
        if not p.exists():
            continue
        a = json.loads(p.read_text(encoding="utf-8"))
        rc = {round(r["coverage"], 2): r["micro_f1"] for r in a["test_risk_coverage"]}
        sp = {round(r["coverage"], 2): r["micro_f1"] for r in a["sparse_policy"]}
        refs[key] = {
            "file": str(p), "arm": a["arm"], "C": a.get("C"),
            "threshold": a["threshold"], "note": note,
            "drug_classes_micro_f1_at_coverage": {"1.00": rc.get(1.0),
                                                  "0.80": rc.get(0.8)},
            "sparse_micro_f1_at_coverage": {"1.00": sp.get(1.0),
                                            "0.40": sp.get(0.4)},
            "rich_baseline_micro_f1": a["rich_baseline_micro_f1"],
        }
    report["analysis_reference"] = refs

    (OUT / "abstention_wired.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")

    # --- print -------------------------------------------------------------
    print("=" * 96)
    print(f"WIRED POLICY over the full test split ({len(te)} encounters), "
          f"through Recommender.recommend()")
    print(f"mode={mode}   sparse test rows="
          f"{report['policies']['none']['sparse_test_rows']}")
    print("=" * 96)
    hdr = (f"{'policy':16s} {'component':13s} {'cov':>6s} {'n_cov':>6s} "
           f"{'F1|cov':>8s} {'sp cov':>7s} {'sp F1':>7s} {'rich F1':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for pol, blk in report["policies"].items():
        for comp, d in blk["components"].items():
            print(f"{pol:16s} {comp:13s} {d['coverage']:6.3f} "
                  f"{d['n_covered']:6d} {d['micro_f1_on_covered']:8.4f} "
                  f"{d['sparse']['coverage']:7.3f} "
                  f"{d['sparse']['micro_f1_on_covered']:7.4f} "
                  f"{d['rich']['micro_f1_on_covered']:8.4f}")
    dc = report["policies"]
    print("\n--- abstention rate by complaint count (drug_classes) ---")
    for pol in ("global80", "sparse_targeted"):
        d = dc[pol]["components"].get("drug_classes", {})
        cells = "  ".join(
            f"spans {k}: {v['abstained']:.1%} (n={v['n']})"
            for k, v in d.get("abstained_by_complaint_count", {}).items())
        print(f"   {pol:16s} {cells}")

    print("\n--- does the per-encounter rule pick the same rows as the batch "
          "ranking, over the population it ranks? ---")
    for pol in ("global80", "sparse_targeted"):
        for comp, d in dc[pol]["components"].items():
            sc = d["rank_check_scope"]
            eq = d["rank_equivalent_to_batch_topk"][sc]
            nd = d["rows_differing_from_batch_topk"][sc]
            print(f"   {pol:16s} {comp:13s} scope={sc:6s} "
                  f"{'IDENTICAL' if eq else 'MISMATCH'} "
                  f"(rows differing: {nd})")
    print("   (sparse_targeted covers rich notes unconditionally, so its "
          "ranked population is the sparse subgroup only)")

    for key, ref in report["analysis_reference"].items():
        print(f"\n--- batch analysis '{key}' (C={ref['C']}, thr="
              f"{ref['threshold']:.2f}) vs wired inference ---")
        print(f"{'quantity':38s} {'analysis':>10s} {'wired':>10s} "
              f"{'delta':>9s}")
        rows = [
            ("drug_classes, 100% coverage",
             ref["drug_classes_micro_f1_at_coverage"]["1.00"],
             dc["none"]["components"]["drug_classes"]["micro_f1_on_covered"]),
            ("drug_classes, global80 covered rows",
             ref["drug_classes_micro_f1_at_coverage"]["0.80"],
             dc["global80"]["components"]["drug_classes"]["micro_f1_on_covered"]),
            ("sparse rows, no abstention",
             ref["sparse_micro_f1_at_coverage"]["1.00"],
             dc["none"]["components"]["drug_classes"]["sparse"]["micro_f1_on_covered"]),
            ("sparse rows, sparse_targeted",
             ref["sparse_micro_f1_at_coverage"]["0.40"],
             dc["sparse_targeted"]["components"]["drug_classes"]["sparse"]["micro_f1_on_covered"]),
            ("rich rows, no abstention",
             ref["rich_baseline_micro_f1"],
             dc["none"]["components"]["drug_classes"]["rich"]["micro_f1_on_covered"]),
        ]
        for lab, x, y in rows:
            print(f"{lab:38s} {x:10.4f} {y:10.4f} {y - x:+9.4f}")
        lift_a = (ref["drug_classes_micro_f1_at_coverage"]["0.80"]
                  - ref["drug_classes_micro_f1_at_coverage"]["1.00"])
        lift_w = (dc["global80"]["components"]["drug_classes"]["micro_f1_on_covered"]
                  - dc["none"]["components"]["drug_classes"]["micro_f1_on_covered"])
        print(f"   global80 lift over full coverage: analysis {lift_a:+.4f}, "
              f"wired {lift_w:+.4f}")
        print(f"   {ref['note']}")
    print(f"\nwrote {OUT / 'abstention_wired.json'}")
    return report


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--verify-policy", action="store_true",
                    help="run every policy over the whole test split through "
                         "the inference path and compare with abstain.py")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--policy", default=DEFAULT_POLICY,
                    choices=sorted(POLICIES),
                    help="calibrated abstention policy (default: "
                         f"{DEFAULT_POLICY})")
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--json", default=None,
                    help="raw encounter as a JSON object")
    ap.add_argument("--text", default=None)
    ap.add_argument("--age", type=float, default=None)
    ap.add_argument("--sex", default="NA")
    ap.add_argument("--bp-sys", type=float, default=None)
    ap.add_argument("--bp-dia", type=float, default=None)
    ap.add_argument("--glucose", type=float, default=None)
    ap.add_argument("--bmi", type=float, default=None)
    ap.add_argument("--pulse", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    args = ap.parse_args()

    if args.verify_policy:
        verify_policy(tag=args.tag)
        return

    if args.demo:
        demo(args.n, args.tag, args.seed, args.top_k, args.policy)
        return

    if args.pid is not None:
        df, _ = load_frame()
        comps = build_components(df)
        hit = np.where(df["prescription_id"].to_numpy() == args.pid)[0]
        if not len(hit):
            raise SystemExit(f"prescription_id {args.pid} not in the corpus")
        i = int(hit[0])
        rec = Recommender(tag=args.tag, policy=args.policy)
        r = rec.recommend(df.iloc[[i]], top_k=max(args.top_k, 8))[0]
        g = _gold_for(df, comps, [i], None)[0]
        row = df.iloc[i]
        print(f"pid={args.pid}  split={row['split']}  "
              f"note: {row['symptom_text'][:150]!r}")
        print(f"  abstention policy: {rec.policy.describe()}")
        print(format_recommendation(r, gold=g, top_k=args.top_k))
        return

    kw = json.loads(args.json) if args.json else {}
    kw.setdefault("text", args.text or "")
    for k, v in (("age", args.age), ("sex", args.sex), ("bp_sys", args.bp_sys),
                 ("bp_dia", args.bp_dia), ("blood_glucose", args.glucose),
                 ("bmi", args.bmi), ("pulse_rate", args.pulse),
                 ("temperature", args.temperature)):
        if v is not None and k not in kw:
            kw[k] = v
    X = raw_encounter(**kw)
    rec = Recommender(tag=args.tag, mode="deployable", policy=args.policy)
    r = rec.recommend(X, top_k=max(args.top_k, 8))[0]
    print("walk-in encounter (deployable models: no engineered feature matrix)")
    print(f"  note: {kw.get('text', '')!r}")
    print(f"  feature sets: {json.dumps(r['_feature_sets'])}")
    print(f"  abstention policy: {rec.policy.describe()}")
    print(format_recommendation(r, gold=None, top_k=args.top_k))


if __name__ == "__main__":
    main()
