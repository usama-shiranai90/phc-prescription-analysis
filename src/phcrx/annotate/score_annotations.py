"""Score returned clinician annotations.

Converts the study's agreement figures into accuracy figures, and tests the
prescription model against the attending clinician rather than against a string
match with history.

Analyses
--------
Task A (ICD)
  * accuracy of the automated coder against clinician codes, exact and
    same-chapter, reported per confidence tier
  * whether the confidence gate is calibrated: is the withheld low-confidence
    tier genuinely less codable, or is the gate simply too tight?
  * inter-rater reliability (Cohen's kappa) on the double-annotated overlap,
    which bounds the accuracy any automated coder could show

Task B (prescription)
  * blinded preference for model vs historical prescription, with an exact
    binomial CI, tested for NON-INFERIORITY rather than superiority
  * independent safety ratings, including the rate of prescriptions judged
    unsafe -- the number that actually gates deployment
  * inter-rater reliability on preference

    python -m src.phcrx.annotate.score_annotations \
        --files results/rx_generation/annotation/annotations_*.json
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict

import numpy as np

from ..config import RESULTS

OUT = RESULTS / "annotation"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — reliable at small n and near 0/1."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def cohen_kappa(a: list, b: list) -> float:
    labels = sorted({*a, *b})
    if len(labels) < 2:
        return float("nan")
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(0) @ m.sum(1)) / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def load(files: list[str]) -> dict[str, dict]:
    out = {}
    for f in files:
        d = json.loads(open(f, encoding="utf-8").read())
        out[d.get("annotator", f)] = d.get("answers", {})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+",
                    default=sorted(glob.glob(str(OUT / "annotations_*.json"))))
    args = ap.parse_args()

    if not args.files:
        raise SystemExit(
            f"No annotation files found in {OUT}.\n"
            "Expected annotations_<annotator>.json exported from the tool.")

    key_blob = json.loads(
        (OUT / "KEY_do_not_share_with_annotators.json").read_text())
    key = {k["ann_id"]: k for k in key_blob["key"]}
    ann = load(args.files)
    print(f"annotators: {list(ann)}  files: {len(args.files)}")

    report: dict = {"n_annotators": len(ann)}

    # ---------------- Task A : ICD accuracy ------------------------------
    per_tier = defaultdict(lambda: {"n": 0, "exact": 0, "chapter": 0})
    codability = Counter()
    for who, answers in ann.items():
        for aid, a in answers.items():
            k = key.get(aid)
            if not k or k["task"] != "icd":
                continue
            tier = k["tier"]
            human = (a.get("icd1") or "").strip().upper().split(" ")[0]
            said_none = a.get("no_code") == "yes"
            codability[a.get("codability") or "unset"] += 1
            model = (k.get("model_icd") or "").upper()
            if tier != "confident":
                # Nothing was shipped for these; the question is whether the
                # clinician could code them at all.
                per_tier[tier]["n"] += 1
                per_tier[tier]["exact"] += int(bool(human) and not said_none)
                continue
            if said_none or not human:
                per_tier[tier]["n"] += 1
                continue
            per_tier[tier]["n"] += 1
            per_tier[tier]["exact"] += int(human == model)
            per_tier[tier]["chapter"] += int(bool(model) and human[:1] == model[:1])

    print("\n" + "=" * 70)
    print("TASK A — automated ICD coding vs clinician")
    c = per_tier.get("confident", {"n": 0, "exact": 0, "chapter": 0})
    if c["n"]:
        lo, hi = wilson(c["exact"], c["n"])
        lo2, hi2 = wilson(c["chapter"], c["n"])
        print(f"  confident tier (shipped), n={c['n']}")
        print(f"    exact-code accuracy   : {c['exact']/c['n']:.1%} "
              f"[95% CI {lo:.1%}–{hi:.1%}]")
        print(f"    same-chapter accuracy : {c['chapter']/c['n']:.1%} "
              f"[95% CI {lo2:.1%}–{hi2:.1%}]")
        report["icd_confident"] = {"n": c["n"], "exact": c["exact"] / c["n"],
                                   "chapter": c["chapter"] / c["n"],
                                   "exact_ci": [lo, hi]}
    for tier in ("low_confidence", "no_complaint"):
        t = per_tier.get(tier)
        if t and t["n"]:
            r = t["exact"] / t["n"]
            lo, hi = wilson(t["exact"], t["n"])
            print(f"  {tier} tier (withheld), n={t['n']}: clinician assigned a "
                  f"code in {r:.1%} [95% CI {lo:.1%}–{hi:.1%}]")
            report[f"icd_{tier}_codable"] = {"n": t["n"], "rate": r}
    print(f"  clinician codability judgement: {dict(codability)}")
    print("  → a high codable-rate in the withheld tier means the gate is too "
          "tight;\n    a low rate means the withholding was correct.")

    # ---------------- Task B : blinded preference ------------------------
    pref = Counter()
    safety = {"model": Counter(), "historical": Counter()}
    differ = Counter()
    for who, answers in ann.items():
        for aid, a in answers.items():
            k = key.get(aid)
            if not k or k["task"] != "rx":
                continue
            p = a.get("preference")
            if p in ("1", "2"):
                pref[k[f"option{p}_is"]] += 1
            elif p:
                pref[p] += 1
            for opt in ("1", "2"):
                s = a.get(f"safety{opt}")
                if s:
                    safety[k[f"option{opt}_is"]][s] += 1
            if a.get("would_differ"):
                differ[a["would_differ"]] += 1

    print("\n" + "=" * 70)
    print("TASK B — blinded model vs historical prescription")
    total = sum(pref.values())
    if total:
        m, h = pref.get("model", 0), pref.get("historical", 0)
        eq, bad = pref.get("equal", 0), pref.get("both_bad", 0)
        print(f"  n={total}  model preferred {m}, historical {h}, "
              f"equivalent {eq}, both inappropriate {bad}")
        head = m + h
        if head:
            lo, hi = wilson(m, head)
            print(f"  head-to-head model win rate: {m/head:.1%} "
                  f"[95% CI {lo:.1%}–{hi:.1%}]  (n={head}, ties excluded)")
            # Non-inferiority: is the model within 10 points of parity?
            verdict = ("NON-INFERIOR (CI lower bound above 40%)" if lo > 0.40
                       else "not established")
            print(f"  non-inferiority at a 10-point margin: {verdict}")
            report["rx_preference"] = {"model": m, "historical": h, "equal": eq,
                                       "both_bad": bad, "win_rate": m / head,
                                       "ci": [lo, hi]}
    for src in ("model", "historical"):
        s = safety[src]
        n = sum(s.values())
        if n:
            unsafe = s.get("unsafe", 0)
            lo, hi = wilson(unsafe, n)
            print(f"  {src:11s} safety: appropriate {s.get('appropriate',0)}, "
                  f"suboptimal {s.get('suboptimal',0)}, unsafe {unsafe}"
                  f"  → unsafe rate {unsafe/n:.1%} [95% CI {lo:.1%}–{hi:.1%}]")
            report[f"safety_{src}"] = {"n": n, "unsafe_rate": unsafe / n,
                                       "unsafe_ci": [lo, hi]}
    if differ:
        print(f"  clinician would have prescribed differently: {dict(differ)}")

    # ---------------- inter-rater reliability ----------------------------
    print("\n" + "=" * 70)
    print("INTER-RATER RELIABILITY (double-annotated overlap)")
    names = list(ann)
    if len(names) >= 2:
        a1, a2 = ann[names[0]], ann[names[1]]
        shared = sorted(set(a1) & set(a2))
        for field, task, label in (("icd1", "icd", "ICD primary code"),
                                   ("preference", "rx", "prescription preference"),
                                   ("codability", "icd", "codability judgement")):
            xs, ys = [], []
            for aid in shared:
                k = key.get(aid)
                if not k or k["task"] != task:
                    continue
                x = (a1[aid].get(field) or "").strip().upper()
                y = (a2[aid].get(field) or "").strip().upper()
                if x and y:
                    xs.append(x); ys.append(y)
            if len(xs) >= 10:
                agree = np.mean([x == y for x, y in zip(xs, ys)])
                kap = cohen_kappa(xs, ys)
                print(f"  {label:26s} n={len(xs):3d}  agreement {agree:.1%}  "
                      f"kappa {kap:.3f}")
                report[f"irr_{field}"] = {"n": len(xs), "agreement": float(agree),
                                          "kappa": kap}
            else:
                print(f"  {label:26s} n={len(xs):3d}  (too few to score)")
        print("\n  Clinician-vs-clinician agreement is the ceiling: an automated"
              "\n  coder cannot meaningfully exceed the reliability of the "
              "reference\n  standard it is measured against.")
    else:
        print("  Only one annotator returned — reliability cannot be estimated,"
              "\n  and Task A accuracy has no ceiling to be read against.")

    (OUT / "annotation_results.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")
    print("\nwrote", OUT / "annotation_results.json")


if __name__ == "__main__":
    main()
