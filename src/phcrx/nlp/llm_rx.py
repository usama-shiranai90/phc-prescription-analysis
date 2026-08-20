"""Local-LLM prescription generation, scored against PHC-RxGen.

The question a reviewer will ask: does a purpose-trained 5M-parameter model
still earn its place when a 4B medical LLM can be prompted for free? This
answers it on identical test encounters with identical metrics.

The LLM is constrained to the site formulary (the drugs actually stocked and
prescribed in this corpus). Without that constraint it prescribes generic
international brands that score zero by construction, which would be an unfair
comparison -- the task is to reproduce *this site's* prescribing, not to write
a globally reasonable prescription.

Runs entirely on the local Ollama server; no patient data leaves the machine.

    python -m src.phcrx.nlp.llm_rx --n 150 --model medgemma:latest
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter

import numpy as np
import pandas as pd

from ..config import Config, PROCESSED, RESULTS, MODELS, PAD, BOS, EOS
from ..data import RxCorpus, RxDataset
from .. import metrics as M
from .glossary import prompt_block
from .ollama_client import Ollama, OllamaError

OUT = RESULTS / "nlp"
OUT.mkdir(parents=True, exist_ok=True)

# The first version of this prompt ended with "Many screening visits need no
# medication at all -- when the patient has no treatable complaint, prescribe
# nothing." medgemma then declined to prescribe on 96% of encounters, including
# one with a recorded glucose of 410 mg/dL whose own stated reason was
# "indicates diabetes mellitus". Measuring refusal that the prompt itself
# induced is not a measurement of the model, so the nudge is removed and the
# base rate is left to the few-shot exemplars.
SYSTEM = (
    "You are the prescribing physician in a rural primary-care telemedicine "
    "programme in Bangladesh. You write the prescription for each encounter "
    "using the site formulary. Treat what the note and vitals show, the way "
    "the clinicians at this site do."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "drugs": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "reason": {"type": "string"},
    },
    "required": ["drugs"],
}


def formulary(corpus: RxCorpus, top_n: int = 150) -> list[tuple[str, str]]:
    """Most-prescribed brands with their pharmacological class, for the prompt."""
    o = corpus.orders
    freq = Counter(o["drug_id"].dropna().astype(int))
    names = dict(zip(o["drug_id"].astype("Int64"), o["drug_name"]))
    cats = corpus.vocab.get("category_names", {})
    id2cat = {v: k for k, v in corpus.cat_vocab.items()}
    rows = []
    for did, _ in freq.most_common(top_n):
        nm = names.get(did)
        if not isinstance(nm, str) or not nm.strip():
            continue
        c = corpus.drug2cat.get(int(did), 0)
        cname = cats.get(str(id2cat.get(c, "")), "")
        rows.append((nm.strip(), cname))
    return rows


def build_shots(corpus: RxCorpus, k: int, seed: int = 0) -> str:
    """Few-shot exemplars drawn from TRAIN only (never from the test split).

    Zero-shot is not a fair test of an LLM on a site-specific imitation task:
    the model has no way to learn local prescribing habits from a formulary
    list alone. These exemplars show it what this site actually does.
    """
    if k <= 0:
        return ""
    ds = RxDataset(corpus, "train")
    rows = ds.rows[ds.rows["symptom_text"].fillna("").str.strip() != ""]
    rows = rows.sample(min(k * 6, len(rows)), random_state=seed)
    names = dict(zip(corpus.orders["drug_id"].astype("Int64"),
                     corpus.orders["drug_name"]))

    prescribing, empty = [], []
    for _, r in rows.iterrows():
        g = corpus.rx_orders.get(int(r["prescription_id"]))
        drugs = ([names.get(int(d)) for d in g["drug_id"].dropna().astype(int)][:4]
                 if g is not None else [])
        drugs = [d for d in drugs if isinstance(d, str)]
        age = f"{r['age']:.0f}" if pd.notna(r.get("age")) else "?"
        item = (f'Note: "{r["symptom_text"][:90]}" (age {age}, '
                f'{r.get("sex") or "?"})\n  -> {json.dumps({"drugs": drugs})}')
        (prescribing if drugs else empty).append(item)

    # Hold the empty-prescription share near the corpus rate (21.5%) so the
    # exemplars neither hide nor exaggerate the option to prescribe nothing.
    n_empty = max(1, round(k * 0.215)) if empty else 0
    picked = prescribing[: k - n_empty] + empty[:n_empty]
    if not picked:
        return ""
    # Interleave, and end on a PRESCRIBING example. A 4B model anchors hard on
    # the final exemplar: with an empty one last, medgemma collapsed to
    # near-total non-prescribing (0.07 drugs/encounter) and echoed that
    # exemplar's wording as its stated reason.
    head = picked[:-1]
    rng = np.random.default_rng(seed)
    rng.shuffle(head)
    ordered = head + [prescribing[k - n_empty] if len(prescribing) > k - n_empty
                      else prescribing[-1]]
    return "\nEXAMPLES OF PRESCRIBING AT THIS SITE:\n" + "\n".join(ordered) + "\n"


def build_prompt(row, form: list[tuple[str, str]], shots: str = "") -> str:
    lines = []
    for nm, cls in form:
        lines.append(f"  {nm}" + (f"  [{cls}]" if cls else ""))
    vit = ", ".join(
        f"{lab}={float(row[c]):.0f}" for c, lab in
        (("bp_sys", "BP-sys"), ("bp_dia", "BP-dia"), ("blood_glucose", "glucose"),
         ("bmi", "BMI"), ("pulse_rate", "pulse"), ("temperature", "temp-F"),
         ("blood_hemoglobin", "Hb"))
        if pd.notna(row.get(c)))
    age = f"{row['age']:.0f}" if pd.notna(row.get("age")) else "unknown"
    return f"""{prompt_block()}

PATIENT
  Age: {age}   Sex: {row.get('sex') or 'unknown'}
  Presenting note: "{row.get('symptom_text') or '(none recorded)'}"
  Vitals: {vit or '(none recorded)'}

SITE FORMULARY (prescribe only from this list, using these exact names):
{chr(10).join(lines)}
{shots}
Write the prescription for this encounter.
Rules:
- Use ONLY drug names from the formulary above, spelled exactly as listed.
- Most prescriptions here contain 1-4 drugs; about 1 in 5 encounters gets none.
- Return an empty list only when the note records no treatable complaint.
Return JSON: {{"drugs": [...], "reason": "..."}}"""


def name_resolver(corpus: RxCorpus):
    """Map a generated brand string back to a drug vocabulary id."""
    o = corpus.orders
    lookup: dict[str, int] = {}
    for did, nm in zip(o["drug_id"].astype("Int64"), o["drug_name"]):
        if pd.isna(did) or not isinstance(nm, str):
            continue
        key = re.sub(r"[^a-z0-9]", "", nm.lower())
        if key:
            lookup.setdefault(key, int(did))

    def resolve(s: str):
        if not isinstance(s, str):
            return None
        key = re.sub(r"[^a-z0-9]", "", s.lower())
        if not key:
            return None
        did = lookup.get(key)
        if did is None:  # tolerate minor spelling drift
            for k, v in lookup.items():
                if k.startswith(key[:6]) and abs(len(k) - len(key)) <= 3:
                    did = v
                    break
        return corpus.drug_vocab.get(did) if did is not None else None

    return resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--model", default="medgemma:latest")
    ap.add_argument("--formulary", type=int, default=150)
    ap.add_argument("--shots", type=int, default=0,
                    help="few-shot exemplars drawn from the train split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="llm_rx_baseline.json")
    args = ap.parse_args()

    cfg = Config()
    corpus = RxCorpus(cfg.data)
    ds = RxDataset(corpus, "test")
    rows = ds.rows.sample(min(args.n, len(ds.rows)), random_state=args.seed)
    form = formulary(corpus, args.formulary)
    resolve = name_resolver(corpus)
    llm = Ollama(model=args.model, num_predict=300, timeout=240)

    shots = build_shots(corpus, args.shots, args.seed)
    print(f"model={args.model}  encounters={len(rows)}  "
          f"formulary={len(form)} drugs  shots={args.shots}")

    preds, golds, raw, t0 = [], [], [], time.time()
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        pid = int(r["prescription_id"])
        g = corpus.rx_orders.get(pid)
        gold = ([corpus.drug_vocab.get(int(d), 3)
                 for d in g["drug_id"].dropna().astype(int)][: cfg.data.max_rx_len]
                if g is not None else [])
        parsed = None
        try:
            parsed = llm.generate_json(build_prompt(r, form, shots), SCHEMA,
                                       system=SYSTEM)
        except OllamaError as e:
            parsed = {"drugs": [], "reason": f"error: {e}", "_failed": True}
        # A call that returned nothing parseable is a FAILURE, not a decision to
        # withhold treatment. Conflating the two is what made gpt-oss score a
        # clean 0.000 as if it had deliberately prescribed nothing 150 times.
        failed = parsed is None or parsed.get("_failed", False)
        res = parsed or {}
        names = res.get("drugs") or []
        ids, unresolved = [], []
        for nm in names:
            vid = resolve(nm)
            (ids.append(vid) if vid is not None else unresolved.append(nm))
        ids = list(dict.fromkeys(ids))
        preds.append(ids)
        golds.append(gold)
        raw.append({"prescription_id": pid,
                    "symptom": (r.get("symptom_text") or "")[:120],
                    "llm_names": names, "resolved": ids,
                    "unresolved": unresolved, "failed": failed,
                    "reason": (res.get("reason") or "")[:160]})
        if i % 25 == 0:
            print(f"   … {i}/{len(rows)}  ({(time.time()-t0)/i:.1f}s/case)", flush=True)
            # Incremental checkpoint. A gpt-oss run is ~70 minutes; losing it to
            # a downstream formatting error (which has already happened once) is
            # not acceptable, so raw results hit disk as they are produced.
            (OUT / (args.out + ".partial")).write_text(
                json.dumps({"model": args.model, "done": i, "examples": raw},
                           indent=1, default=float), encoding="utf-8")

    res_llm = M.set_metrics(preds, golds)
    res_llm.update(M.empty_rx_metrics(preds, golds))
    v2c = corpus.vid2cat
    to_cat = lambda s: [[v2c.get(d, 0) for d in x if v2c.get(d, 0) != 0] for x in s]
    res_llm["cat_micro_f1"] = M.set_metrics(to_cat(preds), to_cat(golds))["micro_f1"]

    # --- PHC-RxGen on the identical encounters ---------------------------
    res_nn = None
    ckpt = MODELS / "rxgen_full_patient_seed0.pt"
    if ckpt.exists():
        import torch
        from torch.utils.data import DataLoader, Subset
        from .predict_helper import load_rxgen  # local import keeps deps light
        model, device = load_rxgen(ckpt, corpus)
        # ds.rows carries a RangeIndex, so the sampled labels are already the
        # positional indices the Subset needs.
        loader = DataLoader(Subset(ds, rows.index.tolist()), batch_size=32)
        p2, g2 = [], []
        with torch.no_grad():
            for b in loader:
                b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
                gen = model.generate(b, max_len=cfg.data.max_rx_len)
                p2.extend(gen["drugs"])
                for j, n in enumerate(b["n_drugs"].tolist()):
                    g2.append(b["drug_out"][j, :n].tolist())
        res_nn = M.set_metrics(p2, g2)
        res_nn.update(M.empty_rx_metrics(p2, g2))
        res_nn["cat_micro_f1"] = M.set_metrics(to_cat(p2), to_cat(g2))["micro_f1"]

    unres = Counter(u for r in raw for u in r["unresolved"])
    n_failed = sum(1 for r in raw if r["failed"])
    payload = {"model": args.model, "n": len(rows),
               "formulary_size": len(form), "shots": args.shots,
               "failed_calls": n_failed,
               "llm": res_llm, "phc_rxgen": res_nn,
               "top_unresolved_names": unres.most_common(15),
               # Full per-encounter record, not a 40-item preview: an inference
               # run costs ~70 minutes on this hardware and must never have to
               # be repeated to answer a follow-up question.
               "examples": raw}
    (OUT / args.out).write_text(json.dumps(payload, indent=2, default=float),
                                encoding="utf-8")

    print(f"\n{'='*70}\nPRESCRIPTION GENERATION — {args.model} vs PHC-RxGen  (n={len(rows)})")
    print(f"{'metric':22s} {'LLM':>10s} {'PHC-RxGen':>12s}")
    print("-" * 70)
    for k in ("micro_f1", "jaccard", "exact_match", "cat_micro_f1",
              "empty_f1", "mean_pred_size", "mean_gold_size"):
        a = res_llm.get(k)
        b = res_nn.get(k) if res_nn else None
        print(f"{k:22s} {a:10.4f} {(f'{b:12.4f}' if b is not None else '           —')}")
    if n_failed:
        print(f"\n*** {n_failed}/{len(raw)} calls returned nothing parseable. "
              f"These scored as empty prescriptions and the metrics above are "
              f"NOT a valid measure of this model. ***")
    off = sum(len(r["unresolved"]) for r in raw)
    tot = sum(len(r["llm_names"]) for r in raw)
    print(f"\nfailed calls: {n_failed}/{len(raw)}")
    print(f"off-formulary drug names: {off}/{tot} "
          f"({100*off/max(tot,1):.1f}% of everything the LLM named)")
    if unres:
        print("  most common:", ", ".join(f"{n}({c})" for n, c in unres.most_common(8)))
    print("=" * 70)
    print("wrote", OUT / args.out)
    partial = OUT / (args.out + ".partial")
    if partial.exists():
        partial.unlink()


if __name__ == "__main__":
    main()
