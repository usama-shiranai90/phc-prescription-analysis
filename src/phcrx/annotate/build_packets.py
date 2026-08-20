"""Build the clinician annotation instrument.

Everything measured so far compares automated output against historical
records. That establishes *imitation fidelity*, not clinical validity: ICD
figures are two-automated-coder agreement, and prescription figures are string
match against what a clinician happened to write. Neither says the output is
sound.

This module draws a stratified sample and emits a blinded annotation packet for
two tasks:

  Task A - ICD-10 coding. A clinician codes the encounter from the note and
           vitals with NO model output visible. Converts every agreement figure
           into an accuracy figure.

  Task B - Prescription appropriateness, as a BLINDED PAIRWISE comparison.
           The model's prescription and the historical one are shown as
           "Option 1" / "Option 2" in randomised order. This is deliberate: the
           historical prescription is not ground truth either -- 67 prescribers
           with a prescriber-prior baseline at micro-F1 0.147 means substantial
           clinician-to-clinician variation. Rating the model against history
           as if history were correct would bake that variation into the
           result. A blinded A/B instead tests **non-inferiority to the
           attending clinician**, which is the claim that actually matters.

Patient data stays local. Packets carry an opaque annotation id; the crosswalk
to prescription_id and the A/B key are written to a separate file that
annotators must not receive.

    python -m src.phcrx.annotate.build_packets --n-icd 200 --n-rx 150
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Config, PROCESSED, RESULTS, MODELS
from ..data import RxCorpus, RxDataset, ATTR_KEYS
from ..nlp.predict_helper import load_rxgen

OUT = RESULTS / "annotation"
OUT.mkdir(parents=True, exist_ok=True)

VITAL_LABELS = [
    ("bp_sys", "Systolic BP", "mmHg", "{:.0f}"),
    ("bp_dia", "Diastolic BP", "mmHg", "{:.0f}"),
    ("pulse_rate", "Pulse", "bpm", "{:.0f}"),
    ("temperature", "Temperature", "°F", "{:.1f}"),
    ("blood_glucose", "Blood glucose", "mg/dL", "{:.0f}"),
    ("bmi", "BMI", "kg/m²", "{:.1f}"),
    ("blood_hemoglobin", "Haemoglobin", "g/dL", "{:.1f}"),
    ("oxygen_of_blood", "SpO₂", "%", "{:.0f}"),
]


def vitals_block(row) -> list[dict]:
    out = []
    for col, label, unit, fmt in VITAL_LABELS:
        v = row.get(col)
        if pd.notna(v):
            out.append({"label": label, "value": fmt.format(float(v)), "unit": unit})
    if pd.notna(row.get("blood_glucose")) and row.get("glucose_type") not in (None, "NA"):
        out.append({"label": "Glucose assay", "value": str(row["glucose_type"]), "unit": ""})
    return out


def rx_text(corpus: RxCorpus, drug_vids: list[int],
            attrs: dict | None = None, idx: int | None = None) -> list[dict]:
    """Render a prescription as human-readable lines."""
    id2drug = {v: k for k, v in corpus.drug_vocab.items()}
    names = dict(zip(corpus.orders["drug_id"].astype("Int64"), corpus.orders["drug_name"]))
    id2attr = {k: {v: a for a, v in corpus.attr_vocab[k].items()} for k in ATTR_KEYS}
    cats = corpus.vocab.get("category_names", {})
    id2cat = {v: k for k, v in corpus.cat_vocab.items()}

    lines = []
    for j, vid in enumerate(drug_vids):
        raw = id2drug.get(vid)
        if not isinstance(raw, int):
            continue
        nm = names.get(raw) or str(raw)
        cat = cats.get(str(id2cat.get(corpus.drug2cat.get(raw, 0), "")), "")
        item = {"drug": str(nm), "klass": str(cat)}
        if attrs is not None:
            for k in ATTR_KEYS:
                seq = attrs[k][idx] if idx is not None else attrs[k]
                item[k] = str(id2attr[k].get(seq[j], "")) if j < len(seq) else ""
        lines.append(item)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-icd", type=int, default=200)
    ap.add_argument("--n-rx", type=int, default=150)
    ap.add_argument("--overlap-frac", type=float, default=0.25,
                    help="share double-annotated for inter-rater reliability")
    ap.add_argument("--n-annotators", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--checkpoint",
                    default=str(MODELS / "rxgen_full_patient_seed0.pt"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    cfg = Config()
    corpus = RxCorpus(cfg.data)
    enc = corpus.enc.set_index("prescription_id", drop=False)

    icd = pd.read_parquet(PROCESSED / "icd_autocoded.parquet")
    icd = icd.set_index("prescription_id", drop=False)

    # ---------- Task A sample: stratified across confidence tiers ----------
    # The low-confidence tier is deliberately over-sampled relative to its
    # share: the open question is whether the gate is too conservative, and
    # only coded-by-a-clinician cases can answer that.
    quota_a = {"confident": int(args.n_icd * 0.55),
               "low_confidence": int(args.n_icd * 0.35),
               "no_complaint": args.n_icd - int(args.n_icd * 0.55)
               - int(args.n_icd * 0.35)}
    a_ids: list[int] = []
    for tier, q in quota_a.items():
        pool = icd[icd["tier"] == tier]
        if len(pool) == 0:
            continue
        take = pool.sample(min(q, len(pool)), random_state=args.seed)
        a_ids += [int(x) for x in take["prescription_id"]]

    # ---------- Task B sample: model vs historical prescription -----------
    ds = RxDataset(corpus, "test")
    test_rows = ds.rows
    gold_size = test_rows["prescription_id"].map(
        lambda p: len(corpus.rx_orders.get(int(p), [])) if p in corpus.rx_orders else 0)
    empty_pool = test_rows[gold_size == 0]
    nonempty_pool = test_rows[gold_size > 0]
    n_empty = int(args.n_rx * 0.20)   # mirrors the corpus rate (21.5%)
    b_rows = pd.concat([
        empty_pool.sample(min(n_empty, len(empty_pool)), random_state=args.seed),
        nonempty_pool.sample(min(args.n_rx - n_empty, len(nonempty_pool)),
                             random_state=args.seed),
    ])

    # Generate model prescriptions for the Task B sample.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device = load_rxgen(Path(args.checkpoint), corpus)
    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(Subset(ds, b_rows.index.tolist()), batch_size=32)
    pred_drugs, pred_attrs = [], {k: [] for k in ATTR_KEYS}
    with torch.no_grad():
        for b in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            gen = model.generate(b, max_len=cfg.data.max_rx_len)
            pred_drugs.extend(gen["drugs"])
            for k in ATTR_KEYS:
                pred_attrs[k].extend(gen["attrs"][k])

    # ---------- assemble packets ------------------------------------------
    packets, key = [], []

    for aid, pid in enumerate(sorted(set(a_ids)), start=1):
        if pid not in enc.index:
            continue
        r = enc.loc[pid]
        row = r.iloc[0] if isinstance(r, pd.DataFrame) else r
        packets.append({
            "ann_id": f"A{aid:04d}", "task": "icd",
            "age": (f"{row['age']:.0f}" if pd.notna(row.get("age")) else "unknown"),
            "sex": str(row.get("sex") or "unknown"),
            "note": str(row.get("symptom_text") or ""),
            "vitals": vitals_block(row),
        })
        key.append({"ann_id": f"A{aid:04d}", "task": "icd",
                    "prescription_id": int(pid),
                    "tier": str(icd.loc[pid, "tier"]) if pid in icd.index else "",
                    "model_icd": (str(icd.loc[pid, "icd_final"])
                                  if pid in icd.index and pd.notna(icd.loc[pid, "icd_final"])
                                  else None)})

    for bid, (pos, (_, row)) in enumerate(zip(range(len(b_rows)), b_rows.iterrows()), start=1):
        pid = int(row["prescription_id"])
        g = corpus.rx_orders.get(pid)
        gold_vids = ([corpus.drug_vocab.get(int(d), 3)
                      for d in g["drug_id"].dropna().astype(int)][: cfg.data.max_rx_len]
                     if g is not None else [])
        gold_lines = rx_text(corpus, gold_vids)
        if g is not None:
            for j, (_, orow) in enumerate(g.head(len(gold_lines)).iterrows()):
                gold_lines[j].update({
                    "type": str(orow.get("type_name") or ""),
                    "dose": str(orow.get("dose_canon") or ""),
                    "duration": str(orow.get("duration_bucket") or ""),
                    "instruction": str(orow.get("instruction") or "")})
        model_lines = rx_text(corpus, pred_drugs[pos], pred_attrs, pos)

        # Randomised, blinded presentation order.
        flip = bool(rng.integers(0, 2))
        opt1, opt2 = ((model_lines, gold_lines) if flip else (gold_lines, model_lines))
        packets.append({
            "ann_id": f"B{bid:04d}", "task": "rx",
            "age": (f"{row['age']:.0f}" if pd.notna(row.get("age")) else "unknown"),
            "sex": str(row.get("sex") or "unknown"),
            "note": str(row.get("symptom_text") or ""),
            "vitals": vitals_block(row),
            "option1": opt1, "option2": opt2,
        })
        key.append({"ann_id": f"B{bid:04d}", "task": "rx",
                    "prescription_id": pid,
                    "option1_is": "model" if flip else "historical",
                    "option2_is": "historical" if flip else "model",
                    "n_model": len(model_lines), "n_historical": len(gold_lines)})

    # ---------- assignment across annotators, with overlap ----------------
    ids = [p["ann_id"] for p in packets]
    random.shuffle(ids)
    n_overlap = int(len(ids) * args.overlap_frac)
    overlap, rest = set(ids[:n_overlap]), ids[n_overlap:]
    assignment = {f"annotator_{i+1}": sorted(overlap) for i in range(args.n_annotators)}
    for i, aid in enumerate(rest):
        assignment[f"annotator_{i % args.n_annotators + 1}"].append(aid)
    for k in assignment:
        assignment[k] = sorted(set(assignment[k]))

    (OUT / "packets.json").write_text(json.dumps(packets, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    (OUT / "KEY_do_not_share_with_annotators.json").write_text(
        json.dumps({"key": key, "assignment": assignment,
                    "seed": args.seed,
                    "checkpoint": args.checkpoint}, indent=1), encoding="utf-8")

    n_a = sum(1 for p in packets if p["task"] == "icd")
    n_b = sum(1 for p in packets if p["task"] == "rx")
    print(f"packets: {len(packets)}  (Task A ICD={n_a}, Task B Rx={n_b})")
    print(f"double-annotated for reliability: {n_overlap} "
          f"({args.overlap_frac:.0%}) across {args.n_annotators} annotators")
    for k, v in assignment.items():
        print(f"   {k}: {len(v)} items")
    print("\nTask A tier composition:")
    tiers = pd.Series([k_["tier"] for k_ in key if k_["task"] == "icd"])
    print(tiers.value_counts().to_string())
    print("\nTask B option balance (blinding check):")
    b = [k_ for k_ in key if k_["task"] == "rx"]
    print(f"   model shown as Option 1: {sum(1 for x in b if x['option1_is']=='model')}"
          f"/{len(b)}")
    print(f"\nwrote {OUT/'packets.json'}")
    print(f"wrote {OUT/'KEY_do_not_share_with_annotators.json'}  <-- withhold from annotators")


if __name__ == "__main__":
    main()
