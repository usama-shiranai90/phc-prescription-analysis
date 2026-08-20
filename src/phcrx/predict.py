"""Generate readable prescriptions from a trained checkpoint.

Qualitative output matters as much as the aggregate metrics here: a clinician
reading generated prescriptions can spot failure modes (implausible pairings,
duplicated classes, missing durations) that micro-F1 averages away.

    python -m src.phcrx.predict --checkpoint models/rxgen_full_seed0.pt --n 12
"""
from __future__ import annotations

import argparse
import json

import torch

from .config import Config, ModelConfig, MODELS, RESULTS, PAD, BOS, EOS
from .data import RxCorpus, RxDataset, ATTR_KEYS
from .model import build_model


def invert(d: dict) -> dict:
    return {v: k for k, v in d.items()}


def load_model(path, corpus, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sizes = ck.get("sizes", corpus.sizes)
    if sizes != corpus.sizes:
        raise SystemExit(
            f"Checkpoint/corpus mismatch: this checkpoint was trained on the "
            f"'{ck.get('split', '?')}' split, whose vocabularies differ from the "
            f"currently processed data.\n  checkpoint: {sizes}\n  corpus:     "
            f"{corpus.sizes}\nRe-run preprocess for that split, or pick a "
            f"checkpoint matching the current one.")
    mcfg = ModelConfig(**{k: v for k, v in ck["model_cfg"].items()
                          if k in ModelConfig.__dataclass_fields__})
    model = build_model(mcfg, sizes).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(MODELS / "rxgen_full_patient_seed0.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="qualitative_examples.md")
    args = ap.parse_args()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = RxCorpus(cfg.data)
    model = load_model(args.checkpoint, corpus, device)

    ds = RxDataset(corpus, args.split)
    id2drug = invert(corpus.drug_vocab)
    drug_names = dict(zip(corpus.orders["drug_id"].astype("Int64"),
                          corpus.orders["drug_name"]))
    id2attr = {k: invert(corpus.attr_vocab[k]) for k in ATTR_KEYS}
    cat_names = corpus.vocab.get("category_names", {})
    id2cat = invert(corpus.cat_vocab)

    def name_of(vid: int) -> str:
        raw = id2drug.get(vid)
        if not isinstance(raw, int):
            return str(raw)
        nm = drug_names.get(raw)
        cat = corpus.drug2cat.get(raw, 0)
        cname = cat_names.get(str(id2cat.get(cat, "")), "")
        return f"{nm or raw}" + (f" [{cname}]" if cname else "")

    lines = [f"# Generated prescriptions — {args.split} split", "",
             f"Checkpoint: `{args.checkpoint}`", ""]

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.n, shuffle=False)
    batch = next(iter(loader))
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        gen = model.generate(b, max_len=cfg.data.max_rx_len)

    rows = ds.rows.head(args.n)
    for i in range(min(args.n, len(rows))):
        r = rows.iloc[i]
        n = int(b["n_drugs"][i])
        gold_ids = b["drug_out"][i, :n].tolist()
        gold_attr = {k: b[f"attr_{k}"][i, :n].tolist() for k in ATTR_KEYS}
        pred_ids = gen["drugs"][i]

        age = f"{r['age']:.0f}" if r["age"] == r["age"] else "?"
        vit = ", ".join(
            f"{c}={r[c]:.0f}" for c in ("bp_sys", "bp_dia", "blood_glucose", "bmi")
            if r[c] == r[c])

        lines += [f"### Encounter {int(r['prescription_id'])}", "",
                  f"- **Patient**: {age}y {r['sex'] or '?'}",
                  f"- **Symptoms**: _{r['symptom_text'] or '(none recorded)'}_",
                  f"- **Vitals**: {vit or '(none)'}", ""]

        lines.append("| | Drug | Type | Dose | Duration | Instruction |")
        lines.append("|---|---|---|---|---|---|")
        for j, d in enumerate(gold_ids):
            if d in (PAD, BOS, EOS):
                continue
            a = [id2attr[k].get(gold_attr[k][j], "?") for k in ATTR_KEYS]
            lines.append(f"| gold | {name_of(d)} | {a[0]} | {a[1]} | {a[2]} | {a[3]} |")
        if not [d for d in gold_ids if d not in (PAD, BOS, EOS)]:
            lines.append("| gold | _(no pharmacotherapy)_ | | | | |")
        for j, d in enumerate(pred_ids):
            a = [id2attr[k].get(gen["attrs"][k][i][j], "?") if j < len(gen["attrs"][k][i])
                 else "?" for k in ATTR_KEYS]
            lines.append(f"| **pred** | {name_of(d)} | {a[0]} | {a[1]} | {a[2]} | {a[3]} |")
        if not pred_ids:
            lines.append("| **pred** | _(no pharmacotherapy)_ | | | | |")

        gs, ps = set(gold_ids) - {PAD, BOS, EOS}, set(pred_ids)
        jac = len(gs & ps) / len(gs | ps) if (gs | ps) else 1.0
        lines += ["", f"Jaccard: **{jac:.2f}**", ""]

    out = RESULTS / args.out
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
