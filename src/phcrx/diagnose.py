"""Is the model actually conditioning on the patient, or reciting a prior?

The ablation grid showed every architectural component to be removable without
loss, and removing the fusion encoder *helped*. That pattern is consistent with
a model that has learned the marginal prescribing distribution and ignores its
inputs. These diagnostics decide the question rather than leaving it to
interpretation.

  1. Prediction diversity  - how many distinct prescriptions does it ever emit?
  2. Input-permutation test - shuffle one modality across patients. If the
     metric does not move, that modality was not being used. This is the
     decisive experiment.
  3. Conditional-prior reference - score the single most frequent training
     prescription against every test encounter.

    python -m src.phcrx.diagnose --checkpoint models/rxgen_full_seed0.pt
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import torch

from .config import Config, ModelConfig, MODELS, RESULTS, PAD, BOS, EOS
from .data import RxCorpus, RxDataset, make_loaders
from .model import build_model
from . import metrics as M


def load_model(path, corpus, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sizes = ck.get("sizes", corpus.sizes)
    if sizes != corpus.sizes:
        raise SystemExit(
            f"Checkpoint/corpus mismatch: checkpoint was trained on the "
            f"'{ck.get('split', '?')}' split; re-run preprocess for that split "
            f"or choose a matching checkpoint.")
    mcfg = ModelConfig(**{k: v for k, v in ck["model_cfg"].items()
                          if k in ModelConfig.__dataclass_fields__})
    model = build_model(mcfg, sizes).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect(model, loader, device, cfg, permute: str | None = None, seed: int = 0):
    """Generate over the split, optionally permuting one modality across rows."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    preds, golds = [], []
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        n = b["word_ids"].size(0)
        if n > 1 and permute:
            perm = torch.randperm(n, generator=g).to(device)
            keys = {
                "text": ["word_ids", "char_ids"],
                "vitals": ["vitals", "vitals_mask"],
                "demo": ["demo", "district", "glucose_type"],
                "history": ["hist_feat", "hist_drugs", "hist_mask"],
                "all": ["word_ids", "char_ids", "vitals", "vitals_mask", "demo",
                        "district", "glucose_type", "hist_feat", "hist_drugs",
                        "hist_mask"],
            }[permute]
            for k in keys:
                b[k] = b[k][perm]
        gen = model.generate(b, max_len=cfg.data.max_rx_len)
        preds.extend(gen["drugs"])
        nd = b["n_drugs"].tolist()
        out = b["drug_out"].tolist()
        for i, ni in enumerate(nd):
            golds.append(out[i][:ni])
    return preds, golds


def score(preds, golds, corpus):
    r = M.set_metrics(preds, golds)
    r.update(M.empty_rx_metrics(preds, golds))
    v2c = corpus.vid2cat
    to_cat = lambda s: [[v2c.get(d, 0) for d in x if v2c.get(d, 0) != 0] for x in s]
    r["cat_micro_f1"] = M.set_metrics(to_cat(preds), to_cat(golds))["micro_f1"]
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(MODELS / "rxgen_full_patient_seed0.pt"))
    ap.add_argument("--out", default="diagnostics.json")
    args = ap.parse_args()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = RxCorpus(cfg.data)
    model = load_model(args.checkpoint, corpus, device)
    loaders = make_loaders(corpus, cfg.train.batch_size, num_workers=0)

    report = {}

    # --- 1. Diversity of what the model emits -----------------------------
    preds, golds = collect(model, loaders["test"], device, cfg)
    base = score(preds, golds, corpus)
    pred_sets = Counter(tuple(sorted(set(p))) for p in preds)
    gold_sets = Counter(tuple(sorted(set(g) - {PAD, BOS, EOS})) for g in golds)
    id2drug = {v: k for k, v in corpus.drug_vocab.items()}
    names = dict(zip(corpus.orders["drug_id"].astype("Int64"), corpus.orders["drug_name"]))
    top_pred, top_pred_n = pred_sets.most_common(1)[0]

    report["diversity"] = {
        "n_test": len(preds),
        "distinct_pred_sets": len(pred_sets),
        "distinct_gold_sets": len(gold_sets),
        "modal_pred_set_share": top_pred_n / len(preds),
        "modal_gold_set_share": gold_sets.most_common(1)[0][1] / len(golds),
        "modal_pred_set": [names.get(id2drug.get(d), str(d)) for d in top_pred],
        "top5_pred_sets": [
            {"drugs": [names.get(id2drug.get(d), str(d)) for d in s],
             "share": round(c / len(preds), 4)}
            for s, c in pred_sets.most_common(5)],
    }

    # --- 2. Input-permutation test (the decisive one) ---------------------
    report["baseline"] = base
    report["permutation"] = {}
    for mod in ("text", "vitals", "demo", "history", "all"):
        ps, gs = collect(model, loaders["test"], device, cfg, permute=mod, seed=0)
        s = score(ps, gs, corpus)
        report["permutation"][mod] = {
            "micro_f1": s["micro_f1"], "jaccard": s["jaccard"],
            "cat_micro_f1": s["cat_micro_f1"],
            "delta_micro_f1": s["micro_f1"] - base["micro_f1"],
        }

    # --- 3. Constant-prediction reference ---------------------------------
    tr_preds, tr_golds = [], []
    for batch in loaders["train"]:
        nd = batch["n_drugs"].tolist()
        out = batch["drug_out"].tolist()
        for i, ni in enumerate(nd):
            tr_golds.append(out[i][:ni])
    modal_train = Counter(tuple(sorted(set(g) - {PAD, BOS, EOS})) for g in tr_golds
                          ).most_common(1)[0][0]
    const = [list(modal_train) for _ in golds]
    report["constant_modal_train_set"] = {
        "set": [names.get(id2drug.get(d), str(d)) for d in modal_train],
        **{k: v for k, v in score(const, golds, corpus).items()
           if k in ("micro_f1", "jaccard", "exact_match", "cat_micro_f1")},
    }

    (RESULTS / args.out).write_text(json.dumps(report, indent=2, default=float))

    d = report["diversity"]
    print("=" * 66)
    print("1. PREDICTION DIVERSITY")
    print(f"   test encounters              : {d['n_test']}")
    print(f"   distinct predicted Rx sets   : {d['distinct_pred_sets']}")
    print(f"   distinct gold Rx sets        : {d['distinct_gold_sets']}")
    print(f"   modal predicted set share    : {d['modal_pred_set_share']:.1%}")
    print(f"   modal gold set share         : {d['modal_gold_set_share']:.1%}")
    print(f"   modal predicted set          : {d['modal_pred_set']}")
    print("\n2. INPUT-PERMUTATION TEST  (shuffle a modality across patients)")
    print(f"   {'intact':12s} microF1={base['micro_f1']:.4f}")
    for mod, s in report["permutation"].items():
        flag = "  <-- unused" if abs(s["delta_micro_f1"]) < 0.005 else ""
        print(f"   {mod:12s} microF1={s['micro_f1']:.4f}  "
              f"delta={s['delta_micro_f1']:+.4f}{flag}")
    c = report["constant_modal_train_set"]
    print("\n3. CONSTANT PREDICTION (modal training Rx for every patient)")
    print(f"   set={c['set']}")
    print(f"   microF1={c['micro_f1']:.4f}  jaccard={c['jaccard']:.4f}  "
          f"catF1={c['cat_micro_f1']:.4f}")
    print("=" * 66)


if __name__ == "__main__":
    main()
