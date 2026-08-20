"""Training / evaluation driver for PHC-RxGen.

Usage (inside the WSL conda env, from the project root):
    python -m src.phcrx.train --variant full --seeds 0 1 2
    python -m src.phcrx.train --list-variants
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config, ModelConfig, RESULTS, MODELS, PAD, EOS
from .data import RxCorpus, RxDataset, make_loaders, ATTR_KEYS
from .model import build_model
from . import metrics as M

# --- Ablation grid ---------------------------------------------------------
# Each variant isolates one architectural claim so the contribution of the
# CNN, the RNN, the fusion transformer and the decoder can be attributed.
VARIANTS: dict[str, dict] = {
    "full":            {},
    "no_char_cnn":     {"use_text_cnn": False},
    "no_text_rnn":     {"use_text_rnn": False},
    "no_vitals":       {"use_vitals": False},
    "no_history":      {"use_history": False},
    "no_fusion":       {"use_transformer_fusion": False},
    "gru_decoder":     {"decoder_type": "gru"},
    "text_only":       {"use_vitals": False, "use_history": False},
    # Drops the whole text branch (word embeddings included), not just the
    # CNN/RNN composition layers.
    "tabular_only":    {"use_text": False},
    # Neither text nor physiology: demographics + geography only. This is the
    # conditional-prior floor -- if the full model does not clear it, the
    # multi-modal encoder is not earning its place.
    "prior_only":      {"use_text": False, "use_vitals": False, "use_history": False},
}


def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def compute_loss(out, b, cfg, drug_weight=None):
    tcfg = cfg.train
    drug_logits = out["drug_logits"]                        # (B, T, V)
    tgt = b["drug_out"]
    loss_drug = F.cross_entropy(
        drug_logits.reshape(-1, drug_logits.size(-1)), tgt.reshape(-1),
        ignore_index=PAD, label_smoothing=tcfg.label_smoothing, weight=drug_weight)

    # Attributes are supervised only at real drug positions.
    valid = torch.arange(b["attr_type"].size(1), device=tgt.device
                         ).unsqueeze(0) < b["n_drugs"].unsqueeze(1)
    loss_attr = drug_logits.new_zeros(())
    loss_cat = drug_logits.new_zeros(())
    if valid.any():
        vflat = valid.reshape(-1).float()
        denom = vflat.sum()
        for k in ATTR_KEYS:
            lg = out["attr_logits"][k]
            la = F.cross_entropy(lg.reshape(-1, lg.size(-1)),
                                 b[f"attr_{k}"].reshape(-1), reduction="none")
            loss_attr = loss_attr + (la * vflat).sum() / denom
        loss_attr = loss_attr / len(ATTR_KEYS)

        cl = out["cat_logits"]
        lc = F.cross_entropy(cl.reshape(-1, cl.size(-1)), b["cat_out"].reshape(-1),
                             reduction="none")
        loss_cat = (lc * vflat).sum() / denom

    loss_adv = F.binary_cross_entropy_with_logits(out["advice_logits"], b["advice"])
    loss_tst = F.binary_cross_entropy_with_logits(out["test_logits"], b["tests"])

    total = (tcfg.w_drug * loss_drug + tcfg.w_attr * loss_attr
             + tcfg.w_cat * loss_cat
             + tcfg.w_advice * loss_adv + tcfg.w_test * loss_tst)
    return total, {"drug": float(loss_drug), "attr": float(loss_attr),
                   "cat": float(loss_cat), "advice": float(loss_adv),
                   "test": float(loss_tst)}


@torch.no_grad()
def evaluate(model, loader, corpus, device, cfg, full: bool = False):
    model.eval()
    pred_drugs, gold_drugs = [], []
    pred_attrs = {k: [] for k in ATTR_KEYS}
    gold_attrs = {k: [] for k in ATTR_KEYS}
    adv_p, adv_g, tst_p, tst_g, scores = [], [], [], [], []

    for batch in loader:
        b = move(batch, device)
        gen = model.generate(b, max_len=cfg.data.max_rx_len)
        pred_drugs.extend(gen["drugs"])
        for k in ATTR_KEYS:
            pred_attrs[k].extend(gen["attrs"][k])

        n = b["n_drugs"].tolist()
        out_ids = b["drug_out"].tolist()
        for i, ni in enumerate(n):
            gold_drugs.append([d for d in out_ids[i][:ni]])
            for k in ATTR_KEYS:
                gold_attrs[k].append(b[f"attr_{k}"][i, :ni].tolist())

        adv_p.append(gen["advice_prob"].float().cpu().numpy())
        adv_g.append(b["advice"].cpu().numpy())
        tst_p.append(gen["test_prob"].float().cpu().numpy())
        tst_g.append(b["tests"].cpu().numpy())

        if full:
            # Ranking scores: max drug probability across teacher-forced steps.
            out = model(b)
            p = out["drug_logits"].softmax(-1).amax(dim=1)
            scores.append(p.float().cpu().numpy())

    res = M.set_metrics(pred_drugs, gold_drugs)
    res.update(M.empty_rx_metrics(pred_drugs, gold_drugs))

    # Category-level view: brand choice within a class (16 PPI brands, 14
    # paracetamol brands) is a formulary decision, so class-level agreement is
    # the clinically meaningful score. Unmapped brands (cat 0) are dropped
    # rather than collapsed together, so they cannot inflate agreement.
    v2c = corpus.vid2cat
    to_cat = lambda seqs: [[v2c.get(d, 0) for d in s if v2c.get(d, 0) != 0]
                           for s in seqs]
    cat_res = M.set_metrics(to_cat(pred_drugs), to_cat(gold_drugs))
    res["category_level"] = {f"cat_{k}": v for k, v in cat_res.items()}

    if full:
        strata = {corpus.drug_vocab.get(int(k), -1): v
                  for k, v in corpus.vocab["drug_stratum"].items()}
        res["macro_f1"] = M.macro_f1_by_label(pred_drugs, gold_drugs, len(corpus.drug_vocab))
        res["by_stratum"] = M.stratified_metrics(pred_drugs, gold_drugs, strata)
        res.update(M.attribute_accuracy(pred_attrs, gold_attrs, pred_drugs, gold_drugs))
        res.update(M.ranking_metrics(np.concatenate(scores), gold_drugs))
        ap, ag = np.concatenate(adv_p), np.concatenate(adv_g)
        tp_, tg = np.concatenate(tst_p), np.concatenate(tst_g)
        res["advice"] = M.multilabel_metrics(ap, ag)
        res["advice"]["ece"] = M.expected_calibration_error(ap, ag)
        res["test"] = M.multilabel_metrics(tp_, tg)
        res["test"]["ece"] = M.expected_calibration_error(tp_, tg)
    return res


def train_one(variant: str, seed: int, cfg: Config, corpus: RxCorpus,
              device: torch.device, quiet: bool = False) -> dict:
    set_seed(seed)
    mcfg = replace(cfg.model, **VARIANTS[variant])
    model = build_model(mcfg, corpus.sizes).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    loaders = make_loaders(corpus, cfg.train.batch_size, cfg.train.num_workers)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    steps = max(1, len(loaders["train"]) * cfg.train.epochs)
    warmup = int(steps * cfg.train.warmup_frac)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup) if s < warmup
        else max(0.0, 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, steps - warmup)))))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")

    best = {"micro_f1": -1.0}
    best_state, bad = None, 0
    # The split is part of the identity: vocabularies are refit per split, so a
    # temporal checkpoint is not loadable against patient-split data.
    ckpt = MODELS / f"rxgen_{variant}_{cfg.data.split}_seed{seed}.pt"

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in loaders["train"]:
            b = move(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(b)
                loss, parts = compute_loss(out, b, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss); nb += 1

        val = evaluate(model, loaders["val"], corpus, device, cfg)
        if not quiet:
            print(f"  [{variant} s{seed}] ep{epoch:02d} loss={tot/max(nb,1):.4f} "
                  f"val_microF1={val['micro_f1']:.4f} jac={val['jaccard']:.4f} "
                  f"emptyF1={val['empty_f1']:.4f}", flush=True)

        if val["micro_f1"] > best["micro_f1"]:
            best = val; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.train.patience:
                if not quiet:
                    print(f"  early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "model_cfg": vars(mcfg),
                    "sizes": corpus.sizes, "split": cfg.data.split}, ckpt)
    test = evaluate(model, loaders["test"], corpus, device, cfg, full=True)
    return {"variant": variant, "seed": seed, "n_params": n_params,
            "val": best, "test": test}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="+", default=["full"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="run every ablation variant")
    ap.add_argument("--list-variants", action="store_true")
    ap.add_argument("--out", default="rxgen_results.json")
    args = ap.parse_args()

    if args.list_variants:
        for k, v in VARIANTS.items():
            print(f"{k:16s} {v}")
        return

    cfg = Config()
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    seeds = args.seeds or list(cfg.train.seeds)
    variants = list(VARIANTS) if args.all else args.variant

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} variants={variants} seeds={seeds}")

    corpus = RxCorpus(cfg.data)
    print("sizes:", corpus.sizes)
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(RxDataset(corpus, split))}")

    runs = []
    t0 = time.time()
    for v in variants:
        for s in seeds:
            r = train_one(v, s, cfg, corpus, device)
            runs.append(r)
            print(f"[{v} seed={s}] TEST microF1={r['test']['micro_f1']:.4f} "
                  f"jaccard={r['test']['jaccard']:.4f} "
                  f"macroF1={r['test'].get('macro_f1', 0):.4f} "
                  f"exact={r['test']['exact_match']:.4f} "
                  f"({r['n_params']/1e6:.2f}M params)", flush=True)

    out_path = RESULTS / args.out
    out_path.write_text(json.dumps(runs, indent=2, default=float))
    print(f"\nwrote {out_path}  ({time.time()-t0:.0f}s)")

    # Aggregate mean +/- std across seeds -- small corpus, so variance matters.
    print(f"\n{'variant':16s} {'microF1':>16s} {'jaccard':>16s} {'exact':>16s}")
    for v in variants:
        rs = [r for r in runs if r["variant"] == v]
        if not rs:
            continue
        def ms(key):
            a = np.array([r["test"][key] for r in rs])
            return f"{a.mean():.4f}+/-{a.std():.4f}"
        print(f"{v:16s} {ms('micro_f1'):>16s} {ms('jaccard'):>16s} {ms('exact_match'):>16s}")


if __name__ == "__main__":
    main()
