"""Train the multi-label reframing of PHC-RxGen.

    python -m src.phcrx.neural_mlc.train --all --seeds 0 1 2
    python -m src.phcrx.neural_mlc.train --variant full_tab --label-space norm46

Protocol
--------
* Fit on TRAIN. Select the epoch on VAL micro-F1, where VAL micro-F1 is
  measured at the *threshold tuned on VAL* -- the linear arms get a tuned
  operating point, so withholding one from the neural model would bias the
  comparison in the opposite direction.
* Threshold grid, threshold objective and micro-F1 arithmetic are imported from
  `bench.head_to_head`, so nothing about the decision rule differs between the
  systems being compared.
* Three seeds; mean +/- s.d. reported. Seed s.d. on the autoregressive side was
  +/- 0.008, so any claimed difference smaller than that is noise.
* TEST is scored once per run and its per-encounter probabilities are cached to
  `results/rx_generation/neural_mlc/probs/`; `evaluate.py` re-scores those under
  the published protocols without retraining.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

from ..config import Config, RESULTS
from ..bench import head_to_head as H
from . import data as D
from .model import MLCConfig, VARIANTS, build_mlc, variant_configs

OUT = RESULTS / "neural_mlc"
PROBS = OUT / "probs"


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, st: D.SplitTensors, batch_size: int, amp: bool) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(st), batch_size):
        idx = torch.arange(i, min(i + batch_size, len(st)), device=st.y.device)
        with torch.amp.autocast("cuda", enabled=amp):
            logit = model(st.batch(idx))
        out.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(out)


def score(Y: np.ndarray, P: np.ndarray, thr: float) -> dict:
    """head_to_head's own metric block, on this label space."""
    return H.set_metrics(Y, P >= thr)


def train_one(variant: str, seed: int, ls: D.LabelSpace, splits: dict,
              sizes: dict, cfg: Config, base_mlc: MLCConfig, args, device) -> dict:
    set_seed(seed)
    tab_dim = splits["train"].tab.shape[1]
    mcfg, mlccfg = variant_configs(variant, cfg.model, base_mlc, tab_dim)

    tr, va, te = splits["train"], splits["val"], splits["test"]
    Ytr = tr.y.cpu().numpy() > 0.5
    Yva = va.y.cpu().numpy() > 0.5
    Yte = te.y.cpu().numpy() > 0.5
    prior = torch.from_numpy(Ytr.mean(0).astype(np.float32))

    model = build_mlc(mcfg, mlccfg, sizes, ls.n, prior).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    pos_weight = None
    if args.pos_weight_alpha > 0:
        p = prior.clamp(1e-4, 1 - 1e-4)
        pos_weight = ((1 - p) / p).pow(args.pos_weight_alpha).clamp(1.0, 50.0).to(device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=cfg.train.weight_decay)
    n_tr = len(tr)
    steps_per_epoch = max(1, (n_tr + args.batch_size - 1) // args.batch_size)
    steps = steps_per_epoch * args.epochs
    warmup = max(1, int(steps * cfg.train.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else max(0.0, 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, steps - warmup)))))
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    best = {"val_micro_f1": -1.0}
    best_state, best_epoch, bad = None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, generator=gen).to(device)
        tot = 0.0
        for i in range(0, n_tr, args.batch_size):
            b = tr.batch(perm[i:i + args.batch_size])
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = lossf(model(b), b["y"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss)

        Pva = predict(model, va, args.batch_size, amp)
        thr = H.tune_threshold(Yva, Pva)
        vm = H.micro_f1_from_counts(*H.row_counts(Yva, Pva >= thr))
        if not args.quiet:
            print(f"  [{variant} s{seed}] ep{epoch:02d} "
                  f"loss={tot/steps_per_epoch:.5f} thr={thr:.2f} "
                  f"val_microF1={vm:.4f}", flush=True)
        if vm > best["val_micro_f1"] + 1e-6:
            best = {"val_micro_f1": vm, "threshold": thr}
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch, bad = epoch, 0
        else:
            bad += 1
            if bad >= args.patience:
                if not args.quiet:
                    print(f"  early stop at epoch {epoch} (best {best_epoch})")
                break

    model.load_state_dict(best_state)
    Pva = predict(model, va, args.batch_size, amp)
    Pte = predict(model, te, args.batch_size, amp)
    thr = H.tune_threshold(Yva, Pva)

    np.savez_compressed(
        PROBS / f"{ls.kind}_{variant}_seed{seed}.npz",
        val_pid=va.pid.cpu().numpy(), val_prob=Pva.astype(np.float32),
        test_pid=te.pid.cpu().numpy(), test_prob=Pte.astype(np.float32),
        labels=np.asarray(ls.labels), threshold=np.float32(thr))

    return {
        "variant": variant, "seed": seed, "label_space": ls.kind,
        "n_params": int(n_params), "best_epoch": best_epoch, "threshold": thr,
        "model_cfg": {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in asdict(mcfg).items()},
        "mlc_cfg": asdict(mlccfg),
        "val": score(Yva, Pva, thr),
        "test": score(Yte, Pte, thr),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="+", default=["full"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--label-space", default="cat89", choices=["cat89", "norm46"])
    ap.add_argument("--tab-families", default="all",
                    choices=["all", "clinical", "physio"])
    ap.add_argument("--pool", default="cls_mean")
    ap.add_argument("--head-hidden", type=int, default=0)
    ap.add_argument("--pos-weight-alpha", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="runs.json")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    PROBS.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = list(VARIANTS) if args.all else args.variant

    t0 = time.time()
    corpus, enc, ls, splits = D.load_all(args.label_space, args.tab_families, cfg.data)
    for s in splits:
        splits[s] = splits[s].to(device)
    print(f"device={device} label_space={ls.kind} n_labels={ls.n} "
          f"tab_dim={splits['train'].tab.shape[1]} "
          f"rows={{'train': {len(splits['train'])}, 'val': {len(splits['val'])}, "
          f"'test': {len(splits['test'])}}}  (load {time.time()-t0:.0f}s)", flush=True)
    print(f"variants={variants} seeds={args.seeds}", flush=True)

    base_mlc = MLCConfig(pool=args.pool, head_hidden=args.head_hidden)
    runs = []
    for v in variants:
        for s in args.seeds:
            r = train_one(v, s, ls, splits, corpus.sizes, cfg, base_mlc,
                          args, device)
            runs.append(r)
            print(f"[{v} seed={s}] val_microF1={r['val']['micro_f1']:.4f} "
                  f"TEST microF1={r['test']['micro_f1']:.4f} "
                  f"macroF1={r['test']['macro_f1']:.4f} "
                  f"jac={r['test']['jaccard']:.4f} "
                  f"set={r['test']['mean_pred_size']:.2f} "
                  f"thr={r['threshold']:.2f} ep{r['best_epoch']} "
                  f"({r['n_params']/1e6:.2f}M)", flush=True)

    name = args.out if not args.tag else f"runs_{args.tag}.json"
    (OUT / name).write_text(json.dumps(runs, indent=2, default=float))

    print(f"\n{'variant':20s} {'val microF1':>18s} {'test microF1':>18s} "
          f"{'test macroF1':>18s}")
    for v in variants:
        rs = [r for r in runs if r["variant"] == v]
        if not rs:
            continue

        def ms(split, key):
            a = np.array([r[split][key] for r in rs])
            return f"{a.mean():.4f}+/-{a.std():.4f}"
        print(f"{v:20s} {ms('val','micro_f1'):>18s} {ms('test','micro_f1'):>18s} "
              f"{ms('test','macro_f1'):>18s}")
    print(f"\nwrote {OUT/name}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
