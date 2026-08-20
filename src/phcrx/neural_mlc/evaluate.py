"""Score the multi-label reframing under the published head-to-head protocol.

    python -m src.phcrx.neural_mlc.evaluate --bootstrap 2000

Everything that defines the comparison -- rows, label space, gold matrix,
micro-F1 arithmetic, threshold grid, the logistic-regression estimator and the
paired bootstrap -- is imported from `bench.head_to_head`. This module adds the
neural multi-label arms and nothing else, so any difference in the numbers is a
difference between systems, not between evaluations.

Protocols
---------
``full89`` / ``restricted47``
    Reproduced exactly (`head_to_head.PROTOCOLS`). The MLC head is trained in
    the `cat89` label space, which *is* the `full89` label space, so scoring it
    under either protocol is a column subset plus a row subset -- no re-mapping,
    no coarsening, no lost labels.
``norm46``
    The 46 normalised pharmacological classes. Not a coarsening of `cat89`, so
    every system is re-derived in that space: the linear arms are re-fitted, and
    the autoregressive checkpoint is re-decoded with its generated *brands*
    mapped through `drug_normalization.parquet` instead of `drug2cat`.

Operating points
----------------
Each arm is reported twice, exactly as `head_to_head` reports the linear arms:
at the threshold that maximises VAL micro-F1, and at the threshold whose VAL
mean predicted-set size matches the autoregressive decoder's. The decoder has
no threshold, so the second column is the one that cannot be dismissed as
operating-point tuning.

Uncertainty
-----------
Paired bootstrap over test encounters. Multi-seed arms are resampled *as a
seed-ensemble of scores*: within each resample the micro-F1 is computed per
seed and averaged, so the interval is on the expected performance of a
randomly-seeded run and the pairing against the reference systems is preserved.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..config import PROCESSED, MODELS, RESULTS, DataConfig
from ..bench import head_to_head as H
from . import data as D

OUT = RESULTS / "neural_mlc"
PROBS = OUT / "probs"
SEED = H.SEED

# Linear references. `lr_text_tab_all` is the arm that beat the autoregressive
# model by +0.092; the other two are kept for context on where the signal is.
LINEAR_ARMS = {
    "lr_text": ["text"],
    "lr_tab_all": ["tab_all"],
    "lr_text_tab_all": ["text", "tab_all"],
}


# ---------------------------------------------------------------------------
# autoregressive predictions in the normalised-class space
# ---------------------------------------------------------------------------
def neural_norm_sets(ckpt: Path, batch_size: int = 64, use_cache: bool = True):
    """As `head_to_head.neural_category_sets`, but the generated brands are
    mapped through `drug_normalization.parquet` rather than `drug2cat`.

    Same checkpoint, same greedy no-repeat decoding, same max length -- only the
    taxonomy the emitted brands are read into differs.
    """
    cache = OUT / "neural_norm_cache.json"
    key = f"{ckpt.name}:{int(ckpt.stat().st_mtime)}"
    if use_cache and cache.exists():
        blob = json.loads(cache.read_text())
        if blob.get("key") == key:
            return {s: {int(k): v for k, v in d.items()}
                    for s, d in blob["preds"].items()}

    import torch
    from torch.utils.data import DataLoader
    from ..data import RxCorpus, RxDataset
    from ..nlp.predict_helper import load_rxgen

    dn = pd.read_parquet(PROCESSED / "drug_normalization.parquet")
    d2n = {int(r.drug_id): r.drug_class for r in dn.itertuples()
           if r.drug_class is not None and pd.notna(r.drug_class)}
    classes = sorted({c for c in d2n.values()})
    c2i = {c: i + 1 for i, c in enumerate(classes)}

    corpus = RxCorpus(DataConfig())
    # vocabulary id -> normalised class id (0 = unmapped, dropped like cat 0)
    vid2norm = {vid: c2i.get(d2n.get(int(d), ""), 0)
                for d, vid in corpus.drug_vocab.items() if isinstance(d, int)}

    model, device = load_rxgen(ckpt, corpus)
    preds = {}
    with torch.no_grad():
        for split in ("val", "test"):
            dl = DataLoader(RxDataset(corpus, split), batch_size=batch_size,
                            shuffle=False, num_workers=0)
            out = {}
            for batch in dl:
                b = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
                gen = model.generate(b, max_len=corpus.cfg.max_rx_len)
                for i, pid in enumerate(b["pid"].tolist()):
                    cls = {vid2norm.get(d, 0) for d in gen["drugs"][i]}
                    out[int(pid)] = sorted(cls - {0})
            preds[split] = out
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "preds": preds}))
    return preds


# ---------------------------------------------------------------------------
# bootstrap over a multi-seed arm
# ---------------------------------------------------------------------------
def boot_index(m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, m, (n, m)).astype(np.int32)


def boot_micro(tps: np.ndarray, nps: np.ndarray, ng: np.ndarray,
               I: np.ndarray, chunk: int = 200) -> np.ndarray:
    """Bootstrap draws of micro-F1, averaged over seeds within each resample.

    tps / nps are (n_seeds, n_rows) per-row true-positive and predicted-size
    counts; `ng` is shared because the gold is the same for every seed.
    """
    n = I.shape[0]
    out = np.empty(n)
    for a in range(0, n, chunk):
        J = I[a:a + chunk]
        NG = ng[J].sum(1)
        f = np.zeros(len(J))
        for s in range(tps.shape[0]):
            f += 2 * tps[s][J].sum(1) / np.maximum(nps[s][J].sum(1) + NG, 1e-9)
        out[a:a + chunk] = f / tps.shape[0]
    return out


def summarise(draws: np.ndarray, point: float) -> dict:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"micro_f1": point, "micro_f1_lo95": float(lo), "micro_f1_hi95": float(hi)}


def delta(arm: np.ndarray, ref: np.ndarray, point_arm: float,
          point_ref: float, tag: str) -> dict:
    d = arm - ref
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {f"vs_{tag}_delta": point_arm - point_ref,
            f"vs_{tag}_delta_lo95": float(lo),
            f"vs_{tag}_delta_hi95": float(hi),
            f"vs_{tag}_p_better": float((d > 0).mean())}


def mean_metrics(mlist: list[dict]) -> dict:
    keys = mlist[0].keys()
    out = {}
    for k in keys:
        a = np.array([m[k] for m in mlist], dtype=float)
        out[k] = float(a.mean())
        if len(a) > 1:
            out[k + "_sd"] = float(a.std())
    return out


# ---------------------------------------------------------------------------
# protocol driver
# ---------------------------------------------------------------------------
def run_protocol(name: str, pcfg: dict, ctx: dict, args) -> dict:
    enc, feat, FF = ctx["enc"], ctx["feat"], ctx["families"]
    ls: D.LabelSpace = ctx["label_spaces"][pcfg["label_space"]]
    o = ls.orders

    labels = H.label_space(o, enc, pcfg["min_label_freq"])
    Yall = H.gold_matrix(enc, o, labels)
    cat2col = {c: j for j, c in enumerate(labels)}
    # column positions of these labels inside the model's own (full) label space
    lscol = {c: j for j, c in enumerate(ls.labels)}
    keep_cols = np.array([lscol[c] for c in labels])

    split = enc["split"].to_numpy()
    mask = np.ones(len(enc), dtype=bool)
    if pcfg["require_text"]:
        mask &= (enc["symptom_text"].str.strip() != "").to_numpy()
    if pcfg["require_label"]:
        mask &= Yall.any(1)
    idx = {s: np.where(mask & (split == s))[0] for s in D.SPLITS}
    Y = {s: Yall[idx[s]] for s in D.SPLITS}
    pids = {s: enc["prescription_id"].to_numpy()[idx[s]].astype(int)
            for s in ("val", "test")}

    ar = ctx["ar"][pcfg["label_space"]]

    def ar_P(s):
        P = np.zeros((len(pids[s]), len(labels)), dtype=bool)
        for i, pid in enumerate(pids[s]):
            for c in ar[s].get(int(pid), []):
                j = cat2col.get(int(c))
                if j is not None:
                    P[i, j] = True
        return P

    Par = {s: ar_P(s) for s in ("val", "test")}
    ar_val_size = float(Par["val"].sum(1).mean())

    print(f"\n### protocol {name}: label_space={pcfg['label_space']} "
          f"labels={len(labels)} rows train={len(idx['train'])} "
          f"val={len(idx['val'])} test={len(idx['test'])}", flush=True)
    print(f"    gold classes/encounter test={Y['test'].sum(1).mean():.3f}; "
          f"empty-gold test rows={float((Y['test'].sum(1)==0).mean()):.3f}; "
          f"autoregressive val set size={ar_val_size:.3f}", flush=True)

    # --- systems: per-row count triples, one entry per seed -----------------
    counts: dict[str, list] = {}
    meta: dict[str, dict] = {}

    def register(nm, Pte_list, extra):
        counts[nm] = [H.row_counts(Y["test"], P) for P in Pte_list]
        meta[nm] = {**extra, **mean_metrics([H.set_metrics(Y["test"], P)
                                             for P in Pte_list])}

    # autoregressive decoder (the incumbent neural system)
    register("neural_rxgen", [Par["test"]],
             {"threshold": None,
              "val_micro_f1": H.micro_f1_from_counts(*H.row_counts(Y["val"], Par["val"]))})

    # linear arms, re-fitted inside this protocol
    text = enc["symptom_text"].to_numpy()
    cols = [c for c in FF["feature_columns"] if c in feat.columns]
    blocks = {"text": H.build_text_blocks(text, idx),
              "tab_all": H.build_tabular_blocks(
                  feat, cols, list(FF["categorical_features"]), idx)}
    for arm, parts in LINEAR_ARMS.items():
        t0 = time.time()
        X = [sp.hstack([blocks[p][i] for p in parts]).tocsr() if len(parts) > 1
             else blocks[parts[0]][i] for i in range(3)]
        Pva, Pte = H.fit_ovr(X[0], Y["train"], X[1], X[2])
        thr = H.tune_threshold(Y["val"], Pva)
        register(arm, [Pte >= thr],
                 {"threshold": thr,
                  "val_micro_f1": H.micro_f1_from_counts(
                      *H.row_counts(Y["val"], Pva >= thr))})
        sthr = H.size_matched_threshold(Pva, ar_val_size)
        register(arm + "@sizematch", [Pte >= sthr],
                 {"threshold": sthr,
                  "val_micro_f1": H.micro_f1_from_counts(
                      *H.row_counts(Y["val"], Pva >= sthr))})
        print(f"    {arm:26s} thr={thr:.2f} "
              f"test_micro={meta[arm]['micro_f1']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # neural multi-label arms
    vrow = {int(p): i for i, p in enumerate(pids["val"])}
    trow = {int(p): i for i, p in enumerate(pids["test"])}
    for variant in ctx["variants"]:
        Pte_t, Pte_s, thrs, sthrs, vfs = [], [], [], [], []
        for seed in ctx["seeds"]:
            f = PROBS / f"{ls.kind}_{variant}_seed{seed}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            assert np.array_equal(z["labels"], np.asarray(ls.labels))
            vsel = np.array([vrow[int(p)] for p in z["val_pid"] if int(p) in vrow])
            vsrc = np.array([i for i, p in enumerate(z["val_pid"]) if int(p) in vrow])
            tsel = np.array([trow[int(p)] for p in z["test_pid"] if int(p) in trow])
            tsrc = np.array([i for i, p in enumerate(z["test_pid"]) if int(p) in trow])
            Pva = np.zeros((len(pids["val"]), len(labels)), dtype=np.float32)
            Pva[vsel] = z["val_prob"][vsrc][:, keep_cols]
            Pte = np.zeros((len(pids["test"]), len(labels)), dtype=np.float32)
            Pte[tsel] = z["test_prob"][tsrc][:, keep_cols]

            thr = H.tune_threshold(Y["val"], Pva)
            sthr = H.size_matched_threshold(Pva, ar_val_size)
            Pte_t.append(Pte >= thr)
            Pte_s.append(Pte >= sthr)
            thrs.append(thr)
            sthrs.append(sthr)
            vfs.append(H.micro_f1_from_counts(*H.row_counts(Y["val"], Pva >= thr)))
        if not Pte_t:
            continue
        nm = "mlc_" + variant
        register(nm, Pte_t, {"threshold": float(np.mean(thrs)),
                             "val_micro_f1": float(np.mean(vfs)),
                             "n_seeds": len(Pte_t)})
        register(nm + "@sizematch", Pte_s,
                 {"threshold": float(np.mean(sthrs)), "val_micro_f1": None,
                  "n_seeds": len(Pte_s)})
        print(f"    {nm:26s} thr={np.mean(thrs):.2f} "
              f"test_micro={meta[nm]['micro_f1']:.4f} "
              f"(+/-{meta[nm].get('micro_f1_sd', 0):.4f}, {len(Pte_t)} seeds)",
              flush=True)

    # --- uncertainty --------------------------------------------------------
    ng = counts["neural_rxgen"][0][2]
    I = boot_index(len(ng), args.bootstrap, SEED)
    draws, points = {}, {}
    for nm, cl in counts.items():
        tps = np.stack([c[0] for c in cl])
        nps = np.stack([c[1] for c in cl])
        draws[nm] = boot_micro(tps, nps, ng, I)
        points[nm] = float(np.mean([H.micro_f1_from_counts(c[0], c[1], ng)
                                    for c in cl]))

    results = {}
    for nm in counts:
        r = {**meta[nm], **summarise(draws[nm], points[nm])}
        for ref, tag in (("neural_rxgen", "neural_rxgen"),
                         ("lr_text_tab_all", "lr_text_tab_all")):
            if nm != ref and ref in draws:
                r.update(delta(draws[nm], draws[ref], points[nm], points[ref], tag))
        results[nm] = r

    return {
        "label_space": pcfg["label_space"],
        "n_labels": len(labels),
        "n_rows": {s: int(len(idx[s])) for s in D.SPLITS},
        "ar_val_mean_set_size": ar_val_size,
        "gold_mean_size_test": float(Y["test"].sum(1).mean()),
        "empty_gold_rate_test": float((Y["test"].sum(1) == 0).mean()),
        "results": results,
    }


PROTOCOLS = {
    "full89": {**H.PROTOCOLS["full89"], "label_space": "cat89"},
    "restricted47": {**H.PROTOCOLS["restricted47"], "label_space": "cat89"},
    # Same row/label rules as full89, in the normalised taxonomy.
    "norm46": {"min_label_freq": 0, "require_text": False,
               "require_label": False, "label_space": "norm46"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(MODELS / "rxgen_full_patient_seed0.pt"))
    ap.add_argument("--protocols", nargs="+", default=["full89", "restricted47"])
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="benchmark")
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    enc, orders, V = H.load_tables()
    feat = D.align_features(enc, pd.read_parquet(PROCESSED / "features.parquet"))
    FF = json.loads((PROCESSED / "feature_families.json").read_text())

    needed = {PROTOCOLS[p]["label_space"] for p in args.protocols}
    label_spaces = {k: D.build_label_space(k, enc, orders, V) for k in needed}

    variants = args.variants
    if variants is None:
        variants = sorted({f.name.split("_seed")[0].split("_", 1)[1]
                           for f in PROBS.glob("*_seed*.npz")})
    print(f"variants={variants} seeds={args.seeds}", flush=True)

    ar = {}
    if "cat89" in needed:
        print("decoding autoregressive predictions (cat89) ...", flush=True)
        ar["cat89"] = H.neural_category_sets(Path(args.ckpt), args.batch_size)
    if "norm46" in needed:
        print("decoding autoregressive predictions (norm46) ...", flush=True)
        ar["norm46"] = neural_norm_sets(Path(args.ckpt), args.batch_size)

    ctx = {"enc": enc, "feat": feat, "families": FF, "ar": ar,
           "label_spaces": label_spaces, "variants": variants,
           "seeds": args.seeds}

    blob = {"checkpoint": Path(args.ckpt).name, "bootstrap": args.bootstrap,
            "seeds": args.seeds, "protocols": {}}
    for p in args.protocols:
        blob["protocols"][p] = run_protocol(p, PROTOCOLS[p], ctx, args)

    rows = []
    for p, b in blob["protocols"].items():
        for nm, r in b["results"].items():
            rows.append({"protocol": p, "system": nm, **r})
    df = pd.DataFrame(rows)
    cols = ["protocol", "system", "n_seeds", "threshold", "val_micro_f1",
            "micro_f1", "micro_f1_sd", "micro_f1_lo95", "micro_f1_hi95",
            "macro_f1", "jaccard", "exact_match", "micro_precision",
            "micro_recall", "mean_pred_size", "mean_gold_size",
            "vs_neural_rxgen_delta", "vs_neural_rxgen_delta_lo95",
            "vs_neural_rxgen_delta_hi95", "vs_neural_rxgen_p_better",
            "vs_lr_text_tab_all_delta", "vs_lr_text_tab_all_delta_lo95",
            "vs_lr_text_tab_all_delta_hi95", "vs_lr_text_tab_all_p_better"]
    df = df.reindex(columns=cols)

    (OUT / f"{args.out}.json").write_text(json.dumps(blob, indent=2, default=float))
    df.to_csv(OUT / f"{args.out}.csv", index=False)

    print("\n" + "=" * 130)
    print("MULTI-LABEL REFRAMING vs AUTOREGRESSIVE DECODER vs LINEAR BASELINES")
    print("=" * 130)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df.round(4).to_string(index=False))
    print(f"\nwrote {OUT/(args.out+'.csv')}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
