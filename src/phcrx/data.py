"""Torch dataset for prescription generation.

One sample = one clinical encounter. Inputs are multi-modal (symptom free text,
vitals with explicit missingness, demographics, geography, prior visits);
targets are the prescription: an ordered drug sequence with per-drug structured
attributes, plus advice and test sets as auxiliary multi-label targets.

Drug order within a prescription is a canonical descending-frequency ordering.
The underlying data is a *set*, so imposing a deterministic order removes the
ordering ambiguity that would otherwise inject irreducible loss.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import PROCESSED, VITAL_COLS, DataConfig, PAD, BOS, EOS, UNK

ATTR_KEYS = ("type", "dose", "duration", "instruction")
ATTR_COL = {"type": "type_name", "dose": "dose_canon",
            "duration": "duration_bucket", "instruction": "instruction"}


class RxCorpus:
    """Loads the processed Parquet + vocab once and shares it across splits."""

    def __init__(self, cfg: DataConfig | None = None):
        self.cfg = cfg or DataConfig()
        self.vocab = json.loads((PROCESSED / "rxgen_vocab.json").read_text())
        self.stats = json.loads((PROCESSED / "rxgen_norm_stats.json").read_text())

        self.enc = pd.read_parquet(PROCESSED / "rxgen_encounters.parquet")
        self.orders = pd.read_parquet(PROCESSED / "rxgen_orders.parquet")
        self.advice = pd.read_parquet(PROCESSED / "rxgen_advice.parquet")
        self.tests = pd.read_parquet(PROCESSED / "rxgen_tests.parquet")

        self.word_vocab = self.vocab["word"]
        self.char_vocab = self.vocab["char"]
        self.drug_vocab = {int(k) if k.lstrip("-").isdigit() else k: v
                           for k, v in self.vocab["drug"].items()}
        self.attr_vocab = self.vocab["attr"]
        self.advice_vocab = {int(k): v for k, v in self.vocab["advice"].items()}
        self.test_vocab = self.vocab["test"]
        self.district_vocab = self.vocab["district"]
        self.glucose_vocab = self.vocab["glucose_type"]
        self.drug_freq = {int(k): v for k, v in self.vocab["drug_train_freq"].items()}
        self.cat_vocab = self.vocab["category"]
        self.drug2cat = {int(k): v for k, v in self.vocab["drug2cat"].items()}
        # Index by *vocab* id so predictions can be mapped to categories directly.
        self.vid2cat = {vid: self.drug2cat.get(int(d), 0)
                        for d, vid in self.drug_vocab.items() if isinstance(d, int)}

        self._index_targets()

    # --- target assembly ---------------------------------------------------
    def _index_targets(self) -> None:
        o = self.orders.copy()
        o["drug_id"] = o["drug_id"].astype("Int64")
        o["freq"] = o["drug_id"].map(lambda d: self.drug_freq.get(int(d), 0)
                                     if pd.notna(d) else 0)
        # Canonical order: most frequent drug first, drug_id as deterministic tiebreak.
        o = o.sort_values(["prescription_id", "freq", "drug_id"],
                          ascending=[True, False, True])
        self.rx_orders = {
            pid: g for pid, g in o.groupby("prescription_id", sort=False)
        }

        av = self.advice.copy()
        av["advice_id"] = pd.to_numeric(av["advice_id"], errors="coerce").astype("Int64")
        self.rx_advice = av.dropna(subset=["advice_id"]).groupby("prescription_id")[
            "advice_id"].apply(lambda s: [int(x) for x in s]).to_dict()

        te = self.tests.dropna(subset=["test_id"])
        self.rx_tests = te.groupby("prescription_id")["test_id"].apply(list).to_dict()

        # Prior-visit index: encounters sorted per patient for the history RNN.
        e = self.enc.sort_values(["user_id", "checkup_date"])
        self.patient_visits = {
            uid: g[["prescription_id", "checkup_date"] + VITAL_COLS].to_dict("records")
            for uid, g in e.groupby("user_id", sort=False)
        }

    def split_ids(self, split: str) -> list[int]:
        return self.enc.loc[self.enc["split"] == split, "prescription_id"].tolist()

    # --- sizes exposed to the model ---------------------------------------
    @property
    def sizes(self) -> dict[str, int]:
        return {
            "word": len(self.word_vocab),
            "char": len(self.char_vocab),
            "drug": len(self.drug_vocab),
            "advice": len(self.advice_vocab),
            "test": len(self.test_vocab),
            "district": len(self.district_vocab),
            "glucose": len(self.glucose_vocab),
            "n_vitals": len(VITAL_COLS),
            "category": len(self.cat_vocab),
            **{f"attr_{k}": len(self.attr_vocab[k]) for k in ATTR_KEYS},
        }


class RxDataset(Dataset):
    def __init__(self, corpus: RxCorpus, split: str):
        self.c = corpus
        self.cfg = corpus.cfg
        self.split = split
        self.rows = corpus.enc[corpus.enc["split"] == split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    # --- feature builders --------------------------------------------------
    def _text(self, tokens) -> tuple[np.ndarray, np.ndarray]:
        mw, mc = self.cfg.max_symptom_words, self.cfg.max_word_chars
        toks = list(tokens)[:mw] if tokens is not None else []
        wid = np.full(mw, PAD, dtype=np.int64)
        cid = np.full((mw, mc), PAD, dtype=np.int64)
        for i, t in enumerate(toks):
            wid[i] = self.c.word_vocab.get(t, UNK)
            for j, ch in enumerate(t[:mc]):
                cid[i, j] = self.c.char_vocab.get(ch, UNK)
        return wid, cid

    def _vitals(self, row) -> tuple[np.ndarray, np.ndarray]:
        v = np.zeros(len(VITAL_COLS), dtype=np.float32)
        m = np.zeros(len(VITAL_COLS), dtype=np.float32)
        for i, col in enumerate(VITAL_COLS):
            x = row[col]
            if pd.notna(x):
                s = self.c.stats[col]
                v[i] = (float(x) - s["mean"]) / (s["std"] or 1.0)
                m[i] = 1.0
        return np.clip(v, -6, 6), m

    def _history(self, row):
        """Prior encounters strictly before this one (no look-ahead)."""
        mh, mr = self.cfg.max_history, self.cfg.max_rx_len
        n_v = len(VITAL_COLS)
        hv = np.zeros((mh, n_v * 2 + 2), dtype=np.float32)
        hd = np.full((mh, mr), PAD, dtype=np.int64)
        hm = np.zeros(mh, dtype=np.float32)

        visits = self.c.patient_visits.get(row["user_id"], [])
        now = row["checkup_date"]
        prior = [v for v in visits if pd.notna(v["checkup_date"]) and v["checkup_date"] < now]
        prior = prior[-mh:]
        for k, v in enumerate(prior):
            feat = np.zeros(n_v * 2 + 2, dtype=np.float32)
            for i, col in enumerate(VITAL_COLS):
                x = v[col]
                if pd.notna(x):
                    s = self.c.stats[col]
                    feat[i] = np.clip((float(x) - s["mean"]) / (s["std"] or 1.0), -6, 6)
                    feat[n_v + i] = 1.0
            gap = (now - v["checkup_date"]).days if pd.notna(v["checkup_date"]) else 0
            feat[-2] = np.log1p(max(gap, 0)) / 10.0
            g = self.c.rx_orders.get(v["prescription_id"])
            if g is not None:
                ids = [self.c.drug_vocab.get(int(d), UNK)
                       for d in g["drug_id"].dropna().astype(int)][:mr]
                hd[k, :len(ids)] = ids
                feat[-1] = len(ids) / 10.0
            hv[k] = feat
            hm[k] = 1.0
        return hv, hd, hm

    def _targets(self, pid):
        mr = self.cfg.max_rx_len
        g = self.c.rx_orders.get(pid)
        drug_in = np.full(mr + 1, PAD, dtype=np.int64)   # decoder input  (BOS ...)
        drug_out = np.full(mr + 1, PAD, dtype=np.int64)  # decoder target (... EOS)
        cat_out = np.zeros(mr, dtype=np.int64)           # category per position
        attrs = {k: np.zeros(mr, dtype=np.int64) for k in ATTR_KEYS}
        n = 0
        drug_in[0] = BOS
        if g is not None:
            rows = g.head(mr)
            n = len(rows)
            for i, (_, r) in enumerate(rows.iterrows()):
                did = self.c.drug_vocab.get(int(r["drug_id"]), UNK)
                drug_out[i] = did
                cat_out[i] = self.c.drug2cat.get(int(r["drug_id"]), 0)
                if i + 1 <= mr:
                    drug_in[i + 1] = did
                for k in ATTR_KEYS:
                    val = r[ATTR_COL[k]]
                    val = "<na>" if (pd.isna(val) or val == "") else str(val)
                    attrs[k][i] = self.c.attr_vocab[k].get(val, 0)
        drug_out[n] = EOS  # empty prescription -> BOS then immediate EOS

        adv = np.zeros(len(self.c.advice_vocab), dtype=np.float32)
        for a in self.c.rx_advice.get(pid, []):
            j = self.c.advice_vocab.get(int(a))
            if j is not None:
                adv[j] = 1.0
        tst = np.zeros(len(self.c.test_vocab), dtype=np.float32)
        for t in self.c.rx_tests.get(pid, []):
            j = self.c.test_vocab.get(str(t))
            if j is not None:
                tst[j] = 1.0
        return drug_in, drug_out, cat_out, attrs, n, adv, tst

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        pid = int(row["prescription_id"])
        wid, cid = self._text(row["symptom_tokens"])
        vit, vmask = self._vitals(row)
        hv, hd, hm = self._history(row)
        drug_in, drug_out, cat_out, attrs, n_drugs, adv, tst = self._targets(pid)

        age_s = self.c.stats["age"]
        age = float(row["age"]) if pd.notna(row["age"]) else age_s["mean"]
        demo = np.array([
            (age - age_s["mean"]) / (age_s["std"] or 1.0),
            1.0 if pd.notna(row["age"]) else 0.0,
            1.0 if row["sex"] == "F" else 0.0,
            1.0 if row["sex"] == "M" else 0.0,
            float(row["smoker_flag"]),
        ], dtype=np.float32)

        district = self.c.district_vocab.get(
            str(int(row["site_district"])) if pd.notna(row["site_district"]) else "<na>", 0)
        glucose = self.c.glucose_vocab.get(str(row["glucose_type"]), 0)

        return {
            "pid": torch.tensor(pid, dtype=torch.long),
            "word_ids": torch.from_numpy(wid),
            "char_ids": torch.from_numpy(cid),
            "vitals": torch.from_numpy(vit),
            "vitals_mask": torch.from_numpy(vmask),
            "demo": torch.from_numpy(demo),
            "district": torch.tensor(district, dtype=torch.long),
            "glucose_type": torch.tensor(glucose, dtype=torch.long),
            "hist_feat": torch.from_numpy(hv),
            "hist_drugs": torch.from_numpy(hd),
            "hist_mask": torch.from_numpy(hm),
            "drug_in": torch.from_numpy(drug_in),
            "drug_out": torch.from_numpy(drug_out),
            "cat_out": torch.from_numpy(cat_out),
            "n_drugs": torch.tensor(n_drugs, dtype=torch.long),
            "advice": torch.from_numpy(adv),
            "tests": torch.from_numpy(tst),
            **{f"attr_{k}": torch.from_numpy(v) for k, v in attrs.items()},
        }


def make_loaders(corpus: RxCorpus, batch_size: int, num_workers: int = 2):
    from torch.utils.data import DataLoader
    out = {}
    for split in ("train", "val", "test"):
        ds = RxDataset(corpus, split)
        out[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=True, drop_last=False,
            persistent_workers=num_workers > 0,
        )
    return out
