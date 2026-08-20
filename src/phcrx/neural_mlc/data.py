"""Label spaces, multi-hot targets and tensor materialisation for the MLC head.

Nothing here re-implements feature engineering. `RxCorpus`/`RxDataset` build the
encoder inputs exactly as the autoregressive model saw them; this module only

  * builds the label space (drug-class ids) and the per-encounter multi-hot
    target, using `bench.head_to_head.gold_matrix` so the neural targets are
    *the same matrix* the benchmark scores against;
  * attaches the engineered tabular block via
    `bench.head_to_head.build_tabular_blocks`, so the neural model sees exactly
    the columns `lr_tab_all` sees, fitted on TRAIN only;
  * materialises the 14,074-row corpus into flat tensors once, so 24 training
    runs do not pay the pandas `__getitem__` cost 24 times.

Two label spaces are supported:

``cat89``   the `drug2cat` categories used by `head_to_head` (88 non-<na>
            categories present in the corpus). This is the space the published
            benchmark scores, so predictions map into `full89` / `restricted47`
            with no lossy re-mapping.
``norm46``  the 46 normalised pharmacological classes from
            `data/processed/drug_normalization.parquet` (the `drugmap`
            workstream). A better taxonomy -- 97.6% order coverage against
            86.8% for the raw `rx_category` -- but *not* a coarsening of
            `cat89` (several categories fold into one class), so it cannot be
            scored under the published protocols. It gets its own protocol in
            `evaluate.py`, where the linear arms are re-fitted in the same
            space so the comparison stays matched.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ..config import PROCESSED, DataConfig
from ..data import RxCorpus, RxDataset
from ..bench import head_to_head as H

SPLITS = ("train", "val", "test")

# Encoder inputs carried through to the model. Decoder-side fields
# (drug_in / drug_out / attr_* / n_drugs) are deliberately dropped: this model
# has no decoder, no ordering and no EOS.
ENC_KEYS = ("word_ids", "char_ids", "vitals", "vitals_mask", "demo",
            "district", "glucose_type", "hist_feat", "hist_drugs", "hist_mask")


# ---------------------------------------------------------------------------
# label spaces
# ---------------------------------------------------------------------------
@dataclass
class LabelSpace:
    kind: str
    labels: list[int]
    names: list[str]
    orders: pd.DataFrame          # prescription_id, cat -- protocol ready
    Y: np.ndarray                 # (n_enc, n_labels) bool, aligned to enc rows

    @property
    def n(self) -> int:
        return len(self.labels)


def build_label_space(kind: str, enc: pd.DataFrame, orders: pd.DataFrame,
                      V: dict) -> LabelSpace:
    if kind == "cat89":
        o = H.encounter_categories(enc, orders, V)
        labels = H.label_space(o, enc, 0)
        cn = V.get("category_names", {})
        names = [str(cn.get(str(c), c)) for c in labels]
    elif kind == "norm46":
        dn = pd.read_parquet(PROCESSED / "drug_normalization.parquet")
        d2n = {int(r.drug_id): r.drug_class for r in dn.itertuples()
               if r.drug_class is not None and pd.notna(r.drug_class)}
        o = orders.dropna(subset=["drug_id", "prescription_id"]).copy()
        o["cls"] = o["drug_id"].astype(int).map(d2n)
        o = o.dropna(subset=["cls"])
        classes = sorted(o["cls"].unique().tolist())
        c2i = {c: i + 1 for i, c in enumerate(classes)}
        o["cat"] = o["cls"].map(c2i).astype(int)
        o["prescription_id"] = o["prescription_id"].astype(int)
        labels, names = list(range(1, len(classes) + 1)), classes
    else:
        raise ValueError("unknown label space " + repr(kind))
    return LabelSpace(kind, labels, names, o, H.gold_matrix(enc, o, labels))


# ---------------------------------------------------------------------------
# tabular block
# ---------------------------------------------------------------------------
def split_index(enc: pd.DataFrame) -> dict[str, np.ndarray]:
    s = enc["split"].to_numpy()
    return {k: np.where(s == k)[0] for k in SPLITS}


def align_features(enc: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Reindex features.parquet onto encounter row order, as head_to_head does."""
    return (feat.set_index("prescription_id")
                .reindex(enc["prescription_id"].astype(int))
                .reset_index())


def tabular_matrix(enc, feat, FF, families: str = "all") -> np.ndarray:
    """Dense (n_enc, d_tab) block: median-imputed and standardised numerics plus
    one-hot codes, all fitted on TRAIN rows only.

    Identical construction to the `lr_tab_*` arms, so when the neural model is
    given tabular input it is given exactly the block the linear arm gets.
    """
    fam = FF["families"]
    cols = [c for c in FF["feature_columns"] if c in feat.columns]
    if families == "clinical":
        cols = [c for c in cols if fam.get(c) not in H.NON_CLINICAL_FAMILIES]
    elif families == "physio":
        cols = [c for c in cols if fam.get(c) in H.PHYSIO_FAMILIES]
    elif families != "all":
        raise ValueError(families)
    idx = split_index(enc)
    blocks = H.build_tabular_blocks(feat, cols, list(FF["categorical_features"]), idx)
    X = np.zeros((len(enc), blocks[0].shape[1]), dtype=np.float32)
    for s, B in zip(SPLITS, blocks):
        X[idx[s]] = B.toarray().astype(np.float32)
    return X


# ---------------------------------------------------------------------------
# materialisation
# ---------------------------------------------------------------------------
@dataclass
class SplitTensors:
    pid: torch.Tensor
    y: torch.Tensor
    tab: torch.Tensor
    enc_rows: np.ndarray
    x: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.pid.numel())

    def to(self, device) -> "SplitTensors":
        self.pid = self.pid.to(device)
        self.y = self.y.to(device)
        self.tab = self.tab.to(device)
        self.x = {k: v.to(device) for k, v in self.x.items()}
        return self

    def batch(self, idx: torch.Tensor) -> dict:
        b = {k: v[idx] for k, v in self.x.items()}
        b["y"] = self.y[idx]
        b["tab"] = self.tab[idx]
        return b


def materialize(corpus: RxCorpus, split: str, enc: pd.DataFrame,
                Y: np.ndarray, tab: np.ndarray) -> SplitTensors:
    """One pass over RxDataset -> flat tensors.

    Targets and tabular rows are attached by encounter row index; the alignment
    is asserted on prescription_id rather than assumed.
    """
    ds = RxDataset(corpus, split)
    rows = split_index(enc)[split]
    assert len(ds) == len(rows), split + ": " + str(len(ds)) + " vs " + str(len(rows))

    buf = {k: [] for k in ENC_KEYS}
    pids = []
    for i in range(len(ds)):
        item = ds[i]
        for k in ENC_KEYS:
            buf[k].append(item[k])
        pids.append(int(item["pid"]))
    pid = torch.tensor(pids, dtype=torch.long)

    enc_pid = enc["prescription_id"].to_numpy().astype(int)[rows]
    assert np.array_equal(pid.numpy(), enc_pid), split + ": pid misalignment"

    return SplitTensors(
        pid=pid,
        y=torch.from_numpy(Y[rows].astype(np.float32)),
        tab=torch.from_numpy(tab[rows]),
        enc_rows=rows,
        x={k: torch.stack(v) for k, v in buf.items()},
    )


def load_all(label_kind: str = "cat89", tab_families: str = "all",
             dcfg: DataConfig | None = None):
    """corpus + label space + per-split tensors. Called once per process."""
    corpus = RxCorpus(dcfg or DataConfig())
    enc, orders, V = H.load_tables()
    ls = build_label_space(label_kind, enc, orders, V)
    feat = align_features(enc, pd.read_parquet(PROCESSED / "features.parquet"))
    FF = json.loads((PROCESSED / "feature_families.json").read_text())
    tab = tabular_matrix(enc, feat, FF, tab_families)
    splits = {s: materialize(corpus, s, enc, ls.Y, tab) for s in SPLITS}
    return corpus, enc, ls, splits
