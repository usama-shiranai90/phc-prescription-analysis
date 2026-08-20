"""PHC-RxGen encoder + multi-label head. No decoder.

What is kept
------------
Every encoder submodule is imported from `..model` and used unmodified:
`TextEncoder` (char-CNN over words + word BiLSTM), `VitalsEncoder` (one token
per vital, value and missingness), `HistoryEncoder` (GRU over prior
encounters), and the same `nn.TransformerEncoder` fusion stack over
[CLS] + text + vitals + demo + geo + history. `RxMLC.encode` is a line-for-line
mirror of `PHCRxGen.encode` with two additions: an optional tabular token, and
switches for the demographic and geographic tokens so a genuine text-only arm
is expressible.

What is removed
---------------
The transformer/GRU decoder, the tied drug-embedding output head, the four
per-drug attribute heads, positional encoding over drug positions, BOS/EOS, the
canonical descending-frequency drug ordering, and the advice/test auxiliary
heads. What remains is one linear layer over the pooled context producing an
independent logit per drug class, trained with `BCEWithLogitsLoss`.

Pooling
-------
The autoregressive model cross-attends to the whole token sequence, so it never
needed [CLS] to be informative. With fusion disabled, [CLS] is a *constant*
parameter -- pooling on it alone would give the no-fusion arm a constant
representation and guarantee it fails for the wrong reason. The default
`cls_mean` pooling concatenates [CLS] with a mask-aware mean over context
tokens, so every ablation is scored on a representation that actually carries
its inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from ..config import ModelConfig, PAD
from ..model import TextEncoder, VitalsEncoder, HistoryEncoder

POOLS = ("cls", "mean", "cls_mean", "attn")


@dataclass
class MLCConfig:
    """Head-side options. Encoder options stay in `ModelConfig`."""
    pool: str = "cls_mean"
    head_hidden: int = 0          # 0 -> plain linear head
    head_dropout: float = 0.2
    use_demo: bool = True
    use_geo: bool = True
    use_tabular: bool = False
    tab_dim: int = 0              # filled in by the caller
    tab_tokens: int = 1
    bias_init_prior: bool = True  # init head bias at the train log-odds


class RxMLC(nn.Module):
    def __init__(self, cfg: ModelConfig, mlc: MLCConfig, sizes: dict,
                 n_labels: int):
        super().__init__()
        self.cfg, self.mlc, self.sizes = cfg, mlc, sizes
        d = cfg.d_model
        if mlc.pool not in POOLS:
            raise ValueError("pool must be one of " + str(POOLS))

        self.text = (TextEncoder(cfg, sizes["word"], sizes["char"])
                     if cfg.use_text else None)
        self.vitals = VitalsEncoder(cfg, sizes["n_vitals"]) if cfg.use_vitals else None

        self.demo = (nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Dropout(cfg.dropout))
                     if mlc.use_demo else None)
        if mlc.use_geo:
            self.district_emb = nn.Embedding(sizes["district"], d)
            self.glucose_emb = nn.Embedding(sizes["glucose"], d)

        # Only the history GRU needs drug embeddings (it pools prior orders);
        # there is no drug-embedding output head any more.
        if cfg.use_history:
            self.drug_emb = nn.Embedding(sizes["drug"], d, padding_idx=PAD)
            self.history = HistoryEncoder(cfg, sizes["n_vitals"] * 2 + 2, self.drug_emb)
        else:
            self.history = None

        if mlc.use_tabular:
            if mlc.tab_dim <= 0:
                raise ValueError("use_tabular requires tab_dim > 0")
            self.tab = nn.Sequential(
                nn.LayerNorm(mlc.tab_dim),
                nn.Linear(mlc.tab_dim, d * mlc.tab_tokens),
                nn.GELU(), nn.Dropout(cfg.dropout))
        else:
            self.tab = None

        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        # cls / text / vital / demo / geo / hist / tab
        self.type_emb = nn.Embedding(7, d)

        if cfg.use_transformer_fusion:
            layer = nn.TransformerEncoderLayer(
                d, cfg.n_heads, cfg.d_ff, cfg.dropout,
                batch_first=True, norm_first=True, activation="gelu")
            self.fusion = nn.TransformerEncoder(layer, cfg.n_fusion_layers)
        else:
            self.fusion = None
        self.enc_norm = nn.LayerNorm(d)

        if mlc.pool == "attn":
            self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        pool_dim = d * (2 if mlc.pool == "cls_mean" else 1)

        head: list[nn.Module] = [nn.Dropout(mlc.head_dropout)]
        if mlc.head_hidden > 0:
            head += [nn.Linear(pool_dim, mlc.head_hidden), nn.GELU(),
                     nn.Dropout(mlc.head_dropout)]
            pool_dim = mlc.head_hidden
        self.out = nn.Linear(pool_dim, n_labels)
        head.append(self.out)
        self.head = nn.Sequential(*head)

    # -- encoder ------------------------------------------------------------
    def encode(self, b):
        toks, masks = [], []
        B = b["demo"].size(0)
        dev = b["demo"].device
        t = self.type_emb.weight

        toks.append(self.cls.expand(B, -1, -1) + t[0])
        masks.append(torch.ones(B, 1, dtype=torch.bool, device=dev))

        def add(tok, m=None):
            toks.append(tok)
            masks.append(m if m is not None else torch.ones(
                B, tok.size(1), dtype=torch.bool, device=dev))

        if self.text is not None:
            x, tmask = self.text(b["word_ids"], b["char_ids"])
            add(x + t[1], tmask)
        if self.vitals is not None:
            add(self.vitals(b["vitals"], b["vitals_mask"]) + t[2])
        if self.demo is not None:
            add(self.demo(b["demo"]).unsqueeze(1) + t[3])
        if self.mlc.use_geo:
            geo = (self.district_emb(b["district"])
                   + self.glucose_emb(b["glucose_type"])).unsqueeze(1)
            add(geo + t[4])
        if self.history is not None:
            h = self.history(b["hist_feat"], b["hist_drugs"], b["hist_mask"])
            add(h.unsqueeze(1) + t[5])
        if self.tab is not None:
            tb = self.tab(b["tab"]).view(B, self.mlc.tab_tokens, -1)
            add(tb + t[6])

        x = torch.cat(toks, dim=1)
        mask = torch.cat(masks, dim=1)
        if self.fusion is not None:
            x = self.fusion(x, src_key_padding_mask=~mask)
        return self.enc_norm(x), mask

    # -- pooling ------------------------------------------------------------
    def pool(self, x, mask):
        p = self.mlc.pool
        if p == "cls":
            return x[:, 0]
        m = mask.unsqueeze(-1).float()
        if p in ("mean", "cls_mean"):
            mean = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
            return mean if p == "mean" else torch.cat([x[:, 0], mean], dim=-1)
        q = self.pool_q.expand(x.size(0), -1, -1)
        return self.pool_attn(q, x, x, key_padding_mask=~mask,
                              need_weights=False)[0].squeeze(1)

    def forward(self, b):
        x, mask = self.encode(b)
        return self.head(self.pool(x, mask))


def build_mlc(cfg: ModelConfig, mlc: MLCConfig, sizes: dict, n_labels: int,
              prior: torch.Tensor | None = None) -> RxMLC:
    m = RxMLC(cfg, mlc, sizes, n_labels)
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    if mlc.bias_init_prior and prior is not None:
        # Start each label at its train base rate. With 88 labels at a mean
        # prevalence of ~2.6%, a zero-init head spends its first epochs just
        # learning to say "no", and the early-stopping trace becomes noise.
        p = prior.clamp(1e-4, 1 - 1e-4)
        with torch.no_grad():
            m.out.bias.copy_(torch.log(p / (1 - p)))
    return m


# ---------------------------------------------------------------------------
# Ablation grid. Encoder switches reuse `ModelConfig`'s own flags, so an arm
# here means exactly what the same-named arm meant in `train.VARIANTS`.
# ---------------------------------------------------------------------------
VARIANTS: dict[str, dict] = {
    # required arms
    "full":              {},
    "no_fusion":         {"use_transformer_fusion": False},
    "text_only":         {"use_vitals": False, "use_history": False},
    "text_vitals":       {"use_history": False},
    # tabular-augmented arms: lr_tab_all beat lr_text, so withholding the
    # engineered columns from the neural model would confound the comparison
    "full_tab":          {"_tab": True},
    "full_tab_nofusion": {"_tab": True, "use_transformer_fusion": False},
    "text_tab":          {"_tab": True, "use_vitals": False, "use_history": False},
    "tab_only":          {"_tab": True, "use_text": False, "use_vitals": False,
                          "use_history": False},
    # floors / diagnostics
    "prior_only":        {"use_text": False, "use_vitals": False,
                          "use_history": False},
}


def variant_configs(name: str, base_model: ModelConfig, base_mlc: MLCConfig,
                    tab_dim: int) -> tuple[ModelConfig, MLCConfig]:
    spec = dict(VARIANTS[name])
    use_tab = bool(spec.pop("_tab", False))
    mcfg = replace(base_model, **spec)
    kw = {"use_tabular": use_tab, "tab_dim": tab_dim if use_tab else 0}
    return mcfg, replace(base_mlc, **kw)
