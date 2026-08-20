"""PHC-RxGen: hierarchical multi-modal prescription generator.

Architecture
------------
Encoder (multi-modal -> a sequence of context tokens)
  * CharCNN      : character-level convolutions per word. Motivated directly by
                   the data -- `extra_symptom` is clinician shorthand riddled
                   with abbreviations ('D/M', 'H/O') and typos ('Loos motion',
                   'Incresed'), so a purely word-level lookup hits OOV often.
  * Word BiLSTM  : sequential composition over word+char representations.
  * VitalsEnc    : per-vital value/missingness embedding -> one token each, so
                   the fusion attention can address individual physiology.
  * HistoryGRU   : recurrent pass over prior encounters (vitals + prescribed
                   drug bag), producing one patient-trajectory token.
  * Transformer fusion encoder over [CLS] + text + vitals + demo + history.

Decoder (autoregressive over drug orders)
  * Transformer decoder cross-attending to the fused context; at each step it
    emits a drug id, then four attribute heads (type / dose / duration /
    instruction) conditioned on the decoder state.
  * A GRU decoder variant is provided for the RNN-vs-Transformer ablation.

Auxiliary heads (multi-task) predict the advice and test sets from [CLS];
they regularise the shared encoder on a 14k-sample corpus.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, PAD, BOS, EOS
from .data import ATTR_KEYS


# --------------------------------------------------------------------------
class CharCNN(nn.Module):
    """Character-level CNN producing a fixed vector per word."""

    def __init__(self, n_chars: int, emb: int, kernels, filters: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(n_chars, emb, padding_idx=PAD)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb, filters, k, padding=k // 2) for k in kernels])
        self.out_dim = filters * len(kernels)
        self.drop = nn.Dropout(dropout)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: (B, W, C)
        B, W, C = char_ids.shape
        x = self.emb(char_ids.view(B * W, C)).transpose(1, 2)   # (B*W, E, C)
        feats = [torch.relu(conv(x)).amax(dim=2) for conv in self.convs]
        return self.drop(torch.cat(feats, dim=1)).view(B, W, self.out_dim)


class TextEncoder(nn.Module):
    """CharCNN + word embedding -> BiLSTM over the symptom phrase."""

    def __init__(self, cfg: ModelConfig, n_words: int, n_chars: int):
        super().__init__()
        self.cfg = cfg
        self.word_emb = nn.Embedding(n_words, cfg.word_emb, padding_idx=PAD)
        in_dim = cfg.word_emb
        self.char_cnn = None
        if cfg.use_text_cnn:
            self.char_cnn = CharCNN(n_chars, cfg.char_emb, cfg.char_kernels,
                                    cfg.char_filters, cfg.dropout)
            in_dim += self.char_cnn.out_dim

        if cfg.use_text_rnn:
            self.rnn = nn.LSTM(in_dim, cfg.lstm_hidden, cfg.lstm_layers,
                               batch_first=True, bidirectional=True,
                               dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0)
            self.out_dim = cfg.lstm_hidden * 2
        else:
            self.rnn = None
            self.out_dim = in_dim
        self.proj = nn.Linear(self.out_dim, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, word_ids, char_ids):
        mask = word_ids.ne(PAD)                       # (B, W)
        x = self.word_emb(word_ids)
        if self.char_cnn is not None:
            x = torch.cat([x, self.char_cnn(char_ids)], dim=-1)
        x = self.drop(x)
        if self.rnn is not None:
            lengths = mask.sum(1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=word_ids.size(1))
        return self.proj(x), mask


class VitalsEncoder(nn.Module):
    """One token per vital: learned channel embedding gated by value+mask.

    Keeping vitals as separate tokens (rather than one dense vector) lets the
    fusion attention weight individual physiology against the symptom text.
    """

    def __init__(self, cfg: ModelConfig, n_vitals: int):
        super().__init__()
        self.channel = nn.Parameter(torch.randn(n_vitals, cfg.d_model) * 0.02)
        self.value = nn.Linear(2, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, vitals, vmask):
        # vitals/vmask: (B, V)
        vv = torch.stack([vitals, vmask], dim=-1)              # (B, V, 2)
        x = self.channel.unsqueeze(0) + self.value(vv)
        return self.drop(self.norm(x))


class HistoryEncoder(nn.Module):
    """GRU over prior encounters; prior drugs are mean-pooled drug embeddings."""

    def __init__(self, cfg: ModelConfig, feat_dim: int, drug_emb: nn.Embedding):
        super().__init__()
        self.drug_emb = drug_emb
        d = drug_emb.embedding_dim
        self.gru = nn.GRU(feat_dim + d, cfg.hist_hidden, batch_first=True)
        self.proj = nn.Linear(cfg.hist_hidden, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, hist_feat, hist_drugs, hist_mask):
        B, H, _ = hist_feat.shape
        dm = hist_drugs.ne(PAD).float().unsqueeze(-1)          # (B,H,R,1)
        de = self.drug_emb(hist_drugs) * dm
        de = de.sum(2) / dm.sum(2).clamp(min=1.0)              # (B,H,D)
        x = torch.cat([hist_feat, de], dim=-1)
        lengths = hist_mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = self.proj(self.drop(h[-1]))
        # Patients with no prior visit contribute a zero token.
        return h * (hist_mask.sum(1, keepdim=True) > 0).float()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------
class PHCRxGen(nn.Module):
    def __init__(self, cfg: ModelConfig, sizes: dict[str, int]):
        super().__init__()
        self.cfg, self.sizes = cfg, sizes
        d = cfg.d_model

        self.text = (TextEncoder(cfg, sizes["word"], sizes["char"])
                     if cfg.use_text else None)
        self.vitals = VitalsEncoder(cfg, sizes["n_vitals"]) if cfg.use_vitals else None

        self.demo = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Dropout(cfg.dropout))
        self.district_emb = nn.Embedding(sizes["district"], d)
        self.glucose_emb = nn.Embedding(sizes["glucose"], d)

        self.drug_emb = nn.Embedding(sizes["drug"], d, padding_idx=PAD)
        hist_feat_dim = sizes["n_vitals"] * 2 + 2
        self.history = (HistoryEncoder(cfg, hist_feat_dim, self.drug_emb)
                        if cfg.use_history else None)

        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.type_emb = nn.Embedding(6, d)   # modality tag: cls/text/vital/demo/geo/hist

        if cfg.use_transformer_fusion:
            layer = nn.TransformerEncoderLayer(
                d, cfg.n_heads, cfg.d_ff, cfg.dropout,
                batch_first=True, norm_first=True, activation="gelu")
            self.fusion = nn.TransformerEncoder(layer, cfg.n_fusion_layers)
        else:
            self.fusion = None
        self.enc_norm = nn.LayerNorm(d)

        # --- decoder ---
        self.pos = PositionalEncoding(d)
        if cfg.decoder_type == "transformer":
            dlayer = nn.TransformerDecoderLayer(
                d, cfg.n_heads, cfg.d_ff, cfg.dropout,
                batch_first=True, norm_first=True, activation="gelu")
            self.decoder = nn.TransformerDecoder(dlayer, cfg.n_decoder_layers)
        else:
            self.decoder = nn.GRU(d, d, cfg.n_decoder_layers,
                                  batch_first=True, dropout=cfg.dropout)
        self.dec_norm = nn.LayerNorm(d)

        self.drug_head = nn.Linear(d, sizes["drug"])
        self.drug_head.weight = self.drug_emb.weight        # tied embeddings
        # Keys are prefixed: ModuleDict forbids names that shadow nn.Module
        # attributes, and 'type' collides with Module.type().
        self.attr_heads = nn.ModuleDict(
            {f"attr_{k}": nn.Linear(d, sizes[f"attr_{k}"]) for k in ATTR_KEYS})
        # Hierarchical head: the pharmacological category of the drug emitted at
        # each step. 606/717 brands collapse into 90 categories, so this is the
        # clinically meaningful level and supervising it shapes the drug space.
        self.cat_head = nn.Linear(d, sizes["category"])
        self.advice_head = nn.Linear(d, sizes["advice"])
        self.test_head = nn.Linear(d, sizes["test"])

    # -- encoder ------------------------------------------------------------
    def encode(self, b):
        toks, masks = [], []
        B = b["word_ids"].size(0)
        dev = b["word_ids"].device

        cls = self.cls.expand(B, -1, -1) + self.type_emb.weight[0]
        toks.append(cls)
        masks.append(torch.ones(B, 1, dtype=torch.bool, device=dev))

        if self.text is not None:
            t, tmask = self.text(b["word_ids"], b["char_ids"])
            toks.append(t + self.type_emb.weight[1])
            masks.append(tmask)

        if self.vitals is not None:
            v = self.vitals(b["vitals"], b["vitals_mask"])
            toks.append(v + self.type_emb.weight[2])
            masks.append(torch.ones(B, v.size(1), dtype=torch.bool, device=dev))

        demo = self.demo(b["demo"]).unsqueeze(1) + self.type_emb.weight[3]
        toks.append(demo)
        masks.append(torch.ones(B, 1, dtype=torch.bool, device=dev))

        geo = (self.district_emb(b["district"]) + self.glucose_emb(b["glucose_type"])
               ).unsqueeze(1) + self.type_emb.weight[4]
        toks.append(geo)
        masks.append(torch.ones(B, 1, dtype=torch.bool, device=dev))

        if self.history is not None:
            h = self.history(b["hist_feat"], b["hist_drugs"], b["hist_mask"]
                             ).unsqueeze(1) + self.type_emb.weight[5]
            toks.append(h)
            masks.append(torch.ones(B, 1, dtype=torch.bool, device=dev))

        x = torch.cat(toks, dim=1)
        mask = torch.cat(masks, dim=1)
        if self.fusion is not None:
            x = self.fusion(x, src_key_padding_mask=~mask)
        return self.enc_norm(x), mask

    # -- decoder ------------------------------------------------------------
    def decode(self, mem, mem_mask, drug_in):
        x = self.pos(self.drug_emb(drug_in))
        if isinstance(self.decoder, nn.TransformerDecoder):
            T = drug_in.size(1)
            causal = torch.triu(torch.ones(T, T, device=drug_in.device, dtype=torch.bool), 1)
            h = self.decoder(x, mem, tgt_mask=causal,
                             memory_key_padding_mask=~mem_mask)
        else:
            # GRU ablation: condition the initial state on the pooled context.
            h0 = mem[:, 0].unsqueeze(0).repeat(self.decoder.num_layers, 1, 1).contiguous()
            h, _ = self.decoder(x, h0)
        return self.dec_norm(h)

    def forward(self, b):
        mem, mem_mask = self.encode(b)
        h = self.decode(mem, mem_mask, b["drug_in"])
        cls = mem[:, 0]
        return {
            "drug_logits": self.drug_head(h),
            "cat_logits": self.cat_head(h[:, :-1]),
            "attr_logits": {k: self.attr_heads[f"attr_{k}"](h[:, :-1]) for k in ATTR_KEYS},
            "advice_logits": self.advice_head(cls),
            "test_logits": self.test_head(cls),
        }

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def generate(self, b, max_len: int = 12, beam: int = 1, no_repeat: bool = True):
        """Greedy / beam decoding of the drug sequence + attributes."""
        mem, mem_mask = self.encode(b)
        B = mem.size(0)
        dev = mem.device
        seq = torch.full((B, 1), BOS, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        out_drugs = [[] for _ in range(B)]
        attr_out = {k: [[] for _ in range(B)] for k in ATTR_KEYS}

        for _ in range(max_len):
            h = self.decode(mem, mem_mask, seq)
            logits = self.drug_head(h[:, -1])
            logits[:, PAD] = -1e9
            logits[:, BOS] = -1e9
            if no_repeat:
                for i in range(B):
                    if out_drugs[i]:
                        logits[i, torch.tensor(out_drugs[i], device=dev)] = -1e9
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
            done = done | nxt.eq(EOS)
            step_h = h[:, -1]
            for k in ATTR_KEYS:
                a = self.attr_heads[f"attr_{k}"](step_h).argmax(-1)
                for i in range(B):
                    if not done[i]:
                        attr_out[k][i].append(int(a[i]))
            for i in range(B):
                if not done[i]:
                    out_drugs[i].append(int(nxt[i]))
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
            if bool(done.all()):
                break

        cls = mem[:, 0]
        return {
            "drugs": out_drugs,
            "attrs": attr_out,
            "advice_prob": torch.sigmoid(self.advice_head(cls)),
            "test_prob": torch.sigmoid(self.test_head(cls)),
        }


def build_model(cfg: ModelConfig, sizes: dict[str, int]) -> PHCRxGen:
    m = PHCRxGen(cfg, sizes)
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return m
