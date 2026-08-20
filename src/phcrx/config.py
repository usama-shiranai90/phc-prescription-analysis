"""Central configuration for the PHC prescription-generation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

# Project root = two levels above this file (src/phcrx/config.py -> project root)
ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results" / "rx_generation"

for _p in (PROCESSED, MODELS, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

# --- Physiologically plausible ranges. Values outside these are treated as
# --- recording errors and set to missing (the mask bit then carries the signal).
VITAL_RANGES: dict[str, tuple[float, float]] = {
    "height": (60.0, 220.0),          # cm
    "weight": (10.0, 250.0),          # kg
    "bmi": (10.0, 60.0),              # kg/m^2
    "waist_hip_ratio": (0.4, 1.8),
    "temperature": (90.0, 110.0),     # F (harmonised from C upstream)
    "oxygen_of_blood": (50.0, 100.0), # % SpO2
    "bp_sys": (60.0, 260.0),          # mmHg
    "bp_dia": (30.0, 180.0),          # mmHg
    "blood_glucose": (20.0, 600.0),   # mg/dL (harmonised from mmol/L upstream)
    "blood_hemoglobin": (3.0, 22.0),  # g/dL
    "pulse_rate": (30.0, 200.0),      # bpm
    "cholesterol": (50.0, 500.0),     # mg/dL
    "uric_acid": (1.0, 20.0),         # mg/dL
}
VITAL_COLS: list[str] = list(VITAL_RANGES)


@dataclass
class DataConfig:
    max_symptom_words: int = 48      # p99 of symptom length is ~150 chars
    max_word_chars: int = 16         # char-CNN window per word
    max_rx_len: int = 12             # observed max drugs per prescription
    max_history: int = 4             # prior encounters fed to the history RNN
    min_word_freq: int = 2           # word vocab cutoff (train split only)
    min_char_freq: int = 5
    split: str = "patient"           # "patient" | "temporal"
    val_frac: float = 0.10
    test_frac: float = 0.20
    temporal_train_end: int = 2015   # used when split == "temporal"
    temporal_val_end: int = 2016
    seed: int = 42


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 4
    n_fusion_layers: int = 3
    n_decoder_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.2

    # Char-CNN over words (handles the typos/abbreviations in extra_symptom)
    char_emb: int = 32
    char_kernels: tuple[int, ...] = (2, 3, 4, 5)
    char_filters: int = 32           # per kernel -> 128 total

    # Word-level BiLSTM
    word_emb: int = 128
    lstm_hidden: int = 128
    lstm_layers: int = 1

    # History GRU over prior encounters
    hist_hidden: int = 128

    # Which encoder branches are active (ablation switches).
    # use_text gates the ENTIRE text branch; use_text_cnn/rnn only swap the
    # composition layers and leave word embeddings in place.
    use_text: bool = True
    use_text_cnn: bool = True
    use_text_rnn: bool = True
    use_vitals: bool = True
    use_history: bool = True
    use_transformer_fusion: bool = True
    decoder_type: str = "transformer"  # "transformer" | "gru"


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_frac: float = 0.06
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    patience: int = 10               # early stopping on val drug micro-F1
    # Multi-task loss weights (advice/test heads regularise the shared encoder)
    w_drug: float = 1.0
    w_attr: float = 0.3
    w_cat: float = 0.5               # hierarchical category head (brand -> class)
    w_advice: float = 0.2
    w_test: float = 0.2
    amp: bool = True
    num_workers: int = 2
    seeds: tuple[int, ...] = (0, 1, 2)   # repeated runs -> mean +/- std


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, default=str))


# Special token ids shared by the drug vocabulary and the decoder.
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
# "unspecified" class for structured attributes that are genuinely absent
# (instruction missing 25%, size 29%, duration-unit 11% of orders).
NA = 0
