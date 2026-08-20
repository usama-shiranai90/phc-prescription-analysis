"""Shared checkpoint loader, so the NLP modules can score PHC-RxGen."""
from __future__ import annotations

import torch

from ..config import ModelConfig
from ..model import build_model


def load_rxgen(path, corpus):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    sizes = ck.get("sizes", corpus.sizes)
    if sizes != corpus.sizes:
        raise SystemExit(
            f"Checkpoint was trained on the '{ck.get('split','?')}' split; its "
            f"vocabularies do not match the processed data.")
    mcfg = ModelConfig(**{k: v for k, v in ck["model_cfg"].items()
                          if k in ModelConfig.__dataclass_fields__})
    model = build_model(mcfg, sizes).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, device
