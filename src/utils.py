from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path); path.mkdir(parents=True, exist_ok=True); return path


def save_json(value: dict, path: str | Path) -> None:
    path = Path(path); ensure_dir(path.parent)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))
