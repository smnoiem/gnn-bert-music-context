from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure consistent, timestamped logs for command-line entry points."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logging.getLogger("music-context")


def progress(iterable, *, desc: str, total: int | None = None):
    """Render progress on stderr so logs and machine-readable stdout stay usable."""
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit="item",
        dynamic_ncols=True,
        file=sys.stderr,
        leave=False,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path); path.mkdir(parents=True, exist_ok=True); return path


def save_json(value: dict, path: str | Path) -> None:
    path = Path(path); ensure_dir(path.parent)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))
