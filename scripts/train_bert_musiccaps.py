"""Train the text-only BERT baseline on MusicCaps captions.

MusicCaps provides captions rather than human tag annotations.  When a ``tags``
column is not present, this script creates a reproducible proxy task by
matching caption phrases against the small, documented vocabulary below.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Allow the documented ``python scripts/...`` invocation from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bert_encoder import BertTagClassifier
from src.metrics import multilabel_metrics
from src.utils import ensure_dir, save_json, seed_everything


PROXY_TAGS = {
    "acoustic": ("acoustic", "unplugged", "acoustic guitar"),
    "ambient": ("ambient", "atmospheric", "soundscape"),
    "blues": ("blues", "bluesy"),
    "classical": ("classical", "orchestra", "orchestral", "symphony"),
    "electronic": ("electronic", "synth", "synthesizer", "techno"),
    "folk": ("folk", "folksy", "banjo"),
    "hip hop": ("hip hop", "hip-hop", "rap"),
    "jazz": ("jazz", "saxophone", "swing"),
    "metal": ("metal", "heavy metal", "guitar riff"),
    "piano": ("piano", "keyboard"),
    "pop": ("pop", "catchy"),
    "rock": ("rock", "rock guitar", "distorted guitar"),
}
TAG_COLUMNS = ("tags", "labels", "label")
TEXT_COLUMNS = ("caption", "text", "description")


def _read_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("rows", []))
        if not isinstance(payload, list):
            raise ValueError("JSON input must contain a list of rows or a data/rows list.")
        return payload
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Reading parquet requires pandas, which is listed in requirements.txt.") from exc
        return pd.read_parquet(path).to_dict("records")
    raise ValueError(f"Unsupported input format {path.suffix!r}; use CSV, JSON, JSONL, or parquet.")


def _first_value(row: dict, names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def proxy_tags(caption: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", caption.lower())
    return [
        tag for tag, phrases in PROXY_TAGS.items()
        if any(re.search(rf"\b{re.escape(phrase)}\b", normalized) for phrase in phrases)
    ]


def _parse_tags(value: str) -> list[str]:
    if not value:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
        return [str(tag).strip().lower() for tag in parsed if str(tag).strip()]
    return [tag.strip().lower() for tag in re.split(r"[|,;]", value) if tag.strip()]


def prepare_rows(path: Path) -> tuple[list[dict], list[str]]:
    prepared = []
    for row in _read_rows(path):
        caption = _first_value(row, TEXT_COLUMNS)
        if not caption:
            continue
        tags = _parse_tags(_first_value(row, TAG_COLUMNS)) or proxy_tags(caption)
        if not tags:
            continue
        identifier = _first_value(row, ("ytid", "track_id", "id")) or caption
        prepared.append({"id": identifier, "text": caption, "tags": sorted(set(tags))})
    if not prepared:
        raise ValueError("No caption rows with at least one tag or proxy tag were found.")
    vocabulary = sorted({tag for row in prepared for tag in row["tags"]})
    for row in prepared:
        row["target"] = [float(tag in row["tags"]) for tag in vocabulary]
    return prepared, vocabulary


def split_rows(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """Split by stable row identity so rerunning never changes the examples."""
    buckets = {"train": [], "val": [], "test": []}
    for row in rows:
        digest = hashlib.sha256(f"{seed}:{row['id']}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        split = "train" if value < 0.8 else "val" if value < 0.9 else "test"
        buckets[split].append(row)
    if not all(buckets.values()):
        raise ValueError("The input must contain enough rows for non-empty train, val, and test splits.")
    return buckets


class CaptionDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        row = self.rows[index]
        return row["text"], torch.tensor(row["target"], dtype=torch.float32)


def _collate(batch: list[tuple[str, torch.Tensor]]) -> tuple[list[str], torch.Tensor]:
    texts, targets = zip(*batch)
    return list(texts), torch.stack(targets)


def run_epoch(model, loader, optimizer, device) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()
    losses, logits, targets = [], [], []
    for texts, target in loader:
        target = target.to(device)
        prediction = model(texts)
        loss = criterion(prediction, target)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        logits.append(prediction.detach().cpu().numpy())
        targets.append(target.cpu().numpy())
    return float(np.mean(losses)), multilabel_metrics(np.concatenate(logits), np.concatenate(targets))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="MusicCaps CSV, JSON, JSONL, or parquet file.")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="bert_musiccaps")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")

    seed_everything(args.seed)
    rows, vocabulary = prepare_rows(args.input)
    splits = split_rows(rows, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertTagClassifier(
        len(vocabulary),
        hidden_size=args.hidden_size,
        model_name=args.model_name,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
    ).to(device)
    loaders = {
        name: DataLoader(CaptionDataset(items), batch_size=args.batch_size, shuffle=name == "train", collate_fn=_collate)
        for name, items in splits.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history, best = [], -1.0
    output_dir = ensure_dir(args.output_dir)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, loaders["train"], optimizer, device)
        with torch.no_grad():
            val_loss, val_metrics = run_epoch(model, loaders["val"], None, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        print(record)
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            torch.save(
                {"model": model.state_dict(), "vocab": vocabulary, "model_name": args.model_name},
                output_dir / f"{args.run_name}_best.pt",
            )

    with torch.no_grad():
        test_loss, test_metrics = run_epoch(model, loaders["test"], None, device)
    save_json(
        {"dataset": "musiccaps", "proxy_task": True, "labels": vocabulary, "history": history},
        output_dir / f"{args.run_name}_metrics.json",
    )
    save_json({"loss": test_loss, **test_metrics}, output_dir / f"{args.run_name}_test_metrics.json")

    import matplotlib.pyplot as plt

    epochs = [record["epoch"] for record in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [record["val_macro_f1"] for record in history], label="validation Macro-F1")
    plt.plot(epochs, [record["val_micro_f1"] for record in history], label="validation Micro-F1")
    plt.xlabel("epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{args.run_name}_f1_curve.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
