from __future__ import annotations

from pathlib import Path
import json
import torch
from torch.utils.data import Dataset


def label_vocabulary(rows: list[dict]) -> list[str]:
    return sorted({tag for row in rows for tag in str(row.get("labels", "")).split("|") if tag})


def encode_labels(tags: str, vocab: list[str]) -> torch.Tensor:
    tagged = set(str(tags).split("|")); return torch.tensor([x in tagged for x in vocab], dtype=torch.float32)


def assert_no_artist_leakage(rows: list[dict]) -> None:
    seen = {}
    for row in rows:
        artist, split = row.get("artist_id"), row.get("split")
        if artist and artist in seen and seen[artist] != split: raise ValueError(f"artist leakage: {artist} in {seen[artist]} and {split}")
        if artist: seen[artist] = split


class MusicGraphDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str | None = None, vocab: list[str] | None = None):
        with open(manifest) as f: rows = [json.loads(x) for x in f if x.strip()]
        assert_no_artist_leakage(rows)
        self.rows = [r for r in rows if split is None or r.get("split") == split]
        self.vocab = vocab or label_vocabulary(rows)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]; graph = torch.load(row["graph"], weights_only=False)
        graph["y"] = encode_labels(row["labels"], self.vocab); graph["text"] = row.get("text", "")
        graph["track_id"] = row["track_id"]
        if "valence" in row: graph["emotion"] = torch.tensor([float(row["valence"]), float(row["arousal"])])
        return graph
