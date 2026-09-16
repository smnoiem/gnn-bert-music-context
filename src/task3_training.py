"""Task 3-only data, models, and evaluation helpers.

This module deliberately does not share the Task 2 training data path.  Task 3
manifests contain a fixed multihot ``target`` and non-target text, and graphs
must be collated before they are passed to GraphSAGE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .bert_encoder import BertTextEncoder
from .gnn_model import GraphSAGEEncoder
from .utils import progress


SPLITS = ("train", "val", "test")


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Task 3 manifest is empty: {path}")
    return rows


def load_labels(path: str | Path) -> list[str]:
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    labels = spec.get("labels") if isinstance(spec, dict) else spec
    if not isinstance(labels, list) or not labels or any(
        not isinstance(label, str) or not label.strip() for label in labels
    ):
        raise ValueError(f"Labels JSON must contain a non-empty labels list: {path}")
    if len(set(labels)) != len(labels):
        raise ValueError("Labels JSON contains duplicate labels")
    return labels


def _target_for_row(row: dict, labels: Sequence[str]) -> list[float]:
    if "target" in row:
        target = row["target"]
        if not isinstance(target, (list, tuple)) or len(target) != len(labels):
            raise ValueError(
                f"Target for track {row.get('track_id')} must have {len(labels)} values"
            )
        if any(float(value) not in (0.0, 1.0) for value in target):
            raise ValueError(f"Non-binary target for track {row.get('track_id')}")
        return [float(value) for value in target]
    values = row.get("labels", [])
    if isinstance(values, str):
        values = [value for value in values.split("|") if value.strip()]
    if not isinstance(values, list):
        raise ValueError(f"Missing labels for track {row.get('track_id')}")
    values = set(str(value) for value in values)
    return [float(label in values) for label in labels]


def validate_manifest(
    rows: Sequence[dict], labels: Sequence[str], manifest: str | Path
) -> None:
    """Validate paths, canonical splits, targets, and artist split isolation."""
    parent = Path(manifest).resolve().parent
    counts = {split: 0 for split in SPLITS}
    artists: dict[str, str] = {}
    track_ids: set[str] = set()
    for row in progress(rows, desc="Validating Task 3 manifest", total=len(rows)):
        track_id = str(row.get("track_id", "")).strip()
        if not track_id or track_id in track_ids:
            raise ValueError(f"Missing or duplicate track_id in {manifest}")
        track_ids.add(track_id)
        split = str(row.get("split", "")).strip()
        if split not in SPLITS:
            raise ValueError(
                f"Invalid split {split!r} for track {track_id}; expected {SPLITS}"
            )
        counts[split] += 1
        graph = Path(str(row.get("graph", "")))
        if not graph.is_absolute():
            graph = parent / graph
        if not graph.is_file():
            raise FileNotFoundError(f"Graph file was not found: {graph}")
        text = str(row.get("bert_text", row.get("text", ""))).strip()
        if not text:
            raise ValueError(f"Empty bert_text for track {track_id}")
        _target_for_row(row, labels)
        artist = row.get("artist_id")
        if artist is not None and str(artist).strip():
            artist = str(artist)
            previous = artists.setdefault(artist, split)
            if previous != split:
                raise ValueError(f"Artist leakage detected for artist_id {artist}")
    missing = [split for split, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Task 3 manifest has empty split(s): {', '.join(missing)}")


class Task3GraphDataset(Dataset):
    """A verified split of the Task 3 graph/text/multilabel manifest."""

    def __init__(
        self,
        manifest: str | Path,
        labels: Sequence[str],
        split: str,
        rows: Sequence[dict] | None = None,
    ):
        self.manifest = Path(manifest).resolve()
        self.labels = list(labels)
        all_rows = list(rows) if rows is not None else load_jsonl(self.manifest)
        if rows is None:
            validate_manifest(all_rows, self.labels, self.manifest)
        self.rows = [row for row in all_rows if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"No Task 3 rows found for split {split!r}")
        self.split = split
        self._graph_paths = [self._resolve_graph(row["graph"]) for row in self.rows]
        self._input_dim: int | None = None

    def _resolve_graph(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.manifest.parent / path
        if not path.is_file():
            raise FileNotFoundError(f"Graph file was not found: {path}")
        return path

    @property
    def input_dim(self) -> int:
        if self._input_dim is None:
            graph = self._load(0)
            self._input_dim = int(graph["x"].shape[1])
        return self._input_dim

    def _load(self, index: int) -> dict:
        graph = torch.load(self._graph_paths[index], weights_only=False)
        if not isinstance(graph, dict) or "x" not in graph or "edge_index" not in graph:
            raise ValueError(f"Graph must contain x and edge_index: {self._graph_paths[index]}")
        x = torch.as_tensor(graph["x"], dtype=torch.float32)
        edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)
        if x.ndim != 2:
            raise ValueError(f"Graph x must be [nodes, features]: {self._graph_paths[index]}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"Graph edge_index must be [2, edges]: {self._graph_paths[index]}"
            )
        return {"x": x, "edge_index": edge_index}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        graph = self._load(index)
        if graph["x"].shape[1] != self.input_dim:
            raise ValueError("All Task 3 graphs must have the same feature dimension")
        graph["target"] = torch.tensor(
            _target_for_row(row, self.labels), dtype=torch.float32
        )
        graph["text"] = str(row.get("bert_text", row.get("text", ""))).strip()
        graph["track_id"] = row["track_id"]
        return graph


def collate_task3_graphs(samples: list[dict]) -> dict:
    """Concatenate graph nodes and offset edges for a mini-batch."""
    if not samples:
        raise ValueError("Cannot collate an empty Task 3 batch")
    node_offset = 0
    xs, edges, batch = [], [], []
    for graph_index, sample in enumerate(samples):
        x = sample["x"]
        edge_index = sample["edge_index"]
        xs.append(x)
        edges.append(edge_index + node_offset)
        batch.append(torch.full((x.shape[0],), graph_index, dtype=torch.long))
        node_offset += x.shape[0]
    return {
        "graph": {
            "x": torch.cat(xs, dim=0),
            "edge_index": torch.cat(edges, dim=1),
            "batch": torch.cat(batch, dim=0),
        },
        "texts": [sample["text"] for sample in samples],
        "targets": torch.stack([sample["target"] for sample in samples]),
        "track_ids": [sample["track_id"] for sample in samples],
    }


def positive_weight(dataset: Task3GraphDataset) -> torch.Tensor:
    """Compute ``negative / positive`` weights using only the given split."""
    targets = torch.stack(
        [
            dataset[index]["target"]
            for index in progress(
                range(len(dataset)),
                desc=f"Computing {dataset.split} positive weights",
                total=len(dataset),
            )
        ]
    )
    positives = targets.sum(dim=0)
    weights = torch.where(positives > 0, (len(dataset) - positives) / positives, torch.ones_like(positives))
    return weights.float()


class Task3FusionModel(nn.Module):
    """Batched GNN/BERT fusion model with cross-attention or early concat."""

    def __init__(
        self,
        num_labels: int,
        graph_input: int,
        graph_hidden: int = 256,
        graph_layers: int = 3,
        text_hidden: int = 256,
        text_model: str = "distilbert-base-uncased",
        freeze_text: bool = False,
        max_length: int = 128,
        cross_attention: bool = True,
        dropout: float = 0.2,
        local_files_only: bool = False,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.graph = GraphSAGEEncoder(
            input_dim=graph_input,
            hidden_dim=graph_hidden,
            layers=graph_layers,
            dropout=dropout,
        )
        self.text = BertTextEncoder(
            model_name=text_model,
            hidden_size=text_hidden,
            freeze=freeze_text,
            local_files_only=local_files_only,
            max_length=max_length,
        )
        self.graph_size = graph_hidden * 3
        self.cross_attention = cross_attention
        self.attention = None
        if cross_attention:
            if text_hidden % attention_heads:
                raise ValueError("text_hidden must be divisible by attention_heads")
            self.query = nn.Linear(self.graph_size, text_hidden)
            self.attention = nn.MultiheadAttention(
                text_hidden, attention_heads, batch_first=True
            )
        combined = self.graph_size + text_hidden
        self.head = nn.Sequential(
            nn.Linear(combined, combined),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined, num_labels),
        )

    def fused_embedding(self, graph: dict, texts: Sequence[str]) -> torch.Tensor:
        graph_embedding = self.graph(graph)
        tokens, pooled = self.text(list(texts), return_tokens=True)
        if self.cross_attention:
            query = self.query(graph_embedding).unsqueeze(1)
            semantic = self.attention(query, tokens, tokens, need_weights=False)[0].squeeze(1)
        else:
            semantic = pooled
        return torch.cat((graph_embedding, semantic), dim=1)

    def forward(self, graph: dict, texts: Sequence[str]) -> torch.Tensor:
        return self.head(self.fused_embedding(graph, texts))


class Task3BertOnlyModel(nn.Module):
    def __init__(self, num_labels: int, **kwargs):
        super().__init__()
        self.text = BertTextEncoder(**kwargs)
        self.head = nn.Linear(self.text.hidden_size, num_labels)

    def forward(self, graph: dict, texts: Sequence[str]) -> torch.Tensor:
        return self.head(self.text(list(texts)))


class Task3GNNOnlyModel(nn.Module):
    def __init__(self, num_labels: int, graph_input: int, **kwargs):
        super().__init__()
        hidden = kwargs.pop("graph_hidden", 256)
        self.graph = GraphSAGEEncoder(
            input_dim=graph_input,
            hidden_dim=hidden,
            layers=kwargs.pop("graph_layers", 3),
            dropout=kwargs.pop("dropout", 0.2),
        )
        self.head = nn.Linear(hidden * 3, num_labels)

    def forward(self, graph: dict, texts: Sequence[str]) -> torch.Tensor:
        return self.head(self.graph(graph))


def multilabel_metrics(logits: Iterable, targets: Iterable) -> dict[str, float]:
    scores = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    actual = np.asarray(targets).astype(int)
    predicted = (scores >= 0.5).astype(int)
    tp = ((actual == 1) & (predicted == 1)).sum()
    fp = ((actual == 0) & (predicted == 1)).sum()
    fn = ((actual == 1) & (predicted == 0)).sum()
    micro_f1 = float(2 * tp / max(2 * tp + fp + fn, 1))
    per_label = []
    average_precisions = []
    for index in range(actual.shape[1]):
        y, p = actual[:, index], predicted[:, index]
        ltp = ((y == 1) & (p == 1)).sum()
        lfp = ((y == 0) & (p == 1)).sum()
        lfn = ((y == 1) & (p == 0)).sum()
        per_label.append(float(2 * ltp / max(2 * ltp + lfp + lfn, 1)))
        if y.sum():
            order = np.argsort(-scores[:, index])
            ranked = y[order]
            precision = np.cumsum(ranked) / np.arange(1, len(y) + 1)
            average_precisions.append(float((precision * ranked).sum() / y.sum()))
    mean_ap = float(np.mean(average_precisions)) if average_precisions else float("nan")
    return {
        "macro_f1": float(np.mean(per_label)),
        "micro_f1": micro_f1,
        "mean_ap": mean_ap,
        "auc_pr": mean_ap,
    }


# Descriptive aliases make the Task 3-only API convenient for small fixture
# tests and downstream experiments without coupling them to Task 2 classes.
Task3Dataset = Task3GraphDataset
collate_graphs = collate_task3_graphs
compute_pos_weight = positive_weight
