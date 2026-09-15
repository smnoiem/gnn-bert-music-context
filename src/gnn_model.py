from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder implemented with PyTorch Geometric SAGEConv layers."""

    def __init__(
        self,
        input_dim: int = 182,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least one")
        dimensions = [input_dim] + [hidden_dim] * layers
        self.layers = nn.ModuleList(
            SAGEConv(dimensions[index], dimensions[index + 1])
            for index in range(layers)
        )
        self.dropout = dropout
        self.hidden_size = hidden_dim

    def forward(self, graph: dict) -> torch.Tensor:
        x, edge_index = graph["x"], graph["edge_index"]
        batch = graph.get("batch")
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)




class GenreGraphSAGEClassifier(nn.Module):
    """Task 2 GraphSAGE classifier for single-label genre prediction."""

    def __init__(
        self,
        num_genres: int,
        input_dim: int = 182,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
        )
        self.head = nn.Linear(self.encoder.hidden_size, num_genres)

    def forward(self, graph: dict) -> torch.Tensor:
        return self.head(self.encoder(graph))


class MelCNN(nn.Module):
    """Audio-only fair baseline, taking a [batch, 1, mel, frames] tensor."""
    def __init__(self, num_labels: int):
        super().__init__(); self.features = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)); self.head = nn.Linear(64, num_labels)
    def forward(self, x): return self.head(self.features(x).flatten(1))
