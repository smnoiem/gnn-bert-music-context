from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_max_pool, global_mean_pool


class GraphSAGEEncoder(nn.Module):
    """Residual GraphSAGE encoder with a multi-statistic graph readout."""

    def __init__(
        self,
        input_dim: int = 182,
        hidden_dim: int = 256,
        layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least one")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1)")
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(min(dropout, 0.1)),
        )
        self.layers = nn.ModuleList(
            SAGEConv(hidden_dim, hidden_dim) for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.dropout = dropout
        self.hidden_size = hidden_dim

    def forward(self, graph: dict) -> torch.Tensor:
        x, edge_index = graph["x"], graph["edge_index"]
        batch = graph.get("batch")
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x, edge_index)
            x = F.relu(norm(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        mean = global_mean_pool(x, batch)
        maximum = global_max_pool(x, batch)
        second_moment = global_mean_pool(x.square(), batch)
        standard_deviation = (second_moment - mean.square()).clamp_min(1e-6).sqrt()
        return torch.cat((mean, maximum, standard_deviation), dim=1)




class GenreGraphSAGEClassifier(nn.Module):
    """Task 2 GraphSAGE classifier for single-label genre prediction."""

    def __init__(
        self,
        num_genres: int,
        input_dim: int = 182,
        hidden_dim: int = 256,
        layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
        )
        representation_size = self.encoder.hidden_size * 3
        self.head = nn.Sequential(
            nn.Linear(representation_size, self.encoder.hidden_size),
            nn.LayerNorm(self.encoder.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.hidden_size, num_genres),
        )

    def forward(self, graph: dict) -> torch.Tensor:
        return self.head(self.encoder(graph))


class MelCNN(nn.Module):
    """Audio-only fair baseline, taking a [batch, 1, mel, frames] tensor."""
    def __init__(self, num_labels: int):
        super().__init__(); self.features = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)); self.head = nn.Linear(64, num_labels)
    def forward(self, x): return self.head(self.features(x).flatten(1))
