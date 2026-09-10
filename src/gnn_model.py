from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int): super().__init__(); self.linear = nn.Linear(in_dim * 2, out_dim)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index; aggregate = torch.zeros_like(x).index_add_(0, dst, x[src]); degree = torch.zeros(x.size(0), device=x.device).index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype)).clamp_min(1)
        return F.relu(self.linear(torch.cat([x, aggregate / degree[:, None]], dim=-1)))


class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 128, layers: int = 2, dropout: float = .2):
        super().__init__(); dims = [input_dim] + [hidden_dim] * layers; self.layers = nn.ModuleList(GraphSAGELayer(dims[i], dims[i+1]) for i in range(layers)); self.dropout = dropout; self.hidden_size = hidden_dim
    def forward(self, graph: dict) -> torch.Tensor:
        x, edge = graph["x"], graph["edge_index"]
        for layer in self.layers: x = F.dropout(layer(x, edge), p=self.dropout, training=self.training)
        return x.mean(0)


class GNNClassifier(nn.Module):
    def __init__(self, num_labels: int, **kwargs): super().__init__(); self.encoder = GraphSAGEEncoder(**kwargs); self.head = nn.Linear(self.encoder.hidden_size, num_labels)
    def forward(self, graph: dict): return self.head(self.encoder(graph))


class MelCNN(nn.Module):
    """Audio-only fair baseline, taking a [batch, 1, mel, frames] tensor."""
    def __init__(self, num_labels: int):
        super().__init__(); self.features = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)); self.head = nn.Linear(64, num_labels)
    def forward(self, x): return self.head(self.features(x).flatten(1))
