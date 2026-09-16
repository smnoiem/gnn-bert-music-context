from __future__ import annotations

import torch
from torch import nn

from .bert_encoder import BertTextEncoder
from .gnn_model import GraphSAGEEncoder


class FusionModel(nn.Module):
    """Task 3 paired GNN-BERT classifier; cross_attention=False is early-concat ablation."""
    def __init__(
        self,
        num_labels: int,
        graph_input: int = 182,
        graph_hidden: int = 128,
        text_hidden: int = 256,
        cross_attention: bool = True,
        emotion_head: bool = False,
        **kwargs,
    ):
        super().__init__()
        graph_kwargs = {key: kwargs[key] for key in ("layers", "dropout") if key in kwargs}
        text_kwargs = {
            key: kwargs[key]
            for key in ("model_name", "freeze", "local_files_only", "max_length")
            if key in kwargs
        }
        self.graph = GraphSAGEEncoder(graph_input, graph_hidden, **graph_kwargs)
        self.text = BertTextEncoder(hidden_size=text_hidden, **text_kwargs)
        self.graph_size = self.graph.hidden_size * 3
        self.cross_attention = cross_attention
        if text_hidden % 4:
            raise ValueError("text_hidden must be divisible by four for cross-attention")
        self.query = nn.Linear(self.graph_size, text_hidden)
        self.attention = nn.MultiheadAttention(
            text_hidden, 4, batch_first=True
        )
        combined = self.graph_size + text_hidden
        dropout = kwargs.get("dropout", 0.2)
        self.head = nn.Sequential(
            nn.Linear(combined, combined),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined, num_labels),
        )
        self.emotion = nn.Linear(combined, 2) if emotion_head else None

    def forward(self, graph: dict, text: str, return_attention: bool = False):
        graph_embedding = self.graph(graph)
        if graph_embedding.size(0) != 1:
            raise ValueError("FusionModel.forward expects one graph and one text")
        graph_embedding = graph_embedding.unsqueeze(1)
        tokens, pooled = self.text([text], return_tokens=True)
        if self.cross_attention:
            attended, weights = self.attention(
                self.query(graph_embedding), tokens, tokens, need_weights=True
            )
            semantic = attended.squeeze(1)
        else:
            semantic, weights = pooled, None
        embedding = torch.cat([graph_embedding.squeeze(1), semantic], dim=-1)
        output = {"logits": self.head(embedding), "embedding": embedding}
        if self.emotion is not None:
            output["emotion"] = self.emotion(embedding)
        if return_attention:
            output["attention"] = weights
        return output
