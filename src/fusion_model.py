from __future__ import annotations

import torch
from torch import nn

from .bert_encoder import BertTextEncoder
from .gnn_model import GraphSAGEEncoder


class FusionModel(nn.Module):
    """Task 3 paired GNN-BERT classifier; cross_attention=False is early-concat ablation."""
    def __init__(self, num_labels: int, graph_input: int = 32, graph_hidden: int = 128, text_hidden: int = 256, cross_attention: bool = True, emotion_head: bool = False, **kwargs):
        super().__init__(); self.graph = GraphSAGEEncoder(graph_input, graph_hidden, **{k:v for k,v in kwargs.items() if k in {"layers", "dropout"}})
        self.text = BertTextEncoder(hidden_size=text_hidden, **{k:v for k,v in kwargs.items() if k in {"model_name", "freeze", "local_files_only"}})
        self.cross_attention = cross_attention; self.query = nn.Linear(graph_hidden, text_hidden); self.attention = nn.MultiheadAttention(text_hidden, 4, batch_first=True)
        combined = graph_hidden + text_hidden; self.head = nn.Sequential(nn.Linear(combined, combined), nn.ReLU(), nn.Dropout(.2), nn.Linear(combined, num_labels)); self.emotion = nn.Linear(combined, 2) if emotion_head else None
    def forward(self, graph: dict, text: str, return_attention: bool = False):
        g = self.graph(graph).unsqueeze(0); tokens, pooled = self.text([text], return_tokens=True)
        if self.cross_attention: attended, weights = self.attention(self.query(g).unsqueeze(1), tokens, tokens, need_weights=True); semantic = attended.squeeze(1)
        else: semantic, weights = pooled, None
        z = torch.cat([g, semantic], dim=-1); logits = self.head(z); output = {"logits": logits, "embedding": z}
        if self.emotion: output["emotion"] = self.emotion(z)
        if return_attention: output["attention"] = weights
        return output
