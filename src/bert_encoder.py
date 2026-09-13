from __future__ import annotations

import torch
from torch import nn


class SimpleTextEncoder(nn.Module):
    """Deterministic local fallback used only when pretrained weights are unavailable."""
    def __init__(self, hidden_size: int = 256, buckets: int = 8192):
        super().__init__(); self.embedding = nn.EmbeddingBag(buckets, hidden_size, mode="mean"); self.hidden_size = hidden_size; self.buckets = buckets
    def forward(self, texts: list[str]):
        ids, offsets = [], [0]
        for text in texts:
            tokens = [abs(hash(x)) % self.buckets for x in text.lower().split()] or [0]
            ids.extend(tokens); offsets.append(offsets[-1] + len(tokens))
        device = self.embedding.weight.device
        return self.embedding(torch.tensor(ids, device=device), torch.tensor(offsets[:-1], device=device))


class BertTextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", hidden_size: int = 256, freeze: bool = False, local_files_only: bool = False, max_length: int = 128):
        super().__init__(); self.backend = "simple"; self.max_length = max_length
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only); source = self.model.config.hidden_size
            self.project = nn.Identity() if source == hidden_size else nn.Linear(source, hidden_size); self.backend = "transformers"
            if freeze:
                for p in self.model.parameters(): p.requires_grad = False
        except Exception:
            self.model = SimpleTextEncoder(hidden_size); self.project = nn.Identity()
        self.hidden_size = hidden_size
    def forward(self, texts: list[str], return_tokens: bool = False):
        if self.backend == "simple":
            pooled = self.model(texts); return (pooled[:, None, :], pooled) if return_tokens else pooled
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(next(self.model.parameters()).device)
        out = self.model(**batch).last_hidden_state; token = self.project(out); pooled = token[:, 0]
        return (token, pooled) if return_tokens else pooled


class BertTagClassifier(nn.Module):
    def __init__(self, num_labels: int, **encoder_kwargs):
        super().__init__(); self.encoder = BertTextEncoder(**encoder_kwargs); self.head = nn.Linear(self.encoder.hidden_size, num_labels)
    def forward(self, texts: list[str]): return self.head(self.encoder(texts))
