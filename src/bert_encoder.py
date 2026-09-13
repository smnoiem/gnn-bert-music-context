from __future__ import annotations

import torch
from torch import nn


class BertTextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", hidden_size: int = 256, freeze: bool = False, local_files_only: bool = False, max_length: int = 128):
        super().__init__(); self.backend = "transformers"; self.max_length = max_length
        try:
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only); source = self.model.config.hidden_size
            self.project = nn.Identity() if source == hidden_size else nn.Linear(source, hidden_size)
            if freeze:
                for p in self.model.parameters(): p.requires_grad = False
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load HuggingFace model {model_name!r}. "
                "Install transformers and allow model downloads, or use a locally cached checkpoint."
            ) from exc
        self.hidden_size = hidden_size
    def forward(self, texts: list[str], return_tokens: bool = False):
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(next(self.model.parameters()).device)
        out = self.model(**batch).last_hidden_state; token = self.project(out); pooled = token[:, 0]
        return (token, pooled) if return_tokens else pooled


class BertTagClassifier(nn.Module):
    def __init__(self, num_labels: int, **encoder_kwargs):
        super().__init__(); self.encoder = BertTextEncoder(**encoder_kwargs); self.head = nn.Linear(self.encoder.hidden_size, num_labels)
    def forward(self, texts: list[str]): return self.head(self.encoder(texts))
