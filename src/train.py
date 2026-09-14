from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import yaml

from .bert_encoder import BertTagClassifier
from .fusion_model import FusionModel
from .gnn_model import GNNClassifier
from .metrics import multilabel_metrics
from .utils import ensure_dir, save_json, seed_everything


def load_manifest(manifest: str | Path) -> list[dict]:
    with open(manifest, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest}")
    return rows


def label_vocabulary(rows: list[dict]) -> list[str]:
    return sorted(
        {
            tag
            for row in rows
            for tag in str(row.get("labels", "")).split("|")
            if tag
        }
    )


def genre_vocabulary(rows: list[dict]) -> list[str]:
    genres = sorted({str(row.get("genre", "")).strip() for row in rows})
    if not genres or "" in genres:
        raise ValueError("Every graph manifest row must contain a non-empty genre")
    return genres


def encode_genre(genre: str, vocabulary: list[str]) -> torch.Tensor:
    try:
        return torch.tensor(vocabulary.index(str(genre).strip()), dtype=torch.long)
    except ValueError as exc:
        raise ValueError(f"Unknown genre {genre!r}; expected one of {vocabulary}") from exc


def assert_no_artist_leakage(rows: list[dict]) -> None:
    seen = {}
    for row in rows:
        artist, split = row.get("artist_id"), row.get("split")
        if artist and artist in seen and seen[artist] != split:
            raise ValueError(
                f"artist leakage: {artist} in {seen[artist]} and {split}"
            )
        if artist:
            seen[artist] = split


class MusicGraphDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        split: str | None = None,
        vocab: list[str] | None = None,
    ):
        rows = load_manifest(manifest)
        assert_no_artist_leakage(rows)
        self.rows = [row for row in rows if split is None or row.get("split") == split]
        self.vocab = vocab or label_vocabulary(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        graph = torch.load(row["graph"], weights_only=False)
        tagged = set(str(row["labels"]).split("|"))
        graph["y"] = torch.tensor(
            [label in tagged for label in self.vocab], dtype=torch.float32
        )
        graph["text"] = row.get("text", "")
        graph["track_id"] = row["track_id"]
        if "valence" in row:
            graph["emotion"] = torch.tensor(
                [float(row["valence"]), float(row["arousal"])]
            )
        return graph


class GenreGraphDataset(Dataset):
    """Load Task 2 graph samples with single-label genre targets."""

    def __init__(
        self,
        manifest: str | Path,
        split: str | None = None,
        vocabulary: list[str] | None = None,
    ):
        rows = load_manifest(manifest)
        assert_no_artist_leakage(rows)
        self.rows = [row for row in rows if split is None or row.get("split") == split]
        if not self.rows:
            raise ValueError(f"No graph samples found for split {split!r}")
        self.vocabulary = vocabulary or genre_vocabulary(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        graph_path = Path(row["graph"])
        if not graph_path.is_file():
            raise FileNotFoundError(f"Graph file was not found: {graph_path}")
        graph = torch.load(graph_path, weights_only=False)
        if "x" not in graph or "edge_index" not in graph:
            raise ValueError(f"Graph is missing x or edge_index: {graph_path}")
        graph["y"] = encode_genre(row["genre"], self.vocabulary)
        graph["track_id"] = int(row["track_id"])
        graph["genre"] = str(row["genre"])
        return graph


def device_graph(graph, device): return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in graph.items()}

def run_epoch(model, data, optimizer, task, device):
    training = optimizer is not None; model.train(training); criterion = nn.BCEWithLogitsLoss(); losses=[]; logits=[]; targets=[]
    for graph in data:
        graph = device_graph(graph, device); y = graph["y"].unsqueeze(0).to(device)
        if task == "bert": prediction = model([graph["text"]])
        elif task == "gnn": prediction = model(graph).unsqueeze(0)
        else: prediction = model(graph, graph["text"])["logits"]
        loss = criterion(prediction, y)
        if task == "fusion" and "emotion" in graph:
            output = model(graph, graph["text"]); loss = loss + .1 * nn.functional.mse_loss(output["emotion"], graph["emotion"].unsqueeze(0))
        if training: optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item()); logits.append(prediction.detach().cpu().numpy()[0]); targets.append(y.cpu().numpy()[0])
    return float(np.mean(losses)), multilabel_metrics(np.asarray(logits), np.asarray(targets))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--task", choices=["bert","gnn","fusion"], required=True); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--manifest", default="data/processed/manifest.jsonl"); ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--epochs", type=int); ap.add_argument("--early-concat", action="store_true"); ap.add_argument("--run-name"); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config)); seed_everything(cfg["seed"])
    if args.synthetic: args.manifest="data/processed/manifest.jsonl"
    train=MusicGraphDataset(args.manifest, "train"); val=MusicGraphDataset(args.manifest, "val", train.vocab)
    num_labels=len(train.vocab); model_args=dict(num_labels=num_labels, text_hidden=cfg["model"]["text_hidden"], graph_hidden=cfg["model"]["gnn_hidden"], layers=cfg["model"]["gnn_layers"], dropout=cfg["model"]["dropout"], model_name=cfg["model"]["text_model"], freeze=cfg["training"]["freeze_text_encoder"])
    if args.task == "bert": model=BertTagClassifier(num_labels, hidden_size=cfg["model"]["text_hidden"], model_name=cfg["model"]["text_model"], freeze=cfg["training"]["freeze_text_encoder"], local_files_only=args.synthetic)
    elif args.task == "gnn": model=GNNClassifier(num_labels, hidden_dim=cfg["model"]["gnn_hidden"], layers=cfg["model"]["gnn_layers"])
    else: model=FusionModel(**model_args, cross_attention=not args.early_concat, local_files_only=args.synthetic)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device); opt=torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    results=ensure_dir("results"); run_name=args.run_name or args.task; history=[]; best=-1
    for epoch in range(args.epochs or cfg["training"]["epochs"]):
        tl,tm=run_epoch(model, train, opt, args.task, device); vl,vm=run_epoch(model, val, None, args.task, device); row={"epoch":epoch+1,"train_loss":tl,"val_loss":vl,**{f"train_{k}":v for k,v in tm.items()},**{f"val_{k}":v for k,v in vm.items()}}; history.append(row); print(row)
        if vm["macro_f1"] > best: best=vm["macro_f1"]; torch.save({"model":model.state_dict(),"vocab":train.vocab,"config":cfg,"task":args.task,"early_concat":args.early_concat}, results / f"{run_name}_best.pt")
    save_json({"task":args.task,"labels":train.vocab,"history":history}, results / f"{run_name}_metrics.json")
    # Required learning curves: metrics are also retained as JSON for the report table.
    import matplotlib.pyplot as plt
    epochs = [x["epoch"] for x in history]
    plt.figure(figsize=(6, 4)); plt.plot(epochs, [x["train_macro_f1"] for x in history], label="train macro-F1"); plt.plot(epochs, [x["val_macro_f1"] for x in history], label="validation macro-F1"); plt.plot(epochs, [x["val_micro_f1"] for x in history], label="validation micro-F1"); plt.xlabel("epoch"); plt.ylabel("F1"); plt.legend(); plt.tight_layout(); plt.savefig(results / f"{run_name}_f1_curve.png", dpi=160); plt.close()

if __name__ == "__main__": main()
