from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import yaml

from .audio_features import load_audio, mel_spectrogram
from .bert_encoder import BertTagClassifier
from .fusion_model import FusionModel
from .gnn_model import GNNClassifier, GenreGraphSAGEClassifier, MelCNN
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


def load_audio_metadata(metadata_path: str | Path) -> dict[int, dict[str, str]]:
    with open(metadata_path, newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        required = {"track_id", "path"}
        if not required.issubset(rows.fieldnames or set()):
            raise ValueError(f"Audio metadata must contain {sorted(required)}")
        metadata = {}
        for row in rows:
            track_id = int(row["track_id"])
            if track_id in metadata:
                raise ValueError(f"Duplicate audio metadata track ID: {track_id}")
            metadata[track_id] = row
    if not metadata:
        raise ValueError(f"Audio metadata is empty: {metadata_path}")
    return metadata


class GenreMelDataset(Dataset):
    """Load one fixed-size log-mel spectrogram per FMA track."""

    def __init__(
        self,
        manifest: str | Path,
        metadata: str | Path,
        audio_root: str | Path,
        split: str,
        vocabulary: list[str] | None = None,
        sample_rate: int = 22050,
        segment_seconds: float = 5.0,
        n_mels: int = 128,
    ):
        rows = load_manifest(manifest)
        assert_no_artist_leakage(rows)
        self.rows = [row for row in rows if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"No audio samples found for split {split!r}")
        self.vocabulary = vocabulary or genre_vocabulary(rows)
        self.audio_metadata = load_audio_metadata(metadata)
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        self.segment_samples = round(sample_rate * segment_seconds)
        self.n_mels = n_mels
        for row in self.rows:
            track_id = int(row["track_id"])
            if track_id not in self.audio_metadata:
                raise ValueError(f"Track {track_id} is missing from audio metadata")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        track_id = int(row["track_id"])
        audio_path = self.audio_root / self.audio_metadata[track_id]["path"]
        waveform, _ = load_audio(str(audio_path), self.sample_rate)
        waveform = waveform[: self.segment_samples]
        if len(waveform) < self.segment_samples:
            waveform = np.pad(waveform, (0, self.segment_samples - len(waveform)))
        mel = mel_spectrogram(waveform, self.sample_rate, self.n_mels)
        return {
            "x": torch.from_numpy(mel).unsqueeze(0),
            "y": encode_genre(row["genre"], self.vocabulary),
            "track_id": track_id,
        }


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


def single_label_metrics(predictions: list[int], targets: list[int], num_classes: int) -> dict:
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    per_class = []
    for class_index in range(num_classes):
        true_positive = np.sum((predictions == class_index) & (targets == class_index))
        false_positive = np.sum((predictions == class_index) & (targets != class_index))
        false_negative = np.sum((predictions != class_index) & (targets == class_index))
        denominator = 2 * true_positive + false_positive + false_negative
        per_class.append(float(2 * true_positive / denominator) if denominator else 0.0)
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": float(np.mean(per_class)),
        "micro_f1": float(np.mean(predictions == targets)),
    }


def run_genre_epoch(model, data, optimizer, device, num_classes: int) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    for graph in data:
        graph = device_graph(graph, device)
        target = graph["y"].reshape(1).to(device)
        logits = model(graph).reshape(1, num_classes)
        loss = criterion(logits, target)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        predictions.append(int(logits.argmax(dim=1).item()))
        targets.append(int(target.item()))
    return float(np.mean(losses)), single_label_metrics(predictions, targets, num_classes)


def run_cnn_epoch(model, data, optimizer, device, num_classes: int) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    for sample in data:
        inputs = sample["x"].unsqueeze(0).to(device)
        target = sample["y"].reshape(1).to(device)
        logits = model(inputs)
        loss = criterion(logits, target)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        predictions.append(int(logits.argmax(dim=1).item()))
        targets.append(int(target.item()))
    return float(np.mean(losses)), single_label_metrics(predictions, targets, num_classes)


def train_genre_gnn(args, cfg) -> None:
    train = GenreGraphDataset(args.manifest, "train")
    val = GenreGraphDataset(args.manifest, "val", train.vocabulary)
    test = GenreGraphDataset(args.manifest, "test", train.vocabulary)
    model = GenreGraphSAGEClassifier(
        num_genres=len(train.vocabulary),
        input_dim=32,
        hidden_dim=cfg["model"]["gnn_hidden"],
        layers=cfg["model"]["gnn_layers"],
        dropout=cfg["model"]["dropout"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    run_name = args.run_name or "task2_gnn"
    results = ensure_dir(Path("results") / Path(run_name).parent)
    history = []
    best = -1.0
    for epoch in range(args.epochs or cfg["training"]["epochs"]):
        train_loss, train_metrics = run_genre_epoch(
            model, train, optimizer, device, len(train.vocabulary)
        )
        val_loss, val_metrics = run_genre_epoch(
            model, val, None, device, len(train.vocabulary)
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(row)
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocabulary": train.vocabulary,
                    "config": cfg,
                    "task": "genre_gnn",
                },
                results / f"{run_name}_best.pt",
            )

    checkpoint = torch.load(
        results / f"{run_name}_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    test_loss, test_metrics = run_genre_epoch(
        model, test, None, device, len(train.vocabulary)
    )
    save_json(
        {
            "task": "genre_gnn",
            "genres": train.vocabulary,
            "history": history,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
        },
        results / f"{run_name}_metrics.json",
    )


def train_genre_cnn(args, cfg) -> None:
    train = GenreMelDataset(
        args.manifest,
        args.metadata,
        args.audio_root,
        "train",
        sample_rate=cfg["data"]["sample_rate"],
        segment_seconds=cfg["data"]["segment_seconds"],
        n_mels=cfg["data"]["n_mels"],
    )
    dataset_args = {
        "manifest": args.manifest,
        "metadata": args.metadata,
        "audio_root": args.audio_root,
        "vocabulary": train.vocabulary,
        "sample_rate": cfg["data"]["sample_rate"],
        "segment_seconds": cfg["data"]["segment_seconds"],
        "n_mels": cfg["data"]["n_mels"],
    }
    val = GenreMelDataset(split="val", **dataset_args)
    test = GenreMelDataset(split="test", **dataset_args)
    model = MelCNN(num_labels=len(train.vocabulary))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    run_name = args.run_name or "task2_cnn"
    results = ensure_dir(Path("results") / Path(run_name).parent)
    history = []
    best = -1.0
    for epoch in range(args.epochs or cfg["training"]["epochs"]):
        train_loss, train_metrics = run_cnn_epoch(
            model, train, optimizer, device, len(train.vocabulary)
        )
        val_loss, val_metrics = run_cnn_epoch(
            model, val, None, device, len(train.vocabulary)
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(row)
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocabulary": train.vocabulary,
                    "config": cfg,
                    "task": "genre_cnn",
                },
                results / f"{run_name}_best.pt",
            )

    checkpoint = torch.load(
        results / f"{run_name}_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    test_loss, test_metrics = run_cnn_epoch(
        model, test, None, device, len(train.vocabulary)
    )
    save_json(
        {
            "task": "genre_cnn",
            "genres": train.vocabulary,
            "history": history,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
        },
        results / f"{run_name}_metrics.json",
    )


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--task", choices=["bert","gnn","fusion","genre_gnn","genre_cnn"], required=True); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--manifest", default="data/processed/task2/task2_graph_manifest.jsonl"); ap.add_argument("--metadata", default="data/processed/task2/fma_metadata.csv"); ap.add_argument("--audio-root", default="data/raw/fma/fma_small"); ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--epochs", type=int); ap.add_argument("--early-concat", action="store_true"); ap.add_argument("--run-name"); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config)); seed_everything(cfg["seed"])
    if args.synthetic: args.manifest="data/processed/manifest.jsonl"
    if args.task == "genre_gnn":
        train_genre_gnn(args, cfg)
        return
    if args.task == "genre_cnn":
        train_genre_cnn(args, cfg)
        return
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
