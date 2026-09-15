from __future__ import annotations

import argparse, csv, json
import logging
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from .audio_features import load_audio, mel_spectrogram, segment_audio
from .bert_encoder import BertTagClassifier
from .fusion_model import FusionModel
from .gnn_model import GenreGraphSAGEClassifier, MelCNN
from .metrics import multilabel_metrics
from .utils import configure_logging, ensure_dir, progress, save_json, seed_everything


LOGGER = logging.getLogger("music-context")


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
        if artist is None or str(artist).strip().lower() in {"", "nan"}:
            raise ValueError(f"Missing artist_id for track {row.get('track_id')}")
        if split is None or str(split).strip().lower() in {"", "nan"}:
            raise ValueError(f"Missing split for track {row.get('track_id')}")
        if artist in seen and seen[artist] != split:
            raise ValueError(
                f"artist leakage: {artist} in {seen[artist]} and {split}"
            )
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
        manifest_parent = Path(manifest).resolve().parent
        self.graph_paths = {
            index: self._resolve_graph_path(row["graph"], manifest_parent)
            for index, row in enumerate(self.rows)
        }
        self._cache: dict[int, dict] = {}

    @staticmethod
    def _resolve_graph_path(graph: str | Path, manifest_parent: Path) -> Path:
        graph_path = Path(graph)
        if graph_path.is_absolute():
            return graph_path
        if graph_path.is_file():
            return graph_path
        return manifest_parent / graph_path

    @property
    def input_dim(self) -> int:
        graph = self[0]
        return int(graph["x"].shape[1])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        graph_path = self.graph_paths[index]
        if not graph_path.is_file():
            raise FileNotFoundError(f"Graph file was not found: {graph_path}")
        graph = self._cache.get(index)
        if graph is None:
            graph = torch.load(graph_path, weights_only=False)
            if "x" not in graph or "edge_index" not in graph:
                raise ValueError(f"Graph is missing x or edge_index: {graph_path}")
            self._cache[index] = graph
        return {
            **graph,
            "y": encode_genre(row["genre"], self.vocabulary),
            "track_id": int(row["track_id"]),
            "genre": str(row["genre"]),
        }


def collate_graphs(graphs: list[dict]) -> dict:
    """Batch graph dictionaries by concatenating nodes and offsetting edges."""
    if not graphs:
        raise ValueError("Cannot collate an empty graph batch")
    node_offsets = []
    offset = 0
    for graph in graphs:
        node_offsets.append(offset)
        offset += graph["x"].size(0)
    x = torch.cat([graph["x"] for graph in graphs], dim=0)
    edge_index = torch.cat(
        [
            graph["edge_index"] + node_offset
            for graph, node_offset in zip(graphs, node_offsets)
        ],
        dim=1,
    )
    batch = torch.cat(
        [
            torch.full(
                (graph["x"].size(0),),
                index,
                dtype=torch.long,
            )
            for index, graph in enumerate(graphs)
        ]
    )
    return {
        "x": x,
        "edge_index": edge_index,
        "batch": batch,
        "y": torch.stack([graph["y"] for graph in graphs]),
        "track_id": [graph["track_id"] for graph in graphs],
        "genre": [graph["genre"] for graph in graphs],
    }


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
    """Load all fixed-size log-mel spectrogram segments for each FMA track."""

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
        mel_dir: str | Path | None = None,
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
        self.mel_dir = Path(mel_dir) if mel_dir is not None else None
        if self.mel_dir is None:
            raise ValueError("Task 2 CNN training requires a preprocessed mel_dir")
        self._cache: dict[int, torch.Tensor] = {}
        for row in self.rows:
            track_id = int(row["track_id"])
            if track_id not in self.audio_metadata:
                raise ValueError(f"Track {track_id} is missing from audio metadata")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        track_id = int(row["track_id"])
        mel_path = self.mel_dir / f"{track_id:06d}.pt"
        if not mel_path.is_file():
            raise FileNotFoundError(
                f"Cached mel input was not found: {mel_path}. "
                "Run python -m src.prepare_task2 first."
            )
        cached = self._cache.get(track_id)
        if cached is None:
            cached = torch.load(mel_path, weights_only=False)["x"]
            self._cache[track_id] = cached
        return {
            "x": cached,
            "y": encode_genre(row["genre"], self.vocabulary),
            "track_id": track_id,
        }


def collate_mels(samples: list[dict]) -> dict:
    """Flatten variable-length track segments while retaining track ownership."""
    if not samples:
        raise ValueError("Cannot collate an empty mel batch")
    inputs = torch.cat([sample["x"] for sample in samples], dim=0)
    track_index = torch.cat(
        [
            torch.full((sample["x"].size(0),), index, dtype=torch.long)
            for index, sample in enumerate(samples)
        ]
    )
    return {
        "x": inputs,
        "track_index": track_index,
        "y": torch.stack([sample["y"] for sample in samples]),
        "track_id": [sample["track_id"] for sample in samples],
    }


def device_graph(graph, device): return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in graph.items()}

def run_epoch(model, data, optimizer, task, device):
    training = optimizer is not None; model.train(training); criterion = nn.BCEWithLogitsLoss(); losses=[]; logits=[]; targets=[]
    phase = "train" if training else "validation"
    for graph in progress(data, desc=f"{phase} batches", total=len(data)):
        graph = device_graph(graph, device); y = graph["y"].unsqueeze(0).to(device)
        if task == "bert": prediction = model([graph["text"]])
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
    phase = "train" if training else "validation"
    with torch.set_grad_enabled(training):
        for graph in progress(data, desc=f"{phase} batches", total=len(data)):
            graph = device_graph(graph, device)
            target = graph["y"].reshape(-1)
            logits = model(graph).reshape(-1, num_classes)
            loss = criterion(logits, target)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(float(loss.item()))
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(target.detach().cpu().tolist())
    return float(np.mean(losses)), single_label_metrics(predictions, targets, num_classes)


def run_cnn_epoch(model, data, optimizer, device, num_classes: int) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    phase = "train" if training else "validation"
    for sample in progress(data, desc=f"{phase} batches", total=len(data)):
        inputs = sample["x"].to(device, non_blocking=True)
        track_index = sample["track_index"].to(device, non_blocking=True)
        target = sample["y"].to(device, non_blocking=True)
        segment_logits = model(inputs)
        logits = torch.zeros(
            target.size(0), num_classes, device=device, dtype=segment_logits.dtype
        )
        logits.index_add_(0, track_index, segment_logits)
        counts = torch.bincount(track_index, minlength=target.size(0)).to(
            device=device, dtype=segment_logits.dtype
        )
        logits = logits / counts.unsqueeze(1)
        loss = criterion(logits, target)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(target.detach().cpu().tolist())
    return float(np.mean(losses)), single_label_metrics(predictions, targets, num_classes)


def save_genre_learning_curves(
    history: list[dict],
    output: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()
    axes[1].plot(
        epochs, [row["train_macro_f1"] for row in history], label="train Macro-F1"
    )
    axes[1].plot(
        epochs, [row["val_macro_f1"] for row in history], label="validation Macro-F1"
    )
    axes[1].set_title("Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def train_gnn(args, cfg) -> None:
    train = GenreGraphDataset(args.manifest, "train")
    val = GenreGraphDataset(args.manifest, "val", train.vocabulary)
    test = GenreGraphDataset(args.manifest, "test", train.vocabulary)
    input_dim = train.input_dim
    model = GenreGraphSAGEClassifier(
        num_genres=len(train.vocabulary),
        input_dim=input_dim,
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
    run_name = args.run_name or "graphsage_genre"
    results = ensure_dir(cfg["data"]["results_dir"])
    history = []
    best = -1.0
    total_epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = int(cfg["training"].get("graph_batch_size", cfg["training"]["batch_size"]))
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "collate_fn": collate_graphs,
    }
    train_loader = DataLoader(train, **loader_kwargs)
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )
    LOGGER.info(
        "Starting %s for %d epochs on %s (graph batch size=%d)",
        run_name,
        total_epochs,
        device,
        batch_size,
    )
    for epoch in range(total_epochs):
        train_loss, train_metrics = run_genre_epoch(
            model, train_loader, optimizer, device, len(train.vocabulary)
        )
        val_loss, val_metrics = run_genre_epoch(
            model, val_loader, None, device, len(train.vocabulary)
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        LOGGER.info("Epoch %d/%d: %s", epoch + 1, total_epochs, row)
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocabulary": train.vocabulary,
                    "input_dim": input_dim,
                    "config": cfg,
                    "task": "gnn",
                },
                results / f"{run_name}_best.pt",
            )

    checkpoint = torch.load(
        results / f"{run_name}_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    test_loss, test_metrics = run_genre_epoch(
        model, test_loader, None, device, len(train.vocabulary)
    )
    save_json(
        {
            "task": "gnn",
            "genres": train.vocabulary,
            "history": history,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
        },
        results / f"{run_name}_metrics.json",
    )
    save_genre_learning_curves(
        history,
        ensure_dir(results / "plots") / "task2_graphsage_learning_curves.png",
        "Task 2 GraphSAGE learning curves",
    )
    LOGGER.info("Completed %s; test metrics: %s", run_name, test_metrics)


def train_genre_cnn(args, cfg) -> None:
    train = GenreMelDataset(
        args.manifest,
        args.metadata,
        args.audio_root,
        "train",
        sample_rate=cfg["data"]["sample_rate"],
        segment_seconds=cfg["data"]["segment_seconds"],
        n_mels=cfg["data"]["n_mels"],
        mel_dir=cfg["data"]["mel_dir"],
    )
    dataset_args = {
        "manifest": args.manifest,
        "metadata": args.metadata,
        "audio_root": args.audio_root,
        "vocabulary": train.vocabulary,
        "sample_rate": cfg["data"]["sample_rate"],
        "segment_seconds": cfg["data"]["segment_seconds"],
        "n_mels": cfg["data"]["n_mels"],
        "mel_dir": cfg["data"]["mel_dir"],
    }
    val = GenreMelDataset(split="val", **dataset_args)
    test = GenreMelDataset(split="test", **dataset_args)
    batch_size = int(cfg["training"].get("cnn_batch_size", cfg["training"]["batch_size"]))
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_mels,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test, shuffle=False, **loader_kwargs)
    model = MelCNN(num_labels=len(train.vocabulary))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    run_name = args.run_name or "cnn_melspectrogram_genre"
    results = ensure_dir(cfg["data"]["results_dir"])
    history = []
    best = -1.0
    total_epochs = args.epochs or cfg["training"]["epochs"]
    LOGGER.info("Starting %s for %d epochs on %s", run_name, total_epochs, device)
    for epoch in range(total_epochs):
        train_loss, train_metrics = run_cnn_epoch(
            model, train_loader, optimizer, device, len(train.vocabulary)
        )
        val_loss, val_metrics = run_cnn_epoch(
            model, val_loader, None, device, len(train.vocabulary)
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        LOGGER.info("Epoch %d/%d: %s", epoch + 1, total_epochs, row)
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
        model, test_loader, None, device, len(train.vocabulary)
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
    save_genre_learning_curves(
        history,
        ensure_dir(results / "plots") / "task2_cnn_learning_curves.png",
        "Task 2 CNN learning curves",
    )
    LOGGER.info("Completed %s; test metrics: %s", run_name, test_metrics)


def main():
    configure_logging()
    ap=argparse.ArgumentParser(); ap.add_argument("--task", choices=["bert","gnn","fusion","genre_cnn"], required=True); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--manifest"); ap.add_argument("--metadata"); ap.add_argument("--audio-root"); ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--epochs", type=int); ap.add_argument("--early-concat", action="store_true"); ap.add_argument("--run-name"); args=ap.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    seed_everything(cfg["seed"])
    LOGGER.info("Task=%s, seed=%s, synthetic=%s", args.task, cfg["seed"], args.synthetic)
    task2 = args.task in {"gnn", "genre_cnn"}
    if task2 and args.synthetic:
        raise ValueError(
            "Synthetic data uses the legacy multilabel labels contract and cannot "
            "be used for Task 2 single-label genre training. Build the FMA Task 2 "
            "manifest and omit --synthetic."
        )
    if task2:
        args.manifest = args.manifest or cfg["data"]["manifest"]
        args.metadata = args.metadata or cfg["data"]["metadata_csv"]
        args.audio_root = args.audio_root or cfg["data"]["audio_root"]
    elif args.synthetic:
        args.manifest = args.manifest or "data/processed/manifest.jsonl"
    elif not args.manifest:
        ap.error("--manifest is required for non-Task-2 training")
    if args.task == "gnn":
        train_gnn(args, cfg)
        return
    if args.task == "genre_cnn":
        train_genre_cnn(args, cfg)
        return
    train=MusicGraphDataset(args.manifest, "train"); val=MusicGraphDataset(args.manifest, "val", train.vocab)
    num_labels=len(train.vocab); model_args=dict(num_labels=num_labels, text_hidden=cfg["model"]["text_hidden"], graph_hidden=cfg["model"]["gnn_hidden"], layers=cfg["model"]["gnn_layers"], dropout=cfg["model"]["dropout"], model_name=cfg["model"]["text_model"], freeze=cfg["training"]["freeze_text_encoder"])
    if args.task == "bert": model=BertTagClassifier(num_labels, hidden_size=cfg["model"]["text_hidden"], model_name=cfg["model"]["text_model"], freeze=cfg["training"]["freeze_text_encoder"], local_files_only=args.synthetic)
    else: model=FusionModel(**model_args, cross_attention=not args.early_concat, local_files_only=args.synthetic)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device); opt=torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    results=ensure_dir("results"); run_name=args.run_name or args.task; history=[]; best=-1
    total_epochs = args.epochs or cfg["training"]["epochs"]
    LOGGER.info("Starting %s for %d epochs on %s", run_name, total_epochs, device)
    for epoch in range(total_epochs):
        tl,tm=run_epoch(model, train, opt, args.task, device); vl,vm=run_epoch(model, val, None, args.task, device); row={"epoch":epoch+1,"train_loss":tl,"val_loss":vl,**{f"train_{k}":v for k,v in tm.items()},**{f"val_{k}":v for k,v in vm.items()}}; history.append(row); LOGGER.info("Epoch %d/%d: %s", epoch + 1, total_epochs, row)
        if vm["macro_f1"] > best: best=vm["macro_f1"]; torch.save({"model":model.state_dict(),"vocab":train.vocab,"config":cfg,"task":args.task,"early_concat":args.early_concat}, results / f"{run_name}_best.pt")
    save_json({"task":args.task,"labels":train.vocab,"history":history}, results / f"{run_name}_metrics.json")
    # Required learning curves: metrics are also retained as JSON for the report table.
    import matplotlib.pyplot as plt
    epochs = [x["epoch"] for x in history]
    plt.figure(figsize=(6, 4)); plt.plot(epochs, [x["train_macro_f1"] for x in history], label="train macro-F1"); plt.plot(epochs, [x["val_macro_f1"] for x in history], label="validation macro-F1"); plt.plot(epochs, [x["val_micro_f1"] for x in history], label="validation micro-F1"); plt.xlabel("epoch"); plt.ylabel("F1"); plt.legend(); plt.tight_layout(); plt.savefig(results / f"{run_name}_f1_curve.png", dpi=160); plt.close()
    LOGGER.info("Completed %s; metrics written to %s", run_name, results)

if __name__ == "__main__": main()
