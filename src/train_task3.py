"""Train Task 3 graph/text multilabel models without touching Task 2 paths."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .task3_training import (
    SPLITS,
    Task3BertOnlyModel,
    Task3FusionModel,
    Task3GNNOnlyModel,
    Task3GraphDataset,
    collate_task3_graphs,
    load_jsonl,
    load_labels,
    multilabel_metrics,
    positive_weight,
    progress,
    validate_manifest,
)


def _write_task3_artifacts(
    output: Path,
    run_name: str,
    history: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    embeddings: np.ndarray | None,
    labels: list[str],
) -> None:
    """Write the plots and qualitative outputs produced by every real run."""
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (output / f"{run_name}_predictions.json").write_text(
        json.dumps(predictions, indent=2, allow_nan=True), encoding="utf-8"
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import average_precision_score, precision_recall_curve

        epochs = [item["epoch"] for item in history]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
        axes[0].plot(epochs, [item["val_loss"] for item in history], label="validation")
        axes[0].set(title="Task 3 loss", xlabel="epoch", ylabel="BCE loss")
        axes[1].plot(epochs, [item["train"]["macro_f1"] for item in history], label="train")
        axes[1].plot(epochs, [item["val"]["macro_f1"] for item in history], label="validation")
        axes[1].set(title="Task 3 macro-F1", xlabel="epoch", ylabel="macro-F1")
        for axis in axes:
            axis.legend()
            axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / f"{run_name}_learning_curves.png", dpi=150)
        plt.close(fig)

        y_true = np.asarray([row["target"] for row in predictions])
        y_score = np.asarray([row["probabilities"] for row in predictions])
        fig, axis = plt.subplots(figsize=(6, 5))
        for index, label in enumerate(labels):
            if y_true[:, index].sum() == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_true[:, index], y_score[:, index])
            ap = average_precision_score(y_true[:, index], y_score[:, index])
            axis.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
        axis.set(title="Task 3 precision-recall curves", xlabel="recall", ylabel="precision")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / f"{run_name}_pr_curve.png", dpi=150)
        plt.close(fig)

        if embeddings is not None and len(embeddings) >= 2:
            from sklearn.manifold import TSNE
            perplexity = min(30, max(1, len(embeddings) - 1))
            points = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(embeddings)
            colors = [int(np.argmax(row["target"])) if any(row["target"]) else -1 for row in predictions]
            fig, axis = plt.subplots(figsize=(7, 5))
            scatter = axis.scatter(points[:, 0], points[:, 1], c=colors, cmap="tab20", alpha=0.85)
            axis.set(title="Task 3 fused embedding t-SNE", xlabel="t-SNE 1", ylabel="t-SNE 2")
            fig.colorbar(scatter, ax=axis, label="dominant label index")
            fig.tight_layout()
            fig.savefig(plots / f"{run_name}_fused_tsne.png", dpi=150)
            plt.close(fig)
    except (ImportError, ValueError):
        # Training and JSON deliverables remain usable in minimal environments.
        pass

    ranked = sorted(predictions, key=lambda row: row["confidence"], reverse=True)
    case_studies = []
    for row in ranked[: min(3, len(ranked))]:
        case_studies.append({
            "track_id": row["track_id"],
            "text": row["text"],
            "graph": row["graph"],
            "true_labels": [label for label, value in zip(labels, row["target"]) if value],
            "predicted_labels": [
                label for label, value in zip(labels, row["probabilities"]) if value >= 0.5
            ],
            "probabilities": dict(zip(labels, row["probabilities"])),
        })
    (output / f"{run_name}_case_studies.json").write_text(
        json.dumps(case_studies, indent=2), encoding="utf-8"
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _value(args: argparse.Namespace, cfg: dict, section: str, name: str, default: Any):
    value = getattr(args, name, None)
    if value is not None:
        return value
    return cfg.get(section, {}).get(name, cfg.get("model", {}).get(name, default))


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    clip_norm: float | None = 1.0,
    desc: str = "Batches",
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss, examples = 0.0, 0
    all_logits, all_targets = [], []
    batches = progress(loader, desc=desc, total=len(loader))
    for batch in batches:
        graph = {key: value.to(device) for key, value in batch["graph"].items()}
        targets = batch["targets"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(graph, batch["texts"])
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                if clip_norm is not None and clip_norm > 0:
                    clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
        count = targets.shape[0]
        total_loss += float(loss.detach().cpu()) * count
        examples += count
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
    if not all_logits:
        raise ValueError("DataLoader produced no batches")
    metrics = multilabel_metrics(torch.cat(all_logits), torch.cat(all_targets))
    metrics["loss"] = total_loss / max(examples, 1)
    batches.set_postfix(loss=f"{metrics['loss']:.4f}", macro_f1=f"{metrics['macro_f1']:.4f}")
    return metrics["loss"], metrics


def train_task3(
    manifest: str | Path,
    labels_path: str | Path,
    output_dir: str | Path = "results/task3",
    run_name: str = "task3_fusion",
    model_kind: str = "fusion",
    cross_attention: bool = True,
    config_path: str | Path | None = None,
    **overrides: Any,
) -> dict[str, dict[str, float]]:
    """Train and evaluate one Task 3 condition, returning split metrics."""
    cfg = _config(config_path)
    seed = int(overrides.get("seed", cfg.get("seed", 42)))
    _seed_everything(seed)
    manifest = Path(manifest)
    labels = load_labels(labels_path)
    rows = load_jsonl(manifest)
    validate_manifest(rows, labels, manifest)
    datasets = {
        split: Task3GraphDataset(manifest, labels, split, rows=rows)
        for split in SPLITS
    }

    batch_size = int(overrides.get("batch_size", cfg.get("training", {}).get("batch_size", 8)))
    epochs = int(overrides.get("epochs", cfg.get("training", {}).get("epochs", 50)))
    learning_rate = float(
        overrides.get("learning_rate", cfg.get("training", {}).get("learning_rate", 3e-4))
    )
    weight_decay = float(
        overrides.get("weight_decay", cfg.get("training", {}).get("weight_decay", 1e-4))
    )
    dropout = float(overrides.get("dropout", cfg.get("model", {}).get("dropout", 0.2)))
    graph_hidden = int(overrides.get("graph_hidden", cfg.get("model", {}).get("gnn_hidden", 256)))
    graph_layers = int(overrides.get("graph_layers", cfg.get("model", {}).get("gnn_layers", 3)))
    text_hidden = int(overrides.get("text_hidden", cfg.get("model", {}).get("text_hidden", 256)))
    text_model = str(
        overrides.get("text_model", cfg.get("model", {}).get("text_model", "distilbert-base-uncased"))
    )
    max_length = int(overrides.get("max_length", cfg.get("data", {}).get("max_text_length", 128)))
    freeze_text = bool(
        overrides.get("freeze_text", cfg.get("training", {}).get("freeze_text_encoder", False))
    )
    local_files_only = bool(overrides.get("local_files_only", False))
    requested_device = overrides.get("device")
    if requested_device is None:
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)

    model_kwargs = {
        "graph_hidden": graph_hidden,
        "graph_layers": graph_layers,
        "text_hidden": text_hidden,
        "text_model": text_model,
        "freeze_text": freeze_text,
        "max_length": max_length,
        "dropout": dropout,
        "local_files_only": local_files_only,
    }
    if model_kind == "fusion":
        model = Task3FusionModel(
            num_labels=len(labels),
            graph_input=datasets["train"].input_dim,
            cross_attention=cross_attention,
            **model_kwargs,
        )
    elif model_kind == "bert":
        model = Task3BertOnlyModel(
            num_labels=len(labels),
            model_name=text_model,
            hidden_size=text_hidden,
            freeze=freeze_text,
            max_length=max_length,
            local_files_only=local_files_only,
        )
    elif model_kind == "gnn":
        model = Task3GNNOnlyModel(
            num_labels=len(labels),
            graph_input=datasets["train"].input_dim,
            graph_hidden=graph_hidden,
            graph_layers=graph_layers,
            dropout=dropout,
        )
    else:
        raise ValueError("model_kind must be one of: fusion, bert, gnn")
    model.to(device)

    weighting = str(
        overrides.get("class_weighting", cfg.get("training", {}).get("class_weighting", "none"))
    ).lower()
    if weighting in {"balanced", "positive", "pos_weight"}:
        criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight(datasets["train"]).to(device))
    elif weighting in {"none", "off", "false"}:
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("class_weighting must be none or positive/balanced")

    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=split == "train",
            collate_fn=collate_task3_graphs,
            generator=torch.Generator().manual_seed(seed),
        )
        for split in SPLITS
    }
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg.get("training", {}).get("scheduler_factor", 0.5)),
        patience=int(cfg.get("training", {}).get("scheduler_patience", 5)),
        min_lr=float(cfg.get("training", {}).get("min_learning_rate", 1e-6)),
    )
    patience = int(
        overrides.get(
            "early_stopping_patience",
            cfg.get("training", {}).get("early_stopping_patience", 12),
        )
    )
    min_delta = float(
        overrides.get(
            "early_stopping_min_delta",
            cfg.get("training", {}).get("early_stopping_min_delta", 1e-4),
        )
    )
    clip_norm = float(
        overrides.get("gradient_clip_norm", cfg.get("training", {}).get("gradient_clip_norm", 1.0))
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f"{run_name}_best.pt"
    best_score, stale = -float("inf"), 0
    history: list[dict[str, Any]] = []
    epoch_bar = progress(
        range(1, epochs + 1),
        desc=f"Training {model_kind}",
        total=epochs,
    )
    for epoch in epoch_bar:
        train_loss, train_metrics = _run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
            clip_norm,
            desc=f"Epoch {epoch}/{epochs} train",
        )
        val_loss, val_metrics = _run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            desc=f"Epoch {epoch}/{epochs} val",
        )
        scheduler.step(val_metrics["macro_f1"])
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics,
            "val": val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_macro_f1=f"{val_metrics['macro_f1']:.4f}",
            best="yes" if val_metrics["macro_f1"] >= best_score + min_delta else "no",
        )
        if val_metrics["macro_f1"] > best_score + min_delta:
            best_score = val_metrics["macro_f1"]
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "labels": labels,
                    "model_kind": model_kind,
                    "cross_attention": cross_attention,
                    "input_dim": datasets["train"].input_dim,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "config": {
                        "graph_hidden": graph_hidden,
                        "graph_layers": graph_layers,
                        "text_hidden": text_hidden,
                        "text_model": text_model,
                        "max_length": max_length,
                    },
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break

    if not checkpoint_path.is_file():
        raise RuntimeError("Training did not produce a best checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    result = {}
    evaluation_bar = progress(SPLITS, desc="Evaluating Task 3 splits", total=len(SPLITS))
    for split in evaluation_bar:
        result[split] = _run_epoch(
            model,
            loaders[split],
            criterion,
            device,
            desc=f"Evaluate {split}",
        )[1]
        evaluation_bar.set_postfix(
            split=split,
            macro_f1=f"{result[split]['macro_f1']:.4f}",
        )
    predictions: list[dict[str, Any]] = []
    embedding_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loaders["test"]:
            graph = {key: value.to(device) for key, value in batch["graph"].items()}
            logits = model(graph, batch["texts"])
            probabilities = torch.sigmoid(logits).cpu().numpy()
            if hasattr(model, "fused_embedding"):
                embedding_rows.append(model.fused_embedding(graph, batch["texts"]).cpu().numpy())
            else:
                embedding_rows.append(logits.detach().cpu().numpy())
            for index, track_id in enumerate(batch["track_ids"]):
                row = next(item for item in datasets["test"].rows if item["track_id"] == track_id)
                predictions.append({
                    "track_id": track_id,
                    "text": batch["texts"][index],
                    "graph": str(datasets["test"]._resolve_graph(row["graph"])),
                    "target": batch["targets"][index].tolist(),
                    "probabilities": probabilities[index].tolist(),
                    "true_labels": [
                        label
                        for label, value in zip(labels, batch["targets"][index].tolist())
                        if value
                    ],
                    "predicted_labels": [
                        label
                        for label, value in zip(labels, probabilities[index])
                        if value >= 0.5
                    ],
                    "confidence": float(np.max(probabilities[index])),
                })
    _write_task3_artifacts(
        output,
        run_name,
        history,
        predictions,
        np.concatenate(embedding_rows) if embedding_rows else None,
        labels,
    )
    result["best_epoch"] = {"epoch": checkpoint["epoch"]}
    metrics_path = output / f"{run_name}_metrics.json"
    metrics_path.write_text(
        json.dumps({"metrics": result, "history": history}, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/task3/task3_fusion_manifest.jsonl")
    parser.add_argument("--labels", default="data/processed/task3/labels.json")
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--output-dir", default="results/task3")
    parser.add_argument("--run-name", default="task3_fusion")
    parser.add_argument("--model", dest="model_kind", choices=("fusion", "bert", "gnn"), default="fusion")
    parser.add_argument("--early-concat", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--graph-hidden", type=int)
    parser.add_argument("--graph-layers", type=int)
    parser.add_argument("--text-hidden", type=int)
    parser.add_argument("--text-model")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--class-weighting", choices=("none", "positive", "balanced"))
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--freeze-text", action="store_true", default=None)
    parser.add_argument("--local-files-only", action="store_true", default=None)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = vars(args).copy()
    config_path = overrides.pop("config_path")
    for key in ("manifest", "labels", "output_dir", "run_name", "model_kind", "early_concat", "config_path"):
        overrides.pop(key, None)
    overrides = {key: value for key, value in overrides.items() if value is not None}
    train_task3(
        manifest=args.manifest,
        labels_path=args.labels,
        output_dir=args.output_dir,
        run_name=args.run_name,
        model_kind=args.model_kind,
        cross_attention=not args.early_concat,
        config_path=config_path,
        **overrides,
    )


if __name__ == "__main__":
    main()
