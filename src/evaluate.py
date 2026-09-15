from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .gnn_model import GenreGraphSAGEClassifier, MelCNN
from .train import GenreGraphDataset, GenreMelDataset, device_graph
from .utils import configure_logging, ensure_dir, save_json


def single_label_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    genres: list[str],
) -> dict:
    per_class = {}
    f1_values = []
    for index, genre in enumerate(genres):
        true_positive = np.sum((predictions == index) & (targets == index))
        false_positive = np.sum((predictions == index) & (targets != index))
        false_negative = np.sum((predictions != index) & (targets == index))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = float(2 * true_positive / denominator) if denominator else 0.0
        per_class[genre] = {"f1": f1}
        f1_values.append(f1)
    accuracy = float(np.mean(predictions == targets))
    target_matrix = np.eye(len(genres), dtype=np.float32)[targets]
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "micro_f1": accuracy,
        "auc_pr": float(
            average_precision_score(target_matrix, probabilities, average="macro")
        ),
        "per_class": per_class,
    }


def save_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    genres: list[str],
    output: Path,
    title: str,
) -> None:
    matrix = np.zeros((len(genres), len(genres)), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(genres)),
        yticks=np.arange(len(genres)),
        xticklabels=genres,
        yticklabels=genres,
        xlabel="Predicted genre",
        ylabel="True genre",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for row in range(len(genres)):
        for column in range(len(genres)):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def cross_entropy_loss(probabilities: np.ndarray, targets: np.ndarray) -> float:
    selected = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def evaluate_gnn(checkpoint_path: Path, manifest: Path, output_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    genres = state["vocabulary"]
    config = state["config"]
    input_dim = int(state.get("input_dim", 32))
    model = GenreGraphSAGEClassifier(
        num_genres=len(genres),
        input_dim=input_dim,
        hidden_dim=config["model"]["gnn_hidden"],
        layers=config["model"]["gnn_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    dataset = GenreGraphDataset(manifest, "test", genres)
    predictions, targets, probability_rows, cases = [], [], [], []
    with torch.no_grad():
        for graph in dataset:
            graph = device_graph(graph, device)
            probability_tensor = torch.softmax(model(graph)[0], dim=0)
            prediction = int(probability_tensor.argmax().item())
            target = int(graph["y"].item())
            predictions.append(prediction)
            targets.append(target)
            probabilities_array = probability_tensor.cpu().numpy()
            probability_rows.append(probabilities_array)
            cases.append(
                {
                    "track_id": int(graph["track_id"]),
                    "true_genre": genres[target],
                    "predicted_genre": genres[prediction],
                    "confidence": float(probabilities_array[prediction]),
                }
            )
    predictions_array = np.asarray(predictions)
    targets_array = np.asarray(targets)
    metrics = single_label_metrics(
        predictions_array, targets_array, np.asarray(probability_rows), genres
    )
    metrics["test_loss"] = cross_entropy_loss(
        np.asarray(probability_rows), targets_array
    )
    plots = ensure_dir(output_dir / "plots")
    save_json(metrics, output_dir / "task2_graphsage_test_metrics.json")
    save_json({"predictions": cases}, output_dir / "task2_graphsage_predictions.json")
    save_confusion_matrix(
        predictions_array,
        targets_array,
        genres,
        plots / "task2_graphsage_confusion_matrix.png",
        "Task 2 GraphSAGE confusion matrix",
    )
    return metrics


def evaluate_genre_cnn(
    checkpoint_path: Path,
    manifest: Path,
    metadata: Path,
    audio_root: Path,
    output_dir: Path,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    genres = state["vocabulary"]
    config = state["config"]
    model = MelCNN(num_labels=len(genres)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    dataset = GenreMelDataset(
        manifest=manifest,
        metadata=metadata,
        audio_root=audio_root,
        split="test",
        vocabulary=genres,
        sample_rate=config["data"]["sample_rate"],
        segment_seconds=config["data"]["segment_seconds"],
        n_mels=config["data"]["n_mels"],
        mel_dir=config["data"]["mel_dir"],
    )
    predictions, targets, probability_rows, cases = [], [], [], []
    with torch.no_grad():
        for sample in dataset:
            segment_logits = model(sample["x"].to(device))
            probability_tensor = torch.softmax(
                segment_logits.mean(dim=0), dim=0
            )
            prediction = int(probability_tensor.argmax().item())
            target = int(sample["y"].item())
            probabilities_array = probability_tensor.cpu().numpy()
            predictions.append(prediction)
            targets.append(target)
            probability_rows.append(probabilities_array)
            cases.append(
                {
                    "track_id": int(sample["track_id"]),
                    "true_genre": genres[target],
                    "predicted_genre": genres[prediction],
                    "confidence": float(probabilities_array[prediction]),
                }
            )
    predictions_array = np.asarray(predictions)
    targets_array = np.asarray(targets)
    metrics = single_label_metrics(
        predictions_array, targets_array, np.asarray(probability_rows), genres
    )
    metrics["test_loss"] = cross_entropy_loss(
        np.asarray(probability_rows), targets_array
    )
    plots = ensure_dir(output_dir / "plots")
    save_json(metrics, output_dir / "task2_cnn_test_metrics.json")
    save_json({"predictions": cases}, output_dir / "task2_cnn_predictions.json")
    save_confusion_matrix(
        predictions_array,
        targets_array,
        genres,
        plots / "task2_cnn_confusion_matrix.png",
        "Task 2 CNN confusion matrix",
    )
    return metrics


def main() -> None:
    logger = configure_logging()
    parser = argparse.ArgumentParser(description="Evaluate project models.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--manifest", default="data/processed/task2/task2_graph_manifest.jsonl"
    )
    parser.add_argument("--metadata", default="data/processed/task2/fma_metadata.csv")
    parser.add_argument("--audio-root", default="data/raw/fma/fma_small")
    parser.add_argument(
        "--task", choices=["gnn", "genre_cnn", "bert", "fusion"], required=True
    )
    parser.add_argument("--output-dir", default="results/task2")
    args = parser.parse_args()
    output_dir = ensure_dir(args.output_dir)
    if args.task == "gnn":
        metrics = evaluate_gnn(args.checkpoint, args.manifest, output_dir)
        logger.info("Evaluation complete: %s", metrics)
        print(metrics, flush=True)
        return
    if args.task == "genre_cnn":
        metrics = evaluate_genre_cnn(
            args.checkpoint,
            args.manifest,
            args.metadata,
            args.audio_root,
            output_dir,
        )
        logger.info("Evaluation complete: %s", metrics)
        print(metrics, flush=True)
        return
    raise NotImplementedError("Only Task 2 gnn and genre_cnn evaluation is currently implemented.")


if __name__ == "__main__":
    main()
