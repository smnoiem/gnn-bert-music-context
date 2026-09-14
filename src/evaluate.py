from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .gnn_model import GenreGraphSAGEClassifier
from .train import GenreGraphDataset, device_graph
from .utils import ensure_dir, save_json


def single_label_metrics(predictions: np.ndarray, targets: np.ndarray, genres: list[str]) -> dict:
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
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "micro_f1": accuracy,
        "per_class": per_class,
    }


def save_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    genres: list[str],
    output: Path,
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
        title="Task 2 GraphSAGE confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for row in range(len(genres)):
        for column in range(len(genres)):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def evaluate_genre_gnn(checkpoint_path: Path, manifest: Path, output_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    genres = state["vocabulary"]
    config = state["config"]
    model = GenreGraphSAGEClassifier(
        num_genres=len(genres),
        input_dim=32,
        hidden_dim=config["model"]["gnn_hidden"],
        layers=config["model"]["gnn_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    dataset = GenreGraphDataset(manifest, "test", genres)
    predictions, targets, cases = [], [], []
    with torch.no_grad():
        for graph in dataset:
            graph = device_graph(graph, device)
            probabilities = torch.softmax(model(graph), dim=0)
            prediction = int(probabilities.argmax().item())
            target = int(graph["y"].item())
            predictions.append(prediction)
            targets.append(target)
            cases.append(
                {
                    "track_id": int(graph["track_id"]),
                    "true_genre": genres[target],
                    "predicted_genre": genres[prediction],
                    "confidence": float(probabilities[prediction].item()),
                }
            )
    predictions_array = np.asarray(predictions)
    targets_array = np.asarray(targets)
    metrics = single_label_metrics(predictions_array, targets_array, genres)
    plots = ensure_dir(output_dir / "plots")
    save_json(metrics, output_dir / "task2_graphsage_test_metrics.json")
    save_json({"predictions": cases}, output_dir / "task2_graphsage_predictions.json")
    save_confusion_matrix(
        predictions_array,
        targets_array,
        genres,
        plots / "task2_graphsage_confusion_matrix.png",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate project models.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--manifest", default="data/processed/task2/task2_graph_manifest.jsonl"
    )
    parser.add_argument("--task", choices=["genre_gnn", "bert", "gnn", "fusion"], required=True)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    output_dir = ensure_dir(args.output_dir)
    if args.task == "genre_gnn":
        metrics = evaluate_genre_gnn(args.checkpoint, args.manifest, output_dir)
        print(metrics)
        return
    raise NotImplementedError("Only Task 2 genre_gnn evaluation is currently implemented.")


if __name__ == "__main__":
    main()
