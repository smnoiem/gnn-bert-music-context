from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utils import configure_logging, save_json


METRIC_NAMES = ("accuracy", "macro_f1", "micro_f1", "auc_pr")


def load_test_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file was not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("test_metrics", payload)
    missing = [name for name in METRIC_NAMES if name not in metrics]
    test_loss = payload.get("test_loss", metrics.get("test_loss"))
    if test_loss is None:
        missing.append("test_loss")
    if missing:
        raise ValueError(f"Metrics file is missing {missing}: {path}")
    return {
        "test_loss": float(test_loss),
        **{name: float(metrics[name]) for name in METRIC_NAMES},
    }


def compare_task2_models(
    graphsage_metrics: Path,
    cnn_metrics: Path,
    output: Path,
) -> dict:
    comparison = {
        "task": "task2_genre_classification",
        "models": {
            "graphsage": {
                "description": "PyTorch Geometric GraphSAGE on MFCC+chroma segment graphs",
                "metrics": load_test_metrics(graphsage_metrics),
            },
            "cnn_melspectrogram": {
                "description": "CNN baseline on 5-second log-mel spectrograms",
                "metrics": load_test_metrics(cnn_metrics),
            },
        },
    }
    save_json(comparison, output)
    return comparison


def main() -> None:
    logger = configure_logging()
    parser = argparse.ArgumentParser(
        description="Compare the Task 2 GraphSAGE model with the CNN baseline."
    )
    parser.add_argument(
        "--graphsage-metrics",
        default="results/task2/task2_graphsage_test_metrics.json",
    )
    parser.add_argument(
        "--cnn-metrics",
        default="results/task2/task2_cnn_test_metrics.json",
    )
    parser.add_argument(
        "--output",
        default="results/task2/task2_model_comparison.json",
    )
    args = parser.parse_args()
    comparison = compare_task2_models(
        Path(args.graphsage_metrics),
        Path(args.cnn_metrics),
        Path(args.output),
    )
    logger.info(
        "Compared GraphSAGE metrics from %s with CNN metrics from %s; wrote %s",
        args.graphsage_metrics,
        args.cnn_metrics,
        args.output,
    )
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
