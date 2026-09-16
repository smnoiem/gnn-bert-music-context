"""Compare completed Task 3 runs without starting or modifying training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .utils import progress


DEFAULT_RUNS = (
    "task3_bert_only",
    "task3_gnn_only",
    "task3_early_concat",
    "task3_cross_attention",
)
METRICS = ("macro_f1", "micro_f1", "auc_pr", "loss")
HIGHER_IS_BETTER = {"macro_f1", "micro_f1", "auc_pr"}


def _load_metrics(input_dir: Path, run_names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for name in progress(run_names, desc="Loading Task 3 run metrics", total=len(run_names)):
        path = input_dir / f"{name}_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing completed run metrics for {name!r}: {path}. "
                "Run that condition before comparing ablations."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        test = payload.get("metrics", {}).get("test")
        if not isinstance(test, dict):
            raise ValueError(f"Metrics file has no metrics.test object: {path}")
        missing = [metric for metric in METRICS if metric not in test]
        if missing:
            raise ValueError(f"Metrics file {path} is missing fields: {missing}")
        runs[name] = {metric: float(test[metric]) for metric in METRICS}
    return runs


def compare_task3_runs(
    input_dir: str | Path = "results/task3/ablation",
    output_dir: str | Path | None = None,
    run_names: tuple[str, ...] = DEFAULT_RUNS,
) -> dict[str, Any]:
    """Create comparison data and plots from completed Task 3 metrics files."""
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir is not None else input_path
    output_path.mkdir(parents=True, exist_ok=True)
    if len(set(run_names)) != len(run_names) or not run_names:
        raise ValueError("run_names must be a non-empty sequence of unique names")
    runs = _load_metrics(input_path, run_names)
    baseline = runs.get("task3_bert_only")
    deltas = {}
    if baseline is not None:
        deltas = {
            name: {metric: values[metric] - baseline[metric] for metric in METRICS}
            for name, values in runs.items()
        }
    rankings = {
        metric: sorted(
            runs,
            key=lambda name: runs[name][metric],
            reverse=metric in HIGHER_IS_BETTER,
        )
        for metric in METRICS
    }
    best = {metric: rankings[metric][0] for metric in METRICS}
    analysis = [
        f"Best Macro-F1: {best['macro_f1']} ({runs[best['macro_f1']]['macro_f1']:.4f})",
        f"Best Micro-F1: {best['micro_f1']} ({runs[best['micro_f1']]['micro_f1']:.4f})",
        f"Best PR-AUC: {best['auc_pr']} ({runs[best['auc_pr']]['auc_pr']:.4f})",
        f"Lowest test loss: {best['loss']} ({runs[best['loss']]['loss']:.4f})",
    ]
    if "task3_cross_attention" in deltas:
        delta = deltas["task3_cross_attention"]
        analysis.append(
            "Cross-attention vs BERT-only: "
            f"Macro-F1 {delta['macro_f1']:+.4f}, "
            f"Micro-F1 {delta['micro_f1']:+.4f}, "
            f"PR-AUC {delta['auc_pr']:+.4f}, "
            f"loss {delta['loss']:+.4f}."
        )
    result = {
        "runs": runs,
        "baseline": "task3_bert_only" if baseline is not None else None,
        "deltas_vs_baseline": deltas,
        "rankings": rankings,
        "best_by_metric": best,
        "analysis": analysis,
    }
    (output_path / "task3_ablation_comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output_path / "task3_ablation_analysis.txt").write_text(
        "\n".join(analysis) + "\n", encoding="utf-8"
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(runs)
        figure, axes = plt.subplots(1, 3, figsize=(13, 4))
        for axis, metric in zip(axes, ("macro_f1", "micro_f1", "auc_pr")):
            axis.bar(names, [runs[name][metric] for name in names])
            axis.set_title(metric.replace("_", " ").upper())
            axis.set_ylim(0, max(1.0, max(runs[name][metric] for name in names) * 1.15))
            axis.tick_params(axis="x", rotation=35)
            axis.grid(axis="y", alpha=0.25)
        figure.suptitle("Task 3 ablation comparison on the held-out test split")
        figure.tight_layout()
        plots = output_path / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        figure.savefig(plots / "task3_ablation_comparison.png", dpi=150)
        plt.close(figure)
    except ImportError:
        pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results/task3/ablation")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        help="Completed run names, each matching <name>_metrics.json.",
    )
    args = parser.parse_args()
    result = compare_task3_runs(args.input_dir, args.output_dir, tuple(args.runs))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
