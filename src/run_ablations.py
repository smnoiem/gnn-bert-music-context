"""Run the four Task 3 comparison conditions on one fixed manifest and seed."""
import argparse
import logging
import os
import subprocess
import sys

from .utils import configure_logging


def main():
    configure_logging()
    p = argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=10); p.add_argument("--synthetic", action="store_true"); args = p.parse_args()
    conditions = [("bert", "bert_only", []), ("gnn", "gnn_only", []), ("fusion", "fusion_early_concat", ["--early-concat"]), ("fusion", "fusion_cross_attention", [])]
    logger = logging.getLogger("music-context")
    for index, (task, name, extra) in enumerate(conditions, start=1):
        command = [sys.executable, "-m", "src.train", "--task", task, "--run-name", name, "--epochs", str(args.epochs)] + extra
        if args.synthetic: command.append("--synthetic")
        logger.info("Starting ablation %d/%d: %s", index, len(conditions), " ".join(command))
        subprocess.run(command, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        logger.info("Completed ablation %d/%d: %s", index, len(conditions), name)


if __name__ == "__main__": main()
