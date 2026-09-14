"""Run the four Task 3 comparison conditions on one fixed manifest and seed."""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=10); p.add_argument("--synthetic", action="store_true"); args = p.parse_args()
    conditions = [("bert", "bert_only", []), ("gnn", "gnn_only", []), ("fusion", "fusion_early_concat", ["--early-concat"]), ("fusion", "fusion_cross_attention", [])]
    for task, name, extra in conditions:
        command = [sys.executable, "-m", "src.train", "--task", task, "--run-name", name, "--epochs", str(args.epochs)] + extra
        if args.synthetic: command.append("--synthetic")
        print("Running:", " ".join(command)); subprocess.run(command, check=True)


if __name__ == "__main__": main()
