from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .audio_features import load_audio, segment_features
from .utils import ensure_dir


def build_segment_graph(features: np.ndarray, threshold: float = .75) -> dict:
    """Temporal-adjacency and cosine-similarity segment graph as a portable torch dict."""
    x = torch.tensor(features, dtype=torch.float32); n = len(x)
    x_unit = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sim = x_unit @ x_unit.T
    edges = {(i, i) for i in range(n)}
    for i in range(n - 1): edges.update({(i, i + 1), (i + 1, i)})
    for i, j in zip(*torch.where(torch.triu(sim > threshold, diagonal=1))):
        edges.update({(int(i), int(j)), (int(j), int(i))})
    edge_index = torch.tensor(sorted(edges), dtype=torch.long).T
    return {"x": x, "edge_index": edge_index}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--metadata", required=True); p.add_argument("--audio-root", default=".")
    p.add_argument("--output", default="data/processed/graphs"); p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--segment-seconds", type=float, default=5); p.add_argument("--threshold", type=float, default=.75); args = p.parse_args()
    table, output = pd.read_csv(args.metadata), ensure_dir(args.output)
    required = {"track_id", "path"}; missing = required - set(table.columns)
    if missing: raise ValueError(f"metadata missing columns: {sorted(missing)}")
    manifest = []
    for row in table.itertuples(index=False):
        y, sr = load_audio(str(Path(args.audio_root) / row.path), args.sample_rate)
        graph = build_segment_graph(segment_features(y, sr, args.segment_seconds), args.threshold)
        graph.update({"track_id": str(row.track_id), "text": str(getattr(row, "text", "")), "labels": str(getattr(row, "labels", ""))})
        target = output / f"{row.track_id}.pt"; torch.save(graph, target); manifest.append({"track_id": row.track_id, "graph": str(target), "split": getattr(row, "split", "train")})
    pd.DataFrame(manifest).to_csv(output / "manifest.csv", index=False)


if __name__ == "__main__": main()
