from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .audio_features import load_audio, segment_features
from .utils import ensure_dir


SPLIT_NAMES = {
    "training": "train",
    "train": "train",
    "validation": "val",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def _flatten_columns(columns: pd.MultiIndex) -> list[str]:
    return [
        "_".join(str(part).strip() for part in column if str(part) != "nan")
        for column in columns
    ]


def _track_audio_path(track_id: int) -> str:
    filename = f"{track_id:06d}.mp3"
    return f"{filename[:3]}/{filename}"


def prepare_fma_metadata(
    tracks_csv: str | Path,
    audio_root: str | Path,
    output: str | Path,
) -> pd.DataFrame:
    """Convert official FMA metadata into the Task 2 metadata contract."""
    tracks_csv, audio_root, output = Path(tracks_csv), Path(audio_root), Path(output)
    if not tracks_csv.is_file():
        raise FileNotFoundError(f"FMA tracks metadata was not found: {tracks_csv}")
    if not audio_root.is_dir():
        raise FileNotFoundError(f"FMA-small audio directory was not found: {audio_root}")

    tracks = pd.read_csv(tracks_csv, header=[0, 1], index_col=0)
    tracks.columns = _flatten_columns(tracks.columns)
    required = {"set_subset", "set_split", "track_genre_top", "artist_artist_id"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks.csv is missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    missing_audio: list[str] = []
    for raw_track_id, row in tracks.iterrows():
        if str(row["set_subset"]).strip().lower() != "small":
            continue
        track_id = int(raw_track_id)
        relative_path = _track_audio_path(track_id)
        if not (audio_root / relative_path).is_file():
            missing_audio.append(relative_path)
            continue
        genre = str(row["track_genre_top"]).strip()
        if not genre or genre.lower() == "nan":
            raise ValueError(f"Track {track_id} has no top-level genre")
        split = SPLIT_NAMES.get(str(row["set_split"]).strip().lower())
        if split is None:
            raise ValueError(f"Unsupported FMA split value: {row['set_split']!r}")
        rows.append(
            {
                "track_id": track_id,
                "path": relative_path,
                "genre": genre,
                "split": split,
                "artist_id": int(row["artist_artist_id"]),
            }
        )

    if missing_audio:
        preview = ", ".join(missing_audio[:5])
        raise FileNotFoundError(
            f"{len(missing_audio)} FMA-small audio files are missing; examples: {preview}"
        )
    if not rows:
        raise ValueError("No FMA-small tracks were found in tracks.csv")

    metadata = pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)
    if metadata["track_id"].duplicated().any():
        raise ValueError("Duplicate track IDs found in prepared metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output, index=False)
    return metadata


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
    p = argparse.ArgumentParser()
    p.add_argument("--prepare-metadata", action="store_true")
    p.add_argument("--tracks-csv", default="data/raw/fma/fma_metadata/tracks.csv")
    p.add_argument("--metadata-output", default="data/raw/metadata.csv")
    p.add_argument("--metadata")
    p.add_argument("--audio-root", default="data/raw/fma/fma_small")
    p.add_argument("--output", default="data/processed/graphs")
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--segment-seconds", type=float, default=5)
    p.add_argument("--threshold", type=float, default=.75)
    args = p.parse_args()
    if args.prepare_metadata:
        metadata = prepare_fma_metadata(args.tracks_csv, args.audio_root, args.metadata_output)
        print(f"Wrote {len(metadata)} tracks to {args.metadata_output}")
        print(metadata.groupby("split").size().to_string())
        return
    if not args.metadata:
        p.error("--metadata is required unless --prepare-metadata is used")
    table, output = pd.read_csv(args.metadata), ensure_dir(args.output)
    required = {"track_id", "path"}; missing = required - set(table.columns)
    if missing: raise ValueError(f"metadata missing columns: {sorted(missing)}")
    manifest = []
    for row in table.itertuples(index=False):
        y, sr = load_audio(str(Path(args.audio_root) / row.path), args.sample_rate)
        graph = build_segment_graph(segment_features(y, sr, args.segment_seconds), args.threshold)
        graph.update({"track_id": str(row.track_id), "text": str(getattr(row, "text", "")), "labels": str(getattr(row, "labels", ""))})
        target = output / f"{row.track_id}.pt"; torch.save(graph, target)
        item = {"track_id": str(row.track_id), "graph": str(target), "split": str(getattr(row, "split", "train")), "text": str(getattr(row, "text", "")), "labels": str(getattr(row, "labels", "")), "artist_id": str(getattr(row, "artist_id", ""))}
        if hasattr(row, "valence") and hasattr(row, "arousal"):
            item.update({"valence": float(row.valence), "arousal": float(row.arousal)})
        manifest.append(item)
    manifest_path = output.parent / "manifest.jsonl"
    with open(manifest_path, "w") as stream:
        for item in manifest: stream.write(json.dumps(item) + "\n")
    print(f"Wrote {len(manifest)} graphs and {manifest_path}")


if __name__ == "__main__": main()
