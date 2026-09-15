from __future__ import annotations

import argparse
import logging
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .audio_features import load_audio, segment_features
from .utils import configure_logging, progress
from .utils import ensure_dir


SPLIT_NAMES = {
    "training": "train",
    "train": "train",
    "validation": "val",
    "valid": "val",
    "val": "val",
    "test": "test",
}
REQUIRED_METADATA_COLUMNS = {"track_id", "path", "genre", "split", "artist_id"}


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
    required = {"set_subset", "set_split", "track_genre_top", "artist_id"}
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
                "artist_id": int(row["artist_id"]),
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
    _validate_artist_splits(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output, index=False)
    return metadata


def _validate_artist_splits(metadata: pd.DataFrame) -> None:
    artists = metadata.groupby("artist_id")["split"].nunique()
    leaked = artists[artists > 1]
    if not leaked.empty:
        examples = ", ".join(str(value) for value in leaked.index[:5])
        raise ValueError(f"Artist leakage across splits detected; examples: {examples}")


def build_segment_graph(features: np.ndarray, threshold: float = 0.75) -> dict[str, torch.Tensor]:
    """Build temporal and cosine-similarity edges for segment nodes."""
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")

    x = torch.as_tensor(features, dtype=torch.float32)
    normalized = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    similarity = normalized @ normalized.T
    edges = {(index, index) for index in range(len(x))}
    for index in range(len(x) - 1):
        edges.update({(index, index + 1), (index + 1, index)})
    for source, target in zip(*torch.where(torch.triu(similarity >= threshold, diagonal=1))):
        edges.update({(int(source), int(target)), (int(target), int(source))})

    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    return {"x": x, "edge_index": edge_index}


def _write_split_files(rows: list[dict[str, object]], split_dir: Path) -> None:
    ensure_dir(split_dir)
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            raise ValueError(f"Split {split!r} is empty")
        (split_dir / f"{split}.json").write_text(json.dumps(split_rows, indent=2))


def build_graph_dataset(
    metadata_path: str | Path,
    audio_root: str | Path,
    graph_dir: str | Path,
    manifest_path: str | Path,
    split_dir: str | Path,
    sample_rate: int = 22050,
    segment_seconds: float = 5.0,
    threshold: float = 0.75,
) -> list[dict[str, object]]:
    """Extract node features, serialize graphs, and write the Task 2 manifest."""
    metadata_path, audio_root = Path(metadata_path), Path(audio_root)
    graph_dir, manifest_path, split_dir = (
        ensure_dir(graph_dir),
        Path(manifest_path),
        Path(split_dir),
    )
    metadata = pd.read_csv(metadata_path)
    missing = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    _validate_artist_splits(metadata)

    manifest: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for row in progress(
        metadata.itertuples(index=False),
        desc="Building graph samples",
        total=len(metadata),
    ):
        audio_path = audio_root / str(row.path)
        try:
            waveform, actual_rate = load_audio(str(audio_path), sample_rate)
            features = segment_features(waveform, actual_rate, segment_seconds)
            graph = build_segment_graph(features, threshold)
            graph.update(
                {
                    "track_id": int(row.track_id),
                    "genre": str(row.genre),
                    "artist_id": int(row.artist_id),
                }
            )
            graph_path = graph_dir / f"{int(row.track_id):06d}.pt"
            torch.save(graph, graph_path)
            manifest_graph = Path(os.path.relpath(graph_path, manifest_path.parent))
            manifest.append(
                {
                    "track_id": int(row.track_id),
                    "graph": str(manifest_graph),
                    "genre": str(row.genre),
                    "split": str(row.split),
                    "artist_id": int(row.artist_id),
                }
            )
        except Exception as exc:
            failure = {
                "track_id": int(row.track_id),
                "path": str(audio_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            logging.getLogger("music-context").exception(
                "Skipping track_id=%s after processing failed: %s",
                row.track_id,
                audio_path,
            )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as stream:
        for item in manifest:
            stream.write(json.dumps(item) + "\n")
    _write_split_files(manifest, split_dir)
    failure_path = manifest_path.with_name(f"{manifest_path.stem}_failures.jsonl")
    with failure_path.open("w", encoding="utf-8") as stream:
        for failure in failures:
            stream.write(json.dumps(failure) + "\n")
    logger = logging.getLogger("music-context")
    if failures:
        logger.warning(
            "Completed graph build with %d skipped file(s); failure report: %s",
            len(failures),
            failure_path,
        )
    else:
        logger.info("Completed graph build with no skipped files")
    return manifest


def main() -> None:
    logger = configure_logging()
    parser = argparse.ArgumentParser(description="Prepare FMA metadata or build Task 2 graphs.")
    parser.add_argument("--prepare-metadata", action="store_true")
    parser.add_argument("--tracks-csv", default="data/raw/fma/fma_metadata/tracks.csv")
    parser.add_argument(
        "--metadata-output", default="data/processed/task2/fma_metadata.csv"
    )
    parser.add_argument("--metadata")
    parser.add_argument("--audio-root", default="data/raw/fma/fma_small")
    parser.add_argument("--output", default="data/processed/task2/graphs")
    parser.add_argument(
        "--manifest", default="data/processed/task2/task2_graph_manifest.jsonl"
    )
    parser.add_argument("--split-dir", default="data/splits/task2")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    if args.prepare_metadata:
        metadata = prepare_fma_metadata(args.tracks_csv, args.audio_root, args.metadata_output)
        logger.info("Wrote %d tracks to %s", len(metadata), args.metadata_output)
        logger.info("Split counts:\n%s", metadata.groupby("split").size().to_string())
        return
    if not args.metadata:
        parser.error("--metadata is required unless --prepare-metadata is used")

    manifest = build_graph_dataset(
        args.metadata,
        args.audio_root,
        args.output,
        args.manifest,
        args.split_dir,
        args.sample_rate,
        args.segment_seconds,
        args.threshold,
    )
    logger.info("Wrote %d graphs to %s", len(manifest), args.output)
    logger.info("Wrote manifest to %s", args.manifest)


if __name__ == "__main__":
    main()
