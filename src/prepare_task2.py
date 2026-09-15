from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .audio_features import load_audio, mel_spectrogram, segment_audio
from .train import load_manifest
from .utils import configure_logging, ensure_dir, progress


def load_audio_metadata(path: str | Path) -> dict[int, str]:
    with open(path, newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        required = {"track_id", "path"}
        if not required.issubset(rows.fieldnames or set()):
            raise ValueError(f"Audio metadata must contain {sorted(required)}")
        metadata = {}
        for row in rows:
            track_id = int(row["track_id"])
            if track_id in metadata:
                raise ValueError(f"Duplicate audio metadata track ID: {track_id}")
            metadata[track_id] = row["path"]
    if not metadata:
        raise ValueError(f"Audio metadata is empty: {path}")
    return metadata


def build_mel_cache(
    manifest: str | Path,
    metadata: str | Path,
    audio_root: str | Path,
    output: str | Path,
    sample_rate: int = 22050,
    segment_seconds: float = 5.0,
    n_mels: int = 128,
) -> int:
    rows = load_manifest(manifest)
    audio_paths = load_audio_metadata(metadata)
    output = ensure_dir(output)
    segment_samples = round(sample_rate * segment_seconds)
    written = 0
    for row in progress(rows, desc="Caching mel spectrograms", total=len(rows)):
        track_id = int(row["track_id"])
        cache_path = output / f"{track_id:06d}.pt"
        if cache_path.is_file():
            continue
        if track_id not in audio_paths:
            raise ValueError(f"Track {track_id} is missing from audio metadata")
        audio_path = Path(audio_root) / audio_paths[track_id]
        waveform, _ = load_audio(str(audio_path), sample_rate)
        segments = segment_audio(
            waveform,
            sample_rate,
            segment_seconds=segment_samples / sample_rate,
            minimum_seconds=1.0,
        )
        if not segments:
            raise ValueError(f"Audio file contains no valid segments: {audio_path}")
        mel_segments = []
        for segment in segments:
            if len(segment) < segment_samples:
                segment = np.pad(segment, (0, segment_samples - len(segment)))
            mel_segments.append(
                torch.from_numpy(
                    mel_spectrogram(segment, sample_rate, n_mels)
                ).unsqueeze(0)
            )
        torch.save({"x": torch.stack(mel_segments), "track_id": track_id}, cache_path)
        written += 1
    return written


def main() -> None:
    logger = configure_logging()
    parser = argparse.ArgumentParser(description="Cache Task 2 mel-spectrogram inputs.")
    parser.add_argument("--manifest", default="data/processed/task2/task2_graph_manifest.jsonl")
    parser.add_argument("--metadata", default="data/processed/task2/fma_metadata.csv")
    parser.add_argument("--audio-root", default="data/raw/fma/fma_small")
    parser.add_argument("--output", default="data/processed/task2/mels")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--n-mels", type=int, default=128)
    args = parser.parse_args()
    count = build_mel_cache(
        args.manifest,
        args.metadata,
        args.audio_root,
        args.output,
        args.sample_rate,
        args.segment_seconds,
        args.n_mels,
    )
    logger.info("Cached mel inputs for %d Task 2 tracks in %s", count, args.output)


if __name__ == "__main__":
    main()
