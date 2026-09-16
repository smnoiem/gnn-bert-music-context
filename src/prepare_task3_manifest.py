"""Create a verified graph/text/label manifest for Task 3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .utils import progress


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def build_task3_manifest(
    graph_manifest: str | Path,
    metadata_csv: str | Path,
    labels_json: str | Path,
    output: str | Path,
) -> list[dict]:
    """Join graph rows with BERT text and fixed multihot targets by track_id."""
    graph_manifest = Path(graph_manifest)
    rows = load_jsonl(graph_manifest)
    metadata = pd.read_csv(metadata_csv, low_memory=False)
    label_spec = json.loads(Path(labels_json).read_text(encoding="utf-8"))
    labels = list(label_spec["labels"])
    required = {"track_id", "bert_text", "artist_id", "split"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Prepared metadata is missing columns: {sorted(missing)}")
    label_columns = [f"label_{label}" for label in labels]
    missing_labels = set(label_columns) - set(metadata.columns)
    if missing_labels:
        raise ValueError(
            f"Prepared metadata is missing label columns: {sorted(missing_labels)}"
        )
    if metadata["track_id"].duplicated().any():
        raise ValueError("Prepared metadata contains duplicate track_id values")
    by_track = metadata.set_index("track_id").to_dict("index")
    output_path = Path(output)
    output_parent = output_path.resolve().parent
    output_rows = []
    seen_tracks: set[int] = set()
    artists: dict[int, str] = {}
    for graph_row in progress(
        rows,
        desc="Joining Task 3 graph/text/labels",
        total=len(rows),
    ):
        track_id = int(graph_row["track_id"])
        if track_id in seen_tracks:
            raise ValueError(f"Graph manifest contains duplicate track_id: {track_id}")
        seen_tracks.add(track_id)
        metadata_row = by_track.get(track_id)
        if metadata_row is None:
            raise ValueError(f"No prepared metadata exists for track_id {track_id}")
        text = str(metadata_row["bert_text"]).strip()
        if not text or text.lower() == "nan":
            raise ValueError(f"Empty bert_text for track_id {track_id}")
        target = [int(metadata_row[column]) for column in label_columns]
        if any(value not in {0, 1} for value in target):
            raise ValueError(f"Non-binary target for track_id {track_id}")
        split = str(metadata_row["split"]).strip()
        artist_id = int(metadata_row["artist_id"])
        if artist_id in artists and artists[artist_id] != split:
            raise ValueError(f"Artist leakage detected for artist_id {artist_id}")
        artists[artist_id] = split
        graph_file = Path(graph_row["graph"])
        if not graph_file.is_absolute():
            graph_file = graph_manifest.resolve().parent / graph_file
        if not graph_file.is_file():
            raise FileNotFoundError(f"Graph file was not found: {graph_file}")
        output_rows.append(
            {
                "track_id": track_id,
                "graph": os.path.relpath(graph_file, output_parent),
                "bert_text": text,
                "labels": [label for label, value in zip(labels, target) if value],
                "target": target,
                "split": split,
                "artist_id": artist_id,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row) + "\n")
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-manifest", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = build_task3_manifest(
        args.graph_manifest, args.metadata, args.labels, args.output
    )
    print(f"Wrote {len(rows)} verified Task 3 rows.")


if __name__ == "__main__":
    main()
