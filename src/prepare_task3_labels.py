"""Build a deterministic Task 3 label vocabulary from FMA track metadata."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


def _flatten_columns(columns: pd.MultiIndex) -> list[str]:
    return [
        "_".join(str(part).strip() for part in column if str(part) != "nan")
        for column in columns
    ]


def _read_tracks(path: Path) -> pd.DataFrame:
    """Read either the FMA two-row-header export or a normal CSV."""
    normal = pd.read_csv(path)
    expected = {
        "track_genre_top",
        "genre",
        "genres",
        "track_genre",
        "track_tags",
        "tags",
        "tag",
        "track_tag",
    }
    if expected.intersection(str(column).strip().lower() for column in normal.columns):
        return normal
    try:
        tracks = pd.read_csv(path, header=[0, 1], index_col=0)
        if isinstance(tracks.columns, pd.MultiIndex):
            tracks.columns = _flatten_columns(tracks.columns)
            if expected.intersection(
                str(column).strip().lower() for column in tracks.columns
            ):
                return tracks.reset_index()
    except (pd.errors.ParserError, UnicodeDecodeError):
        pass
    return normal


def _find_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        f"Could not find any of {list(candidates)} in tracks.csv columns: "
        f"{sorted(normalized.values())}"
    )


def parse_array_cell(value: object) -> list[str]:
    """Parse list-like CSV cells while accepting JSON and Python list syntax."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "[]"}:
            return []
        try:
            values = json.loads(text)
        except json.JSONDecodeError:
            try:
                values = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                values = [part.strip() for part in text.split(",")]
        if not isinstance(values, (list, tuple, set)):
            values = [values]
    return sorted(
        {
            str(item).strip()
            for item in values
            if str(item).strip() and str(item).strip().lower() != "nan"
        }
    )


def scan_task3_metadata(tracks_csv: str | Path, output: str | Path) -> dict:
    """Scan all tracks and persist complete genre/tag frequencies."""
    tracks = _read_tracks(Path(tracks_csv))
    genre_column = _find_column(
        tracks.columns, ("track_genre_top", "genre", "genres", "track_genre")
    )
    tag_column = _find_column(
        tracks.columns, ("track_tags", "tags", "tag", "track_tag")
    )
    genre_counts = Counter(
        str(value).strip()
        for value in tracks[genre_column]
        if str(value).strip() and str(value).strip().lower() != "nan"
    )
    tag_counts = Counter(
        tag for value in tracks[tag_column] for tag in parse_array_cell(value)
    )
    result = {
        "tracks_scanned": len(tracks),
        "genre_column": genre_column,
        "tag_column": tag_column,
        "genres": sorted(genre_counts),
        "genre_frequencies": dict(sorted(genre_counts.items())),
        "tags": sorted(tag_counts),
        "tag_frequencies": dict(sorted(tag_counts.items())),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def select_task3_labels_from_scan(
    scan_path: str | Path, output: str | Path, max_labels: int = 100
) -> dict:
    """Select all genres and then the most frequent non-duplicate tags."""
    scan = json.loads(Path(scan_path).read_text(encoding="utf-8"))
    genres = list(scan["genres"])
    if len(genres) > max_labels:
        raise ValueError(
            f"Found {len(genres)} genres, exceeding the {max_labels}-label budget"
        )
    genre_set = set(genres)
    tag_counts = Counter(scan["tag_frequencies"])
    selected_tags = [
        tag
        for tag, _count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))
        if tag not in genre_set
    ][: max_labels - len(genres)]
    result = {
        "max_labels": max_labels,
        "genre_column": scan["genre_column"],
        "tag_column": scan["tag_column"],
        "labels": genres + selected_tags,
        "genres": genres,
        "tags": selected_tags,
        "genre_count": len(genres),
        "tag_count": len(selected_tags),
        "tracks_scanned": scan["tracks_scanned"],
        "tag_frequencies": {tag: tag_counts[tag] for tag in selected_tags},
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def select_task3_genres_from_scan(
    scan_path: str | Path, output: str | Path
) -> dict:
    """Persist the complete sorted top-level genre vocabulary."""
    scan = json.loads(Path(scan_path).read_text(encoding="utf-8"))
    genres = list(scan["genres"])
    if not genres:
        raise ValueError("No genres were found in the metadata scan")
    result = {
        "task": "single_label_multiclass_genre",
        "genre_column": scan["genre_column"],
        "genres": genres,
        "labels": genres,
        "genre_count": len(genres),
        "tracks_scanned": scan["tracks_scanned"],
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def prepare_task3_genre_dataset(
    tracks_csv: str | Path, genres_path: str | Path, output: str | Path
) -> pd.DataFrame:
    """Add scalar genre labels and indices for multi-class training."""
    tracks = _read_tracks(Path(tracks_csv))
    genre_spec = json.loads(Path(genres_path).read_text(encoding="utf-8"))
    genre_column = genre_spec["genre_column"]
    genres = list(genre_spec["genres"])
    index_by_genre = {genre: index for index, genre in enumerate(genres)}
    normalized = tracks[genre_column].map(lambda value: str(value).strip())
    unknown = sorted(set(normalized) - set(index_by_genre))
    if unknown:
        raise ValueError(f"Unknown or invalid genres found: {unknown[:10]}")
    tracks["genre_label"] = normalized
    tracks["genre_index"] = normalized.map(index_by_genre).astype("int64")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(output_path, index=False)
    return tracks


def prepare_task3_text_dataset(
    input_csv: str | Path,
    output: str | Path,
    text_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Append deterministic BERT text made from safe metadata text columns."""
    tracks = pd.read_csv(input_csv, low_memory=False)
    excluded_exact = {
        "track_genre_top",
        "track_genres",
        "track_genres_all",
        "track_tags",
        "genre_label",
        "genre_index",
        "task3_labels",
        "split",
        "artist_id",
        "track_id",
        "path",
    }
    excluded_prefixes = ("label_",)
    if text_columns is None:
        candidates = [
            str(column)
            for column in tracks.columns
            if pd.api.types.is_string_dtype(tracks[column])
            and str(column).strip().lower() not in excluded_exact
            and not str(column).strip().lower().startswith(excluded_prefixes)
        ]
    else:
        missing = set(text_columns) - set(tracks.columns)
        if missing:
            raise ValueError(f"Text columns were not found: {sorted(missing)}")
        candidates = text_columns
    if not candidates:
        raise ValueError("No safe text columns are available for BERT input")

    def format_value(column: str, value: object) -> str:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return ""
        return f"{column}: {text}"

    tracks["bert_text"] = [
        ". ".join(
            value
            for column in candidates
            for value in [format_value(column, row[column])]
            if value
        )
        or "no track metadata available"
        for _, row in tracks.iterrows()
    ]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(output_path, index=False)
    return tracks


def prepare_task3_dataset(
    tracks_csv: str | Path, labels_path: str | Path, output: str | Path
) -> pd.DataFrame:
    """Add one binary target column per selected label and write a CSV."""
    tracks = _read_tracks(Path(tracks_csv))
    label_spec = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    genre_column = label_spec["genre_column"]
    tag_column = label_spec["tag_column"]
    missing = {genre_column, tag_column} - set(tracks.columns)
    if missing:
        raise ValueError(f"Input tracks.csv is missing columns: {sorted(missing)}")
    labels = list(label_spec["labels"])
    track_genres = tracks[genre_column].map(
        lambda value: (
            {str(value).strip()}
            if str(value).strip() and str(value).strip().lower() != "nan"
            else set()
        )
    )
    track_tags = tracks[tag_column].map(parse_array_cell).map(set)
    for label in labels:
        tracks[f"label_{label}"] = [
            int(label in genres or label in tags)
            for genres, tags in zip(track_genres, track_tags)
        ]
    tracks["task3_labels"] = [
        "|".join(label for label in labels if row[f"label_{label}"])
        for _, row in tracks.iterrows()
    ]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(output_path, index=False)
    return tracks


def select_task3_labels(
    tracks_csv: str | Path,
    output: str | Path,
    max_labels: int = 100,
) -> dict:
    """Select all genres, then the most frequent tags, up to ``max_labels``."""
    if max_labels < 1:
        raise ValueError("max_labels must be positive")
    scan_path = Path(output).with_name("task3_metadata_scan.json")
    scan_task3_metadata(tracks_csv, scan_path)
    return select_task3_labels_from_scan(scan_path, output, max_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    scan = commands.add_parser("scan")
    scan.add_argument("--tracks-csv", required=True)
    scan.add_argument("--output", required=True)

    select = commands.add_parser("select")
    select.add_argument("--scan", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--max-labels", type=int, default=100)

    select_genres = commands.add_parser("select-genres")
    select_genres.add_argument("--scan", required=True)
    select_genres.add_argument("--output", required=True)

    prepare_genre = commands.add_parser("prepare-genre")
    prepare_genre.add_argument("--tracks-csv", required=True)
    prepare_genre.add_argument("--genres", required=True)
    prepare_genre.add_argument("--output", required=True)

    prepare_text = commands.add_parser("prepare-text")
    prepare_text.add_argument("--input", required=True)
    prepare_text.add_argument("--output", required=True)
    prepare_text.add_argument(
        "--columns",
        nargs="+",
        help="Optional explicit safe text columns; otherwise object columns are detected.",
    )

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--tracks-csv", required=True)
    prepare.add_argument("--labels", required=True)
    prepare.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "scan":
        result = scan_task3_metadata(args.tracks_csv, args.output)
        print(f"Scanned {result['tracks_scanned']} tracks and wrote all genre/tag frequencies.")
    elif args.command == "select":
        result = select_task3_labels_from_scan(args.scan, args.output, args.max_labels)
        print(f"Selected {result['genre_count']} genres and {result['tag_count']} tags.")
    elif args.command == "select-genres":
        result = select_task3_genres_from_scan(args.scan, args.output)
        print(f"Selected {result['genre_count']} genre classes.")
    elif args.command == "prepare-genre":
        result = prepare_task3_genre_dataset(args.tracks_csv, args.genres, args.output)
        print(f"Wrote {len(result)} tracks with genre_label and genre_index columns.")
    elif args.command == "prepare-text":
        result = prepare_task3_text_dataset(args.input, args.output, args.columns)
        print(f"Wrote {len(result)} tracks with bert_text from metadata columns.")
    else:
        result = prepare_task3_dataset(args.tracks_csv, args.labels, args.output)
        print(f"Wrote {len(result)} tracks with {len(json.loads(Path(args.labels).read_text())['labels'])} label columns.")


if __name__ == "__main__":
    main()
