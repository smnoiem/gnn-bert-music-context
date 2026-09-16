# Task 3 Dataset and Label Decisions

This file records the confirmed Task 3 dataset and target decisions for future
implementation and report generation.

## Confirmed dataset

- **Dataset:** FMA-small only.
- **Metadata source:** the complete FMA `tracks.csv`.
- **Audio representation:** 5-second audio segments converted into feature
  graphs for the GNN branch.
- **Text representation:** only non-target FMA metadata text, such as track
  title and album title, may be passed to BERT.
- **Artist split rule:** all tracks by one artist must remain in exactly one of
  train, validation, or test.
- **No external pairing:** MusicCaps captions, random text, and genre-based
  text pairing are not valid substitutes for missing FMA text.

## Confirmed prediction task

Task 3 is **multilabel genre/tag prediction**.

- The target combines the normalized `track_genre_top` value and normalized
  values from the `track_tags` array.
- A track may have multiple positive labels.
- The model outputs one logit per selected label.
- Training uses `BCEWithLogitsLoss`.
- The target is a fixed-length multihot vector.
- A label is present when it matches the track genre or appears in the track's
  tag array.

The fixed vocabulary contains at most 100 labels:

1. Include all unique normalized genres first.
2. Fill remaining slots with the most frequent normalized tags.
3. Exclude tags already represented as genres.
4. Resolve frequency ties deterministically.

The exact ordered vocabulary must be persisted in JSON and reused for every
split and experiment.

## Metadata preparation decisions

The preparation process is intentionally split into reproducible stages:

1. **Scan:** read the complete `tracks.csv`, parse the genre and plain
   array-valued tag columns, normalize values, and write all unique values and
   frequencies.
2. **Select:** include all genres and then the most frequent tags up to the
   100-label limit.
3. **Prepare:** write a new derived dataset containing the original metadata,
   one binary `label_<name>` column per selected label, and a
   `task3_labels` convenience string. The source `tracks.csv` is never
   modified.

## BERT text preparation

The `prepare-text` command appends a `bert_text` column to a derived copy of
the prepared dataset. It uses this compact descriptive set:

```text
album_information
album_title
album_type
artist_bio
artist_location
artist_members
track_composer
track_information
track_title
```

These fields are preferred over administrative metadata and artist identity
to reduce irrelevant signals and memorization shortcuts. Genre/tag sources
(`track_genre_top`, `track_genres`, `track_genres_all`, `track_tags`,
`album_tags`, and `artist_tags`) plus IDs, paths, dates, URLs, licenses,
publishers, and split fields are excluded. The command fails if a fixed text
column is missing.

The resulting text uses labeled fields, for example:

   ```text
   album: Ambient Sessions. track: Sunrise
   ```

Null/NaN fields are omitted from each row's text. If all nine selected fields
are missing, the row is excluded from the BERT/fusion dataset rather than
represented by placeholder text.

The Task 3 fusion manifest is an inner join on `track_id` between the Task 2
graph manifest and this prepared metadata. Tracks skipped during graph
construction and tracks removed during text preparation therefore do not enter
the fusion dataset.

The current helper is:

```text
src/prepare_task3_labels.py
```

Its `scan`, `select`, and `prepare` commands define the primary label
preparation flow. The genre-only commands are not the primary Task 3 target.

## Target encoding

For a fixed vocabulary such as:

```text
["guitar", "jazz", "live", "rock"]
```

a track with genre `rock` and tags `["guitar", "live"]` has:

```text
[1, 0, 1, 1]
```

Each selected label has its own binary position.

For a batch of 32 tracks and 100 labels:

```text
logits:  [32, 100]
targets: [32, 100]
```

The expected training contract is:

```python
loss = torch.nn.BCEWithLogitsLoss()(logits, targets.float())
probabilities = torch.sigmoid(logits)
```

## Normalization and tag parsing

- Normalize every genre and tag by trimming, lowercasing, and collapsing
  repeated whitespace.
- Treat `track_tags` as an array of tag strings.
- Values such as `80s` are literal tag names, not counts.
- Do not interpret tag values as `(name, count)` pairs.
- Malformed tag entries are silently skipped according to the confirmed
  preparation behavior.
- Each track contributes a tag at most once to its target, even if a malformed
  or repeated source entry contains duplicates.

## Reproducibility requirements

- Preserve the exact ordered genre vocabulary in a JSON artifact.
- Resolve genre and tag normalization explicitly and consistently.
- Record track counts per split and per genre.
- Report missing genre values instead of silently assigning a fallback label.
- Generate the derived CSV under `data/processed/task3/`.
