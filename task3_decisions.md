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

Task 3 is **single-label multi-class genre classification**, not multilabel
genre/tag prediction.

- The target is the scalar `track_genre_top` value.
- Each track has exactly one target genre.
- The model predicts exactly one genre using `argmax`.
- Training uses `CrossEntropyLoss`.
- The output dimension is the number of selected genre classes.
- FMA tags are not primary targets.

Tags may be retained for descriptive statistics or a future auxiliary
multi-label experiment, but they must not be combined with the primary genre
target for the main Task 3 result.

## Metadata preparation decisions

The preparation process is intentionally split into reproducible stages:

1. **Scan:** read the complete `tracks.csv`, parse the genre and array-valued
   tag columns, and write all unique values and frequencies.
2. **Select:** use the complete genre list as the fixed multi-class vocabulary.
   Tags are reported for analysis only and do not become primary classes.
3. **Prepare:** write a new derived dataset containing the original metadata,
   a deterministic integer `genre_index` target, and a `genre_label` display
   column. The source `tracks.csv` is never modified.

   ## BERT text preparation

   The `prepare-text` command appends a `bert_text` column to a derived copy of
   the genre dataset. Automatic column selection combines object/string metadata
   columns while excluding genre fields, tag fields, target columns, IDs, paths,
   and split fields. This avoids target leakage. The selected columns can be
   overridden with an explicit safe `--columns` list after inspecting the scan.

   The resulting text uses labeled fields, for example:

   ```text
   album: Ambient Sessions. track: Sunrise
   ```

   The fallback text for a row with no usable metadata is
   `no track metadata available`; genre and tags are never used as fallback text.

The current helper is:

```text
src/prepare_task3_labels.py
```

Its existing `scan` command is retained for full genre/tag inventory. The
primary Task 3 training preparation must use the genre-only target contract
above; the earlier 100-label multihot output is only an exploratory artifact
and must not be used as the main multi-class target.

## Target encoding

For a fixed vocabulary such as:

```text
["Blues", "Electronic", "Folk", "Jazz", "Rock"]
```

the prepared target is:

```text
genre_label: Rock
genre_index: 4
```

For a batch of 32 tracks and five genres:

```text
logits: [32, 5]
targets: [32]
```

The expected training contract is:

```python
loss = torch.nn.CrossEntropyLoss()(logits, targets)
prediction = logits.argmax(dim=1)
```

## Reproducibility requirements

- Preserve the exact ordered genre vocabulary in a JSON artifact.
- Resolve genre normalization explicitly and consistently.
- Record track counts per split and per genre.
- Report missing or invalid genre values instead of silently assigning a
  fallback class.
- Generate the derived CSV under `data/processed/task3/`.
