# GNN-BERT Music Context

This repository implements:

- **Task 1:** a DistilBERT/BERT text-only multi-label tag classifier.
- **Task 2:** an audio-segment graph classifier.
- **Task 3:** graph/text fusion with early concatenation or cross-attention.

The commands below use real MusicCaps captions and real audio metadata. Synthetic
data is not part of the documented workflow or the reported results.

## Windows setup

Open **PowerShell** and change to the project directory. Replace the example
path with the directory where this repository is located:

```powershell
Set-Location -LiteralPath "C:\Users\YOUR_USERNAME\Documents\gnn-bert-music-context"
```

Create the virtual environment once, then install the dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in an Administrator PowerShell:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

For later sessions, run these commands from the project directory:

```powershell
.\.venv\Scripts\Activate.ps1
```

The first training run downloads `distilbert-base-uncased` from Hugging Face.
An internet connection is required unless the model has already been cached.

## Logs and progress

Command-line workflows emit timestamped `INFO` logs for setup, phase changes,
epoch summaries, output files, and failures. Dataset and graph-building loops
also show live progress bars. Progress is written to stderr so JSON metrics and
other structured output on stdout remain usable. Ablation child processes inherit
unbuffered output, so each command's logs appear as it runs rather than only
after the process exits.

Graph construction handles failures per audio file: the failing track and full
exception are logged, later files continue processing, and skipped rows are
written to a `<manifest>_failures.jsonl` report next to the graph manifest.


The relevant layout is:

```text
data\
  raw\
    fma\
      fma_metadata\
      fma_small\
  processed\
    task2\
      fma_metadata.csv
      graphs\
      mels\
      task2_graph_manifest.jsonl
  splits\
    task2\
      train.json
      val.json
      test.json
results\
  task2\
    graphsage_genre_best.pt
    graphsage_genre_metrics.json
    cnn_melspectrogram_genre_best.pt
    cnn_melspectrogram_genre_metrics.json
```


## Task 1: train the DistilBERT text classifier

### 1. Export the MusicCaps captions

Create the raw-data directory and export only the caption and identifier
columns needed by the Task 1 runner:

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
python -c "from datasets import load_dataset; ds=load_dataset('google/MusicCaps', split='train'); ds.select_columns(['ytid','caption','start_s','end_s']).to_csv('data/raw/musiccaps.csv', index=False)"
```

The runner also accepts a local CSV, JSON, JSONL, or parquet file. A CSV must
contain a caption/text/description column and either a `tags`, `labels`, or
`label` column. When those label columns are absent, the runner creates the
documented proxy tags by matching music phrases in each caption. These are
proxy-task labels, not human-annotated MusicCaps tags.

### 2. Run the actual Task 1 DistilBERT training command

Run this from the repository root:

```powershell
python -m src.train_bert_musiccaps --input data\raw\musiccaps\musiccaps_public.csv --model-name distilbert-base-uncased --output-dir results --run-name task1_distilbert --epochs 10 --batch-size 8 --learning-rate 2e-5 --max-length 128 --hidden-size 256 --seed 42
```

This command uses the best validation Macro-F1 checkpoint and evaluates it on
the held-out test split. The stable hash split is 80% train, 10% validation,
and 10% test.

Outputs are written to `results\`:

- `task1_distilbert_best.pt`
- `task1_distilbert_metrics.json`
- `task1_distilbert_test_metrics.json`
- `task1_distilbert_predictions.json`
- `task1_distilbert_f1_curve.png`

To use a locally cached model without network access, add
`--local-files-only` to the training command.

## Tasks 2 GNN on Music Structure Graphs

Task 2 keeps its derived metadata, graphs, manifest, splits, and results in
task-specific locations. The original FMA files remain under `data\raw\fma`.

Run these commands sequentially from the repository root after the FMA-small
archives have been extracted under `data\raw\fma`.

### 1. Prepare Task 2 metadata

```powershell
python -m src.graph_builder `
  --prepare-metadata `
  --tracks-csv data\raw\fma\fma_metadata\tracks.csv `
  --audio-root data\raw\fma\fma_small `
  --metadata-output data\processed\task2\fma_metadata.csv
```

### 2. Build Task 2 segment graphs, manifest, and split files

```powershell
python -m src.graph_builder `
  --metadata data\processed\task2\fma_metadata.csv `
  --audio-root data\raw\fma\fma_small `
  --output data\processed\task2\graphs `
  --manifest data\processed\task2\task2_graph_manifest.jsonl `
  --split-dir data\splits\task2 `
  --sample-rate 22050 `
  --segment-seconds 5 `
  --threshold 0.75
```

### 3. Train GraphSAGE

```powershell
python -m src.train `
  --task gnn `
  --config config.yaml `
  --manifest data\processed\task2\task2_graph_manifest.jsonl `
  --run-name graphsage_genre `
  --epochs 10
```

### 4. Cache mel-spectrogram inputs for the CNN baseline

```powershell
python -m src.prepare_task2 `
  --manifest data\processed\task2\task2_graph_manifest.jsonl `
  --metadata data\processed\task2\fma_metadata.csv `
  --audio-root data\raw\fma\fma_small `
  --output data\processed\task2\mels `
  --sample-rate 22050 `
  --segment-seconds 5 `
  --n-mels 128
```

This phase opens each successful Task 2 audio file once and stores its
5-second log-mel segments under `data\processed\task2\mels`. CNN training and
evaluation then load those cached tensors instead of decoding audio again.
Training batches segments from multiple tracks together on the accelerator and
averages the segment logits per track before computing the loss.

### 5. Train the mel-spectrogram CNN baseline

```powershell
python -m src.train `
  --task genre_cnn `
  --config config.yaml `
  --manifest data\processed\task2\task2_graph_manifest.jsonl `
  --metadata data\processed\task2\fma_metadata.csv `
  --audio-root data\raw\fma\fma_small `
  --run-name cnn_melspectrogram_genre `
  --epochs 10
```

### 6. Evaluate GraphSAGE on the held-out test split

```powershell
python -m src.evaluate `
  --task gnn `
  --checkpoint results\task2\graphsage_genre_best.pt `
  --manifest data\processed\task2\task2_graph_manifest.jsonl `
  --output-dir results\task2
```

### 7. Evaluate the CNN baseline on the held-out test split

```powershell
python -m src.evaluate `
  --task genre_cnn `
  --checkpoint results\task2\cnn_melspectrogram_genre_best.pt `
  --manifest data\processed\task2\task2_graph_manifest.jsonl `
  --metadata data\processed\task2\fma_metadata.csv `
  --audio-root data\raw\fma\fma_small `
  --output-dir results\task2
```

### 8. Compare GraphSAGE and CNN results

```powershell
python -m src.compare_task2_models `
  --graphsage-metrics results\task2\task2_graphsage_test_metrics.json `
  --cnn-metrics results\task2\task2_cnn_test_metrics.json `
  --output results\task2\task2_model_comparison.json
```

## Task 3 The GNN-BERT Fusion

### Phase 1: Create the fixed Task 3 label vocabulary

Task 3 uses **FMA-small only**. Each example is an FMA track represented by:

- a graph of its 5-second audio segments;
- its FMA genre and tag metadata;
- one top-level genre target.

Task 3 is a **multilabel genre/tag prediction** task. The target is the union
of the normalized `track_genre_top` value and normalized values in the
`track_tags` array, so a track may have multiple positive labels. Training uses
`BCEWithLogitsLoss` with a fixed-length multihot target.

The complete decision record is in
[`task3_decisions.md`](C:/Users/user5/projects/gnn-bert-music-context.worktrees/gnn-bert-fusion-multimodal-improvements/task3_decisions.md).

The fixed vocabulary contains at most 100 labels: all normalized genres first,
then the most frequent normalized tags, excluding tags already present as
genres. Frequency ties must be resolved deterministically.

Run the metadata inventory and vocabulary-selection commands in order:

```powershell
python -m src.prepare_task3_labels scan `
  --tracks-csv data\raw\fma\fma_metadata\tracks.csv `
  --output data\processed\task3\task3_metadata_scan.json
```

The scan output contains every unique normalized genre and tag and their
frequencies. Select the fixed vocabulary:

```powershell
python -m src.prepare_task3_labels select `
  --scan data\processed\task3\task3_metadata_scan.json `
  --output data\processed\task3\labels.json `
  --max-labels 100
```

### Phase 2: Prepare multilabel metadata and BERT text

Create the derived multilabel CSV, then append the non-target BERT text:

```powershell
python -m src.prepare_task3_labels prepare `
  --tracks-csv data\raw\fma\fma_metadata\tracks.csv `
  --labels data\processed\task3\labels.json `
  --output data\processed\task3\fma_task3_labels.csv
```

The derived CSV contains the original metadata plus one binary
`label_<normalized_label>` column per selected label and a pipe-delimited
`task3_labels` convenience column. The source `tracks.csv` is never modified.

```powershell
python -m src.prepare_task3_labels prepare-text `
  --input data\processed\task3\fma_task3_labels.csv `
  --output data\processed\task3\fma_task3_text.csv
```

This appends `bert_text` from the fixed compact set of descriptive FMA
columns:

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

These fields were selected for descriptive musical context rather than
administrative metadata or artist identity memorization. Genre and tag sources
(`track_genre_top`, `track_genres`, `track_genres_all`, `track_tags`,
`album_tags`, and `artist_tags`) plus artist identity, IDs, paths, dates,
URLs, licenses, and split fields are excluded by construction.
Null/NaN values are omitted from each row's text. Tracks with no usable value
in any of the nine selected fields are excluded from the BERT/fusion copy, and
the command reports the retained row count.

### Phase 3: Create the verified graph/text/label manifest

Run this after the Task 2 graph manifest exists:

```powershell
python -m src.prepare_task3_manifest `
  --graph-manifest data\processed\task2\task2_graph_manifest.jsonl `
  --metadata data\processed\task3\fma_task3_text.csv `
  --labels data\processed\task3\labels.json `
  --output data\processed\task3\task3_fusion_manifest.jsonl
```

This joins graph, text, labels, split, and artist identity by `track_id`. It
keeps only the intersection of tracks present in both the graph manifest and
the prepared metadata CSV: graph tracks skipped during graph construction and
metadata tracks removed because they have no usable BERT text are excluded.
The split is taken from the Task 2 graph manifest, whose canonical values are
only `train`, `val`, and `test`. The prepared text metadata does not define a
second split contract.
The command reports how many graph tracks had no metadata match. It still
rejects duplicate tracks, non-binary targets, invalid graph files, and artist
leakage.

## Task 3 sequential training pipeline

Continue with these phases after Phase 3. Every phase uses the same prepared
metadata, fixed label vocabulary, and artist-disjoint split inherited from the
Task 2 graph manifest.

### Phase 4: Train the primary cross-attention model

This is the main Task 3 result. The trainer restores the checkpoint with the
best validation Macro-F1, then evaluates train, validation, and test splits.
`--epochs` is the maximum number of epochs; early stopping is already enabled
through `config.yaml` and stops sooner when validation Macro-F1 does not
improve.

```powershell
python -m src.train_task3 `
  --manifest data\processed\task3\task3_fusion_manifest.jsonl `
  --labels data\processed\task3\labels.json `
  --config config.yaml `
  --model fusion `
  --run-name task3_cross_attention `
  --output-dir results\task3 `
  --class-weighting positive `
  --epochs 20 `
  --seed 42
```

### Phase 5: Run the ablation models individually

Because each model may require substantial time and GPU memory, the recommended
workflow is to run and inspect each condition separately. A failed or
interrupted condition can then be rerun without repeating completed runs.
Use the same manifest, labels, seed, and training settings for every command:

```powershell
# 1. BERT-only
python -m src.train_task3 `
  --manifest data\processed\task3\task3_fusion_manifest.jsonl `
  --labels data\processed\task3\labels.json `
  --config config.yaml `
  --model bert `
  --run-name task3_bert_only `
  --output-dir results\task3\ablation `
  --epochs 20 `
  --seed 42

# 2. GNN-only
python -m src.train_task3 `
  --manifest data\processed\task3\task3_fusion_manifest.jsonl `
  --labels data\processed\task3\labels.json `
  --config config.yaml `
  --model gnn `
  --run-name task3_gnn_only `
  --output-dir results\task3\ablation `
  --epochs 20 `
  --seed 42

# 3. Early-concatenation fusion
python -m src.train_task3 `
  --manifest data\processed\task3\task3_fusion_manifest.jsonl `
  --labels data\processed\task3\labels.json `
  --config config.yaml `
  --model fusion `
  --early-concat `
  --run-name task3_early_concat `
  --output-dir results\task3\ablation `
  --epochs 20 `
  --seed 42

# 4. Cross-attention fusion
python -m src.train_task3 `
  --manifest data\processed\task3\task3_fusion_manifest.jsonl `
  --labels data\processed\task3\labels.json `
  --config config.yaml `
  --model fusion `
  --run-name task3_cross_attention `
  --output-dir results\task3\ablation `
  --epochs 20 `
  --seed 42
```

Live terminal output includes batch loss, epoch
train/validation loss, validation Macro-F1, and the current best-checkpoint
status. After each run, the filesystem contains the full metrics history,
checkpoint, predictions, case studies, learning curves, PR curves, and (for
fusion runs) the fused-embedding t-SNE plot. These individual commands store
the files under `results\task3\ablation`.

### Phase 6: Review generated metrics and plots

After all four individual ablation runs are complete, run the comparison-only
analytics command. It reads existing metrics files; it does not start
training, load the dataset, or modify any model checkpoint.

```powershell
python -m src.compare_task3_runs `
  --input-dir results\task3\ablation `
  --output-dir results\task3\ablation
```
c
By default, the command looks for the four standard files:
`task3_bert_only_metrics.json`, `task3_gnn_only_metrics.json`,
`task3_early_concat_metrics.json`, and
`task3_cross_attention_metrics.json`. These names match the individual
training commands in Phase 5. Use `--runs` only when custom run names were
used.

Each completed model run writes a checkpoint, metrics history, predictions,
and qualitative case studies. Plots are kept in the `plots` subdirectory so
they are easy to distinguish from JSON and checkpoint files.

```text
results\task3\<run-name>_best.pt
results\task3\<run-name>_metrics.json
results\task3\<run-name>_predictions.json
results\task3\<run-name>_case_studies.json
results\task3\plots\<run-name>_learning_curves.png
results\task3\plots\<run-name>_pr_curve.png
results\task3\plots\<run-name>_fused_tsne.png
```

The learning curves show train/validation BCE loss and Macro-F1. The
precision-recall plot shows one curve per label with positive test examples.
The t-SNE plot uses fused test embeddings and dominant target labels; the
confirmed Task 3 target has genre/tag labels, not DEAM mood labels.

The ablation directory contains the same files for each condition plus:

```text
results\task3\ablation\task3_ablation_comparison.json
results\task3\ablation\task3_ablation_analysis.txt
results\task3\ablation\plots\task3_ablation_comparison.png
```

The comparison JSON reports held-out test Macro-F1, Micro-F1, mean AP/PR-AUC,
and loss for all four conditions, ranks the models for every metric, and
computes deltas against the BERT-only baseline. The text file summarizes the
best model for each metric and the cross-attention change relative to BERT-only.
The plot compares Macro-F1, Micro-F1, and PR-AUC side by side. If a metrics
file is missing, the command stops with the exact run that must be completed.

The dedicated trainer uses the fixed vocabulary order from `labels.json`,
creates one logit per selected label, and trains with
`BCEWithLogitsLoss`. Long-running phases display progress bars for manifest
validation, class-weight computation, batches, epochs, and final evaluation.

The Task 3 architecture is:

```text
FMA audio -> 5-second feature graph -> residual GraphSAGE
          -> mean/max/std pooling -> graph embedding
FMA track title/album text -> BERT -> text embedding
graph embedding + text embedding -> cross-attention or early-concat classifier
FMA genre/tags -> fixed 100-label multilabel target
```

The BERT input must use only non-target metadata fields such as track title and
album title. It must not serialize the genre or tag columns into the input,
because those columns define the prediction target and would leak the labels.
If the chosen FMA export has no usable title/album fields, the honest fallback
is a GNN-only model; a fabricated caption or random text pairing is invalid.

Build the segment graphs:

```powershell
python -m src.graph_builder --prepare-metadata --tracks-csv data\raw\fma\fma_metadata\tracks.csv --audio-root data\raw\fma\fma_small --metadata-output data\processed\task2\fma_metadata.csv
python -m src.graph_builder --metadata data\processed\task2\fma_metadata.csv --audio-root data\raw\fma\fma_small --output data\processed\task2\graphs --manifest data\processed\task2\task2_graph_manifest.jsonl --split-dir data\splits\task2 --sample-rate 22050 --segment-seconds 5 --threshold 0.75
python -m src.prepare_task2 --manifest data\processed\task2\task2_graph_manifest.jsonl --metadata data\processed\task2\fma_metadata.csv --audio-root data\raw\fma\fma_small --output data\processed\task2\mels --sample-rate 22050 --segment-seconds 5 --n-mels 128
```

Train the real-data models:

```powershell
python -m src.train --task gnn --config config.yaml --manifest data\processed\task2\task2_graph_manifest.jsonl --run-name graphsage_genre
python -m src.train --task genre_cnn --config config.yaml --manifest data\processed\task2\task2_graph_manifest.jsonl --metadata data\processed\task2\fma_metadata.csv --audio-root data\raw\fma\fma_small --run-name cnn_melspectrogram_genre --epochs 10
```

The mature GraphSAGE path uses a learned input projection, three residual
GraphSAGE blocks, and mean/max/standard-deviation graph pooling before its
classifier head. Training uses configurable balanced class weights, AdamW,
validation Macro-F1 learning-rate reduction, gradient clipping, best-checkpoint
restoration, and early stopping. The defaults are configured in `config.yaml`;
pass `--epochs` to cap a run explicitly.

Evaluate the best Task 2 checkpoint:

```powershell
python -m src.evaluate --task gnn --checkpoint results\task2\graphsage_genre_best.pt --manifest data\processed\task2\task2_graph_manifest.jsonl --output-dir results\task2
```

Task 2 evaluation writes `task2_graphsage_test_metrics.json`,
`task2_graphsage_predictions.json`, and
`plots\task2_graphsage_confusion_matrix.png` under `results\task2`.

Evaluate the CNN baseline:

```powershell
python -m src.evaluate `
  --task genre_cnn `
  --checkpoint results\task2\cnn_melspectrogram_genre_best.pt `
  --manifest data\processed\task2\task2_graph_manifest.jsonl `
  --metadata data\processed\task2\fma_metadata.csv `
  --audio-root data\raw\fma\fma_small `
  --output-dir results\task2
```

This writes `task2_cnn_test_metrics.json`, `task2_cnn_predictions.json`, and
`plots\task2_cnn_confusion_matrix.png` under `results\task2`.

Both Task 2 training commands also save learning curves under
`results\task2\plots`:

```text
task2_graphsage_learning_curves.png
task2_cnn_learning_curves.png
```

Compare the trained Task 2 models:

```powershell
python -m src.compare_task2_models `
  --graphsage-metrics results\task2\task2_graphsage_test_metrics.json `
  --cnn-metrics results\task2\task2_cnn_test_metrics.json `
  --output results\task2\task2_model_comparison.json
```

The graph nodes capture 182 normalized descriptors per 5-second segment:
MFCC distribution and temporal coefficients, chroma, spectral contrast, tonnetz,
spectral shape, energy, and zero-crossing statistics. The comparison file
contains test loss, Accuracy, Macro-F1, Micro-F1, and
one-vs-rest Macro PR-AUC for the GraphSAGE model and the mel-spectrogram CNN
baseline. The CNN averages logits from every 5-second segment in each track,
matching the full-track coverage of the graph model.
Graph feature scaling is fitted only on training-track segments and reused for
validation and test tracks, so similarity edges remain comparable without
leaking evaluation statistics.

## Project directories

- `data\raw\`: FMA, MagnaTagATune, MusicCaps, audio, and metadata downloads.
- `data\processed\`: serialized graphs, mel-spectrograms, and BERT caches.
- `data\splits\`: train/validation/test split files.
- `notebooks\`: exploratory analysis and the end-to-end demo notebook.
- `src\`: preprocessing, graph, model, training, evaluation, and metric code.
- `results\`: metrics, plots, checkpoints, and retrieval examples.
- `report\`: final report PDF and related report assets.
- `config.yaml`: default model and training settings.

Do not use illustrative scores from the assignment PDF. Report only metrics
generated from the real dataset and record whether Task 1 uses proxy labels.
