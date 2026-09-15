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

## Tasks 2 and 3: real audio graphs and fusion

Task 2 keeps its derived metadata, graphs, manifest, splits, and results in
task-specific locations. The original FMA files remain under `data\raw\fma`.

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

## Task 2 complete pipeline

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

For the Task 3 early-concatenation comparison:

```powershell
python -m src.train --task fusion --config config.yaml --manifest data\processed\manifest.jsonl --run-name task3_fusion_early_concat --epochs 10 --early-concat
```

All checkpoints, metric histories, and learning curves are written to
`results\`. The text model configured by default is
`distilbert-base-uncased`; change `model.text_model` in `config.yaml` only when
you intentionally want to use another Hugging Face checkpoint.

## Ablation comparison

After building the real graph manifest, run all four real-data conditions:

```powershell
python -m src.run_ablations --epochs 10
```

The ablation script runs BERT-only, GNN-only, early-concatenation fusion, and
cross-attention fusion. Compare their test Macro-F1, Micro-F1, and mean
PR-AUC in the report.

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
