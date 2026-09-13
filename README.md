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
python scripts\train_bert_musiccaps.py --input data\raw\musiccaps\musiccaps_public.csv --model-name distilbert-base-uncased --output-dir results --run-name task1_distilbert --epochs 10 --batch-size 8 --learning-rate 2e-5 --max-length 128 --hidden-size 256 --seed 42
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

These tasks require a real `data\raw\metadata.csv` and the corresponding audio
files. The metadata must contain `track_id`, `path`, `text`, `labels`, `split`,
and `artist_id`. `path` is relative to `data\raw\audio`; `labels` are separated
by `|`; and each artist must occur in only one split.

Place the files as follows:

```text
data\
  raw\
    audio\
      <real audio files>
    metadata.csv
```

Build the segment graphs:

```powershell
python -m src.graph_builder --metadata data\raw\metadata.csv --audio-root data\raw\audio --output data\processed\graphs --sample-rate 22050 --segment-seconds 5 --threshold 0.75
```

Train the real-data models:

```powershell
python -m src.train --task bert --config config.yaml --manifest data\processed\manifest.jsonl --run-name task1_bert --epochs 10
python -m src.train --task gnn --config config.yaml --manifest data\processed\manifest.jsonl --run-name task2_gnn --epochs 10
python -m src.train --task fusion --config config.yaml --manifest data\processed\manifest.jsonl --run-name task3_fusion --epochs 10
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
python scripts\run_ablations.py --epochs 10
```

The ablation script runs BERT-only, GNN-only, early-concatenation fusion, and
cross-attention fusion. Compare their test Macro-F1, Micro-F1, and mean
PR-AUC in the report.

## Project directories

- `data\raw\`: real input audio and metadata; raw files are ignored by Git.
- `data\processed\graphs\`: serialized audio graphs.
- `data\processed\manifest.jsonl`: graph paths, text, labels, and split metadata.
- `results\`: checkpoints, metrics, plots, and prediction examples.
- `src\`: model, graph, training, evaluation, and metric implementations.
- `scripts\`: dataset export/training and ablation utilities.
- `config.yaml`: default model and training settings.

Do not use illustrative scores from the assignment PDF. Report only metrics
generated from the real dataset and record whether Task 1 uses proxy labels.
