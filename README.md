# GNN-BERT Music Context

Implementation of CSE425 Tasks 1--3: text-only multi-label tagging, audio-structure graphs, and cross-attention GNN-BERT fusion. Task 4 contrastive retrieval is deliberately excluded.

## Run it step by step

All commands below must be run in a terminal. First enter the project folder and create an isolated Python environment (only needed once):

```bash
cd "/home/noiem/Documents/ChatGPT/gnn-bert-music-context"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Every time you open a new terminal later, return to the folder and activate the environment again:

```bash
cd "/home/noiem/Documents/ChatGPT/gnn-bert-music-context"
source .venv/bin/activate
```

### 1. Run the safe synthetic demo

This creates fake, small input data. It confirms that the code runs but its scores are **not** suitable for your report.

```bash
python scripts/make_synthetic_data.py --count 24
```

### 2. Train one model at a time

```bash
# Task 1: text/tag classifier
python -m src.train --task bert --synthetic --epochs 5

# Task 2: music-structure graph classifier
python -m src.train --task gnn --synthetic --epochs 5

# Task 3: GNN + text cross-attention fusion model
python -m src.train --task fusion --synthetic --epochs 5
```

`--epochs 5` is just a quick test. Increase it for real experiments once the pipeline works.

### 3. Evaluate a trained model

```bash
python -m src.evaluate --checkpoint results/fusion_best.pt --task fusion
```

Look in `results/` afterward:

- `*_best.pt`: saved best model checkpoint.
- `*_metrics.json`: metrics for each training epoch.
- `*_f1_curve.png`: train/validation F1 plot.
- `fusion_test_metrics.json`: Macro-F1, Micro-F1, and AUC-PR on the test split.
- `fusion_tsne.png`: 2D embedding visualization.
- `fusion_case_studies.json`: three prediction examples.

### 4. Run the required ablation comparison

```bash
python scripts/run_ablations.py --synthetic --epochs 5
```

This runs and saves four comparisons: BERT-only, GNN-only, simple early-concatenation fusion, and cross-attention fusion. For a real report, compare their test Macro-F1, Micro-F1, and AUC-PR in one table.

### 5. Run on a real dataset

Place audio in `data/raw/audio/`. Create `data/raw/metadata.csv` with these columns:

```csv
track_id,path,text,labels,split,artist_id
001,rock/song001.mp3,"energetic distorted guitar rock","rock|energetic",train,artist_001
002,jazz/song002.mp3,"calm piano jazz","jazz|calm",val,artist_002
```

`path` is relative to `data/raw/audio/`; `labels` are separated by `|`; and an artist must appear in only one split to prevent leakage. Then build graphs, train, and evaluate:

```bash
python -m src.graph_builder --metadata data/raw/metadata.csv --audio-root data/raw/audio
python -m src.train --task fusion --config config.yaml --epochs 10
python -m src.evaluate --checkpoint results/fusion_best.pt --task fusion
```

For a full real-data ablation, omit `--synthetic`:

```bash
python scripts/run_ablations.py --epochs 10
```

For actual BERT weights, use an internet connection on the first real-data run. The default model is `distilbert-base-uncased`, chosen to be more practical on a weak computer. Change `model.text_model` in `config.yaml` to `bert-base-uncased` only if you want to test the larger model.

### MusicCaps caption-to-tag proxy baseline

MusicCaps supplies captions but not a conventional multi-label tag column. The
standalone Task 1 runner therefore accepts a local MusicCaps CSV, JSON, JSONL,
or parquet export and creates reproducible proxy tags by matching documented
music phrases in each caption. If the input already has a `tags`, `labels`, or
`label` column, those labels are used instead. Rows are split by a stable hash
of `ytid` (or `track_id`/`id`) so the split is reproducible:

```bash
python scripts/train_bert_musiccaps.py --input data/raw/musiccaps.csv \
  --model-name distilbert-base-uncased --epochs 10
```

The command writes `bert_musiccaps_best.pt`, `bert_musiccaps_metrics.json`,
`bert_musiccaps_test_metrics.json`, and `bert_musiccaps_f1_curve.png` to
`results/`. The JSON history contains Macro-F1 and Micro-F1 for every epoch;
the PNG plots both validation curves. These are proxy-task results and should
be reported as such, not as human-annotated MusicCaps tag accuracy.

## Implemented deliverables

- Task 1: HuggingFace BERT/DistilBERT tag classifier with offline lightweight fallback, BCE loss, macro/micro F1 and PR-AUC curves.
- Task 2: segment graph construction and GraphSAGE message passing, plus a mel-spectrogram CNN baseline.
- Task 3: paired graph/text fusion with both early concatenation and cross-attention, optional valence/arousal auxiliary regression, ablation runner, t-SNE, and case-study export.
- `scripts/make_synthetic_data.py` generates 24 serialised graph samples (meeting the 20-sample submission requirement) and reproducible train/validation/test splits.

Results are written below `results/`; plots and case studies are generated by `src.evaluate`. Do not use the illustrative scores from the assignment PDF: all metrics produced here come from your actual data.
