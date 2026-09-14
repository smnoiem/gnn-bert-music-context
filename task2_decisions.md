# Task 2 Implementation Decisions

This file records confirmed Task 2 decisions so they can be reused in later
sessions. Only confirmed decisions are recorded here.

## Confirmed decisions

1. **Dataset:** FMA-small.
2. **Prediction target:** Single-label genre classification.
3. **GNN layer:** GraphSAGE.
4. **Node features:** MFCC + chroma.
5. **Similarity edges:** Add cosine-similarity edges for MFCC+chroma vectors when similarity is at least 0.75, in addition to temporal-adjacency edges.
6. **Segment duration:** 5 seconds.
7. **Audio sample rate:** 22050 Hz.
8. **Baseline:** CNN on mel-spectrograms.

## Implementation scope

The requirements PDF is authoritative. Existing Task 2 code and the repository
project or directory structure may be refactored or reorganized when they are
inconsistent with the requirements or prevent a coherent stepwise pipeline.
The existing `src/` and `scripts/` split is not mandatory. Changes should
remain scoped to Task 2 and should preserve Task 1 and Task 3 behavior unless a
directly coupled change is necessary.

The Task 2 implementation follows the PDF project tree: reusable preprocessing,
graph, model, training, and evaluation code belongs under `src/`; raw inputs
belong under `data/raw/`; generated features and graphs belong under
`data/processed/`; split files belong under `data/splits/`; and outputs belong
under `results/`. The PDF-listed `scripts/` directory is not used for the new
Task 2 pipeline.
