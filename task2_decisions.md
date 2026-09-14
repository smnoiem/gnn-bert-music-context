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
