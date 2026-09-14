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

The requirements PDF is authoritative. Existing Task 2 code may be refactored
or improved when it is inconsistent with the requirements or prevents a
coherent stepwise pipeline. Changes should remain scoped to Task 2 and should
preserve Task 1 and Task 3 behavior unless a directly coupled change is
necessary.
