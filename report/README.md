# IEEE Conference Report

`main.tex` and `references.bib` are an Overleaf-ready IEEE conference
manuscript. Upload the contents of this folder to an Overleaf project using
the IEEE Conference template, then compile `main.tex`.

This is a full project report, not only a final-score summary. It documents
the synthetic smoke-test stage, dataset selection, preprocessing decisions,
graph construction, model iterations, hyperparameters, training histories,
Task 2 baseline comparison, Task 3 four-condition ablation, post-ablation
interpretation, artifact map, limitations, and future-work roadmap. The
appendix embeds the complete plot archive. Task 1 is explicitly identified
as a MusicCaps caption-to-tag proxy experiment; the FMA-small Task 2 and
Task 3 results are reported separately.

The manuscript reports only real-run artifacts under `results/`; synthetic
data is discussed as engineering history and is not presented as an
experimental result. Exported JSON predictions and case studies remain in
the repository for detailed qualitative inspection.

Before submission:

1. Replace the placeholder author email in `main.tex`.
2. Confirm the course/institution formatting and submission date.
3. Upload the generated figures under `figures/`.
4. Add exact dataset counts and selected case-study details if the course
   requires them in the final version.
