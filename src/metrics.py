from __future__ import annotations

import numpy as np


def multilabel_metrics(logits, targets) -> dict:
    probs, y = 1 / (1 + np.exp(-np.asarray(logits))), np.asarray(targets)
    pred = (probs >= .5).astype(int)
    def f1(a, b):
        tp = np.sum((a == 1) & (b == 1)); fp = np.sum((a == 0) & (b == 1)); fn = np.sum((a == 1) & (b == 0))
        return 2 * tp / max(2 * tp + fp + fn, 1)
    per_label = [f1(y[:, i], pred[:, i]) for i in range(y.shape[1])]
    result = {"macro_f1": float(np.mean(per_label)), "micro_f1": float(f1(y.ravel(), pred.ravel()))}
    def average_precision(a, score):
        order = np.argsort(-score); ranked = a[order]; hits = np.cumsum(ranked); precision = hits / np.arange(1, len(a) + 1)
        return float(np.sum(precision * ranked) / max(np.sum(ranked), 1))
    valid = [average_precision(y[:, i], probs[:, i]) for i in range(y.shape[1]) if len(np.unique(y[:, i])) > 1]
    result["auc_pr"] = float(np.mean(valid)) if valid else float("nan")
    return result
