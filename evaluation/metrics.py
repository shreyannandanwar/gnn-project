"""
evaluation/metrics.py
ROC-AUC computation with multi-task and missing-label support.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_roc_auc(y_true, y_pred):
    """
    Compute ROC-AUC, handling:
      - Single-task (1-D) tensors / arrays
      - Multi-task (2-D) tensors / arrays (e.g. Tox21 has 12 sub-tasks)
      - Missing labels encoded as -1  →  those samples are masked out
      - Tasks where all labels are identical (undefined AUC)  →  skipped

    Returns
    -------
    float  –  mean ROC-AUC across valid tasks, or nan if none are valid.
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # ── Single-task ──────────────────────────────────────────────────────────
    if y_true.ndim == 1:
        mask = y_true != -1
        if mask.sum() < 2:
            return float("nan")
        if len(np.unique(y_true[mask])) < 2:
            return float("nan")
        return float(roc_auc_score(y_true[mask], y_pred[mask]))

    # ── Multi-task ───────────────────────────────────────────────────────────
    aucs = []
    for t in range(y_true.shape[1]):
        col_true = y_true[:, t]
        col_pred = y_pred[:, t]
        mask = col_true != -1
        if mask.sum() < 2:
            continue
        if len(np.unique(col_true[mask])) < 2:
            continue
        aucs.append(roc_auc_score(col_true[mask], col_pred[mask]))

    return float(np.mean(aucs)) if aucs else float("nan")
