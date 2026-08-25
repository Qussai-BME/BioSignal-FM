"""Classification metrics for biosignal evaluation.

All metrics are computed with NumPy (no sklearn dependency at this layer)
to ensure deterministic behavior and easy audit.

Critical: :func:`confusion_matrix` accepts a list of per-fold predictions
and labels (NOT a single fold) to enforce aggregation across folds —
this is the fix for the MyoControl v2.0 audit defect where the UI showed
in-sample confusion matrices.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["accuracy", "f1_score", "confusion_matrix", "classification_report"]


def accuracy(y_true: np.ndarray | Sequence[int], y_pred: np.ndarray | Sequence[int]) -> float:
    """Compute classification accuracy.

    Examples
    --------
    >>> accuracy([0, 1, 2, 0], [0, 1, 1, 0])
    0.75
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def f1_score(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
    average: str = "macro",
) -> float:
    """Compute F1 score.

    Parameters
    ----------
    average : str
        "macro" (default), "micro", or "weighted".
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        return 0.0

    labels = np.unique(np.concatenate([y_true, y_pred]))
    if average == "micro":
        tp = int(np.sum(y_true == y_pred))
        fp_fn = int(np.sum(y_true != y_pred))
        if tp == 0:
            return 0.0
        return float(2 * tp / (2 * tp + fp_fn))

    f1s = []
    weights = []
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1s.append(f1)
        weights.append(int(np.sum(y_true == label)))

    if average == "macro":
        return float(np.mean(f1s))
    elif average == "weighted":
        total = sum(weights)
        return (
            float(sum(f * w for f, w in zip(f1s, weights, strict=True)) / total) if total else 0.0
        )
    else:
        raise ValueError(f"Unknown average: {average}")


def confusion_matrix(
    y_true_per_fold: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
    y_pred_per_fold: np.ndarray | Sequence[Sequence[int]] | Sequence[int],
    n_classes: int | None = None,
) -> np.ndarray:
    """Aggregate confusion matrix across folds.

    CRITICAL: This function accepts either:

    - A flat array (single fold) — backward compatible
    - A list of arrays (multiple folds) — aggregates across all folds

    The aggregation-across-folds behavior is the fix for the MyoControl
    v2.0 audit defect where the UI displayed in-sample confusion matrices.

    Parameters
    ----------
    y_true_per_fold : array-like
        Either 1D (single fold) or list of 1D arrays (multiple folds).
    y_pred_per_fold : array-like
        Same structure as ``y_true_per_fold``.
    n_classes : int, optional
        Number of classes. If None, inferred from data.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape ``(n_classes, n_classes)`` where row i,
        column j = number of samples with true label i and predicted label j.

    Examples
    --------
    >>> # Single fold
    >>> cm = confusion_matrix([0, 1, 2, 0], [0, 1, 1, 0], n_classes=3)
    >>> cm.shape
    (3, 3)
    >>> # Multiple folds (aggregated)
    >>> cm = confusion_matrix([[0, 1], [2, 0]], [[0, 1], [2, 1]], n_classes=3)
    >>> int(cm.sum())  # 4 total samples
    4
    """
    # Detect single fold vs multiple folds
    if len(y_true_per_fold) == 0:
        return np.zeros((n_classes or 1, n_classes or 1), dtype=np.int64)

    first = y_true_per_fold[0]
    is_multi_fold = isinstance(first, (list, tuple, np.ndarray)) and np.asarray(first).ndim == 1

    if is_multi_fold:
        # Multiple folds — aggregate
        y_true_all = np.concatenate([np.asarray(y) for y in y_true_per_fold])
        y_pred_all = np.concatenate([np.asarray(y) for y in y_pred_per_fold])
    else:
        # Single fold
        y_true_all = np.asarray(y_true_per_fold)
        y_pred_all = np.asarray(y_pred_per_fold)

    if n_classes is None:
        n_classes = int(max(y_true_all.max(), y_pred_all.max())) + 1 if len(y_true_all) > 0 else 1

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true_all, y_pred_all, strict=True):
        cm[int(t), int(p)] += 1
    return cm


def classification_report(
    y_true: np.ndarray | Sequence[int],
    y_pred: np.ndarray | Sequence[int],
    label_names: Sequence[str] | None = None,
) -> dict:
    """Generate a per-class classification report.

    Returns
    -------
    dict
        {"per_class": {label: {precision, recall, f1, support}},
         "macro_f1", "micro_f1", "accuracy"}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if label_names is None:
        label_names = [str(int(lab)) for lab in labels]

    per_class: dict[str, dict[str, float]] = {}
    for label, name in zip(labels, label_names, strict=True):
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        support = int(np.sum(y_true == label))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }

    return {
        "per_class": per_class,
        "macro_f1": float(np.mean([c["f1"] for c in per_class.values()])),
        "micro_f1": f1_score(y_true, y_pred, average="micro"),
        "accuracy": accuracy(y_true, y_pred),
    }
