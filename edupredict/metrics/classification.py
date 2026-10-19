"""
Classification evaluation metrics implemented from scratch.

Provides:
- accuracy_score
- confusion_matrix
- precision_score
- recall_score
- f1_score
- classification_report
"""

from typing import Sequence, List, Any, Dict, Optional, Tuple


def _check_metrics_inputs(y_true: Sequence[Any], y_pred: Sequence[Any]) -> None:
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Inputs cannot be empty")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Lengths do not match: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}")


def accuracy_score(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """
    Accuracy classification score: (number of correct predictions) / (total predictions).
    """
    _check_metrics_inputs(y_true, y_pred)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / float(len(y_true))


def confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Sequence[Any]] = None
) -> List[List[int]]:
    """
    Compute confusion matrix where rows are actual classes and columns are predicted classes:
    C[i][j] is the number of observations known to be in group i and predicted to be in group j.
    """
    _check_metrics_inputs(y_true, y_pred)
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)), key=lambda x: str(x))

    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    k = len(labels)
    matrix = [[0 for _ in range(k)] for _ in range(k)]

    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            r = label_to_idx[yt]
            c = label_to_idx[yp]
            matrix[r][c] += 1

    return matrix


def precision_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    pos_label: Any = 1
) -> float:
    """
    Precision = TP / (TP + FP).
    Returns 0.0 if denominator is 0.
    """
    _check_metrics_inputs(y_true, y_pred)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == pos_label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != pos_label and yp == pos_label)
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def recall_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    pos_label: Any = 1
) -> float:
    """
    Recall (Sensitivity) = TP / (TP + FN).
    Returns 0.0 if denominator is 0.
    """
    _check_metrics_inputs(y_true, y_pred)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == pos_label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp != pos_label)
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def f1_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    pos_label: Any = 1
) -> float:
    """
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall).
    """
    p = precision_score(y_true, y_pred, pos_label=pos_label)
    r = recall_score(y_true, y_pred, pos_label=pos_label)
    denom = p + r
    return 2.0 * (p * r) / denom if denom > 0 else 0.0


def classification_report(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Sequence[Any]] = None
) -> str:
    """
    Build a text report showing main classification metrics per class.
    """
    _check_metrics_inputs(y_true, y_pred)
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)), key=lambda x: str(x))

    lines = [f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}"]
    lines.append("-" * 58)

    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    total_support = len(y_true)

    for lbl in labels:
        p = precision_score(y_true, y_pred, pos_label=lbl)
        r = recall_score(y_true, y_pred, pos_label=lbl)
        f1 = f1_score(y_true, y_pred, pos_label=lbl)
        supp = sum(1 for yt in y_true if yt == lbl)

        macro_p += p
        macro_r += r
        macro_f1 += f1

        lines.append(f"{str(lbl):<12} {p:<12.4f} {r:<12.4f} {f1:<12.4f} {supp:<10}")

    lines.append("-" * 58)
    k = len(labels)
    acc = accuracy_score(y_true, y_pred)
    lines.append(f"{'Accuracy':<12} {'':<12} {'':<12} {acc:<12.4f} {total_support:<10}")
    lines.append(f"{'Macro Avg':<12} {macro_p/k:<12.4f} {macro_r/k:<12.4f} {macro_f1/k:<12.4f} {total_support:<10}")

    return "\n".join(lines)
