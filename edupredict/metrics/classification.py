"""
Classification evaluation metrics implemented from first principles.

Strictly zero third-party dependencies and zero external library imports (no Scikit-learn, no math).
Supports accuracy, confusion matrix, precision, recall, F1-score, ROC curves,
and Area Under the ROC Curve (ROC-AUC) using the trapezoidal integration rule.
"""

from typing import Sequence, List, Any, Dict, Optional, Tuple, Union

Number = Union[int, float]


def _check_metrics_inputs(y_true: Sequence[Any], y_pred: Sequence[Any]) -> None:
    """Validate input sequences for non-emptiness and equal dimensions."""
    if not hasattr(y_true, "__len__") or not hasattr(y_pred, "__len__"):
        raise TypeError("y_true and y_pred must be sequence collections")
    if isinstance(y_true, (str, dict, set)) or isinstance(y_pred, (str, dict, set)):
        raise TypeError("y_true and y_pred must be indexed sequences (not str, dict, or set)")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Inputs cannot be empty")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Lengths do not match: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}")


def accuracy_score(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Accuracy score: (number of correct predictions) / (total predictions)."""
    _check_metrics_inputs(y_true, y_pred)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / float(len(y_true))


def confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Sequence[Any]] = None
) -> List[List[int]]:
    """
    Compute confusion matrix to evaluate classification accuracy.
    By definition entry C_{i, j} is the number of observations known to be in group i
    and predicted to be in group j.
    """
    _check_metrics_inputs(y_true, y_pred)
    if labels is None:
        unique_labels = sorted(list(set(y_true) | set(y_pred)), key=lambda x: str(x))
    else:
        unique_labels = list(labels)

    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    k = len(unique_labels)
    matrix = [[0 for _ in range(k)] for _ in range(k)]

    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            matrix[label_to_idx[yt]][label_to_idx[yp]] += 1

    return matrix


def precision_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    pos_label: Any = 1,
    zero_division: float = 0.0
) -> float:
    """Precision: TP / (TP + FP)."""
    _check_metrics_inputs(y_true, y_pred)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == pos_label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != pos_label and yp == pos_label)
    if tp + fp == 0:
        return float(zero_division)
    return tp / float(tp + fp)


def recall_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    pos_label: Any = 1,
    zero_division: float = 0.0
) -> float:
    """Recall (Sensitivity): TP / (TP + FN)."""
    _check_metrics_inputs(y_true, y_pred)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == pos_label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp != pos_label)
    if tp + fn == 0:
        return float(zero_division)
    return tp / float(tp + fn)


def f1_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    pos_label: Any = 1,
    zero_division: float = 0.0
) -> float:
    """F1-Score: Harmonic mean of precision and recall: 2 * (P * R) / (P + R)."""
    prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    if prec + rec == 0.0:
        return float(zero_division)
    return 2.0 * (prec * rec) / (prec + rec)


def classification_report(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Sequence[Any]] = None
) -> str:
    """Generate a readable text report showing precision, recall, and F1-score."""
    _check_metrics_inputs(y_true, y_pred)
    if labels is None:
        labels_list = sorted(list(set(y_true) | set(y_pred)), key=lambda x: str(x))
    else:
        labels_list = list(labels)

    lines = [
        f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}",
        "-" * 58
    ]
    for lbl in labels_list:
        support = sum(1 for y in y_true if y == lbl)
        prec = precision_score(y_true, y_pred, pos_label=lbl)
        rec = recall_score(y_true, y_pred, pos_label=lbl)
        f1 = f1_score(y_true, y_pred, pos_label=lbl)
        lines.append(f"{str(lbl):<12} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {support:<10}")

    acc = accuracy_score(y_true, y_pred)
    lines.append("-" * 58)
    lines.append(f"{'Accuracy':<12} {'':<12} {'':<12} {acc:<12.4f} {len(y_true):<10}")
    return "\n".join(lines)


def roc_curve(
    y_true: Sequence[Any],
    y_score: Sequence[float],
    pos_label: Any = 1
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute Receiver Operating Characteristic (ROC) coordinates: (FPR, TPR, thresholds).
    Evaluates empirical thresholds formed by unique predicted probability scores.
    """
    _check_metrics_inputs(y_true, y_score)
    y_binary = [1 if yt == pos_label else 0 for yt in y_true]
    n_pos = sum(y_binary)
    n_neg = len(y_binary) - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC curve requires both positive and negative samples in y_true")

    # Collect unique thresholds in descending order
    unique_thresholds = sorted(list(set(float(s) for s in y_score)), reverse=True)
    # Append extreme threshold above 1.0 for (0, 0) coordinate
    thresholds = [unique_thresholds[0] + 1.0] + unique_thresholds

    fpr_list: List[float] = []
    tpr_list: List[float] = []

    for thresh in thresholds:
        tp = 0
        fp = 0
        for yt, score in zip(y_binary, y_score):
            if score >= thresh:
                if yt == 1:
                    tp += 1
                else:
                    fp += 1
        tpr = tp / float(n_pos)
        fpr = fp / float(n_neg)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Ensure endpoints reach (1.0, 1.0)
    if fpr_list[-1] < 1.0 or tpr_list[-1] < 1.0:
        fpr_list.append(1.0)
        tpr_list.append(1.0)
        thresholds.append(0.0)

    return fpr_list, tpr_list, thresholds


def roc_auc_score(
    y_true: Sequence[Any],
    y_score: Sequence[float],
    pos_label: Any = 1
) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
    using the trapezoidal numerical integration rule:
    AUC = sum_{i=1}^n ((TPR_i + TPR_{i-1}) / 2) * (FPR_i - FPR_{i-1}).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    auc = 0.0
    for i in range(1, len(fpr)):
        dx = fpr[i] - fpr[i - 1]
        avg_y = (tpr[i] + tpr[i - 1]) / 2.0
        auc += avg_y * dx
    return auc
