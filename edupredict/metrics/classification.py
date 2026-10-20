"""
Classification evaluation metrics implemented from scratch.

Provides:
- accuracy_score
- confusion_matrix
- precision_score
- recall_score
- f1_score
- classification_report
- roc_curve
- roc_auc_score
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
    pos_label: Any = 1,
    zero_division: float = 0.0
) -> float:
    """
    Precision = TP / (TP + FP).
    Returns `zero_division` (default 0.0) if denominator is 0.
    """
    _check_metrics_inputs(y_true, y_pred)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == pos_label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != pos_label and yp == pos_label)
    denom = tp + fp
    return (tp / float(denom)) if denom > 0 else zero_division


def recall_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    pos_label: Any = 1,
    zero_division: float = 0.0
) -> float:
    """
    Recall (Sensitivity) = TP / (TP + FN).
    Returns `zero_division` (default 0.0) if denominator is 0.
    """
    _check_metrics_inputs(y_true, y_pred)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp == pos_label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp != pos_label)
    denom = tp + fn
    return (tp / float(denom)) if denom > 0 else zero_division


def f1_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    pos_label: Any = 1,
    zero_division: float = 0.0
) -> float:
    """
    F1 Score = 2 * (Precision * Recall) / (Precision + Recall).
    Returns `zero_division` (default 0.0) if Precision + Recall is 0.
    """
    p = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    r = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    denom = p + r
    return (2.0 * p * r / float(denom)) if denom > 0 else zero_division


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


def roc_curve(
    y_true: Sequence[Any],
    y_score: Sequence[Number],
    pos_label: Any = 1
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute Receiver Operating Characteristic (ROC) curve from first principles.
    
    Args:
        y_true: True binary labels in {0, 1}.
        y_score: Target scores, can either be probability estimates of positive class or confidence scores.
        pos_label: Label of the positive class (default 1).
        
    Returns:
        (fpr, tpr, thresholds) where fpr is False Positive Rate, tpr is True Positive Rate.
    """
    _check_metrics_inputs(y_true, y_score)
    n = len(y_true)

    # Count total positive and negative instances
    total_pos = sum(1 for yt in y_true if yt == pos_label)
    total_neg = n - total_pos

    if total_pos == 0 or total_neg == 0:
        raise ValueError("ROC curve requires both positive and negative samples in y_true")

    # Sort samples by predicted score in descending order
    pairs = sorted(zip(y_score, y_true), key=lambda item: float(item[0]), reverse=True)

    # Distinct score thresholds descending
    distinct_scores = sorted(list(set(float(p[0]) for p in pairs)), reverse=True)
    # Include an initial threshold higher than max score
    thresholds = [distinct_scores[0] + 1.0] + distinct_scores

    fpr_list: List[float] = []
    tpr_list: List[float] = []

    for thresh in thresholds:
        tp = 0
        fp = 0
        for score, label in pairs:
            if float(score) >= thresh:
                if label == pos_label:
                    tp += 1
                else:
                    fp += 1

        tpr = tp / float(total_pos)
        fpr = fp / float(total_neg)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    return fpr_list, tpr_list, thresholds


def roc_auc_score(
    y_true: Sequence[Any],
    y_score: Sequence[Number],
    pos_label: Any = 1
) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    using the trapezoidal numerical integration rule from first principles.
    
    AUC = sum_{i=1}^{M-1} (tpr[i] + tpr[i-1]) / 2 * (fpr[i] - fpr[i-1])
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)

    auc = 0.0
    for i in range(1, len(fpr)):
        delta_fpr = fpr[i] - fpr[i - 1]
        avg_tpr = (tpr[i] + tpr[i - 1]) / 2.0
        auc += avg_tpr * delta_fpr

    return max(0.0, min(1.0, auc))
