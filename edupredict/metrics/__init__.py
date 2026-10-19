"""
Metrics module for EduPredict.
"""

from edupredict.metrics.regression import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from edupredict.metrics.classification import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score,
)

__all__ = [
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "accuracy_score",
    "confusion_matrix",
    "precision_score",
    "recall_score",
    "f1_score",
    "classification_report",
    "roc_curve",
    "roc_auc_score",
]
