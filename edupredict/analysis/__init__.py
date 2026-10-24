"""
Model diagnostic and complexity analysis package for EduPredict.
"""

from edupredict.analysis.overfitting import (
    analyze_knn_complexity,
    analyze_decision_tree_complexity,
    analyze_linear_regression_regularization,
)

__all__ = [
    "analyze_knn_complexity",
    "analyze_decision_tree_complexity",
    "analyze_linear_regression_regularization",
]
