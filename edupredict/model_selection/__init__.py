"""
Model selection module for EduPredict.
"""

from edupredict.model_selection.split import (
    train_test_split,
    train_test_split_indices,
    KFold,
    StratifiedKFold,
)
from edupredict.model_selection.validation import cross_val_score

__all__ = [
    "train_test_split",
    "train_test_split_indices",
    "KFold",
    "StratifiedKFold",
    "cross_val_score",
]

