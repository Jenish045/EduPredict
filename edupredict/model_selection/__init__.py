"""
Model selection module for EduPredict.
"""

from edupredict.model_selection.split import (
    train_test_split,
    KFold,
    StratifiedKFold,
)
from edupredict.model_selection.validation import cross_val_score

__all__ = [
    "train_test_split",
    "KFold",
    "StratifiedKFold",
    "cross_val_score",
]
