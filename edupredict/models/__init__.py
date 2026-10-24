"""
Models module for EduPredict.

Exposes custom estimators built from first principles for predictive tasks along with
model registry and serialization helpers.
"""

from edupredict.models.base import BaseModel
from edupredict.models.linear_regression import LinearRegression
from edupredict.models.knn import KNNClassifier
from edupredict.models.decision_tree import DecisionTreeClassifier
from edupredict.models.registry import (
    get_model,
    list_models,
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
)
from edupredict.models.serialization import save_model, load_model

__all__ = [
    "BaseModel",
    "LinearRegression",
    "KNNClassifier",
    "DecisionTreeClassifier",
    "get_model",
    "list_models",
    "save_model",
    "load_model",
    "REGRESSION_MODELS",
    "CLASSIFICATION_MODELS",
]
