"""
Models module for EduPredict.

Exposes Scikit-learn estimators for predictive tasks along with
model registry and serialization helpers.
"""

from edupredict.models.registry import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    KNNClassifier,
    KNNRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    GaussianNaiveBayes,
    get_model,
    list_models,
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
)
from edupredict.models.serialization import save_model, load_model

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "Ridge",
    "KNNClassifier",
    "KNNRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "GaussianNaiveBayes",
    "get_model",
    "list_models",
    "save_model",
    "load_model",
    "REGRESSION_MODELS",
    "CLASSIFICATION_MODELS",
]
