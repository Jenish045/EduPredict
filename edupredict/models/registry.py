"""
Custom model registry and factory for EduPredict.

Provides validated model instantiation for regression and classification tasks,
exposing custom estimators built from first principles without external ML libraries.
"""

from typing import Dict, Any, List, Optional, Type

from edupredict.models.base import BaseModel
from edupredict.models.linear_regression import LinearRegression
from edupredict.models.knn import KNNClassifier
from edupredict.models.decision_tree import DecisionTreeClassifier

REGRESSION_MODELS: Dict[str, Dict[str, Any]] = {
    "Linear Regression": {
        "class": LinearRegression,
        "default_params": {"solver": "gradient_descent"},
        "description": "Custom Linear Regression (Normal Equation & Gradient Descent)",
    },
}

CLASSIFICATION_MODELS: Dict[str, Dict[str, Any]] = {
    "KNN Classifier": {
        "class": KNNClassifier,
        "default_params": {"n_neighbors": 5, "weights": "uniform"},
        "description": "Custom K-Nearest Neighbors Classifier",
    },
    "Decision Tree Classifier": {
        "class": DecisionTreeClassifier,
        "default_params": {"max_depth": 5, "min_samples_split": 2, "random_seed": 42},
        "description": "Custom CART Decision Tree Classifier",
    },
}


def list_models(task: str = "regression") -> List[str]:
    """Return available model display names for the specified task."""
    if task == "regression":
        return list(REGRESSION_MODELS.keys())
    elif "classification" in task:
        return list(CLASSIFICATION_MODELS.keys())
    else:
        raise ValueError(f"Unknown task '{task}'. Choose 'regression' or 'classification'")


def get_model(name: str, task: str = "regression", **kwargs) -> BaseModel:
    """
    Instantiate and return a configured custom EduPredict estimator.

    Parameters
    ----------
    name : str
        Model name or key (e.g. 'Linear Regression', 'KNN Classifier', 'Decision Tree Classifier').
    task : str
        'regression' or 'classification'.
    kwargs : Any
        Hyperparameters passed to the estimator constructor.
    """
    registry = REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS

    # Case-insensitive / normalized lookup
    matched_key = None
    for k in registry:
        if k.lower() == name.lower() or k.lower().replace(" ", "") == name.lower().replace(" ", ""):
            matched_key = k
            break

    if matched_key is None:
        raise ValueError(
            f"Model '{name}' not found for task '{task}'. Available: {list(registry.keys())}"
        )

    model_meta = registry[matched_key]
    model_cls = model_meta["class"]
    params = dict(model_meta["default_params"])
    params.update(kwargs)

    return model_cls(**params)
