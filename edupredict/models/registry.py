"""
Scikit-learn model registry and factory for EduPredict.

Provides validated model instantiation for regression and classification tasks,
exposing standard Scikit-learn estimators configured for educational benchmarking.
"""

from typing import Dict, Any, List, Optional, Type
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

# Aliases for consistent naming
KNNClassifier = KNeighborsClassifier
KNNRegressor = KNeighborsRegressor
GaussianNaiveBayes = GaussianNB

REGRESSION_MODELS: Dict[str, Dict[str, Any]] = {
    "Linear Regression": {
        "class": LinearRegression,
        "default_params": {"fit_intercept": True},
        "description": "Ordinary Least Squares Linear Regression",
    },
    "Ridge Regression": {
        "class": Ridge,
        "default_params": {"alpha": 1.0},
        "description": "Linear least squares with L2 regularization",
    },
    "KNN Regressor": {
        "class": KNeighborsRegressor,
        "default_params": {"n_neighbors": 5, "weights": "uniform"},
        "description": "k-Nearest Neighbors Regressor",
    },
    "Decision Tree Regressor": {
        "class": DecisionTreeRegressor,
        "default_params": {"max_depth": 5, "min_samples_split": 2, "random_state": 42},
        "description": "Decision Tree Regressor",
    },
}

CLASSIFICATION_MODELS: Dict[str, Dict[str, Any]] = {
    "Logistic Regression": {
        "class": LogisticRegression,
        "default_params": {"max_iter": 1000, "random_state": 42},
        "description": "Binary and Multiclass Logistic Regression",
    },
    "KNN Classifier": {
        "class": KNeighborsClassifier,
        "default_params": {"n_neighbors": 5, "weights": "uniform"},
        "description": "k-Nearest Neighbors Classifier",
    },
    "Decision Tree Classifier": {
        "class": DecisionTreeClassifier,
        "default_params": {"max_depth": 5, "min_samples_split": 2, "random_state": 42},
        "description": "Decision Tree Classifier",
    },
    "Gaussian Naive Bayes": {
        "class": GaussianNB,
        "default_params": {"var_smoothing": 1e-9},
        "description": "Gaussian Naive Bayes Classifier",
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


def get_model(name: str, task: str = "regression", **kwargs) -> Any:
    """
    Instantiate and return a configured Scikit-learn estimator.

    Parameters
    ----------
    name : str
        Model name or key (e.g. 'Linear Regression', 'Logistic Regression').
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
