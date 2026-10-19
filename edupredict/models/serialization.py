"""
Model serialization and persistence implemented completely from scratch.

Provides clean JSON export and import for all EduPredict models:
- LinearRegression
- LogisticRegression
- KNNClassifier
- KNNRegressor
- DecisionTreeClassifier
- GaussianNaiveBayes

Eliminates reliance on pickle or third-party serialization libraries,
ensuring 100% human-readable, auditable, and transparent model checkpoints.
"""

import json
import os
from typing import Dict, Any, Optional

from edupredict.models.base import BaseModel
from edupredict.models.linear_regression import LinearRegression
from edupredict.models.logistic_regression import LogisticRegression
from edupredict.models.knn import KNNClassifier, KNNRegressor
from edupredict.models.decision_tree import DecisionTreeClassifier, TreeNode
from edupredict.models.naive_bayes import GaussianNaiveBayes


def _tree_node_to_dict(node: Optional[TreeNode]) -> Optional[Dict[str, Any]]:
    """Recursively convert a Decision Tree node to a serializable dict."""
    if node is None:
        return None
    return {
        "feature_idx": node.feature_idx,
        "threshold": node.threshold,
        "value": node.value,
        "probabilities": node.probabilities,
        "left": _tree_node_to_dict(node.left),
        "right": _tree_node_to_dict(node.right),
    }


def _dict_to_tree_node(d: Optional[Dict[str, Any]]) -> Optional[TreeNode]:
    """Recursively reconstruct a Decision Tree node from a dict."""
    if d is None:
        return None
    # Reconstruct dictionary keys for probabilities if any
    probs = d.get("probabilities")
    if probs is not None and isinstance(probs, dict):
        probs = {int(k) if k.isdigit() else k: float(v) for k, v in probs.items()}

    return TreeNode(
        feature_idx=d.get("feature_idx"),
        threshold=d.get("threshold"),
        left=_dict_to_tree_node(d.get("left")),
        right=_dict_to_tree_node(d.get("right")),
        value=d.get("value"),
        probabilities=probs,
    )


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """
    Convert a fitted or unfitted EduPredict model into a JSON-serializable dictionary.
    """
    class_name = model.__class__.__name__
    data: Dict[str, Any] = {
        "__model_class__": class_name,
        "is_fitted": model.is_fitted,
    }

    if isinstance(model, LinearRegression):
        data.update({
            "learning_rate": model.learning_rate,
            "max_epochs": model.max_epochs,
            "solver": model.solver,
            "fit_intercept": model.fit_intercept,
            "l2_penalty": model.l2_penalty,
            "weights": model.weights_,
            "bias": model.bias_,
            "loss_history": model.loss_history_,
        })
    elif isinstance(model, LogisticRegression):
        data.update({
            "learning_rate": model.learning_rate,
            "max_epochs": model.max_epochs,
            "threshold": model.threshold,
            "l2_penalty": model.l2_penalty,
            "weights": model.weights_,
            "bias": model.bias_,
            "loss_history": model.loss_history_,
        })
    elif isinstance(model, KNNClassifier):
        data.update({
            "n_neighbors": model.n_neighbors,
            "metric": model.metric,
            "weights": model.weights,
            "classes": model.classes_,
            "X_train": model.X_train_,
            "y_train": model.y_train_,
        })
    elif isinstance(model, KNNRegressor):
        data.update({
            "n_neighbors": model.n_neighbors,
            "metric": model.metric,
            "weights": model.weights,
            "X_train": model.X_train_,
            "y_train": model.y_train_,
        })
    elif isinstance(model, DecisionTreeClassifier):
        data.update({
            "criterion": model.criterion,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "classes": model.classes_,
            "n_features": model.n_features_,
            "feature_importances": model.feature_importances_,
            "root": _tree_node_to_dict(model.root_),
        })
    elif isinstance(model, GaussianNaiveBayes):
        # Convert dictionary keys to string for JSON serialization
        class_priors = {str(k): v for k, v in model.class_priors_.items()}
        means = {str(k): v for k, v in model.means_.items()}
        variances = {str(k): v for k, v in model.variances_.items()}
        data.update({
            "var_smoothing": model.var_smoothing,
            "classes": model.classes_,
            "class_priors": class_priors,
            "means": means,
            "variances": variances,
            "n_features": model.n_features_,
        })
    else:
        raise NotImplementedError(f"Serialization for model class '{class_name}' is not supported.")

    return data


def dict_to_model(data: Dict[str, Any]) -> BaseModel:
    """
    Reconstruct an EduPredict model instance from a serialized dictionary.
    """
    class_name = data.get("__model_class__")
    if not class_name:
        raise ValueError("Missing '__model_class__' key in serialized model data.")

    if class_name == "LinearRegression":
        model = LinearRegression(
            learning_rate=data.get("learning_rate", 0.01),
            max_epochs=data.get("max_epochs", 1000),
            solver=data.get("solver", "gd"),
            fit_intercept=data.get("fit_intercept", True),
            l2_penalty=data.get("l2_penalty", 0.0),
        )
        model.weights_ = data.get("weights")
        model.bias_ = data.get("bias", 0.0)
        model.loss_history_ = data.get("loss_history", [])
        model._is_fitted = data.get("is_fitted", False)
        return model

    elif class_name == "LogisticRegression":
        model = LogisticRegression(
            learning_rate=data.get("learning_rate", 0.01),
            max_epochs=data.get("max_epochs", 1000),
            threshold=data.get("threshold", 0.5),
            l2_penalty=data.get("l2_penalty", 0.0),
        )
        model.weights_ = data.get("weights")
        model.bias_ = data.get("bias", 0.0)
        model.loss_history_ = data.get("loss_history", [])
        model._is_fitted = data.get("is_fitted", False)
        return model

    elif class_name == "KNNClassifier":
        model = KNNClassifier(
            n_neighbors=data.get("n_neighbors", 5),
            metric=data.get("metric", "euclidean"),
            weights=data.get("weights", "uniform"),
        )
        model.classes_ = data.get("classes", [])
        model.X_train_ = data.get("X_train", [])
        model.y_train_ = data.get("y_train", [])
        model._is_fitted = data.get("is_fitted", False)
        return model

    elif class_name == "KNNRegressor":
        model = KNNRegressor(
            n_neighbors=data.get("n_neighbors", 5),
            metric=data.get("metric", "euclidean"),
            weights=data.get("weights", "uniform"),
        )
        model.X_train_ = data.get("X_train", [])
        model.y_train_ = data.get("y_train", [])
        model._is_fitted = data.get("is_fitted", False)
        return model

    elif class_name == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(
            criterion=data.get("criterion", "gini"),
            max_depth=data.get("max_depth", 5),
            min_samples_split=data.get("min_samples_split", 2),
        )
        model.classes_ = data.get("classes", [])
        model.n_features_ = data.get("n_features", 0)
        model.feature_importances_ = data.get("feature_importances", [])
        model.root_ = _dict_to_tree_node(data.get("root"))
        model._is_fitted = data.get("is_fitted", False)
        return model

    elif class_name == "GaussianNaiveBayes":
        model = GaussianNaiveBayes(
            var_smoothing=data.get("var_smoothing", 1e-9)
        )
        model.classes_ = data.get("classes", [])
        model.n_features_ = data.get("n_features", 0)
        raw_priors = data.get("class_priors", {})
        raw_means = data.get("means", {})
        raw_vars = data.get("variances", {})

        # Restore integer class keys if applicable
        model.class_priors_ = {int(k) if k.isdigit() else k: v for k, v in raw_priors.items()}
        model.means_ = {int(k) if k.isdigit() else k: v for k, v in raw_means.items()}
        model.variances_ = {int(k) if k.isdigit() else k: v for k, v in raw_vars.items()}
        model._is_fitted = data.get("is_fitted", False)
        return model

    else:
        raise ValueError(f"Unknown model class name '{class_name}'.")


def save_model(model: BaseModel, filepath: str) -> None:
    """
    Save an EduPredict model to a JSON file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    state_dict = model_to_dict(model)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state_dict, f, indent=2)


def load_model(filepath: str) -> BaseModel:
    """
    Load an EduPredict model from a JSON file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dict_to_model(data)
