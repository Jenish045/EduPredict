"""
Model persistence and serialization for EduPredict custom estimators.

Uses standard Python library persistence (pickle and json) to serialize trained models
without third-party storage dependencies.
"""

import os
import json
import pickle
from typing import Any, Dict

from edupredict.models.linear_regression import LinearRegression
from edupredict.models.knn import KNNClassifier
from edupredict.models.decision_tree import DecisionTreeClassifier

MODEL_CLASSES = {
    "LinearRegression": LinearRegression,
    "KNNClassifier": KNNClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
}


def save_model(model: Any, filepath: str) -> None:
    """Serialize and save an estimator to disk."""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    if filepath.endswith(".json"):
        if hasattr(model, "to_dict"):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(model.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Model {type(model).__name__} does not implement to_dict() for JSON export")
    else:
        with open(filepath, "wb") as f:
            pickle.dump(model, f)


def load_model(filepath: str) -> Any:
    """Load and deserialize an estimator from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    if filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        model_type = data.get("model_type")
        if model_type not in MODEL_CLASSES:
            raise ValueError(f"Unknown serialized model_type: {model_type}")
        return MODEL_CLASSES[model_type].from_dict(data)
    else:
        with open(filepath, "rb") as f:
            return pickle.load(f)
