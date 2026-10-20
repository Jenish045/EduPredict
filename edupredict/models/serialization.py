"""
Model persistence and serialization for Scikit-learn estimators.

Uses standard Python library persistence to serialize trained models and
preprocessors without third-party storage dependencies.
"""

import os
import pickle
from typing import Any


def save_model(model: Any, filepath: str) -> None:
    """Serialize and save an estimator to disk."""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath: str) -> Any:
    """Load and deserialize an estimator from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)
