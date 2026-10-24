"""
Custom K-Nearest Neighbors Classifier implemented entirely from scratch.

Strictly zero third-party dependencies and zero external library imports (no Scikit-learn, no NumPy).
Supports:
- Configurable neighborhood size k.
- Euclidean and Manhattan Minkowski distance metrics.
- Uniform voting and inverse-distance weighting.
- Deterministic tie-breaking on class ordering.
- Class probability estimation via predict_proba.
"""

from typing import List, Sequence, Optional, Dict, Any, Union, Tuple
from collections import Counter

from edupredict.models.base import BaseModel
from edupredict.math.vectors import euclidean_distance, manhattan_distance
from edupredict.metrics.classification import accuracy_score

Number = Union[int, float]
Matrix = Sequence[Sequence[Number]]
Vector = Sequence[Any]


class KNNClassifier(BaseModel):
    """
    Non-parametric K-Nearest Neighbors classifier with explicit distance metrics.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
    ) -> None:
        super().__init__()
        if n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be positive integer, got {n_neighbors}")
        valid_metrics = ("euclidean", "manhattan")
        if metric not in valid_metrics:
            raise ValueError(f"Unknown metric '{metric}'. Expected one of {valid_metrics}")
        valid_weights = ("uniform", "distance")
        if weights not in valid_weights:
            raise ValueError(f"Unknown weights '{weights}'. Expected one of {valid_weights}")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

        # Learned / stored data
        self.X_train_: List[List[float]] = []
        self.y_train_: List[Any] = []
        self.classes_: List[Any] = []

    def fit(self, X: Matrix, y: Vector) -> "KNNClassifier":
        """
        Fit KNN classifier by storing reference training points and class alphabet.
        """
        self._validate_inputs(X, y)
        if self.n_neighbors > len(X):
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) cannot exceed number of training samples ({len(X)})"
            )

        self.X_train_ = [[float(v) for v in row] for row in X]
        self.y_train_ = list(y)
        self.n_features_in_ = len(self.X_train_[0])

        # Extract unique classes sorted deterministically
        unique_classes = list(set(self.y_train_))
        try:
            self.classes_ = sorted(unique_classes)
        except TypeError:
            self.classes_ = sorted(unique_classes, key=lambda x: str(x))

        self.is_fitted = True
        return self

    def _compute_distance(self, u: List[float], v: List[float]) -> float:
        """Compute distance between two feature vectors according to configured metric."""
        if self.metric == "euclidean":
            return euclidean_distance(u, v)
        else:
            return manhattan_distance(u, v)

    def _get_neighbors(self, x: List[float]) -> List[Tuple[float, Any, int]]:
        """
        Compute distances to all training points and return top-k sorted neighbors.
        Each tuple is: (distance, label, original_index).
        """
        distances = []
        for idx, (x_tr, y_tr) in enumerate(zip(self.X_train_, self.y_train_)):
            d = self._compute_distance(x, x_tr)
            distances.append((d, y_tr, idx))

        # Deterministic sort: primarily by distance, then by index
        distances.sort(key=lambda item: (item[0], item[2]))
        return distances[: self.n_neighbors]

    def predict_proba(self, X: Matrix) -> List[List[float]]:
        """
        Predict class probabilities for test samples X.
        Returns a 2D list of shape (n_samples, n_classes) corresponding to self.classes_.
        """
        self._check_is_fitted()
        self._validate_inputs(X, check_y=False)
        if len(X[0]) != self.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: model expects {self.n_features_in_} features, got {len(X[0])}"
            )

        all_probs: List[List[float]] = []
        eps = 1e-9

        for row in X:
            x_query = [float(v) for v in row]
            neighbors = self._get_neighbors(x_query)

            class_scores = {c: 0.0 for c in self.classes_}

            if self.weights == "uniform":
                for _, label, _ in neighbors:
                    class_scores[label] += 1.0
                total_weight = float(len(neighbors))
            else:
                total_weight = 0.0
                for dist, label, _ in neighbors:
                    w = 1.0 / (dist + eps)
                    class_scores[label] += w
                    total_weight += w

            if total_weight > 0.0:
                row_probs = [class_scores[c] / total_weight for c in self.classes_]
            else:
                row_probs = [1.0 / len(self.classes_) for _ in self.classes_]

            all_probs.append(row_probs)

        return all_probs

    def predict(self, X: Matrix) -> List[Any]:
        """
        Predict target class for test samples X with deterministic tie-breaking.
        """
        probs = self.predict_proba(X)
        predictions: List[Any] = []

        for p_row in probs:
            # Pick class with max probability; on tie, choose the first according to self.classes_
            best_idx = 0
            best_p = p_row[0]
            for idx in range(1, len(self.classes_)):
                if p_row[idx] > best_p:
                    best_p = p_row[idx]
                    best_idx = idx
            predictions.append(self.classes_[best_idx])

        return predictions

    def score(self, X: Matrix, y: Vector) -> float:
        """
        Compute classification accuracy on test data.
        """
        preds = self.predict(X)
        return accuracy_score(list(y), preds)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize KNN state to dictionary."""
        return {
            "model_type": "KNNClassifier",
            "n_neighbors": self.n_neighbors,
            "metric": self.metric,
            "weights": self.weights,
            "classes_": self.classes_,
            "X_train_": self.X_train_,
            "y_train_": self.y_train_,
            "is_fitted": self.is_fitted,
            "n_features_in_": self.n_features_in_,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KNNClassifier":
        """Restore KNN classifier from serialized state."""
        model = cls(
            n_neighbors=data.get("n_neighbors", 5),
            metric=data.get("metric", "euclidean"),
            weights=data.get("weights", "uniform"),
        )
        model.classes_ = list(data.get("classes_", []))
        model.X_train_ = [list(r) for r in data.get("X_train_", [])]
        model.y_train_ = list(data.get("y_train_", []))
        model.is_fitted = bool(data.get("is_fitted", False))
        model.n_features_in_ = data.get("n_features_in_")
        return model
