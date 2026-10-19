"""
K-Nearest Neighbors (KNN) classifier and regressor implemented from scratch.

Computes exact pairwise distances using edupredict.math.vectors without external libraries.
"""

from typing import List, Sequence, Union, Any, Optional
from collections import Counter
from edupredict.models.base import BaseModel
from edupredict.math.vectors import euclidean_distance, manhattan_distance
from edupredict.math.stats import mean

Number = Union[int, float]


class KNNClassifier(BaseModel):
    """
    K-Nearest Neighbors Classifier.
    Predicts class based on majority vote of k nearest training points.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform"
    ) -> None:
        super().__init__()
        if n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be positive, got {n_neighbors}")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError(f"Unknown metric '{metric}'. Choose 'euclidean' or 'manhattan'")
        if weights not in ("uniform", "distance"):
            raise ValueError(f"Unknown weights '{weights}'. Choose 'uniform' or 'distance'")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.X_train_: List[List[float]] = []
        self.y_train_: List[Any] = []

    def fit(self, X: Sequence[Sequence[Number]], y: Sequence[Any]) -> "KNNClassifier":
        """Store training points for non-parametric lazy evaluation."""
        self._validate_inputs(X, y)
        if self.n_neighbors > len(X):
            raise ValueError(f"n_neighbors ({self.n_neighbors}) cannot be greater than n_samples ({len(X)})")
        self.X_train_ = [[float(v) for v in row] for row in X]
        self.y_train_ = list(y)
        self._is_fitted = True
        return self

    def _compute_distance(self, u: Sequence[Number], v: Sequence[Number]) -> float:
        if self.metric == "euclidean":
            return euclidean_distance(u, v)
        return manhattan_distance(u, v)

    def predict(self, X: Sequence[Sequence[Number]]) -> List[Any]:
        """Predict majority class for each sample."""
        self._check_is_fitted()
        self._validate_inputs(X, expected_features=len(self.X_train_[0]))

        predictions: List[Any] = []
        for x in X:
            # Calculate distance to all training samples
            dist_label_pairs = []
            for i, x_train in enumerate(self.X_train_):
                d = self._compute_distance(x, x_train)
                dist_label_pairs.append((d, self.y_train_[i]))

            # Find k nearest neighbors
            dist_label_pairs.sort(key=lambda item: item[0])
            neighbors = dist_label_pairs[: self.n_neighbors]

            if self.weights == "uniform":
                # Majority vote
                votes = [label for _, label in neighbors]
                most_common = Counter(votes).most_common(1)[0][0]
                predictions.append(most_common)
            else:
                # Weighted vote: 1 / (d + 1e-5)
                weighted_votes: dict = {}
                for d, label in neighbors:
                    w = 1.0 / (d + 1e-5)
                    weighted_votes[label] = weighted_votes.get(label, 0.0) + w
                best_label = max(weighted_votes.items(), key=lambda item: item[1])[0]
                predictions.append(best_label)

        return predictions


class KNNRegressor(BaseModel):
    """
    K-Nearest Neighbors Regressor.
    Predicts continuous target as the mean (or distance-weighted average) of k neighbors.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform"
    ) -> None:
        super().__init__()
        if n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be positive, got {n_neighbors}")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError(f"Unknown metric '{metric}'")
        if weights not in ("uniform", "distance"):
            raise ValueError(f"Unknown weights '{weights}'")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.X_train_: List[List[float]] = []
        self.y_train_: List[float] = []

    def fit(self, X: Sequence[Sequence[Number]], y: Sequence[Number]) -> "KNNRegressor":
        """Store training points for non-parametric lazy evaluation."""
        self._validate_inputs(X, y)
        if self.n_neighbors > len(X):
            raise ValueError(f"n_neighbors ({self.n_neighbors}) cannot be greater than n_samples ({len(X)})")
        self.X_train_ = [[float(v) for v in row] for row in X]
        self.y_train_ = [float(v) for v in y]
        self._is_fitted = True
        return self

    def predict(self, X: Sequence[Sequence[Number]]) -> List[float]:
        """Predict continuous value for each sample."""
        self._check_is_fitted()
        self._validate_inputs(X, expected_features=len(self.X_train_[0]))

        predictions: List[float] = []
        for x in X:
            dist_target_pairs = []
            for i, x_train in enumerate(self.X_train_):
                d = euclidean_distance(x, x_train) if self.metric == "euclidean" else manhattan_distance(x, x_train)
                dist_target_pairs.append((d, self.y_train_[i]))

            dist_target_pairs.sort(key=lambda item: item[0])
            neighbors = dist_target_pairs[: self.n_neighbors]

            if self.weights == "uniform":
                vals = [t for _, t in neighbors]
                predictions.append(mean(vals))
            else:
                total_w = 0.0
                acc = 0.0
                for d, t in neighbors:
                    w = 1.0 / (d + 1e-5)
                    acc += w * t
                    total_w += w
                predictions.append(acc / total_w)

        return predictions
