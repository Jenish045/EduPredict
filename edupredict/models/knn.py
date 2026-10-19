"""
K-Nearest Neighbors (KNN) classifier and regressor implemented from scratch.

Computes exact pairwise distances using edupredict.math.vectors without external libraries.
Supports:
- Configurable distance metrics: Euclidean and Manhattan
- Weights: 'uniform' or 'distance'
- Deterministic tie-breaking for classification
- Posterior class probability estimation (`predict_proba`)
"""

from typing import List, Sequence, Union, Any, Optional, Dict, Tuple
from collections import Counter
from edupredict.models.base import BaseModel
from edupredict.math.vectors import euclidean_distance, manhattan_distance
from edupredict.math.stats import mean

Number = Union[int, float]
Tuple_Dist_Label = Tuple[float, Any]


class KNNClassifier(BaseModel):
    """
    K-Nearest Neighbors Classifier.
    Predicts class based on majority vote of k nearest training points with deterministic tie-breaking.
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
        self.classes_: List[Any] = []
        self.X_train_: List[List[float]] = []
        self.y_train_: List[Any] = []

    def fit(self, X: Sequence[Sequence[Number]], y: Sequence[Any]) -> "KNNClassifier":
        """Store training points for non-parametric lazy evaluation."""
        self._validate_inputs(X, y)
        if self.n_neighbors > len(X):
            raise ValueError(f"n_neighbors ({self.n_neighbors}) cannot be greater than n_samples ({len(X)})")
        self.X_train_ = [[float(v) for v in row] for row in X]
        self.y_train_ = list(y)
        self.classes_ = sorted(list(set(self.y_train_)), key=lambda c: str(c))
        self._is_fitted = True
        return self

    def _compute_distance(self, u: Sequence[Number], v: Sequence[Number]) -> float:
        if self.metric == "euclidean":
            return euclidean_distance(u, v)
        return manhattan_distance(u, v)

    def _get_neighbors(self, x: Sequence[Number]) -> List[Tuple_Dist_Label]:
        """Compute distances from test sample x to all training samples and return k nearest."""
        dist_label_pairs = []
        for i, x_train in enumerate(self.X_train_):
            d = self._compute_distance(x, x_train)
            dist_label_pairs.append((d, self.y_train_[i]))

        # Sort by distance ascending, then by index for determinism
        dist_label_pairs.sort(key=lambda item: (item[0], str(item[1])))
        return dist_label_pairs[: self.n_neighbors]

    def predict_proba(self, X: Sequence[Sequence[Number]]) -> List[Dict[Any, float]]:
        """
        Estimate class posterior probabilities P(y=c|x) for each query sample in X.
        """
        self._check_is_fitted()
        self._validate_inputs(X, expected_features=len(self.X_train_[0]))

        all_probs: List[Dict[Any, float]] = []

        for x in X:
            neighbors = self._get_neighbors(x)
            prob_dict = {cls: 0.0 for cls in self.classes_}

            if self.weights == "uniform":
                factor = 1.0 / float(self.n_neighbors)
                for _, label in neighbors:
                    prob_dict[label] += factor
            else:
                total_weight = 0.0
                for d, label in neighbors:
                    w = 1.0 / (d + 1e-5)
                    prob_dict[label] += w
                    total_weight += w
                if total_weight > 0.0:
                    for cls in self.classes_:
                        prob_dict[cls] /= total_weight

            all_probs.append(prob_dict)

        return all_probs

    def predict(self, X: Sequence[Sequence[Number]]) -> List[Any]:
        """
        Predict majority class for each sample with deterministic tie-breaking.
        """
        self._check_is_fitted()
        self._validate_inputs(X, expected_features=len(self.X_train_[0]))

        predictions: List[Any] = []
        for x in X:
            neighbors = self._get_neighbors(x)

            if self.weights == "uniform":
                counts: Dict[Any, int] = {}
                closest_dist_per_class: Dict[Any, float] = {}
                for d, label in neighbors:
                    counts[label] = counts.get(label, 0) + 1
                    if label not in closest_dist_per_class or d < closest_dist_per_class[label]:
                        closest_dist_per_class[label] = d

                # Find maximum vote count
                max_count = max(counts.values())
                tied_candidates = [lbl for lbl, c in counts.items() if c == max_count]

                # Deterministic tie-breaking:
                # 1. Candidate with smaller distance to its closest neighbor in k-NN
                # 2. Lexicographical / standard string sort of label
                tied_candidates.sort(key=lambda lbl: (closest_dist_per_class[lbl], str(lbl)))
                best_label = tied_candidates[0]
                predictions.append(best_label)

            else:
                # Weighted vote: 1 / (d + 1e-5)
                weighted_votes: Dict[Any, float] = {}
                for d, label in neighbors:
                    w = 1.0 / (d + 1e-5)
                    weighted_votes[label] = weighted_votes.get(label, 0.0) + w

                max_w = max(weighted_votes.values())
                tied_candidates = [lbl for lbl, w in weighted_votes.items() if abs(w - max_w) < 1e-9]
                tied_candidates.sort(key=lambda lbl: str(lbl))
                predictions.append(tied_candidates[0])

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

            dist_target_pairs.sort(key=lambda item: (item[0], item[1]))
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
                predictions.append(acc / total_w if total_w > 0 else 0.0)

        return predictions

