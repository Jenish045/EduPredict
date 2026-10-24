"""
Custom Decision Tree Classifier implemented entirely from scratch.

Strictly zero third-party dependencies and zero external library imports (no Scikit-learn, no NumPy).
Supports:
- Gini Impurity criterion: 1 - sum(p_k^2).
- Continuous and numerical candidate threshold evaluation.
- Pruning and stopping conditions: max_depth, min_samples_split, min_samples_leaf.
- Feature importance attribution via weighted impurity reduction.
- Tree inspection and recursive dictionary serialization.
"""

from typing import List, Sequence, Optional, Dict, Any, Union, Tuple
from collections import Counter

from edupredict.models.base import BaseModel
from edupredict.metrics.classification import accuracy_score

Number = Union[int, float]
Matrix = Sequence[Sequence[Number]]
Vector = Sequence[Any]


class TreeNode:
    """Node in the recursive decision tree."""

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
        value: Optional[Any] = None,
        probabilities: Optional[Dict[Any, float]] = None,
        gini: float = 0.0,
        n_samples: int = 0,
        is_leaf: bool = False,
    ) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.probabilities = probabilities or {}
        self.gini = gini
        self.n_samples = n_samples
        self.is_leaf = is_leaf

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node recursively to dictionary."""
        d: Dict[str, Any] = {
            "is_leaf": self.is_leaf,
            "gini": self.gini,
            "n_samples": self.n_samples,
            "probabilities": {str(k): v for k, v in self.probabilities.items()},
        }
        if self.is_leaf:
            d["value"] = self.value
        else:
            d["feature_idx"] = self.feature_idx
            d["threshold"] = self.threshold
            d["left"] = self.left.to_dict() if self.left else None
            d["right"] = self.right.to_dict() if self.right else None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TreeNode":
        """Reconstruct node recursively from dictionary."""
        if d.get("is_leaf", False):
            return cls(
                value=d.get("value"),
                probabilities=d.get("probabilities", {}),
                gini=float(d.get("gini", 0.0)),
                n_samples=int(d.get("n_samples", 0)),
                is_leaf=True,
            )
        left_node = cls.from_dict(d["left"]) if d.get("left") else None
        right_node = cls.from_dict(d["right"]) if d.get("right") else None
        return cls(
            feature_idx=d.get("feature_idx"),
            threshold=d.get("threshold"),
            left=left_node,
            right=right_node,
            probabilities=d.get("probabilities", {}),
            gini=float(d.get("gini", 0.0)),
            n_samples=int(d.get("n_samples", 0)),
            is_leaf=False,
        )


class DecisionTreeClassifier(BaseModel):
    """
    Binary and multiclass CART Decision Tree classifier based on Gini impurity.
    """

    def __init__(
        self,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        if max_depth is not None and max_depth <= 0:
            raise ValueError(f"max_depth must be positive integer or None, got {max_depth}")
        if min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_seed = random_seed

        # Learned state
        self.root_: Optional[TreeNode] = None
        self.classes_: List[Any] = []
        self.feature_importances_: List[float] = []

    def _gini_impurity(self, y_subset: Sequence[Any]) -> float:
        """
        Compute Gini impurity: 1 - sum(p_k^2).
        """
        n = len(y_subset)
        if n == 0:
            return 0.0
        counts = Counter(y_subset)
        impurity = 1.0
        for cnt in counts.values():
            p = cnt / n
            impurity -= p * p
        return impurity

    def _majority_vote(self, y_subset: Sequence[Any]) -> Tuple[Any, Dict[Any, float]]:
        """
        Determine plurality class with deterministic tie-breaking and probability map.
        """
        n = len(y_subset)
        counts = Counter(y_subset)
        probs = {c: counts.get(c, 0) / n for c in self.classes_}

        best_cls = self.classes_[0]
        best_cnt = -1
        for c in self.classes_:
            cnt = counts.get(c, 0)
            if cnt > best_cnt:
                best_cnt = cnt
                best_cls = c

        return best_cls, probs

    def _best_split(
        self,
        X_sub: List[List[float]],
        y_sub: List[Any],
        current_gini: float,
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find optimal feature and threshold maximizing Gini gain.
        Returns: (best_feature_idx, best_threshold, best_gain).
        """
        n_samples = len(X_sub)
        n_features = len(X_sub[0])
        best_gain = -1.0
        best_feature = None
        best_threshold = None

        for feat_idx in range(n_features):
            values = sorted(list(set(row[feat_idx] for row in X_sub)))
            if len(values) <= 1:
                continue

            # Generate candidate split thresholds at midpoints
            thresholds = []
            if len(values) > 30:
                # Subsample thresholds if high cardinality
                step = len(values) // 25
                for i in range(0, len(values) - 1, max(1, step)):
                    thresholds.append((values[i] + values[i + 1]) / 2.0)
            else:
                for i in range(len(values) - 1):
                    thresholds.append((values[i] + values[i + 1]) / 2.0)

            for thresh in thresholds:
                left_y: List[Any] = []
                right_y: List[Any] = []

                for r_idx in range(n_samples):
                    if X_sub[r_idx][feat_idx] <= thresh:
                        left_y.append(y_sub[r_idx])
                    else:
                        right_y.append(y_sub[r_idx])

                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue

                left_gini = self._gini_impurity(left_y)
                right_gini = self._gini_impurity(right_y)
                weighted_gini = (len(left_y) / n_samples) * left_gini + (len(right_y) / n_samples) * right_gini
                gain = current_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = thresh

        return best_feature, best_threshold, best_gain

    def _build_tree(
        self,
        X_sub: List[List[float]],
        y_sub: List[Any],
        depth: int,
        feature_gains: List[float],
    ) -> TreeNode:
        """
        Recursively construct tree nodes until stopping criteria are satisfied.
        """
        n_samples = len(X_sub)
        current_gini = self._gini_impurity(y_sub)
        majority_val, probs = self._majority_vote(y_sub)

        # Base / stopping conditions
        if (
            current_gini == 0.0
            or n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return TreeNode(
                value=majority_val,
                probabilities=probs,
                gini=current_gini,
                n_samples=n_samples,
                is_leaf=True,
            )

        best_feat, best_thresh, best_gain = self._best_split(X_sub, y_sub, current_gini)

        if best_feat is None or best_thresh is None or best_gain <= 1e-7:
            return TreeNode(
                value=majority_val,
                probabilities=probs,
                gini=current_gini,
                n_samples=n_samples,
                is_leaf=True,
            )

        # Partition samples
        left_X, left_y = [], []
        right_X, right_y = [], []

        for r_idx in range(n_samples):
            if X_sub[r_idx][best_feat] <= best_thresh:
                left_X.append(X_sub[r_idx])
                left_y.append(y_sub[r_idx])
            else:
                right_X.append(X_sub[r_idx])
                right_y.append(y_sub[r_idx])

        # Accumulate feature importance: weighted impurity decrease
        feature_gains[best_feat] += n_samples * best_gain

        left_child = self._build_tree(left_X, left_y, depth + 1, feature_gains)
        right_child = self._build_tree(right_X, right_y, depth + 1, feature_gains)

        return TreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            probabilities=probs,
            gini=current_gini,
            n_samples=n_samples,
            is_leaf=False,
        )

    def fit(self, X: Matrix, y: Vector) -> "DecisionTreeClassifier":
        """
        Fit decision tree classifier to training dataset.
        """
        self._validate_inputs(X, y)
        X_clean = [[float(v) for v in row] for row in X]
        y_clean = list(y)
        self.n_features_in_ = len(X_clean[0])

        unique_classes = list(set(y_clean))
        try:
            self.classes_ = sorted(unique_classes)
        except TypeError:
            self.classes_ = sorted(unique_classes, key=lambda x: str(x))

        feature_gains = [0.0] * self.n_features_in_
        self.root_ = self._build_tree(X_clean, y_clean, depth=0, feature_gains=feature_gains)

        # Normalize feature importances
        total_gain = sum(feature_gains)
        if total_gain > 0.0:
            self.feature_importances_ = [g / total_gain for g in feature_gains]
        else:
            self.feature_importances_ = [1.0 / self.n_features_in_] * self.n_features_in_

        self.is_fitted = True
        return self

    def _traverse(self, node: TreeNode, x: List[float]) -> TreeNode:
        """Traverse tree down to the leaf node for a single sample."""
        if node.is_leaf or node.feature_idx is None or node.threshold is None:
            return node
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(node.left, x) if node.left else node
        else:
            return self._traverse(node.right, x) if node.right else node

    def predict_proba(self, X: Matrix) -> List[List[float]]:
        """
        Predict class probabilities for samples X.
        """
        self._check_is_fitted()
        self._validate_inputs(X, check_y=False)
        if len(X[0]) != self.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: model expects {self.n_features_in_} features, got {len(X[0])}"
            )

        all_probs: List[List[float]] = []
        for row in X:
            x_query = [float(v) for v in row]
            leaf = self._traverse(self.root_, x_query)
            # Map leaf probabilities to classes_ order
            p_row = [leaf.probabilities.get(c, leaf.probabilities.get(str(c), 0.0)) for c in self.classes_]
            s = sum(p_row)
            if s > 0.0:
                p_row = [p / s for p in p_row]
            else:
                p_row = [1.0 / len(self.classes_) for _ in self.classes_]
            all_probs.append(p_row)

        return all_probs

    def predict(self, X: Matrix) -> List[Any]:
        """
        Predict majority class for samples X.
        """
        self._check_is_fitted()
        self._validate_inputs(X, check_y=False)
        preds: List[Any] = []
        for row in X:
            x_query = [float(v) for v in row]
            leaf = self._traverse(self.root_, x_query)
            preds.append(leaf.value)
        return preds

    def score(self, X: Matrix, y: Vector) -> float:
        """
        Compute classification accuracy on test dataset.
        """
        preds = self.predict(X)
        return accuracy_score(list(y), preds)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Decision Tree state to dictionary."""
        return {
            "model_type": "DecisionTreeClassifier",
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_seed": self.random_seed,
            "classes_": self.classes_,
            "feature_importances_": self.feature_importances_,
            "root_": self.root_.to_dict() if self.root_ else None,
            "is_fitted": self.is_fitted,
            "n_features_in_": self.n_features_in_,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionTreeClassifier":
        """Reconstruct Decision Tree classifier from serialized dictionary."""
        model = cls(
            max_depth=data.get("max_depth", 5),
            min_samples_split=data.get("min_samples_split", 2),
            min_samples_leaf=data.get("min_samples_leaf", 1),
            random_seed=data.get("random_seed", 42),
        )
        model.classes_ = list(data.get("classes_", []))
        model.feature_importances_ = list(data.get("feature_importances_", []))
        if data.get("root_"):
            model.root_ = TreeNode.from_dict(data["root_"])
        model.is_fitted = bool(data.get("is_fitted", False))
        model.n_features_in_ = data.get("n_features_in_")
        return model
