"""
Decision Tree Classifier implemented from first principles.

Supports:
- Gini Impurity: 1 - sum(p_k^2)
- Information Gain / Entropy: -sum(p_k * log2(p_k))
- Max depth and min_samples_split regularization
- Feature importance calculation based on weighted impurity reduction
"""

import math
from typing import List, Sequence, Optional, Union, Any, Dict, Tuple
from collections import Counter
from edupredict.models.base import BaseModel

Number = Union[int, float]


class TreeNode:
    """Internal or leaf node of a Decision Tree."""

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
        *,
        value: Optional[Any] = None,
        probabilities: Optional[Dict[Any, float]] = None
    ) -> None:
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf prediction
        self.probabilities = probabilities  # Leaf class probabilities

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier(BaseModel):
    """
    Classification Decision Tree using recursive binary partitioning.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = 5,
        min_samples_split: int = 2
    ) -> None:
        super().__init__()
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"Unknown criterion '{criterion}'. Choose 'gini' or 'entropy'")
        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {max_depth}")
        if min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root_: Optional[TreeNode] = None
        self.classes_: List[Any] = []
        self.n_features_: int = 0
        self.feature_importances_: List[float] = []

    def _impurity(self, y: Sequence[Any]) -> float:
        """Compute impurity (Gini or Entropy) for a slice of targets y."""
        n = len(y)
        if n == 0:
            return 0.0

        counts = Counter(y)
        if self.criterion == "gini":
            # Gini = 1 - sum(p_k^2)
            sum_sq = sum((count / n) ** 2 for count in counts.values())
            return 1.0 - sum_sq
        else:
            # Entropy = - sum(p_k * log2(p_k))
            ent = 0.0
            for count in counts.values():
                p = count / n
                if p > 0:
                    ent -= p * math.log2(p)
            return ent

    def fit(self, X: Sequence[Sequence[Number]], y: Sequence[Any]) -> "DecisionTreeClassifier":
        """Build decision tree from training set (X, y)."""
        self._validate_inputs(X, y)
        self.n_features_ = len(X[0])
        self.classes_ = sorted(list(set(y)), key=lambda c: str(c))
        self.feature_importances_ = [0.0] * self.n_features_

        self.root_ = self._grow_tree(X, y, depth=0)

        # Normalize feature importances so they sum to 1.0
        total_importance = sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ = [imp / total_importance for imp in self.feature_importances_]

        self._is_fitted = True
        return self

    def _grow_tree(
        self,
        X: Sequence[Sequence[Number]],
        y: Sequence[Any],
        depth: int
    ) -> TreeNode:
        """Recursively build tree."""
        n_samples = len(y)
        counts = Counter(y)
        most_common_class = counts.most_common(1)[0][0]

        # Class probabilities at this node
        probs = {cls: counts.get(cls, 0) / n_samples for cls in self.classes_}

        # Check stopping criteria: max depth reached, min samples reached, or pure node
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(counts) == 1:
            return TreeNode(value=most_common_class, probabilities=probs)

        # Find best split
        best_feat, best_thresh, best_gain = self._best_split(X, y)

        # If no split yields positive information gain, make it a leaf
        if best_feat is None or best_gain <= 0.0:
            return TreeNode(value=most_common_class, probabilities=probs)

        # Accumulate feature importance (weighted gain)
        self.feature_importances_[best_feat] += (n_samples * best_gain)

        # Partition data
        left_X, left_y, right_X, right_y = self._partition(X, y, best_feat, best_thresh)

        # Build subtrees
        left_child = self._grow_tree(left_X, left_y, depth + 1)
        right_child = self._grow_tree(right_X, right_y, depth + 1)

        return TreeNode(
            feature_idx=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child
        )

    def _best_split(
        self,
        X: Sequence[Sequence[Number]],
        y: Sequence[Any]
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find feature and threshold that maximizes impurity reduction."""
        n = len(y)
        current_impurity = self._impurity(y)
        best_gain = 0.0
        best_feat = None
        best_thresh = None

        for feat_idx in range(self.n_features_):
            # Extract distinct values as candidate thresholds
            feat_vals = sorted(list(set(row[feat_idx] for row in X)))
            if len(feat_vals) < 2:
                continue

            # Test midpoints as thresholds
            for i in range(len(feat_vals) - 1):
                thresh = (float(feat_vals[i]) + float(feat_vals[i + 1])) / 2.0

                left_y = [y[idx] for idx, row in enumerate(X) if row[feat_idx] <= thresh]
                right_y = [y[idx] for idx, row in enumerate(X) if row[feat_idx] > thresh]

                if not left_y or not right_y:
                    continue

                # Weighted child impurity
                p_left = len(left_y) / n
                p_right = len(right_y) / n
                child_impurity = p_left * self._impurity(left_y) + p_right * self._impurity(right_y)

                gain = current_impurity - child_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    def _partition(
        self,
        X: Sequence[Sequence[Number]],
        y: Sequence[Any],
        feat_idx: int,
        thresh: float
    ) -> Tuple[List[List[Number]], List[Any], List[List[Number]], List[Any]]:
        """Split X and y into left and right subsets."""
        left_X, left_y = [], []
        right_X, right_y = [], []

        for row, target in zip(X, y):
            if row[feat_idx] <= thresh:
                left_X.append(list(row))
                left_y.append(target)
            else:
                right_X.append(list(row))
                right_y.append(target)

        return left_X, left_y, right_X, right_y

    def predict(self, X: Sequence[Sequence[Number]]) -> List[Any]:
        """Predict classes for samples in X."""
        self._check_is_fitted()
        self._validate_inputs(X, expected_features=self.n_features_)
        return [self._predict_row(row, self.root_) for row in X]

    def _predict_row(self, row: Sequence[Number], node: TreeNode) -> Any:
        if node.is_leaf:
            return node.value
        if row[node.feature_idx] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)

    def predict_proba(self, X: Sequence[Sequence[Number]]) -> List[Dict[Any, float]]:
        """Predict class posterior probabilities for samples in X."""
        self._check_is_fitted()
        self._validate_inputs(X, expected_features=self.n_features_)
        return [self._predict_proba_row(row, self.root_) for row in X]

    def _predict_proba_row(self, row: Sequence[Number], node: TreeNode) -> Dict[Any, float]:
        if node.is_leaf:
            return node.probabilities or {cls: 1.0 if cls == node.value else 0.0 for cls in self.classes_}
        if row[node.feature_idx] <= node.threshold:
            return self._predict_proba_row(row, node.left)
        return self._predict_proba_row(row, node.right)
