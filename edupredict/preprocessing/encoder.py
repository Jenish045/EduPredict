"""
Categorical encoding from scratch.

Provides:
- LabelEncoder: encodes 1D sequence of categorical labels into integers [0, K-1].
- OneHotEncoder: converts categorical columns in a 2D matrix into binary indicator columns.
"""

from typing import List, Dict, Any, Sequence, Optional, Union


class LabelEncoder:
    """
    Encode target labels or single categorical features with value between 0 and n_classes-1.
    """

    def __init__(self) -> None:
        self.classes_: List[Any] = []
        self.class_to_idx_: Dict[Any, int] = {}
        self.idx_to_class_: Dict[int, Any] = {}
        self._is_fitted: bool = False

    def fit(self, y: Sequence[Any]) -> "LabelEncoder":
        """Learn unique classes from y."""
        if not y or len(y) == 0:
            raise ValueError("Cannot fit LabelEncoder on empty sequence")
        
        # Sort classes for deterministic integer assignment
        unique_classes = sorted(list(set(y)), key=lambda x: str(x))
        self.classes_ = unique_classes
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.idx_to_class_ = {idx: cls for idx, cls in enumerate(unique_classes)}
        self._is_fitted = True
        return self

    def transform(self, y: Sequence[Any]) -> List[int]:
        """Transform labels to normalized encodings."""
        if not self._is_fitted:
            raise RuntimeError("LabelEncoder is not fitted yet.")
        encoded = []
        for val in y:
            if val not in self.class_to_idx_:
                raise ValueError(f"Unseen label '{val}' encountered during transform. Known classes: {self.classes_}")
            encoded.append(self.class_to_idx_[val])
        return encoded

    def fit_transform(self, y: Sequence[Any]) -> List[int]:
        """Fit label encoder and return encoded labels."""
        return self.fit(y).transform(y)

    def inverse_transform(self, y_encoded: Sequence[int]) -> List[Any]:
        """Transform integer labels back to original classes."""
        if not self._is_fitted:
            raise RuntimeError("LabelEncoder is not fitted yet.")
        return [self.idx_to_class_[idx] for idx in y_encoded]


class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric matrix.
    Optionally drop first category to avoid multicollinearity (dummy variable trap).
    """

    def __init__(self, drop_first: bool = False) -> None:
        self.drop_first = drop_first
        self.categories_: List[List[Any]] = []
        self._is_fitted: bool = False

    def fit(self, X: Sequence[Sequence[Any]]) -> "OneHotEncoder":
        """Fit OneHotEncoder to 2D matrix X."""
        if not X or len(X) == 0:
            raise ValueError("Cannot fit OneHotEncoder on empty matrix")
        num_cols = len(X[0])
        self.categories_ = []

        for c in range(num_cols):
            vals = [row[c] for row in X]
            cats = sorted(list(set(vals)), key=lambda x: str(x))
            if self.drop_first and len(cats) > 1:
                cats = cats[1:]
            self.categories_.append(cats)

        self._is_fitted = True
        return self

    def transform(self, X: Sequence[Sequence[Any]]) -> List[List[float]]:
        """Transform X to one-hot encoded numeric vectors."""
        if not self._is_fitted:
            raise RuntimeError("OneHotEncoder is not fitted yet.")
        if not X or len(X) == 0:
            return []

        encoded_rows: List[List[float]] = []
        num_cols = len(self.categories_)

        for row in X:
            if len(row) != num_cols:
                raise ValueError(f"Input row length ({len(row)}) does not match fitted columns ({num_cols})")
            encoded_row: List[float] = []
            for col_idx, val in enumerate(row):
                cats = self.categories_[col_idx]
                for cat in cats:
                    encoded_row.append(1.0 if val == cat else 0.0)
            encoded_rows.append(encoded_row)

        return encoded_rows

    def fit_transform(self, X: Sequence[Sequence[Any]]) -> List[List[float]]:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
