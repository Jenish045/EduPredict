"""
Missing value imputation from scratch.

Supports 'mean', 'median', 'mode', and 'constant' strategies on tabular data.
"""

from typing import List, Any, Optional, Union, Sequence
from edupredict.math.stats import mean, median, mode


class SimpleImputer:
    """
    Impute missing values (represented as None, empty str, or float('nan')).
    
    Strategies:
    - 'mean': numeric columns only
    - 'median': numeric columns only
    - 'mode': works for both categorical and numeric
    - 'constant': replaces with fill_value
    """

    def __init__(self, strategy: str = "mean", fill_value: Any = None) -> None:
        valid_strategies = ["mean", "median", "mode", "constant"]
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {valid_strategies}")
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_: List[Any] = []
        self._is_fitted: bool = False

    def _is_missing(self, val: Any) -> bool:
        if val is None:
            return True
        if isinstance(val, str) and val.strip() == "":
            return True
        if isinstance(val, float) and val != val:  # NaN check without math import
            return True
        return False

    def fit(self, X: Sequence[Sequence[Any]]) -> "SimpleImputer":
        """Calculate imputation values for each column of 2D data X."""
        if not X or len(X) == 0:
            raise ValueError("Cannot fit imputer on empty dataset")
        num_cols = len(X[0])
        self.statistics_ = []

        for col_idx in range(num_cols):
            col_vals = [row[col_idx] for row in X]
            valid_vals = [v for v in col_vals if not self._is_missing(v)]

            if not valid_vals:
                # Fallback if entire column is missing
                self.statistics_.append(0.0 if self.strategy != "constant" else self.fill_value)
                continue

            if self.strategy == "mean":
                self.statistics_.append(mean(valid_vals))
            elif self.strategy == "median":
                self.statistics_.append(median(valid_vals))
            elif self.strategy == "mode":
                self.statistics_.append(mode(valid_vals))
            elif self.strategy == "constant":
                self.statistics_.append(self.fill_value)

        self._is_fitted = True
        return self

    def transform(self, X: Sequence[Sequence[Any]]) -> List[List[Any]]:
        """Impute missing values in X."""
        if not self._is_fitted:
            raise RuntimeError("SimpleImputer is not fitted yet. Call fit() first.")
        if not X or len(X) == 0:
            return []

        num_cols = len(self.statistics_)
        transformed: List[List[Any]] = []

        for row_idx, row in enumerate(X):
            if len(row) != num_cols:
                raise ValueError(f"Row {row_idx} length ({len(row)}) does not match fitted columns ({num_cols})")
            new_row = []
            for col_idx, val in enumerate(row):
                if self._is_missing(val):
                    new_row.append(self.statistics_[col_idx])
                else:
                    new_row.append(val)
            transformed.append(new_row)

        return transformed

    def fit_transform(self, X: Sequence[Sequence[Any]]) -> List[List[Any]]:
        """Fit and transform in a single call."""
        return self.fit(X).transform(X)
