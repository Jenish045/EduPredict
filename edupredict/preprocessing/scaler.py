"""
Feature scalers implemented from scratch.

Provides:
- StandardScaler: Standardize features by removing mean and scaling to unit variance: z = (x - mu) / sigma.
- MinMaxScaler: Scale features to a given range [min_range, max_range], typically [0, 1].
"""

from typing import List, Sequence, Tuple, Union
from edupredict.math.stats import mean, standard_deviation, min_val, max_val

Number = Union[int, float]


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    z = (x - mu) / sigma.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: List[float] = []
        self.scale_: List[float] = []
        self._is_fitted: bool = False

    def fit(self, X: Sequence[Sequence[Number]]) -> "StandardScaler":
        """Compute the mean and std to be used for later scaling."""
        if not X or len(X) == 0:
            raise ValueError("Cannot fit StandardScaler on empty matrix")
        num_cols = len(X[0])
        self.mean_ = []
        self.scale_ = []

        for c in range(num_cols):
            col_vals = [float(row[c]) for row in X]
            col_mean = mean(col_vals)
            col_std = standard_deviation(col_vals) if len(col_vals) > 1 else 0.0

            # Guard against division by zero if feature has zero variance
            if col_std < 1e-12:
                col_std = 1.0

            self.mean_.append(col_mean)
            self.scale_.append(col_std)

        self._is_fitted = True
        return self

    def transform(self, X: Sequence[Sequence[Number]]) -> List[List[float]]:
        """Perform standardization by centering and scaling."""
        if not self._is_fitted:
            raise RuntimeError("StandardScaler is not fitted yet.")
        num_cols = len(self.mean_)
        transformed: List[List[float]] = []

        for row_idx, row in enumerate(X):
            if len(row) != num_cols:
                raise ValueError(f"Row {row_idx} length ({len(row)}) does not match fitted features ({num_cols})")
            new_row = []
            for c in range(num_cols):
                val = float(row[c])
                if self.with_mean:
                    val -= self.mean_[c]
                if self.with_std:
                    val /= self.scale_[c]
                new_row.append(val)
            transformed.append(new_row)

        return transformed

    def fit_transform(self, X: Sequence[Sequence[Number]]) -> List[List[float]]:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Sequence[Sequence[Number]]) -> List[List[float]]:
        """Scale back the data to original representation."""
        if not self._is_fitted:
            raise RuntimeError("StandardScaler is not fitted yet.")
        num_cols = len(self.mean_)
        original: List[List[float]] = []

        for row_idx, row in enumerate(X):
            if len(row) != num_cols:
                raise ValueError(f"Row {row_idx} length ({len(row)}) does not match fitted features ({num_cols})")
            new_row = []
            for c in range(num_cols):
                val = float(row[c])
                if self.with_std:
                    val *= self.scale_[c]
                if self.with_mean:
                    val += self.mean_[c]
                new_row.append(val)
            original.append(new_row)

        return original


class MinMaxScaler:
    """
    Transform features by scaling each feature to a given range, default [0, 1].
    x_scaled = (x - min) / (max - min) * (max_range - min_range) + min_range.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        if feature_range[0] >= feature_range[1]:
            raise ValueError(f"Invalid feature_range {feature_range}: min must be less than max")
        self.feature_range = feature_range
        self.data_min_: List[float] = []
        self.data_max_: List[float] = []
        self.data_range_: List[float] = []
        self._is_fitted: bool = False

    def fit(self, X: Sequence[Sequence[Number]]) -> "MinMaxScaler":
        """Compute the minimum and maximum to be used for later scaling."""
        if not X or len(X) == 0:
            raise ValueError("Cannot fit MinMaxScaler on empty matrix")
        num_cols = len(X[0])
        self.data_min_ = []
        self.data_max_ = []
        self.data_range_ = []

        for c in range(num_cols):
            col_vals = [float(row[c]) for row in X]
            mn = min_val(col_vals)
            mx = max_val(col_vals)
            rng = mx - mn
            if rng < 1e-12:
                rng = 1.0  # Avoid zero division for constant feature

            self.data_min_.append(mn)
            self.data_max_.append(mx)
            self.data_range_.append(rng)

        self._is_fitted = True
        return self

    def transform(self, X: Sequence[Sequence[Number]]) -> List[List[float]]:
        """Scale features of X according to feature_range."""
        if not self._is_fitted:
            raise RuntimeError("MinMaxScaler is not fitted yet.")
        num_cols = len(self.data_min_)
        r_min, r_max = self.feature_range
        r_span = r_max - r_min

        transformed: List[List[float]] = []
        for row_idx, row in enumerate(X):
            if len(row) != num_cols:
                raise ValueError(f"Row {row_idx} length ({len(row)}) does not match fitted features ({num_cols})")
            new_row = []
            for c in range(num_cols):
                std_norm = (float(row[c]) - self.data_min_[c]) / self.data_range_[c]
                val = std_norm * r_span + r_min
                new_row.append(val)
            transformed.append(new_row)

        return transformed

    def fit_transform(self, X: Sequence[Sequence[Number]]) -> List[List[float]]:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Sequence[Sequence[Number]]) -> List[List[float]]:
        """Undo the scaling of X according to feature_range."""
        if not self._is_fitted:
            raise RuntimeError("MinMaxScaler is not fitted yet.")
        num_cols = len(self.data_min_)
        r_min, r_max = self.feature_range
        r_span = r_max - r_min

        original: List[List[float]] = []
        for row_idx, row in enumerate(X):
            if len(row) != num_cols:
                raise ValueError(f"Row {row_idx} length ({len(row)}) does not match fitted features ({num_cols})")
            new_row = []
            for c in range(num_cols):
                std_norm = (float(row[c]) - r_min) / r_span
                val = std_norm * self.data_range_[c] + self.data_min_[c]
                new_row.append(val)
            original.append(new_row)

        return original
