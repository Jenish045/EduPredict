"""
Base model interface for EduPredict custom machine learning models.

Strictly zero third-party dependencies and zero external library imports.
Defines common fit/predict interfaces, input validation, and serialization contracts.
"""

from abc import ABC, abstractmethod
from typing import List, Sequence, Any, Optional, Dict, Union

Number = Union[int, float]
Matrix = Sequence[Sequence[Number]]
Vector = Sequence[Any]


def _is_nan(val: Any) -> bool:
    """Check if value is NaN without importing math."""
    return isinstance(val, (int, float)) and val != val


def _is_inf(val: Any) -> bool:
    """Check if value is infinity without importing math."""
    return val == float("inf") or val == float("-inf")


class BaseModel(ABC):
    """
    Abstract base class for all EduPredict estimators.
    """

    def __init__(self) -> None:
        self.is_fitted: bool = False
        self.n_features_in_: Optional[int] = None

    def _validate_inputs(
        self,
        X: Matrix,
        y: Optional[Vector] = None,
        check_y: bool = True,
    ) -> None:
        """
        Validate input matrix X and optional target vector y.
        Raises ValueError or TypeError on invalid dimensions, empty data, or NaN/Inf values.
        """
        if not isinstance(X, (list, tuple)) or len(X) == 0:
            raise ValueError("Input feature matrix X must be a non-empty sequence")

        n_samples = len(X)
        if not isinstance(X[0], (list, tuple)) or len(X[0]) == 0:
            raise ValueError("Feature rows in X must be non-empty sequences")

        n_features = len(X[0])

        for r_idx, row in enumerate(X):
            if not isinstance(row, (list, tuple)):
                raise TypeError(f"Row {r_idx} must be a list or tuple, got {type(row).__name__}")
            if len(row) != n_features:
                raise ValueError(
                    f"Inconsistent feature count at row {r_idx}: expected {n_features}, got {len(row)}"
                )
            for c_idx, val in enumerate(row):
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    raise TypeError(
                        f"X[{r_idx}][{c_idx}] must be a float or int, got {type(val).__name__}"
                    )
                if _is_nan(val) or _is_inf(val):
                    raise ValueError(f"X[{r_idx}][{c_idx}] contains invalid numerical value: {val}")

        if check_y and y is not None:
            if not isinstance(y, (list, tuple)):
                raise TypeError(f"Target vector y must be a list or tuple, got {type(y).__name__}")
            if len(y) != n_samples:
                raise ValueError(
                    f"Sample count mismatch: X has {n_samples} rows, but y has {len(y)} elements"
                )
            for i, val in enumerate(y):
                if _is_nan(val) or _is_inf(val):
                    raise ValueError(f"y[{i}] contains invalid numerical value: {val}")

    def _check_is_fitted(self) -> None:
        """Raise an error if the model has not been fitted yet."""
        if not self.is_fitted:
            raise ValueError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' first.")

    @abstractmethod
    def fit(self, X: Matrix, y: Vector) -> "BaseModel":
        """Fit model to training data."""
        pass

    @abstractmethod
    def predict(self, X: Matrix) -> List[Any]:
        """Generate predictions for feature matrix X."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize model parameters and hyperparameters to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseModel":
        """Instantiate and restore model from serialized dictionary."""
        pass
