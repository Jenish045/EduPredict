"""
Base model interface and validation helpers for EduPredict models.
"""

from abc import ABC, abstractmethod
from typing import Any, Sequence, List, Union, Optional

Number = Union[int, float]
Optional_Sequence = Optional[Sequence[Any]]


class BaseModel(ABC):
    """
    Abstract base class for all Classical ML models in EduPredict.
    Enforces a consistent Scikit-Learn-like fit/predict interface without any dependencies.
    """

    def __init__(self) -> None:
        self._is_fitted: bool = False

    def _check_is_fitted(self) -> None:
        """Check if the model has already been fitted, otherwise raise RuntimeError."""
        if not self._is_fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with training data before using this method."
            )

    def _validate_inputs(
        self,
        X: Sequence[Sequence[Number]],
        y: Optional_Sequence = None,
        expected_features: int = None
    ) -> None:
        """Validate input feature matrix X and target sequence y."""
        if not X or len(X) == 0:
            raise ValueError("Input feature matrix X cannot be empty")
        
        n_features = len(X[0])
        if n_features == 0:
            raise ValueError("Input feature matrix X must have at least one column")

        if expected_features is not None and n_features != expected_features:
            raise ValueError(
                f"Input X has {n_features} features, but model expects {expected_features} features."
            )

        for i, row in enumerate(X):
            if len(row) != n_features:
                raise ValueError(f"Inconsistent number of features at row {i}: expected {n_features}, got {len(row)}")

        if y is not None:
            if len(y) != len(X):
                raise ValueError(
                    f"Number of samples in X ({len(X)}) does not match target y ({len(y)})"
                )

    @abstractmethod
    def fit(self, X: Sequence[Sequence[Number]], y: Sequence[Any]) -> "BaseModel":
        """Fit model on training features X and targets y."""
        pass

    @abstractmethod
    def predict(self, X: Sequence[Sequence[Number]]) -> List[Any]:
        """Predict labels or values for samples in X."""
        pass


