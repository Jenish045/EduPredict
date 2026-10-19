"""
Multiclass One-vs-Rest (OvR) Classifier implemented from scratch.

Allows using any binary classifier (such as LogisticRegression) to predict
multi-category targets (e.g. Letter Grades: A, B, C, D, F, or Academic Risk Bands).
"""

from typing import List, Sequence, Union, Any, Dict, Type, Optional, Tuple
from edupredict.models.base import BaseModel
from edupredict.models.logistic_regression import LogisticRegression
from edupredict.math.activations import softmax

Number = Union[int, float]


class OneVsRestClassifier(BaseModel):
    """
    One-vs-Rest (OvR / One-vs-All) Multiclass Meta-Classifier.

    Trains K independent binary classifiers for K unique classes.
    Predicts the class with the highest probability score.
    """

    def __init__(
        self,
        base_estimator_cls: Optional[Type[BaseModel]] = None,
        estimator_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        self.base_estimator_cls = base_estimator_cls or LogisticRegression
        self.estimator_kwargs = estimator_kwargs or {}
        self.classes_: List[Any] = []
        self.estimators_: List[Tuple[Any, BaseModel]] = []

    def fit(self, X: Sequence[Sequence[Number]], y: Sequence[Any]) -> "OneVsRestClassifier":
        """
        Fit K binary classifiers, one for each class vs all other classes.
        """
        self._validate_inputs(X, y)
        self.classes_ = sorted(list(set(y)), key=lambda c: str(c))
        if len(self.classes_) < 2:
            raise ValueError(f"OneVsRestClassifier requires at least 2 classes, got {len(self.classes_)}")

        self.estimators_ = []
        n_samples = len(y)

        for c in self.classes_:
            # Create binary target: 1 if y == c else 0
            binary_y = [1 if y[i] == c else 0 for i in range(n_samples)]
            estimator = self.base_estimator_cls(**self.estimator_kwargs)
            estimator.fit(X, binary_y)
            self.estimators_.append((c, estimator))

        self._is_fitted = True
        return self

    def predict_proba(self, X: Sequence[Sequence[Number]]) -> List[Dict[Any, float]]:
        """
        Compute normalized posterior class probabilities for each sample.
        """
        self._check_is_fitted()
        n_samples = len(X)
        all_class_probs: List[Dict[Any, float]] = []

        # Collect raw positive-class probabilities for each estimator
        raw_probs: List[List[float]] = []
        for c, estimator in self.estimators_:
            if hasattr(estimator, "predict_proba"):
                p_c = estimator.predict_proba(X)
                # If estimator returned list of dicts or scalars:
                if isinstance(p_c, list) and len(p_c) > 0 and isinstance(p_c[0], dict):
                    raw_probs.append([d.get(1, 0.0) for d in p_c])
                else:
                    raw_probs.append([float(v) for v in p_c])
            else:
                # Fallback to decision scores / predictions
                preds = estimator.predict(X)
                raw_probs.append([1.0 if v == 1 else 0.0 for v in preds])

        # Transpose raw_probs to (n_samples, n_classes) and normalize
        for i in range(n_samples):
            sample_scores = [raw_probs[k][i] for k in range(len(self.classes_))]
            total = sum(sample_scores)
            if total > 1e-12:
                norm_probs = [s / total for s in sample_scores]
            else:
                # Uniform fallback if all scores are zero
                norm_probs = [1.0 / len(self.classes_)] * len(self.classes_)

            sample_dict = {self.classes_[k]: norm_probs[k] for k in range(len(self.classes_))}
            all_class_probs.append(sample_dict)

        return all_class_probs

    def predict(self, X: Sequence[Sequence[Number]]) -> List[Any]:
        """
        Predict class with highest probability score.
        """
        self._check_is_fitted()
        probs = self.predict_proba(X)
        predictions = []
        for sample_dict in probs:
            best_class = max(sample_dict.keys(), key=lambda c: sample_dict[c])
            predictions.append(best_class)
        return predictions
