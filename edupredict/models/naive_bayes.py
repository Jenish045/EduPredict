"""
Gaussian Naive Bayes classifier implemented from first principles.

Calculates class priors, feature means, and feature variances,
and evaluates Gaussian likelihoods in log-space for numerical stability.
"""

import math
from typing import List, Sequence, Union, Any, Dict
from edupredict.models.base import BaseModel
from edupredict.math.stats import mean, variance
from edupredict.math.activations import softmax

Number = Union[int, float]


class GaussianNaiveBayes(BaseModel):
    """
    Gaussian Naive Bayes for continuous feature classification.
    Assumes features within each class follow independent Gaussian distributions.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        super().__init__()
        self.var_smoothing = var_smoothing
        self.classes_: List[Any] = []
        self.class_priors_: Dict[Any, float] = {}
        self.means_: Dict[Any, List[float]] = {}
        self.variances_: Dict[Any, List[float]] = {}
        self.n_features_: int = 0

    def fit(self, X: Sequence[Sequence[Number]], y: Sequence[Any]) -> "GaussianNaiveBayes":
        """Compute mean, variance, and prior for each class."""
        self._validate_inputs(X, y)
        self.n_features_ = len(X[0])
        n_samples = len(y)

        self.classes_ = sorted(list(set(y)), key=lambda c: str(c))
        self.class_priors_ = {}
        self.means_ = {}
        self.variances_ = {}

        for c in self.classes_:
            # Extract samples belonging to class c
            X_c = [X[i] for i in range(n_samples) if y[i] == c]
            n_c = len(X_c)
            self.class_priors_[c] = n_c / float(n_samples)

            means_c = []
            vars_c = []
            for j in range(self.n_features_):
                col_vals = [float(row[j]) for row in X_c]
                mu = mean(col_vals)
                # Compute sample variance if n_c >= 2 else 0.0
                var = variance(col_vals, sample=False) if n_c >= 2 else 0.0
                # Add smoothing variance to prevent division by zero
                var += self.var_smoothing
                means_c.append(mu)
                vars_c.append(var)

            self.means_[c] = means_c
            self.variances_[c] = vars_c

        self._is_fitted = True
        return self

    def _calculate_log_likelihood(self, x: Sequence[Number], c: Any) -> float:
        """
        Compute log P(c) + sum_j log P(x_j | c) using the Gaussian probability density formula:
        log P(x_j | c) = -0.5 * log(2*pi*sigma^2) - ((x_j - mu)^2 / (2*sigma^2))
        """
        log_prior = math.log(self.class_priors_[c])
        log_likelihood = log_prior

        for j in range(self.n_features_):
            mu = self.means_[c][j]
            var = self.variances_[c][j]
            x_j = float(x[j])

            # Gaussian log-density
            log_prob_j = -0.5 * math.log(2.0 * math.pi * var) - ((x_j - mu) ** 2) / (2.0 * var)
            log_likelihood += log_prob_j

        return log_likelihood

    def predict_log_proba(self, X: Sequence[Sequence[Number]]) -> List[Dict[Any, float]]:
        """Return log likelihood for each class."""
        self._check_is_fitted()
        self._validate_inputs(X, expected_features=self.n_features_)
        results = []
        for row in X:
            row_log_probs = {c: self._calculate_log_likelihood(row, c) for c in self.classes_}
            results.append(row_log_probs)
        return results

    def predict_proba(self, X: Sequence[Sequence[Number]]) -> List[List[float]]:
        """Return normalized class probabilities using softmax over log-likelihoods."""
        self._check_is_fitted()
        self._validate_inputs(X, expected_features=self.n_features_)
        probs = []
        for row in X:
            log_vals = [self._calculate_log_likelihood(row, c) for c in self.classes_]
            sm = softmax(log_vals)
            probs.append(sm)
        return probs

    def predict(self, X: Sequence[Sequence[Number]]) -> List[Any]:
        """Predict class with the highest posterior probability."""
        self._check_is_fitted()
        self._validate_inputs(X, expected_features=self.n_features_)
        predictions = []
        for row in X:
            best_class = None
            best_log_likelihood = -float("inf")
            for c in self.classes_:
                ll = self._calculate_log_likelihood(row, c)
                if ll > best_log_likelihood:
                    best_log_likelihood = ll
                    best_class = c
            predictions.append(best_class)
        return predictions
