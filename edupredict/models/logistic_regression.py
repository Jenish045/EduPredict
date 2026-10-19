"""
Logistic Regression implemented from first principles.

Provides binary classification for student academic risk prediction
using numerically stable sigmoid activation and binary cross-entropy gradient descent.
"""

import random
from typing import List, Sequence, Optional, Union
from edupredict.models.base import BaseModel
from edupredict.math.vectors import dot_product
from edupredict.math.activations import sigmoid
from edupredict.losses.classification import binary_cross_entropy
from edupredict.optimization.gradient_descent import GradientDescent

Number = Union[int, float]


class LogisticRegression(BaseModel):
    """
    Logistic Regression classifier for binary outcomes (e.g. academic risk = 0 or 1).
    Probability hypothesis: P(y=1|x) = sigmoid(w^T * x + b).
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_epochs: int = 1000,
        batch_size: Optional[int] = None,
        momentum: float = 0.0,
        l2_penalty: float = 0.0,
        early_stopping: bool = False,
        patience: int = 10,
        tolerance: float = 1e-6,
        random_seed: Optional[int] = 42
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.l2_penalty = l2_penalty
        self.early_stopping = early_stopping
        self.patience = patience
        self.tolerance = tolerance
        self.random_seed = random_seed

        self.weights_: List[float] = []
        self.bias_: float = 0.0
        self.loss_history_: List[float] = []
        self.classes_: List[int] = [0, 1]

    def fit(self, X: Sequence[Sequence[Number]], y: Sequence[Number]) -> "LogisticRegression":
        """
        Fit logistic regression model using gradient descent on binary cross-entropy.
        """
        self._validate_inputs(X, y)
        n_samples = len(X)
        n_features = len(X[0])

        # Validate that y is binary 0/1
        y_int = [int(val) for val in y]
        unique_y = sorted(list(set(y_int)))
        if unique_y not in ([0, 1], [0], [1]):
            raise ValueError(f"LogisticRegression currently supports binary 0/1 targets, got classes: {unique_y}")
        self.classes_ = unique_y

        rng = random.Random(self.random_seed)
        self.weights_ = [rng.gauss(0.0, 0.01) for _ in range(n_features)]
        self.bias_ = 0.0
        self.loss_history_ = []
        self._is_fitted = True

        optimizer = GradientDescent(
            learning_rate=self.learning_rate,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            momentum=self.momentum,
            tolerance=self.tolerance,
            early_stopping=self.early_stopping,
            patience=self.patience,
            random_seed=self.random_seed
        )
        optimizer.init_velocity(n_features)

        for epoch in range(self.max_epochs):
            batches = optimizer.get_batches(n_samples, rng)

            for batch_indices in batches:
                m_b = len(batch_indices)
                grad_w = [0.0] * n_features
                grad_b = 0.0

                for idx in batch_indices:
                    x_i = X[idx]
                    y_i = float(y_int[idx])
                    z = dot_product(self.weights_, x_i) + self.bias_
                    p_i = sigmoid(z)
                    error = p_i - y_i

                    for j in range(n_features):
                        grad_w[j] += error * float(x_i[j])
                    grad_b += error

                for j in range(n_features):
                    grad_w[j] = (1.0 / m_b) * grad_w[j]
                    if self.l2_penalty > 0.0:
                        grad_w[j] += (self.l2_penalty * self.weights_[j])

                grad_b = (1.0 / m_b) * grad_b

                self.weights_, self.bias_ = optimizer.update(
                    self.weights_, self.bias_, grad_w, grad_b
                )

            # Record epoch BCE loss
            probs = self.predict_proba(X)
            epoch_loss = binary_cross_entropy(y_int, probs)
            self.loss_history_.append(epoch_loss)

            if len(self.loss_history_) > 1 and optimizer.check_convergence(self.loss_history_[-2], epoch_loss):
                break

            if optimizer.should_stop_early(epoch_loss, self.weights_, self.bias_):
                if optimizer.best_weights is not None:
                    self.weights_ = list(optimizer.best_weights)
                    self.bias_ = optimizer.best_bias
                break

        self._is_fitted = True
        return self

    def predict_proba(self, X: Sequence[Sequence[Number]]) -> List[float]:
        """
        Estimate posterior probability P(y=1|x) for each sample.
        """
        self._check_is_fitted()
        self._validate_inputs(X, expected_features=len(self.weights_))
        probs = []
        for row in X:
            z = dot_product(self.weights_, row) + self.bias_
            probs.append(sigmoid(z))
        return probs

    def predict(
        self,
        X: Sequence[Sequence[Number]],
        threshold: float = 0.5
    ) -> List[int]:
        """
        Predict binary class 0 or 1 using decision threshold.
        """
        probs = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probs]
