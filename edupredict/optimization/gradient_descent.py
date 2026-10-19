"""
Gradient Descent optimization algorithms implemented from scratch.

Provides:
- GradientDescent: Supports Batch GD, Mini-batch GD, and Stochastic GD (SGD),
  with optional momentum and configurable learning rate.
"""

import random
from typing import List, Tuple, Optional, Sequence


class GradientDescent:
    """
    Gradient descent optimizer for linear and logistic models.
    
    Modes:
    - Batch GD: batch_size = None (uses all samples each step)
    - Mini-Batch GD: batch_size = 16, 32, etc.
    - Stochastic GD (SGD): batch_size = 1
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_epochs: int = 1000,
        batch_size: Optional[int] = None,
        tolerance: float = 1e-6,
        momentum: float = 0.0,
        random_seed: Optional[int] = 42
    ) -> None:
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {learning_rate}")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0.0, 1.0), got {momentum}")

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.momentum = momentum
        self.random_seed = random_seed

        # Velocity buffers for momentum
        self.velocity_weights: List[float] = []
        self.velocity_bias: float = 0.0

    def init_velocity(self, n_features: int) -> None:
        """Initialize momentum velocities."""
        self.velocity_weights = [0.0] * n_features
        self.velocity_bias = 0.0

    def update(
        self,
        weights: List[float],
        bias: float,
        grad_weights: List[float],
        grad_bias: float
    ) -> Tuple[List[float], float]:
        """
        Perform a single parameter update step with optional momentum:
        v_w = momentum * v_w + lr * grad_w
        w = w - v_w
        v_b = momentum * v_b + lr * grad_b
        b = b - v_b
        """
        if not self.velocity_weights:
            self.init_velocity(len(weights))

        new_weights = list(weights)
        for j in range(len(weights)):
            self.velocity_weights[j] = (
                self.momentum * self.velocity_weights[j] + self.learning_rate * grad_weights[j]
            )
            new_weights[j] -= self.velocity_weights[j]

        self.velocity_bias = (
            self.momentum * self.velocity_bias + self.learning_rate * grad_bias
        )
        new_bias = bias - self.velocity_bias

        return new_weights, new_bias

    def get_batches(
        self,
        n_samples: int,
        rng: random.Random
    ) -> List[List[int]]:
        """
        Generate sample index batches for each epoch.
        """
        indices = list(range(n_samples))
        if self.batch_size is None or self.batch_size >= n_samples:
            return [indices]

        # Shuffle indices for SGD / Mini-Batch
        rng.shuffle(indices)
        batches: List[List[int]] = []
        for start_idx in range(0, n_samples, self.batch_size):
            batches.append(indices[start_idx : start_idx + self.batch_size])
        return batches
