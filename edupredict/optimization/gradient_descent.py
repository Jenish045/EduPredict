"""
Gradient Descent optimization algorithms implemented from scratch.

Provides:
- GradientDescent: Supports Batch GD, Mini-batch GD, and Stochastic GD (SGD),
  with parameter initialization schemes, loss history tracking, momentum,
  convergence diagnostics, and early stopping.
Strictly zero third-party dependencies (no autograd / PyTorch / SciPy).
"""

import math
import random
from typing import List, Tuple, Optional, Sequence


def initialize_parameters(
    n_features: int,
    method: str = "random",
    seed: Optional[int] = 42
) -> Tuple[List[float], float]:
    """
    Initialize weight vector and bias term from scratch.
    
    Methods:
    - 'zeros': all weights = 0.0, bias = 0.0
    - 'random': Gaussian N(0, 0.01)
    - 'xavier': Gaussian N(0, sqrt(2 / (n_features + 1)))
    """
    if n_features <= 0:
        raise ValueError(f"n_features must be positive integer, got {n_features}")

    rng = random.Random(seed) if seed is not None else random.Random()

    if method == "zeros":
        return [0.0] * n_features, 0.0
    elif method == "random":
        weights = [rng.gauss(0.0, 0.01) for _ in range(n_features)]
        return weights, 0.0
    elif method == "xavier":
        std = math.sqrt(2.0 / float(n_features + 1))
        weights = [rng.gauss(0.0, std) for _ in range(n_features)]
        return weights, 0.0
    else:
        raise ValueError(f"Unknown initialization method '{method}'. Choose 'zeros', 'random', or 'xavier'")


class GradientDescent:
    """
    Gradient descent optimizer for classical linear and logistic models.
    
    Modes:
    - Batch GD: batch_size = None (evaluates all training samples in each step)
    - Mini-Batch GD: batch_size = 16, 32, etc.
    - Stochastic GD (SGD): batch_size = 1
    
    Features:
    - Momentum acceleration (Polyak momentum)
    - Loss tolerance convergence checks
    - Early stopping with configurable patience and min_delta
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_epochs: int = 1000,
        batch_size: Optional[int] = None,
        tolerance: float = 1e-6,
        momentum: float = 0.0,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 1e-5,
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
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.momentum = momentum
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.random_seed = random_seed

        # Velocity buffers for momentum
        self.velocity_weights: List[float] = []
        self.velocity_bias: float = 0.0

        # Early stopping state
        self.best_loss: float = float("inf")
        self.patience_counter: int = 0
        self.best_weights: Optional[List[float]] = None
        self.best_bias: Optional[float] = None

    def init_velocity(self, n_features: int) -> None:
        """Initialize momentum velocities to zero."""
        self.velocity_weights = [0.0] * n_features
        self.velocity_bias = 0.0
        self.best_loss = float("inf")
        self.patience_counter = 0

    def update(
        self,
        weights: List[float],
        bias: float,
        grad_weights: List[float],
        grad_bias: float
    ) -> Tuple[List[float], float]:
        """
        Perform a single parameter update step with momentum:
        v_w = momentum * v_w + lr * grad_w
        w = w - v_w
        v_b = momentum * v_b + lr * grad_b
        b = b - v_b
        """
        if not self.velocity_weights or len(self.velocity_weights) != len(weights):
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

    def check_convergence(self, prev_loss: float, current_loss: float) -> bool:
        """
        Check if change in loss is smaller than tolerance threshold:
        |L_{t-1} - L_t| < tolerance.
        """
        diff = abs(prev_loss - current_loss)
        return diff < self.tolerance

    def should_stop_early(
        self,
        current_loss: float,
        weights: List[float],
        bias: float
    ) -> bool:
        """
        Check whether early stopping condition is met.
        Returns True if loss fails to improve by at least min_delta for `patience` consecutive epochs.
        """
        if not self.early_stopping:
            return False

        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.patience_counter = 0
            self.best_weights = list(weights)
            self.best_bias = bias
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
            return False

    def get_batches(
        self,
        n_samples: int,
        rng: random.Random
    ) -> List[List[int]]:
        """
        Generate sample index batches for each epoch.
        Full batch if batch_size is None or >= n_samples.
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
