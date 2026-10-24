"""
Gradient descent optimizer implementation from first principles.

Strictly zero third-party dependencies and zero external math module imports.
Supports Batch, Mini-batch, and Stochastic gradient updates, Polyak momentum,
and early stopping based on convergence criteria.
"""

import random
from typing import List, Tuple, Sequence, Optional, Union

Number = Union[int, float]


def _is_nan(val: Number) -> bool:
    """Check if value is NaN without importing math."""
    return val != val


def initialize_parameters(
    n_features: int,
    method: str = "zeros",
    seed: Optional[int] = None,
) -> Tuple[List[float], float]:
    """
    Initialize model weights and bias.

    Supported methods: 'zeros', 'random', 'xavier'.
    """
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")

    rng = random.Random(seed)
    bias = 0.0

    if method == "zeros":
        weights = [0.0] * n_features
    elif method == "random":
        weights = [rng.uniform(-0.05, 0.05) for _ in range(n_features)]
    elif method == "xavier":
        limit = (2.0 / n_features) ** 0.5
        weights = [rng.uniform(-limit, limit) for _ in range(n_features)]
    else:
        raise ValueError(f"Unsupported initialization method: {method}")

    return weights, bias


class GradientDescent:
    """
    First-order gradient descent optimizer for parameter updates.
    Hypothesis: theta_{t+1} = theta_t - alpha * v_t
    where v_t = beta * v_{t-1} + grad_t (with Polyak momentum).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_epochs: int = 1000,
        batch_size: Optional[int] = None,
        momentum: float = 0.0,
        tolerance: float = 1e-6,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 1e-4,
        random_seed: Optional[int] = 42,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"batch_size must be positive or None, got {batch_size}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0.0, 1.0), got {momentum}")
        if tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.tolerance = tolerance
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.random_seed = random_seed

        # Internal state
        self.velocity_w: List[float] = []
        self.velocity_b: float = 0.0
        self.best_loss: float = float("inf")
        self.patience_counter: int = 0
        self.best_weights: Optional[List[float]] = None
        self.best_bias: Optional[float] = None

    def init_velocity(self, n_features: int) -> None:
        """Initialize momentum velocities to zero vectors."""
        self.velocity_w = [0.0] * n_features
        self.velocity_b = 0.0
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.best_weights = None
        self.best_bias = None

    def update(
        self,
        weights: List[float],
        bias: float,
        grad_w: Sequence[float],
        grad_b: float,
    ) -> Tuple[List[float], float]:
        """
        Apply gradient update with momentum:
        v_w = beta * v_w + grad_w
        w = w - alpha * v_w
        """
        n_features = len(weights)
        if len(self.velocity_w) != n_features:
            self.init_velocity(n_features)

        new_weights: List[float] = [0.0] * n_features
        for j in range(n_features):
            self.velocity_w[j] = self.momentum * self.velocity_w[j] + float(grad_w[j])
            new_weights[j] = float(weights[j]) - self.learning_rate * self.velocity_w[j]

        self.velocity_b = self.momentum * self.velocity_b + float(grad_b)
        new_bias = float(bias) - self.learning_rate * self.velocity_b

        return new_weights, new_bias

    def get_batches(self, n_samples: int, rng: random.Random) -> List[List[int]]:
        """Generate sample indices for full-batch or mini-batch training."""
        indices = list(range(n_samples))
        if self.batch_size is None or self.batch_size >= n_samples:
            return [indices]

        rng.shuffle(indices)
        batches: List[List[int]] = []
        for i in range(0, n_samples, self.batch_size):
            batches.append(indices[i : i + self.batch_size])
        return batches

    def check_convergence(self, prev_loss: float, current_loss: float) -> bool:
        """Check if loss decrease is below tolerance threshold."""
        if _is_nan(current_loss):
            return True
        diff = prev_loss - current_loss
        diff_abs = diff if diff >= 0.0 else -diff
        return diff_abs < self.tolerance

    def should_stop_early(self, current_loss: float, weights: List[float], bias: float) -> bool:
        """Check if early stopping patience has been exhausted."""
        if not self.early_stopping:
            return False

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_weights = list(weights)
            self.best_bias = bias
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
            return False
