"""
Classification loss functions and analytical gradients implemented from scratch.

Provides:
- binary_cross_entropy (Log Loss): J = -(1/N) * sum(y*log(p) + (1-y)*log(1-p))
- bce_gradient_wrt_logits: (p - y) / N for logistic regression logits
- hinge_loss: (1/N) * sum(max(0, 1 - y * margin)) for linear SVM / margin classifiers
- hinge_subgradient: subgradient with respect to margin predictions

Expected Input Shapes & Conventions:
- y_true: 1D sequence of length N containing true target labels {0, 1} (or {-1, 1} for hinge loss).
- y_prob: 1D sequence of length N with predicted probabilities in [0.0, 1.0].
- y_logits: 1D sequence of length N with unbounded real values z in (-inf, +inf).
"""

from typing import Sequence, List, Union
from edupredict.math.activations import clip_probability, safe_log

Number = Union[int, float]


def _check_classification_inputs(y_true: Sequence[Number], y_pred: Sequence[Number]) -> None:
    """Validate 1D sequences for length match and non-emptiness."""
    if not isinstance(y_true, (list, tuple)) or not isinstance(y_pred, (list, tuple)):
        raise TypeError("y_true and y_pred must be list or tuple sequences")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Inputs cannot be empty")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}")


def binary_cross_entropy(
    y_true: Sequence[Number],
    y_prob: Sequence[Number],
    eps: float = 1e-15
) -> float:
    """
    Binary Cross-Entropy (Log Loss):
    BCE = - (1 / N) * sum_{i=1}^N [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ].
    
    Numerical Safeguards:
    Probabilities are clipped strictly to [eps, 1 - eps] using `clip_probability`
    to guard against undefined log(0) floating-point errors.
    
    Args:
        y_true: Ground truth binary targets {0, 1} of shape (N,).
        y_prob: Model predicted probabilities of shape (N,).
        eps: Small epsilon for clipping bounds (default 1e-15).
        
    Returns:
        Scalar loss float >= 0.0.
    """
    _check_classification_inputs(y_true, y_prob)
    n = len(y_true)
    total_loss = 0.0

    for i in range(n):
        y = float(y_true[i])
        p = clip_probability(float(y_prob[i]), eps=eps)
        total_loss += y * safe_log(p, eps=eps) + (1.0 - y) * safe_log(1.0 - p, eps=eps)

    return -total_loss / float(n)


def bce_gradient_wrt_logits(
    y_true: Sequence[Number],
    y_prob: Sequence[Number]
) -> List[float]:
    """
    Analytical gradient of BCE with respect to linear logits z (where p = sigmoid(z)):
    d(BCE) / dz_i = (1 / N) * (p_i - y_i).
    
    Args:
        y_true: Binary ground truth labels {0, 1}.
        y_prob: Predicted sigmoid probabilities sigma(z).
        
    Returns:
        Gradient vector of shape (N,).
    """
    _check_classification_inputs(y_true, y_prob)
    n = len(y_true)
    return [(float(y_prob[i]) - float(y_true[i])) / float(n) for i in range(n)]


def hinge_loss(
    y_true: Sequence[Number],
    y_margin: Sequence[Number]
) -> float:
    """
    Hinge Loss for max-margin classifiers:
    L = (1 / N) * sum_{i=1}^N max(0, 1 - y_i * margin_i)
    
    If y_true contains {0, 1}, 0 is automatically re-mapped to -1.
    
    Args:
        y_true: Ground truth labels in {-1, 1} (or {0, 1}).
        y_margin: Raw model decision margins / scores (w^T * x + b).
        
    Returns:
        Scalar hinge loss >= 0.0.
    """
    _check_classification_inputs(y_true, y_margin)
    n = len(y_true)
    total_loss = 0.0

    for i in range(n):
        y = float(y_true[i])
        # Auto-map 0 to -1 if standard binary format
        if y == 0.0:
            y = -1.0
        margin = float(y_margin[i])
        loss_i = max(0.0, 1.0 - y * margin)
        total_loss += loss_i

    return total_loss / float(n)


def hinge_subgradient(
    y_true: Sequence[Number],
    y_margin: Sequence[Number]
) -> List[float]:
    """
    Subgradient of Hinge Loss with respect to margin predictions:
    dL / d(margin_i) = -y_i / N if (y_i * margin_i < 1.0) else 0.0.
    """
    _check_classification_inputs(y_true, y_margin)
    n = len(y_true)
    grads: List[float] = []

    for i in range(n):
        y = float(y_true[i])
        if y == 0.0:
            y = -1.0
        margin = float(y_margin[i])
        if y * margin < 1.0:
            grads.append(-y / float(n))
        else:
            grads.append(0.0)

    return grads
