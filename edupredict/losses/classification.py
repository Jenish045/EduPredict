"""
Classification loss functions and gradients implemented from scratch.

Provides:
- binary_cross_entropy (log loss): J = -(1/N) * sum(y*log(p) + (1-y)*log(1-p))
- bce_gradient_wrt_logits: (p - y) / N for logistic regression logit z
- bce_gradient_wrt_probs: gradient with respect to predicted probability p
"""

import math
from typing import Sequence, List, Union
from edupredict.math.activations import clip_probability, safe_log

Number = Union[int, float]


def _check_classification_inputs(y_true: Sequence[Number], y_pred: Sequence[Number]) -> None:
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
    
    Probabilities are clipped to [eps, 1 - eps] to guarantee numerical stability.
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
    Gradient of Binary Cross-Entropy with respect to linear logits z (where p = sigmoid(z)):
    d(BCE) / dz_i = (1 / N) * (p_i - y_i).
    """
    _check_classification_inputs(y_true, y_prob)
    n = len(y_true)
    return [(float(y_prob[i]) - float(y_true[i])) / float(n) for i in range(n)]
