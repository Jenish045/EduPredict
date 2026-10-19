"""
Model validation and cross-validation procedures implemented from scratch.
"""

from typing import Sequence, List, Any, Callable, Union, Optional
import copy
from edupredict.model_selection.split import KFold
from edupredict.metrics.classification import accuracy_score
from edupredict.metrics.regression import r2_score, mean_squared_error


def cross_val_score(
    estimator: Any,
    X: Sequence[Sequence[Any]],
    y: Sequence[Any],
    cv: Union[int, KFold] = 5,
    scoring: Union[str, Callable] = "accuracy"
) -> List[float]:
    """
    Evaluate a score by cross-validation.

    Args:
        estimator: Model object implementing fit() and predict().
        X: Feature matrix.
        y: Target sequence.
        cv: Number of folds (int) or KFold instance.
        scoring: Metric to compute ('accuracy', 'r2', 'mse', or callable metric(y_true, y_pred)).

    Returns:
        List of validation scores across the folds.
    """
    if isinstance(cv, int):
        splitter = KFold(n_splits=cv, shuffle=True, random_seed=42)
    elif isinstance(cv, KFold):
        splitter = cv
    else:
        raise TypeError("cv must be an integer or KFold instance")

    if isinstance(scoring, str):
        if scoring == "accuracy":
            score_func = accuracy_score
        elif scoring == "r2":
            score_func = r2_score
        elif scoring == "mse":
            score_func = mean_squared_error
        else:
            raise ValueError(f"Unknown scoring string '{scoring}'. Supported: 'accuracy', 'r2', 'mse'")
    elif callable(scoring):
        score_func = scoring
    else:
        raise TypeError("scoring must be a string or callable")

    scores: List[float] = []
    splits = splitter.split(X)

    for train_idx, test_idx in splits:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        # Clone estimator (fresh instance with same hyperparameters)
        model = copy.deepcopy(estimator)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = score_func(y_test, preds)
        scores.append(score)

    return scores
