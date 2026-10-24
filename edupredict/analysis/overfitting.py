"""
Overfitting and underfitting diagnostics through model complexity curves.

Strictly zero third-party dependencies and zero external library imports.
Generates training vs validation evaluation curves across key complexity hyperparameters:
- KNN: k (neighborhood size)
- Decision Tree: max_depth (tree complexity)
- Linear Regression: l2_reg (regularization strength)
"""

from typing import List, Sequence, Dict, Any, Optional, Union, Tuple
from collections import Counter

from edupredict.models.knn import KNNClassifier
from edupredict.models.decision_tree import DecisionTreeClassifier
from edupredict.models.linear_regression import LinearRegression
from edupredict.math.vectors import euclidean_distance, manhattan_distance
from edupredict.metrics.classification import accuracy_score

Number = Union[int, float]
Matrix = Sequence[Sequence[Number]]
Vector = Sequence[Any]


def analyze_knn_complexity(
    X_train: Matrix,
    y_train: Vector,
    X_val: Matrix,
    y_val: Vector,
    k_range: Optional[List[int]] = None,
    metric: str = "euclidean",
    weights: str = "uniform",
) -> Dict[str, Any]:
    """
    Generate train vs validation accuracy curves across neighbor count k.

    Efficiently pre-computes pairwise Minkowski distances once, then slices
    the top-k nearest neighbors for each complexity evaluation.
    """
    X_tr = [[float(v) for v in row] for row in X_train]
    y_tr = list(y_train)
    X_va = [[float(v) for v in row] for row in X_val]
    y_va = list(y_val)

    n_tr = len(X_tr)
    n_va = len(X_va)

    unique_classes = list(set(y_tr))
    try:
        classes = sorted(unique_classes)
    except TypeError:
        classes = sorted(unique_classes, key=lambda x: str(x))

    if k_range is None:
        max_k = min(31, n_tr)
        k_range = [k for k in range(1, max_k, 2)]
        if not k_range:
            k_range = [1]

    dist_fn = euclidean_distance if metric == "euclidean" else manhattan_distance

    # Pre-compute sorted neighbor distances once for validation samples
    val_sorted_neighbors: List[List[Tuple[float, Any, int]]] = []
    for x_query in X_va:
        dists = []
        for idx in range(n_tr):
            d = dist_fn(x_query, X_tr[idx])
            dists.append((d, y_tr[idx], idx))
        dists.sort(key=lambda item: (item[0], item[2]))
        val_sorted_neighbors.append(dists)

    # Pre-compute sorted neighbor distances once for training samples (for train curve)
    # Subsample train points for speed if large, or compute all
    train_eval_indices = list(range(n_tr)) if n_tr <= 200 else list(range(0, n_tr, 2))
    train_sorted_neighbors: List[List[Tuple[float, Any, int]]] = []
    for tr_i in train_eval_indices:
        x_query = X_tr[tr_i]
        dists = []
        for idx in range(n_tr):
            d = dist_fn(x_query, X_tr[idx])
            dists.append((d, y_tr[idx], idx))
        dists.sort(key=lambda item: (item[0], item[2]))
        train_sorted_neighbors.append(dists)

    train_eval_y = [y_tr[i] for i in train_eval_indices]

    train_scores: List[float] = []
    val_scores: List[float] = []
    best_k = k_range[0]
    best_val_score = -1.0
    eps = 1e-9

    def _predict_with_sorted(sorted_list: List[List[Tuple[float, Any, int]]], k_val: int) -> List[Any]:
        preds = []
        for neighbors in sorted_list:
            top_k = neighbors[:k_val]
            class_scores = {c: 0.0 for c in classes}
            if weights == "uniform":
                for _, label, _ in top_k:
                    class_scores[label] += 1.0
            else:
                for dist, label, _ in top_k:
                    class_scores[label] += 1.0 / (dist + eps)
            best_c = classes[0]
            best_sc = -1.0
            for c in classes:
                if class_scores[c] > best_sc:
                    best_sc = class_scores[c]
                    best_c = c
            preds.append(best_c)
        return preds

    for k in k_range:
        tr_preds = _predict_with_sorted(train_sorted_neighbors, k)
        va_preds = _predict_with_sorted(val_sorted_neighbors, k)

        tr_acc = accuracy_score(train_eval_y, tr_preds)
        va_acc = accuracy_score(y_va, va_preds)

        train_scores.append(tr_acc)
        val_scores.append(va_acc)

        if va_acc > best_val_score:
            best_val_score = va_acc
            best_k = k

    summary = (
        f"KNN Complexity Analysis: Evaluated k from {k_range[0]} to {k_range[-1]}. "
        f"Best validation accuracy = {best_val_score:.4f} achieved at k = {best_k}."
    )

    return {
        "model_name": "KNN Classifier",
        "param_name": "n_neighbors",
        "param_values": k_range,
        "train_scores": train_scores,
        "val_scores": val_scores,
        "best_param": best_k,
        "best_val_score": best_val_score,
        "summary": summary,
    }


def analyze_decision_tree_complexity(
    X_train: Matrix,
    y_train: Vector,
    X_val: Matrix,
    y_val: Vector,
    depth_range: Optional[List[int]] = None,
    min_samples_split: int = 2,
    random_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Generate train vs validation accuracy curves across tree max_depth.

    Demonstrates:
    - Low depth (e.g. 1): High bias, low variance -> underfitting.
    - Deep depth (e.g. 10+): Low bias, high variance -> overfitting on training noise.
    """
    if depth_range is None:
        depth_range = list(range(1, 16))

    train_scores: List[float] = []
    val_scores: List[float] = []
    best_depth = depth_range[0]
    best_val_score = -1.0

    for depth in depth_range:
        clf = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=min_samples_split,
            random_seed=random_seed,
        )
        clf.fit(X_train, y_train)

        tr_acc = clf.score(X_train, y_train)
        va_acc = clf.score(X_val, y_val)

        train_scores.append(tr_acc)
        val_scores.append(va_acc)

        if va_acc > best_val_score:
            best_val_score = va_acc
            best_depth = depth

    summary = (
        f"Decision Tree Complexity Analysis: Evaluated max_depth from {depth_range[0]} to {depth_range[-1]}. "
        f"Best validation accuracy = {best_val_score:.4f} achieved at max_depth = {best_depth}."
    )

    return {
        "model_name": "Decision Tree Classifier",
        "param_name": "max_depth",
        "param_values": depth_range,
        "train_scores": train_scores,
        "val_scores": val_scores,
        "best_param": best_depth,
        "best_val_score": best_val_score,
        "summary": summary,
    }


def analyze_linear_regression_regularization(
    X_train: Matrix,
    y_train: Vector,
    X_val: Matrix,
    y_val: Vector,
    alphas: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Generate train vs validation R^2 curves across L2 penalty lambda.
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    train_scores: List[float] = []
    val_scores: List[float] = []
    best_alpha = alphas[0]
    best_val_score = -float("inf")

    for alpha in alphas:
        reg = LinearRegression(
            solver="normal_equation",
            l2_reg=alpha,
            fit_intercept=True,
        )
        reg.fit(X_train, y_train)

        tr_r2 = reg.score(X_train, y_train)
        va_r2 = reg.score(X_val, y_val)

        train_scores.append(tr_r2)
        val_scores.append(va_r2)

        if va_r2 > best_val_score:
            best_val_score = va_r2
            best_alpha = alpha

    summary = (
        f"Linear Regression Regularization Analysis: Evaluated L2 penalty alphas. "
        f"Best validation R^2 = {best_val_score:.4f} at alpha = {best_alpha}."
    )

    return {
        "model_name": "Linear Regression",
        "param_name": "l2_reg",
        "param_values": alphas,
        "train_scores": train_scores,
        "val_scores": val_scores,
        "best_param": best_alpha,
        "best_val_score": best_val_score,
        "summary": summary,
    }
