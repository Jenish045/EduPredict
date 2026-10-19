"""
Dataset splitting algorithms implemented from scratch.

Provides:
- train_test_split: splits X and y into random train and test subsets with optional stratification.
- KFold: K-fold cross-validation splitter.
"""

import random
from typing import Sequence, List, Tuple, Any, Optional, Union

Number = Union[int, float]


def train_test_split(
    X: Sequence[Sequence[Any]],
    y: Sequence[Any],
    test_size: float = 0.2,
    shuffle: bool = True,
    random_seed: Optional[int] = 42,
    stratify: bool = False
) -> Tuple[List[List[Any]], List[List[Any]], List[Any], List[Any]]:
    """
    Split matrices or sequences into random train and test subsets.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Cannot split empty arrays")
    if len(X) != len(y):
        raise ValueError(f"Lengths do not match: len(X)={len(X)} != len(y)={len(y)}")
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    n_samples = len(X)
    rng = random.Random(random_seed)

    if stratify:
        # Group indices by class label in y
        label_groups: dict = {}
        for idx, label in enumerate(y):
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        train_indices: List[int] = []
        test_indices: List[int] = []

        for label, idxs in label_groups.items():
            if shuffle:
                rng.shuffle(idxs)
            n_test = max(1, int(round(len(idxs) * test_size)))
            test_indices.extend(idxs[:n_test])
            train_indices.extend(idxs[n_test:])

        if shuffle:
            rng.shuffle(train_indices)
            rng.shuffle(test_indices)

    else:
        indices = list(range(n_samples))
        if shuffle:
            rng.shuffle(indices)

        n_test = int(round(n_samples * test_size))
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    X_train = [list(X[i]) for i in train_indices]
    X_test = [list(X[i]) for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


class KFold:
    """
    K-Fold cross-validator.
    Provides train/test indices to split data in train/test sets.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_seed: Optional[int] = 42
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed

    def split(self, X: Sequence[Any]) -> List[Tuple[List[int], List[int]]]:
        """
        Generate train and test indices for each fold.
        """
        n_samples = len(X)
        if self.n_splits > n_samples:
            raise ValueError(f"n_splits ({self.n_splits}) cannot exceed number of samples ({n_samples})")

        indices = list(range(n_samples))
        if self.shuffle:
            rng = random.Random(self.random_seed)
            rng.shuffle(indices)

        # Compute fold sizes
        fold_sizes = [n_samples // self.n_splits] * self.n_splits
        for i in range(n_samples % self.n_splits):
            fold_sizes[i] += 1

        splits: List[Tuple[List[int], List[int]]] = []
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = indices[:start] + indices[stop:]
            splits.append((train_idx, test_idx))
            current = stop

        return splits
