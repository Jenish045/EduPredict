"""
Dataset splitting algorithms implemented from scratch.

Provides:
- train_test_split: splits X and y into random train and test subsets with optional stratification.
- KFold: standard K-fold cross-validation splitter.
- StratifiedKFold: stratified K-fold cross-validation splitter preserving class distribution per fold.
"""

import random
from typing import Sequence, List, Tuple, Any, Optional, Union, Dict

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
        label_groups: Dict[Any, List[int]] = {}
        for idx, label in enumerate(y):
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        train_indices: List[int] = []
        test_indices: List[int] = []

        for label, idxs in label_groups.items():
            idxs_copy = list(idxs)
            if shuffle:
                rng.shuffle(idxs_copy)
            n_test = max(1, int(round(len(idxs_copy) * test_size)))
            test_indices.extend(idxs_copy[:n_test])
            train_indices.extend(idxs_copy[n_test:])

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

    def split(self, X: Sequence[Any], y: Optional[Sequence[Any]] = None) -> List[Tuple[List[int], List[int]]]:
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


class StratifiedKFold:
    """
    Stratified K-Fold cross-validator.
    Ensures that each fold contains approximately the same percentage of samples of each target class.
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

    def split(self, X: Sequence[Any], y: Sequence[Any]) -> List[Tuple[List[int], List[int]]]:
        """
        Generate stratified train and test indices for each fold.
        """
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: len(X)={len(X)} != len(y)={len(y)}")
        n_samples = len(X)
        if self.n_splits > n_samples:
            raise ValueError(f"n_splits ({self.n_splits}) cannot exceed number of samples ({n_samples})")

        rng = random.Random(self.random_seed)

        # Group indices by class
        class_indices: Dict[Any, List[int]] = {}
        for idx, label in enumerate(y):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Build folds: initialize empty test indices for each fold
        test_folds: List[List[int]] = [[] for _ in range(self.n_splits)]

        for label, idxs in class_indices.items():
            idxs_copy = list(idxs)
            if self.shuffle:
                rng.shuffle(idxs_copy)

            # Distribute class indices across folds round-robin style
            for i, idx in enumerate(idxs_copy):
                fold_idx = i % self.n_splits
                test_folds[fold_idx].append(idx)

        all_indices_set = set(range(n_samples))
        splits: List[Tuple[List[int], List[int]]] = []

        for test_idx in test_folds:
            if self.shuffle:
                rng.shuffle(test_idx)
            test_set = set(test_idx)
            train_idx = [i for i in range(n_samples) if i not in test_set]
            if self.shuffle:
                rng.shuffle(train_idx)
            splits.append((train_idx, test_idx))

        return splits
