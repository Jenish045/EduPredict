"""
Dataset schema representation and automatic type inference.

Inspects tabular data and categorizes columns into numerical features,
categorical features, target variables, and ignored identifier columns.
Supports explicit user configuration and automatic inference.
"""

from typing import List, Dict, Any, Optional, Set, Tuple


class DatasetSchema:
    """
    Metadata describing the structure, column types, and learning task for a tabular dataset.
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        target_column: str,
        task_type: str = "regression",
        ignored_columns: Optional[List[str]] = None,
        target_classes: Optional[List[Any]] = None,
    ) -> None:
        self.numerical_features = list(numerical_features)
        self.categorical_features = list(categorical_features)
        self.target_column = target_column
        self.task_type = task_type
        self.ignored_columns = list(ignored_columns or [])
        self.target_classes = target_classes

        valid_tasks = ("regression", "binary_classification", "multiclass_classification")
        if self.task_type not in valid_tasks:
            raise ValueError(f"task_type '{self.task_type}' must be one of {valid_tasks}")

    @property
    def feature_names(self) -> List[str]:
        """Return combined feature names in deterministic order (numerical then categorical)."""
        return self.numerical_features + self.categorical_features

    @property
    def n_features(self) -> int:
        return len(self.numerical_features) + len(self.categorical_features)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "ignored_columns": self.ignored_columns,
            "target_classes": self.target_classes,
        }

    def __repr__(self) -> str:
        return (
            f"DatasetSchema(target='{self.target_column}', task='{self.task_type}', "
            f"numerical={self.numerical_features}, categorical={self.categorical_features}, "
            f"ignored={self.ignored_columns})"
        )


def _is_numeric_value(val: Any) -> bool:
    """Check if a cell value represents a real number."""
    if val is None or val == "":
        return False
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return True
    try:
        float(str(val).strip())
        return True
    except (ValueError, TypeError):
        return False


def infer_schema(
    headers: List[str],
    rows: List[List[Any]],
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
    ignored_columns: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> DatasetSchema:
    """
    Infer or validate dataset schema from tabular data.

    Parameters
    ----------
    headers : List[str]
        List of column headers.
    rows : List[List[Any]]
        Data rows.
    target_column : Optional[str]
        Explicit target column name. If omitted, attempts to find standard target names
        or defaults to the last column.
    task_type : Optional[str]
        'regression', 'binary_classification', or 'multiclass_classification'.
        Inferred from target cardinality if omitted.
    ignored_columns : Optional[List[str]]
        Explicit columns to exclude (e.g. ID, timestamp). Auto-detects ID columns if omitted.
    numerical_features : Optional[List[str]]
        Explicit numerical feature names override.
    categorical_features : Optional[List[str]]
        Explicit categorical feature names override.
    """
    if not headers:
        raise ValueError("Cannot infer schema from empty headers list")
    if not rows:
        raise ValueError("Cannot infer schema from empty rows")

    col_to_idx = {h: i for i, h in enumerate(headers)}

    # 1. Determine target column
    if target_column is not None:
        if target_column not in col_to_idx:
            raise KeyError(f"Specified target '{target_column}' not found in headers: {headers}")
        resolved_target = target_column
    else:
        # Check conventional target names
        candidates = ["target", "label", "final_exam_score", "score", "outcome", "y"]
        resolved_target = None
        for cand in candidates:
            if cand in col_to_idx:
                resolved_target = cand
                break
        if resolved_target is None:
            resolved_target = headers[-1]

    # 2. Determine ignored columns
    resolved_ignored: List[str] = []
    if ignored_columns is not None:
        for col in ignored_columns:
            if col in col_to_idx and col != resolved_target:
                resolved_ignored.append(col)
    else:
        # Auto-detect ID-like columns
        for col in headers:
            if col == resolved_target:
                continue
            col_lower = col.lower()
            if col_lower in ("id", "student_id", "uid", "index") or col_lower.endswith("_id"):
                resolved_ignored.append(col)
                continue
            # Check uniqueness for string columns
            col_idx = col_to_idx[col]
            vals = [row[col_idx] for row in rows if row[col_idx] is not None]
            if len(vals) == len(rows) and len(set(vals)) == len(vals) and len(rows) > 30:
                if any(isinstance(v, str) for v in vals[:5]):
                    resolved_ignored.append(col)

    # 3. Determine numerical vs categorical features
    resolved_num: List[str] = []
    resolved_cat: List[str] = []

    if numerical_features is not None or categorical_features is not None:
        resolved_num = [c for c in (numerical_features or []) if c in col_to_idx and c != resolved_target and c not in resolved_ignored]
        resolved_cat = [c for c in (categorical_features or []) if c in col_to_idx and c != resolved_target and c not in resolved_ignored]
    else:
        for col in headers:
            if col == resolved_target or col in resolved_ignored:
                continue
            col_idx = col_to_idx[col]
            non_empty_vals = [row[col_idx] for row in rows if row[col_idx] is not None and str(row[col_idx]).strip() != ""]
            if not non_empty_vals:
                resolved_cat.append(col)
                continue

            num_count = sum(1 for v in non_empty_vals if _is_numeric_value(v))
            ratio = num_count / len(non_empty_vals)
            # If at least 85% of non-empty values are numeric, treat as numeric
            if ratio >= 0.85:
                resolved_num.append(col)
            else:
                resolved_cat.append(col)

    # 4. Infer task type if not specified
    target_idx = col_to_idx[resolved_target]
    target_vals = [row[target_idx] for row in rows if row[target_idx] is not None and str(row[target_idx]).strip() != ""]
    unique_target = sorted(list(set(target_vals)), key=lambda x: str(x))

    if task_type is not None:
        resolved_task = task_type
    else:
        all_numeric = all(_is_numeric_value(v) for v in target_vals)
        if len(unique_target) == 2:
            resolved_task = "binary_classification"
        elif not all_numeric or len(unique_target) <= 10:
            resolved_task = "multiclass_classification" if len(unique_target) > 2 else "binary_classification"
        else:
            resolved_task = "regression"

    target_classes = unique_target if "classification" in resolved_task else None

    return DatasetSchema(
        numerical_features=resolved_num,
        categorical_features=resolved_cat,
        target_column=resolved_target,
        task_type=resolved_task,
        ignored_columns=resolved_ignored,
        target_classes=target_classes,
    )
