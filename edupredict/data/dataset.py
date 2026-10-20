"""
Dataset container and schema validation for educational tabular ML datasets.

Pure Python implementation providing structured dataset operations,
column extraction, validation, and type inspection.
"""

from typing import List, Dict, Any, Optional, Tuple, Sequence
from edupredict.math.stats import mean, standard_deviation, min_val, max_val


class Dataset:
    """
    A 2D tabular dataset representation using pure Python standard collections.
    """

    def __init__(
        self,
        headers: Sequence[str],
        data: Sequence[Sequence[Any]],
        target_col: Optional[str] = None
    ) -> None:
        if not headers or len(headers) == 0:
            raise ValueError("Headers list cannot be empty")
        if not data or len(data) == 0:
            raise ValueError("Data cannot be empty")

        self.headers: List[str] = list(headers)
        if len(self.headers) != len(set(self.headers)):
            raise ValueError("Headers must contain unique column names")

        self.header_to_idx: Dict[str, int] = {h: i for i, h in enumerate(self.headers)}
        num_cols = len(self.headers)

        # Validate rows
        self.rows: List[List[Any]] = []
        for r_idx, row in enumerate(data):
            if len(row) != num_cols:
                raise ValueError(f"Row {r_idx} has {len(row)} elements, expected {num_cols}")
            self.rows.append(list(row))

        self.target_col: Optional[str] = target_col
        if target_col is not None and target_col not in self.header_to_idx:
            raise ValueError(f"Target column '{target_col}' not found in headers: {self.headers}")

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (number of rows, number of columns)."""
        return len(self.rows), len(self.headers)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> List[Any]:
        return self.rows[idx]

    def get_column(self, col_name: str) -> List[Any]:
        """Extract a single column as a list."""
        if col_name not in self.header_to_idx:
            raise KeyError(f"Column '{col_name}' not found. Available: {self.headers}")
        c_idx = self.header_to_idx[col_name]
        return [row[c_idx] for row in self.rows]

    def set_column(self, col_name: str, values: Sequence[Any]) -> None:
        """Replace or add a column."""
        if len(values) != len(self.rows):
            raise ValueError(f"Length of values ({len(values)}) does not match dataset rows ({len(self.rows)})")
        if col_name in self.header_to_idx:
            c_idx = self.header_to_idx[col_name]
            for r_idx in range(len(self.rows)):
                self.rows[r_idx][c_idx] = values[r_idx]
        else:
            self.headers.append(col_name)
            self.header_to_idx[col_name] = len(self.headers) - 1
            for r_idx in range(len(self.rows)):
                self.rows[r_idx].append(values[r_idx])

    def count_missing(self) -> Dict[str, int]:
        """Count missing (None) values in each column."""
        missing_counts = {h: 0 for h in self.headers}
        for row in self.rows:
            for h, val in zip(self.headers, row):
                if val is None:
                    missing_counts[h] += 1
        return missing_counts

    def split_features_target(
        self,
        feature_cols: Optional[Sequence[str]] = None,
        target_col: Optional[str] = None
    ) -> Tuple[List[List[Any]], List[Any]]:
        """
        Split dataset into feature matrix X (List[List[Any]]) and target vector y (List[Any]).
        """
        t_col = target_col or self.target_col
        if t_col is None:
            raise ValueError("Target column must be specified either in Dataset or via argument")
        if t_col not in self.header_to_idx:
            raise KeyError(f"Target column '{t_col}' not found in dataset headers")

        if feature_cols is None:
            feature_cols = [h for h in self.headers if h != t_col]
        else:
            for f in feature_cols:
                if f not in self.header_to_idx:
                    raise KeyError(f"Feature '{f}' not in dataset headers")

        f_indices = [self.header_to_idx[f] for f in feature_cols]
        t_idx = self.header_to_idx[t_col]

        X = [[row[idx] for idx in f_indices] for row in self.rows]
        y = [row[t_idx] for row in self.rows]
        return X, y

    def summary(self) -> str:
        """Generate a concise statistical summary of numeric and categorical columns."""
        lines = [f"Dataset Summary: {len(self.rows)} rows x {len(self.headers)} columns"]
        lines.append("-" * 75)
        lines.append(f"{'Column':<25} {'Type':<12} {'Missing':<8} {'Mean / Unique':<15} {'Std / Top':<15}")
        lines.append("-" * 75)

        missing = self.count_missing()
        for col in self.headers:
            vals = self.get_column(col)
            non_null = [v for v in vals if v is not None]
            col_missing = missing[col]

            if non_null and all(isinstance(v, (int, float)) for v in non_null):
                col_type = "Numeric"
                mu = mean(non_null)
                std = standard_deviation(non_null) if len(non_null) > 1 else 0.0
                val_summary1 = f"{mu:.2f}"
                val_summary2 = f"{std:.2f}"
            else:
                col_type = "Categorical"
                unique_vals = set(non_null)
                val_summary1 = f"{len(unique_vals)} unique"
                val_summary2 = f"{non_null[0]}" if non_null else "N/A"

            lines.append(f"{col:<25} {col_type:<12} {col_missing:<8} {val_summary1:<15} {val_summary2:<15}")

        return "\n".join(lines)

    def get_schema(
        self,
        target_col: Optional[str] = None,
        task_type: Optional[str] = None,
        ignored_columns: Optional[List[str]] = None,
    ):
        """Infer or return the DatasetSchema for this dataset."""
        from edupredict.data.schema import infer_schema
        return infer_schema(
            headers=self.headers,
            rows=self.rows,
            target_column=target_col or self.target_col,
            task_type=task_type,
            ignored_columns=ignored_columns,
        )
