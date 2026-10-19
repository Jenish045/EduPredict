"""
CSV data loading, schema validation, and serialization implemented using
the Python standard library 'csv'.

Strictly zero third-party dependencies (no Pandas).
Supports:
- Header and row length validation
- Automated numerical and categorical type casting
- Missing value detection (None, empty, NaN, NA, null, ?)
- Comprehensive invalid-value reporting with row indices and column names
- Malformed data error raising
"""

import csv
import os
from typing import List, Tuple, Any, Optional, Union, Dict, Set


class ValidationReport:
    """
    Structured report capturing data validation results.
    """

    def __init__(self, headers: List[str], total_rows: int) -> None:
        self.headers = headers
        self.total_rows = total_rows
        self.valid_rows = 0
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.missing_counts: Dict[str, int] = {h: 0 for h in headers}
        self.invalid_values: List[Dict[str, Any]] = []

    @property
    def is_valid(self) -> bool:
        """Returns True if no fatal errors were detected during validation."""
        return len(self.errors) == 0

    def summary(self) -> str:
        """Return human-readable validation summary."""
        lines = [
            f"=== CSV Validation Report ===",
            f"Columns: {len(self.headers)} ({', '.join(self.headers)})",
            f"Total Rows: {self.total_rows}",
            f"Valid Rows: {self.valid_rows}",
            f"Fatal Errors: {len(self.errors)}",
            f"Invalid Value Warnings: {len(self.invalid_values)}",
        ]
        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors[:10]:
                lines.append(f"  - {err}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more errors")

        lines.append("\nMissing Value Counts by Column:")
        for h, count in self.missing_counts.items():
            pct = (count / self.total_rows * 100.0) if self.total_rows > 0 else 0.0
            lines.append(f"  {h:<25}: {count:5d} ({pct:5.1f}%)")

        if self.invalid_values:
            lines.append("\nSample Invalid Values:")
            for item in self.invalid_values[:5]:
                lines.append(
                    f"  Row {item['row_index']}, Col '{item['column']}': "
                    f"value={item['value']!r} ({item['reason']})"
                )
            if len(self.invalid_values) > 5:
                lines.append(f"  ... and {len(self.invalid_values) - 5} more invalid values")

        return "\n".join(lines)


def try_cast_numeric(val: str) -> Union[int, float, str, None]:
    """
    Attempt to convert a raw string cell to int or float.
    Treats empty strings or 'NA', 'NaN', 'null', 'None', '?' as None.
    If conversion fails, returns the stripped string.
    """
    val_strip = val.strip()
    if val_strip == "" or val_strip.lower() in ("na", "nan", "null", "none", "?"):
        return None
    try:
        # Check integer first
        int_val = int(val_strip)
        return int_val
    except ValueError:
        try:
            flt_val = float(val_strip)
            return flt_val
        except ValueError:
            return val_strip


def validate_dataset(
    headers: List[str],
    rows: List[List[Any]],
    schema: Optional[Dict[str, type]] = None,
    allowed_categories: Optional[Dict[str, Set[Any]]] = None,
    numeric_ranges: Optional[Dict[str, Tuple[float, float]]] = None
) -> ValidationReport:
    """
    Validate tabular rows against headers, optional schema, allowed categories, and numeric ranges.
    """
    report = ValidationReport(headers=headers, total_rows=len(rows))
    col_to_idx = {h: i for i, h in enumerate(headers)}
    num_cols = len(headers)

    for r_idx, row in enumerate(rows):
        row_has_fatal = False
        if len(row) != num_cols:
            report.errors.append(
                f"Row {r_idx} length mismatch: expected {num_cols} columns, got {len(row)}"
            )
            continue

        for col_name, c_idx in col_to_idx.items():
            val = row[c_idx]

            # Missing value detection
            if val is None or (isinstance(val, str) and val.strip() == ""):
                report.missing_counts[col_name] += 1
                continue

            # Schema type checking
            if schema and col_name in schema:
                expected_type = schema[col_name]
                if expected_type in (int, float):
                    if not isinstance(val, (int, float)) or isinstance(val, bool):
                        report.invalid_values.append({
                            "row_index": r_idx,
                            "column": col_name,
                            "value": val,
                            "reason": f"Expected numeric {expected_type.__name__}, got {type(val).__name__}"
                        })
                elif expected_type == str:
                    if not isinstance(val, str):
                        report.invalid_values.append({
                            "row_index": r_idx,
                            "column": col_name,
                            "value": val,
                            "reason": f"Expected string, got {type(val).__name__}"
                        })

            # Range check for numeric values
            if numeric_ranges and col_name in numeric_ranges and isinstance(val, (int, float)):
                min_r, max_r = numeric_ranges[col_name]
                if val < min_r or val > max_r:
                    report.invalid_values.append({
                        "row_index": r_idx,
                        "column": col_name,
                        "value": val,
                        "reason": f"Value {val} outside expected range [{min_r}, {max_r}]"
                    })

            # Allowed category check
            if allowed_categories and col_name in allowed_categories:
                valid_set = allowed_categories[col_name]
                if val not in valid_set:
                    report.invalid_values.append({
                        "row_index": r_idx,
                        "column": col_name,
                        "value": val,
                        "reason": f"Value {val!r} not in allowed categories: {valid_set}"
                    })

        if not row_has_fatal:
            report.valid_rows += 1

    return report


def load_csv(
    filepath: str,
    has_header: bool = True,
    delimiter: str = ",",
    auto_cast: bool = True,
    expected_headers: Optional[List[str]] = None
) -> Tuple[List[str], List[List[Any]]]:
    """
    Load a CSV file from disk with strict validation.
    
    Args:
        filepath: Path to CSV file.
        has_header: If True, first row is treated as column headers.
        delimiter: CSV separator character (default ',').
        auto_cast: If True, converts numeric strings to float/int and empty to None.
        expected_headers: Optional list of headers that must match exactly.
        
    Returns:
        (headers, rows) where rows is a list of row lists.
        
    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty, headers are duplicate, or rows are malformed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    headers: List[str] = []
    rows: List[List[Any]] = []

    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError(f"CSV file is empty: {filepath}")

    start_idx = 0
    if has_header:
        headers = [h.strip() for h in raw_rows[0]]
        start_idx = 1
        if not headers or all(h == "" for h in headers):
            raise ValueError(f"CSV header row is empty in {filepath}")
        if len(headers) != len(set(headers)):
            raise ValueError(f"Duplicate column headers found in CSV: {headers}")
    else:
        col_count = len(raw_rows[0])
        headers = [f"col_{i}" for i in range(col_count)]

    if expected_headers is not None:
        if headers != expected_headers:
            raise ValueError(
                f"CSV headers {headers} do not match expected headers {expected_headers}"
            )

    for line_num, raw_row in enumerate(raw_rows[start_idx:], start=start_idx + 1):
        if not raw_row or all(c.strip() == "" for c in raw_row):
            # Skip empty whitespace lines
            continue
        if len(raw_row) != len(headers):
            raise ValueError(
                f"Malformed CSV at line {line_num}: row has {len(raw_row)} columns, expected {len(headers)}"
            )
        if auto_cast:
            parsed_row = [try_cast_numeric(cell) for cell in raw_row]
        else:
            parsed_row = [cell.strip() for cell in raw_row]
        rows.append(parsed_row)

    if not rows:
        raise ValueError(f"No data rows found in CSV: {filepath}")

    return headers, rows


def save_csv(
    filepath: str,
    headers: List[str],
    rows: List[List[Any]],
    delimiter: str = ","
) -> None:
    """
    Save headers and rows to a CSV file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        if headers:
            writer.writerow(headers)
        for row in rows:
            formatted_row = ["" if cell is None else cell for cell in row]
            writer.writerow(formatted_row)
