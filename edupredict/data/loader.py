"""
CSV data loading and serialization implemented using Python standard library 'csv'.

Handles missing values, edge cases, whitespace stripping, and numeric parsing.
"""

import csv
import os
from typing import List, Tuple, Any, Optional, Union


def try_cast_numeric(val: str) -> Union[int, float, str, None]:
    """
    Attempt to convert a raw string to int or float.
    Treats empty strings or 'NA', 'NaN', 'null', 'None' as None.
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


def load_csv(
    filepath: str,
    has_header: bool = True,
    delimiter: str = ",",
    auto_cast: bool = True
) -> Tuple[List[str], List[List[Any]]]:
    """
    Load a CSV file from disk.
    
    Args:
        filepath: Path to CSV file.
        has_header: If True, first row is treated as column headers.
        delimiter: CSV separator character (default ',').
        auto_cast: If True, converts numeric strings to float/int and empty to None.
        
    Returns:
        (headers, rows) where rows is a list of row lists.
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
    else:
        # Generate default column names col_0, col_1...
        col_count = len(raw_rows[0])
        headers = [f"col_{i}" for i in range(col_count)]

    for row_idx, raw_row in enumerate(raw_rows[start_idx:], start=start_idx):
        if not raw_row or all(c.strip() == "" for c in raw_row):
            # Skip empty lines
            continue
        if len(raw_row) != len(headers):
            raise ValueError(
                f"Row {row_idx} length ({len(raw_row)}) does not match header length ({len(headers)})"
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
