"""
Data management module for EduPredict.
"""

from edupredict.data.loader import load_csv, save_csv, try_cast_numeric
from edupredict.data.dataset import Dataset
from edupredict.data.generator import generate_student_data

__all__ = [
    "load_csv",
    "save_csv",
    "try_cast_numeric",
    "Dataset",
    "generate_student_data",
]
