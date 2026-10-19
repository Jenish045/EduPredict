"""
Preprocessing module for EduPredict.
"""

from edupredict.preprocessing.imputer import SimpleImputer
from edupredict.preprocessing.encoder import LabelEncoder, OneHotEncoder
from edupredict.preprocessing.scaler import StandardScaler, MinMaxScaler

__all__ = [
    "SimpleImputer",
    "LabelEncoder",
    "OneHotEncoder",
    "StandardScaler",
    "MinMaxScaler",
]
