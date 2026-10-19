"""
Models module for EduPredict.
"""

from edupredict.models.base import BaseModel
from edupredict.models.linear_regression import LinearRegression
from edupredict.models.logistic_regression import LogisticRegression
from edupredict.models.knn import KNNClassifier, KNNRegressor
from edupredict.models.decision_tree import DecisionTreeClassifier
from edupredict.models.naive_bayes import GaussianNaiveBayes
from edupredict.models.multiclass import OneVsRestClassifier
from edupredict.models.serialization import save_model, load_model

__all__ = [
    "BaseModel",
    "LinearRegression",
    "LogisticRegression",
    "KNNClassifier",
    "KNNRegressor",
    "DecisionTreeClassifier",
    "GaussianNaiveBayes",
    "OneVsRestClassifier",
    "save_model",
    "load_model",
]

