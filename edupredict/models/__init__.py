"""
Models module for EduPredict.
"""

from edupredict.models.base import BaseModel
from edupredict.models.linear_regression import LinearRegression
from edupredict.models.logistic_regression import LogisticRegression
from edupredict.models.knn import KNNClassifier, KNNRegressor
from edupredict.models.decision_tree import DecisionTreeClassifier
from edupredict.models.naive_bayes import GaussianNaiveBayes

__all__ = [
    "BaseModel",
    "LinearRegression",
    "LogisticRegression",
    "KNNClassifier",
    "KNNRegressor",
    "DecisionTreeClassifier",
    "GaussianNaiveBayes",
]
