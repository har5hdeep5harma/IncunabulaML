"""
IncunabulaML
------------

A lightweight educational package with foundational machine learning algorithms implemented from scratch.
"""

from .perceptron import Perceptron
from .adaline_gd import AdalineGD
from .adaline_sgd import AdalineSGD
from .logistic_regression_gd import LogisticRegressionGD

__all__ = [
    "Perceptron",
    "AdalineGD",
    "AdalineSGD",
    "LogisticRegressionGD",
]
