"""
NumPy Naive Bayes Implementation

Manual histogram-based Naive Bayes classifier built from scratch using only NumPy.

This implementation:
- Discretizes continuous features into histograms
- Calculates class priors and feature likelihoods
- Uses Bayes theorem for prediction
- Applies argmax for class selection

Modules:
    - classifier: NaiveBayesNumpy classifier class
    - logger_config: Logging configuration
"""

from .classifier import NaiveBayesNumpy
from .logger_config import get_logger

__all__ = ['NaiveBayesNumpy', 'get_logger']
