"""
Scikit-learn Naive Bayes Implementation

Wrapper for scikit-learn's CategoricalNB with detailed logging.

This implementation:
- Uses the same discretized features as NumPy implementation
- Applies CategoricalNB from scikit-learn
- Provides detailed logging for comparison

Modules:
    - classifier: NaiveBayesSklearn classifier wrapper
    - logger_config: Logging configuration
"""

from .classifier import NaiveBayesSklearn
from .logger_config import get_logger

__all__ = ['NaiveBayesSklearn', 'get_logger']
