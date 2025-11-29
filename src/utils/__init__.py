"""
Utilities Package

Shared utilities for data loading, preprocessing, and evaluation.

Modules:
    - data_loader: Load iris dataset, train/test split, feature binning
    - metrics: Accuracy calculation and comparison utilities
    - visualization: Histogram and plot generation
"""

from .data_loader import load_iris_data, split_data, discretize_features
from .metrics import calculate_accuracy, compare_predictions
from .visualization import plot_feature_histograms, plot_class_priors, plot_prediction_distribution

__all__ = [
    'load_iris_data',
    'split_data',
    'discretize_features',
    'calculate_accuracy',
    'compare_predictions',
    'plot_feature_histograms',
    'plot_class_priors',
    'plot_prediction_distribution'
]
