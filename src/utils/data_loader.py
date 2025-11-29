"""
Data Loading and Preprocessing Module

Provides functions to load iris data, split into train/test sets,
and discretize continuous features into bins for histogram-based Naive Bayes.
"""

import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_iris_data(data_path='data/iris.data'):
    """
    Load iris dataset from CSV file.

    Args:
        data_path: Relative path to iris.data file

    Returns:
        tuple: (features, labels, feature_names, class_names)
            - features: numpy array of shape (n_samples, 4)
            - labels: numpy array of shape (n_samples,)
            - feature_names: list of feature names
            - class_names: list of class names
    """
    logger.info(f"Loading iris dataset from {data_path}")

    # Get relative path from current working directory
    base_path = Path(__file__).parent.parent.parent
    full_path = base_path / data_path

    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    # Load data
    data = []
    labels = []

    with open(full_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            parts = line.split(',')
            if len(parts) == 5:
                features_row = [float(x) for x in parts[:4]]
                label = parts[4]
                data.append(features_row)
                labels.append(label)

    features = np.array(data)
    labels_array = np.array(labels)

    logger.info(f"Loaded {len(features)} samples with {features.shape[1]} features")
    logger.info(f"Classes: {class_names}")
    logger.info(f"Features: {feature_names}")

    # Convert string labels to integers
    label_to_int = {name: idx for idx, name in enumerate(class_names)}
    labels_int = np.array([label_to_int[label] for label in labels_array])

    # Log class distribution
    unique, counts = np.unique(labels_int, return_counts=True)
    for class_idx, count in zip(unique, counts):
        logger.info(f"  {class_names[class_idx]}: {count} samples")

    return features, labels_int, feature_names, class_names


def split_data(features, labels, train_ratio=0.75, random_seed=42):
    """
    Split data into training and testing sets.

    Args:
        features: numpy array of features
        labels: numpy array of labels
        train_ratio: proportion of data for training (default: 0.75)
        random_seed: random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data: {train_ratio*100}% train, {(1-train_ratio)*100}% test")
    logger.info(f"Random seed: {random_seed}")

    # Set random seed
    np.random.seed(random_seed)

    # Get number of samples
    n_samples = len(features)
    n_train = int(n_samples * train_ratio)

    # Shuffle indices
    indices = np.random.permutation(n_samples)

    # Split indices
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Split data
    X_train = features[train_indices]
    X_test = features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Log class distribution in splits
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    logger.info("Training set class distribution:")
    for class_idx, count in zip(unique_train, counts_train):
        logger.info(f"  Class {class_idx}: {count} samples")

    logger.info("Test set class distribution:")
    for class_idx, count in zip(unique_test, counts_test):
        logger.info(f"  Class {class_idx}: {count} samples")

    return X_train, X_test, y_train, y_test


def discretize_features(X_train, X_test, n_bins=10):
    """
    Discretize continuous features into bins.

    Bins are calculated from training data only to avoid data leakage.

    Args:
        X_train: training features (n_samples, n_features)
        X_test: test features (n_samples, n_features)
        n_bins: number of bins per feature

    Returns:
        tuple: (X_train_binned, X_test_binned, bin_edges)
            - X_train_binned: discretized training features
            - X_test_binned: discretized test features
            - bin_edges: list of bin edges for each feature
    """
    logger.info(f"Discretizing features into {n_bins} bins per feature")

    n_features = X_train.shape[1]
    X_train_binned = np.zeros_like(X_train, dtype=int)
    X_test_binned = np.zeros_like(X_test, dtype=int)
    bin_edges = []

    for feature_idx in range(n_features):
        # Calculate bins from training data only
        feature_min = X_train[:, feature_idx].min()
        feature_max = X_train[:, feature_idx].max()

        # Create equal-width bins
        edges = np.linspace(feature_min, feature_max, n_bins + 1)
        bin_edges.append(edges)

        logger.info(f"Feature {feature_idx}: range [{feature_min:.3f}, {feature_max:.3f}], {n_bins} bins")

        # Digitize training data
        # digitize returns 1-indexed bins, we convert to 0-indexed
        X_train_binned[:, feature_idx] = np.digitize(X_train[:, feature_idx], edges[1:-1])

        # Digitize test data using same bins
        X_test_binned[:, feature_idx] = np.digitize(X_test[:, feature_idx], edges[1:-1])

        # Ensure bins are in valid range [0, n_bins-1]
        X_train_binned[:, feature_idx] = np.clip(X_train_binned[:, feature_idx], 0, n_bins - 1)
        X_test_binned[:, feature_idx] = np.clip(X_test_binned[:, feature_idx], 0, n_bins - 1)

    logger.info("Discretization complete")
    logger.info(f"Training binned shape: {X_train_binned.shape}")
    logger.info(f"Test binned shape: {X_test_binned.shape}")

    return X_train_binned, X_test_binned, bin_edges
