"""
Visualization Utilities

Provides functions to generate histograms and other visualizations
for Naive Bayes classifiers.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_feature_histograms(feature_likelihoods, classes, n_features, n_bins,
                            feature_names=None, class_names=None,
                            output_path='output/histograms.png', title_prefix=''):
    """
    Plot histograms of feature likelihoods for each class.

    Creates a grid of subplots showing the probability distribution
    for each feature across all classes.

    Args:
        feature_likelihoods: numpy array of shape (n_classes, n_features, n_bins)
        classes: array of class labels
        n_features: number of features
        n_bins: number of bins per feature
        feature_names: optional list of feature names
        class_names: optional list of class names
        output_path: path to save the plot
        title_prefix: prefix for the plot title
    """
    logger.info(f"Generating feature histogram plot: {output_path}")

    n_classes = len(classes)

    # Default names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(n_features)]
    if class_names is None:
        class_names = [f'Class {c}' for c in classes]

    # Create figure with subplots
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))

    # Handle single feature case
    if n_features == 1:
        axes = [axes]

    # Color map for classes
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    # Plot each feature
    for feature_idx in range(n_features):
        ax = axes[feature_idx]

        # Plot histogram for each class
        bin_positions = np.arange(n_bins)
        width = 0.8 / n_classes

        for class_idx, class_label in enumerate(classes):
            likelihoods = feature_likelihoods[class_idx, feature_idx, :]
            offset = (class_idx - n_classes / 2) * width + width / 2

            ax.bar(bin_positions + offset, likelihoods, width,
                  label=class_names[class_idx],
                  color=colors[class_idx], alpha=0.8)

        ax.set_xlabel('Bin')
        ax.set_ylabel('P(feature|class)')
        ax.set_title(f'{feature_names[feature_idx]} - Probability Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(bin_positions)

    # Overall title
    fig.suptitle(f'{title_prefix}Feature Histograms - Naive Bayes Classifier',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Histogram plot saved to {output_path}")

    plt.close(fig)


def plot_class_priors(class_priors, classes, class_names=None,
                     output_path='output/class_priors.png', title_prefix=''):
    """
    Plot class prior probabilities as a bar chart.

    Args:
        class_priors: array of class prior probabilities
        classes: array of class labels
        class_names: optional list of class names
        output_path: path to save the plot
        title_prefix: prefix for the plot title
    """
    logger.info(f"Generating class priors plot: {output_path}")

    if class_names is None:
        class_names = [f'Class {c}' for c in classes]

    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = np.arange(len(classes))
    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))

    bars = ax.bar(x_pos, class_priors, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Prior Probability', fontsize=12)
    ax.set_title(f'{title_prefix}Class Prior Probabilities', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, max(class_priors) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Class priors plot saved to {output_path}")

    plt.close(fig)


def plot_prediction_distribution(predictions, y_true, classes, class_names=None,
                                 output_path='output/prediction_dist.png', title_prefix=''):
    """
    Plot distribution of predictions vs actual labels.

    Args:
        predictions: array of predicted labels
        y_true: array of true labels
        classes: array of class labels
        class_names: optional list of class names
        output_path: path to save the plot
        title_prefix: prefix for the plot title
    """
    logger.info(f"Generating prediction distribution plot: {output_path}")

    if class_names is None:
        class_names = [f'Class {c}' for c in classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Count predictions and actuals
    pred_counts = np.array([np.sum(predictions == c) for c in classes])
    true_counts = np.array([np.sum(y_true == c) for c in classes])

    x_pos = np.arange(len(classes))
    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))

    # Plot predictions
    bars1 = ax1.bar(x_pos, pred_counts, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Predicted Labels Distribution', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(class_names)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot actual
    bars2 = ax2.bar(x_pos, true_counts, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Actual Labels Distribution', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(class_names)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Overall title
    fig.suptitle(f'{title_prefix}Prediction vs Actual Distribution',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Prediction distribution plot saved to {output_path}")

    plt.close(fig)
