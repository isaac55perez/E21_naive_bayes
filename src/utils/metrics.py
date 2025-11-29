"""
Metrics and Comparison Utilities

Provides functions to calculate accuracy and compare predictions
between different implementations.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.

    Args:
        y_true: true labels
        y_pred: predicted labels

    Returns:
        float: accuracy (percentage of correct predictions)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} samples, y_pred has {len(y_pred)}")

    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total

    logger.info(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total} correct)")

    return accuracy


def compare_predictions(y_pred1, y_pred2, names=('Model 1', 'Model 2')):
    """
    Compare predictions from two models.

    Args:
        y_pred1: predictions from first model
        y_pred2: predictions from second model
        names: tuple of model names for logging

    Returns:
        dict: comparison results including agreement rate and differences
    """
    logger.info(f"Comparing predictions between {names[0]} and {names[1]}")

    if len(y_pred1) != len(y_pred2):
        raise ValueError(f"Length mismatch: {len(y_pred1)} vs {len(y_pred2)}")

    # Calculate agreement
    agreement = np.sum(y_pred1 == y_pred2)
    total = len(y_pred1)
    agreement_rate = agreement / total

    logger.info(f"Agreement: {agreement_rate*100:.2f}% ({agreement}/{total} predictions match)")

    # Find differences
    diff_indices = np.where(y_pred1 != y_pred2)[0]

    if len(diff_indices) > 0:
        logger.info(f"Differences found at {len(diff_indices)} positions:")
        for idx in diff_indices[:10]:  # Log first 10 differences
            logger.info(f"  Sample {idx}: {names[0]} predicted {y_pred1[idx]}, {names[1]} predicted {y_pred2[idx]}")
        if len(diff_indices) > 10:
            logger.info(f"  ... and {len(diff_indices) - 10} more differences")
    else:
        logger.info("All predictions match perfectly!")

    results = {
        'agreement': agreement,
        'total': total,
        'agreement_rate': agreement_rate,
        'diff_indices': diff_indices,
        'n_differences': len(diff_indices)
    }

    return results


def log_confusion_summary(y_true, y_pred, class_names=None):
    """
    Log a summary of predictions per class.

    Args:
        y_true: true labels
        y_pred: predicted labels
        class_names: optional list of class names
    """
    unique_classes = np.unique(y_true)

    logger.info("Per-class prediction summary:")

    for class_idx in unique_classes:
        class_mask = y_true == class_idx
        class_total = np.sum(class_mask)
        class_correct = np.sum((y_true == class_idx) & (y_pred == class_idx))
        class_accuracy = class_correct / class_total if class_total > 0 else 0

        class_label = class_names[class_idx] if class_names else f"Class {class_idx}"

        logger.info(f"  {class_label}: {class_accuracy*100:.2f}% ({class_correct}/{class_total})")
