"""
Scikit-learn based Naive Bayes Classifier

Wrapper around CategoricalNB with detailed logging for comparison.
"""

import numpy as np
from pathlib import Path
from sklearn.naive_bayes import CategoricalNB
from .logger_config import get_logger

logger = get_logger()


class NaiveBayesSklearn:
    """
    Wrapper for scikit-learn's CategoricalNB classifier.

    This wrapper adds detailed logging to match the NumPy implementation
    for fair comparison.
    """

    def __init__(self, alpha=1.0):
        """
        Initialize sklearn Naive Bayes classifier.

        Args:
            alpha: Laplace smoothing parameter (default: 1.0)
        """
        self.alpha = alpha
        self.model = CategoricalNB(alpha=alpha)
        self.classes = None
        self.n_features = None

        logger.info(f"Initialized NaiveBayesSklearn with alpha={alpha}")

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.

        Args:
            X: training features (n_samples, n_features) - discretized/binned
            y: training labels (n_samples,)
        """
        logger.info("=" * 60)
        logger.info("TRAINING SKLEARN NAIVE BAYES CLASSIFIER")
        logger.info("=" * 60)

        n_samples, n_features = X.shape
        self.n_features = n_features
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        logger.info(f"Training samples: {n_samples}")
        logger.info(f"Features: {n_features}")
        logger.info(f"Classes: {self.classes}")
        logger.info(f"Number of classes: {n_classes}")

        # Log class distribution
        logger.info("\n--- Class Distribution ---")
        for class_label in self.classes:
            class_count = np.sum(y == class_label)
            class_prior = class_count / n_samples
            logger.info(f"Class {class_label}: {class_count} samples (prior: {class_prior:.4f})")

        # Log feature value ranges
        logger.info("\n--- Feature Statistics ---")
        for feature_idx in range(n_features):
            unique_values = np.unique(X[:, feature_idx])
            logger.info(f"Feature {feature_idx}: {len(unique_values)} unique values "
                       f"(range: {unique_values.min()}-{unique_values.max()})")

        # Train the model
        logger.info("\n--- Training CategoricalNB ---")
        self.model.fit(X, y)

        # Log learned parameters
        logger.info("\n--- Learned Parameters ---")
        logger.info(f"Class log priors shape: {self.model.class_log_prior_.shape}")
        logger.info(f"Feature log probabilities: list of {len(self.model.feature_log_prob_)} arrays")
        logger.info(f"  Each array shape: {self.model.feature_log_prob_[0].shape}")

        # Log class priors
        class_priors = np.exp(self.model.class_log_prior_)
        for class_idx, class_label in enumerate(self.classes):
            logger.info(f"P(class={class_label}) = {class_priors[class_idx]:.4f}")

        # Log sample feature probabilities for first class and feature
        logger.info("\n--- Sample Feature Probabilities (Class 0, Feature 0) ---")
        feature_probs = np.exp(self.model.feature_log_prob_[0][0, :])
        for bin_idx, prob in enumerate(feature_probs[:5]):  # First 5 bins
            logger.info(f"  P(feature_0={bin_idx}|class=0) = {prob:.4f}")

        logger.info("\n--- Training Complete ---")

    def predict(self, X):
        """
        Predict class labels for samples.

        Args:
            X: features to predict (n_samples, n_features) - discretized/binned

        Returns:
            numpy array of predicted class labels
        """
        logger.info("=" * 60)
        logger.info("PREDICTION PHASE")
        logger.info("=" * 60)

        n_samples = X.shape[0]
        logger.info(f"Predicting {n_samples} samples")

        # Get predictions
        predictions = self.model.predict(X)

        # Log first 5 predictions with details
        logger.info("\n--- Sample Predictions ---")
        if n_samples > 0:
            # Get log probabilities for detailed logging
            log_probs = self.model.predict_log_proba(X)

            for sample_idx in range(min(5, n_samples)):
                logger.info(f"\nSample {sample_idx}: {X[sample_idx]}")
                for class_idx, class_label in enumerate(self.classes):
                    logger.info(f"  log P(class={class_label}|features) = {log_probs[sample_idx, class_idx]:.4f}")
                logger.info(f"  -> Predicted class: {predictions[sample_idx]}")

        logger.info(f"\nPrediction complete: {n_samples} samples predicted")

        # Log prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        logger.info("Prediction distribution:")
        for class_label, count in zip(unique, counts):
            logger.info(f"  Class {class_label}: {count} predictions")

        return predictions

    def score(self, X, y):
        """
        Calculate accuracy on given data.

        Args:
            X: features
            y: true labels

        Returns:
            float: accuracy
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)

        logger.info(f"Accuracy: {accuracy*100:.2f}%")

        return accuracy

    def predict_proba(self, X):
        """
        Predict class probabilities for samples.

        Args:
            X: features to predict (n_samples, n_features)

        Returns:
            numpy array of class probabilities
        """
        return self.model.predict_proba(X)

    def plot_histograms(self, feature_names=None, class_names=None, output_dir='output'):
        """
        Generate and save histogram visualizations.

        Creates three plots:
        1. Feature histograms showing P(feature|class) for all features
        2. Class prior probabilities
        3. Prediction distribution (if predictions have been made)

        Args:
            feature_names: optional list of feature names
            class_names: optional list of class names
            output_dir: directory to save plots (default: 'output')
        """
        from ..utils.visualization import plot_feature_histograms, plot_class_priors

        logger.info("=" * 60)
        logger.info("GENERATING HISTOGRAM VISUALIZATIONS")
        logger.info("=" * 60)

        if not hasattr(self.model, 'feature_log_prob_'):
            logger.warning("Cannot plot histograms: model not trained yet")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract feature likelihoods from sklearn model
        # sklearn stores as list of arrays, we need to convert to (n_classes, n_features, n_bins)
        n_classes = len(self.classes)
        n_features = self.n_features
        n_bins = self.model.feature_log_prob_[0].shape[1]

        feature_likelihoods = np.zeros((n_classes, n_features, n_bins))
        for feature_idx in range(n_features):
            feature_likelihoods[:, feature_idx, :] = np.exp(self.model.feature_log_prob_[feature_idx])

        # Extract class priors
        class_priors = np.exp(self.model.class_log_prior_)

        # Plot feature histograms
        logger.info("Plotting feature histograms...")
        plot_feature_histograms(
            feature_likelihoods,
            self.classes,
            n_features,
            n_bins,
            feature_names=feature_names,
            class_names=class_names,
            output_path=str(output_path / 'sklearn_feature_histograms.png'),
            title_prefix='Sklearn - '
        )

        # Plot class priors
        logger.info("Plotting class priors...")
        plot_class_priors(
            class_priors,
            self.classes,
            class_names=class_names,
            output_path=str(output_path / 'sklearn_class_priors.png'),
            title_prefix='Sklearn - '
        )

        logger.info("Histogram visualizations complete")
        logger.info(f"Plots saved to {output_dir}/")
