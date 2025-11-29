"""
NumPy-based Histogram Naive Bayes Classifier

Manual implementation of Naive Bayes using histogram-based probability estimation.
"""

import numpy as np
from pathlib import Path
from .logger_config import get_logger

logger = get_logger()


class NaiveBayesNumpy:
    """
    Histogram-based Naive Bayes classifier using only NumPy.

    This classifier:
    1. Calculates class priors P(class)
    2. Builds histograms for each feature per class
    3. Estimates P(feature=value|class) from histograms
    4. Predicts using Bayes theorem: P(class|features) ∝ P(class) * ∏P(feature|class)
    5. Selects class with maximum posterior probability (argmax)
    """

    def __init__(self, n_bins=10, alpha=1.0):
        """
        Initialize Naive Bayes classifier.

        Args:
            n_bins: number of bins per feature (must match discretization)
            alpha: Laplace smoothing parameter (default: 1.0)
        """
        self.n_bins = n_bins
        self.alpha = alpha  # Laplace smoothing
        self.class_priors = None
        self.feature_likelihoods = None
        self.classes = None
        self.n_features = None

        logger.info(f"Initialized NaiveBayesNumpy with {n_bins} bins, alpha={alpha}")

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.

        Args:
            X: training features (n_samples, n_features) - discretized/binned
            y: training labels (n_samples,)
        """
        logger.info("=" * 60)
        logger.info("TRAINING NUMPY NAIVE BAYES CLASSIFIER")
        logger.info("=" * 60)

        n_samples, n_features = X.shape
        self.n_features = n_features
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        logger.info(f"Training samples: {n_samples}")
        logger.info(f"Features: {n_features}")
        logger.info(f"Classes: {self.classes}")
        logger.info(f"Number of classes: {n_classes}")

        # Step 1: Calculate class priors P(class)
        logger.info("\n--- Step 1: Calculating Class Priors ---")
        self.class_priors = np.zeros(n_classes)

        for class_idx, class_label in enumerate(self.classes):
            class_count = np.sum(y == class_label)
            self.class_priors[class_idx] = class_count / n_samples
            logger.info(f"P(class={class_label}) = {self.class_priors[class_idx]:.4f} "
                       f"({class_count}/{n_samples} samples)")

        # Step 2: Build histograms and calculate likelihoods P(feature=value|class)
        logger.info("\n--- Step 2: Building Histograms and Calculating Likelihoods ---")

        # Shape: (n_classes, n_features, n_bins)
        self.feature_likelihoods = np.zeros((n_classes, n_features, self.n_bins))

        for class_idx, class_label in enumerate(self.classes):
            class_mask = y == class_label
            X_class = X[class_mask]
            n_class_samples = len(X_class)

            logger.info(f"\nClass {class_label} ({n_class_samples} samples):")

            for feature_idx in range(n_features):
                feature_values = X_class[:, feature_idx]

                # Count occurrences of each bin value
                for bin_idx in range(self.n_bins):
                    count = np.sum(feature_values == bin_idx)

                    # Apply Laplace smoothing
                    # P(feature=bin|class) = (count + alpha) / (n_class_samples + alpha * n_bins)
                    likelihood = (count + self.alpha) / (n_class_samples + self.alpha * self.n_bins)
                    self.feature_likelihoods[class_idx, feature_idx, bin_idx] = likelihood

                    if count > 0:  # Only log non-zero counts for brevity
                        logger.debug(f"  Feature {feature_idx}, Bin {bin_idx}: "
                                   f"count={count}, P(f{feature_idx}={bin_idx}|c={class_label})={likelihood:.4f}")

                # Log summary for this feature
                logger.info(f"  Feature {feature_idx}: histogram built with {self.n_bins} bins")

        logger.info("\n--- Training Complete ---")
        logger.info(f"Class priors shape: {self.class_priors.shape}")
        logger.info(f"Feature likelihoods shape: {self.feature_likelihoods.shape}")

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

        predictions = np.zeros(n_samples, dtype=int)

        for sample_idx in range(n_samples):
            sample = X[sample_idx]
            log_posteriors = self._calculate_log_posterior(sample)

            # Argmax: select class with highest posterior
            predicted_class_idx = np.argmax(log_posteriors)
            predictions[sample_idx] = self.classes[predicted_class_idx]

            # Log first 5 predictions in detail
            if sample_idx < 5:
                logger.info(f"\nSample {sample_idx}: {sample}")
                for class_idx, class_label in enumerate(self.classes):
                    logger.info(f"  log P(class={class_label}|features) = {log_posteriors[class_idx]:.4f}")
                logger.info(f"  -> Predicted class: {predictions[sample_idx]}")

        logger.info(f"\nPrediction complete: {n_samples} samples predicted")

        # Log prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        logger.info("Prediction distribution:")
        for class_label, count in zip(unique, counts):
            logger.info(f"  Class {class_label}: {count} predictions")

        return predictions

    def _calculate_log_posterior(self, sample):
        """
        Calculate log posterior probabilities for a single sample.

        Uses log probabilities to avoid numerical underflow.

        log P(class|features) = log P(class) + Σ log P(feature|class)

        Args:
            sample: single feature vector (n_features,)

        Returns:
            numpy array of log posterior probabilities for each class
        """
        n_classes = len(self.classes)
        log_posteriors = np.zeros(n_classes)

        for class_idx in range(n_classes):
            # Start with log prior
            log_posterior = np.log(self.class_priors[class_idx])

            # Add log likelihoods for each feature
            for feature_idx in range(self.n_features):
                bin_value = int(sample[feature_idx])

                # Get likelihood for this feature value given the class
                likelihood = self.feature_likelihoods[class_idx, feature_idx, bin_value]

                # Add log likelihood (product becomes sum in log space)
                log_posterior += np.log(likelihood)

            log_posteriors[class_idx] = log_posterior

        return log_posteriors

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

        if self.feature_likelihoods is None:
            logger.warning("Cannot plot histograms: model not trained yet")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Plot feature histograms
        logger.info("Plotting feature histograms...")
        plot_feature_histograms(
            self.feature_likelihoods,
            self.classes,
            self.n_features,
            self.n_bins,
            feature_names=feature_names,
            class_names=class_names,
            output_path=str(output_path / 'numpy_feature_histograms.png'),
            title_prefix='NumPy - '
        )

        # Plot class priors
        logger.info("Plotting class priors...")
        plot_class_priors(
            self.class_priors,
            self.classes,
            class_names=class_names,
            output_path=str(output_path / 'numpy_class_priors.png'),
            title_prefix='NumPy - '
        )

        logger.info("Histogram visualizations complete")
        logger.info(f"Plots saved to {output_dir}/")
