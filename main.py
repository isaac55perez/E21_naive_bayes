"""
Main Application - Iris Classification with Naive Bayes

Runs both NumPy and scikit-learn implementations of Naive Bayes
and compares their results.
"""

import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.data_loader import load_iris_data, split_data, discretize_features
from src.utils.metrics import calculate_accuracy, compare_predictions, log_confusion_summary
from src.naive_bayes_numpy import NaiveBayesNumpy, get_logger as get_numpy_logger
from src.naive_bayes_sklearn import NaiveBayesSklearn, get_logger as get_sklearn_logger


def setup_main_logger():
    """Configure main application logger."""
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)

    # Create output directory
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    # File handler
    log_file = output_dir / 'main.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    """Main application entry point."""
    logger = setup_main_logger()

    logger.info("=" * 80)
    logger.info("IRIS CLASSIFICATION - NAIVE BAYES COMPARISON")
    logger.info("NumPy (Histogram-based) vs Scikit-learn (CategoricalNB)")
    logger.info("=" * 80)

    # Configuration
    TRAIN_RATIO = 0.75
    RANDOM_SEED = 42
    N_BINS = 10
    ALPHA = 1.0  # Laplace smoothing

    logger.info(f"\nConfiguration:")
    logger.info(f"  Train/Test Split: {TRAIN_RATIO*100:.0f}% / {(1-TRAIN_RATIO)*100:.0f}%")
    logger.info(f"  Random Seed: {RANDOM_SEED}")
    logger.info(f"  Number of Bins: {N_BINS}")
    logger.info(f"  Laplace Smoothing (alpha): {ALPHA}")

    # Step 1: Load data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 80)

    features, labels, feature_names, class_names = load_iris_data()

    logger.info(f"Dataset loaded: {len(features)} samples, {len(feature_names)} features, {len(class_names)} classes")

    # Step 2: Split data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: SPLITTING DATA")
    logger.info("=" * 80)

    X_train, X_test, y_train, y_test = split_data(features, labels, TRAIN_RATIO, RANDOM_SEED)

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Step 3: Discretize features
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: DISCRETIZING FEATURES")
    logger.info("=" * 80)

    X_train_binned, X_test_binned, bin_edges = discretize_features(X_train, X_test, N_BINS)

    logger.info(f"Features discretized into {N_BINS} bins")

    # Step 4: Train and evaluate NumPy implementation
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: NUMPY IMPLEMENTATION")
    logger.info("=" * 80)

    logger.info("\n--- Training NumPy Classifier ---")
    nb_numpy = NaiveBayesNumpy(n_bins=N_BINS, alpha=ALPHA)
    nb_numpy.fit(X_train_binned, y_train)

    logger.info("\n--- Evaluating on Training Set ---")
    train_acc_numpy = nb_numpy.score(X_train_binned, y_train)
    logger.info(f"NumPy Training Accuracy: {train_acc_numpy*100:.2f}%")

    logger.info("\n--- Evaluating on Test Set ---")
    test_predictions_numpy = nb_numpy.predict(X_test_binned)
    test_acc_numpy = calculate_accuracy(y_test, test_predictions_numpy)
    logger.info(f"NumPy Test Accuracy: {test_acc_numpy*100:.2f}%")

    logger.info("\n--- Generating Histograms ---")
    nb_numpy.plot_histograms(feature_names=feature_names, class_names=class_names)

    # Step 5: Train and evaluate sklearn implementation
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: SCIKIT-LEARN IMPLEMENTATION")
    logger.info("=" * 80)

    logger.info("\n--- Training Sklearn Classifier ---")
    nb_sklearn = NaiveBayesSklearn(alpha=ALPHA)
    nb_sklearn.fit(X_train_binned, y_train)

    logger.info("\n--- Evaluating on Training Set ---")
    train_acc_sklearn = nb_sklearn.score(X_train_binned, y_train)
    logger.info(f"Sklearn Training Accuracy: {train_acc_sklearn*100:.2f}%")

    logger.info("\n--- Evaluating on Test Set ---")
    test_predictions_sklearn = nb_sklearn.predict(X_test_binned)
    test_acc_sklearn = calculate_accuracy(y_test, test_predictions_sklearn)
    logger.info(f"Sklearn Test Accuracy: {test_acc_sklearn*100:.2f}%")

    logger.info("\n--- Generating Histograms ---")
    nb_sklearn.plot_histograms(feature_names=feature_names, class_names=class_names)

    # Step 6: Compare results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: COMPARISON AND VERIFICATION")
    logger.info("=" * 80)

    logger.info("\n--- Accuracy Comparison ---")
    logger.info(f"Training Set:")
    logger.info(f"  NumPy:   {train_acc_numpy*100:.2f}%")
    logger.info(f"  Sklearn: {train_acc_sklearn*100:.2f}%")
    logger.info(f"  Difference: {abs(train_acc_numpy - train_acc_sklearn)*100:.2f}%")

    logger.info(f"\nTest Set:")
    logger.info(f"  NumPy:   {test_acc_numpy*100:.2f}%")
    logger.info(f"  Sklearn: {test_acc_sklearn*100:.2f}%")
    logger.info(f"  Difference: {abs(test_acc_numpy - test_acc_sklearn)*100:.2f}%")

    logger.info("\n--- Prediction Comparison (Training Set) ---")
    train_predictions_numpy = nb_numpy.predict(X_train_binned)
    train_predictions_sklearn = nb_sklearn.predict(X_train_binned)
    train_comparison = compare_predictions(
        train_predictions_numpy,
        train_predictions_sklearn,
        names=('NumPy', 'Sklearn')
    )

    logger.info("\n--- Prediction Comparison (Test Set) ---")
    test_comparison = compare_predictions(
        test_predictions_numpy,
        test_predictions_sklearn,
        names=('NumPy', 'Sklearn')
    )

    # Step 7: Detailed per-class analysis
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: PER-CLASS ANALYSIS")
    logger.info("=" * 80)

    logger.info("\n--- NumPy Test Set Performance ---")
    log_confusion_summary(y_test, test_predictions_numpy, class_names)

    logger.info("\n--- Sklearn Test Set Performance ---")
    log_confusion_summary(y_test, test_predictions_sklearn, class_names)

    # Step 8: Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)

    logger.info("\nResults:")
    logger.info(f"  1. Both implementations trained successfully")
    logger.info(f"  2. NumPy test accuracy: {test_acc_numpy*100:.2f}%")
    logger.info(f"  3. Sklearn test accuracy: {test_acc_sklearn*100:.2f}%")
    logger.info(f"  4. Prediction agreement on test set: {test_comparison['agreement_rate']*100:.2f}%")

    # Compatibility verification
    logger.info("\nCompatibility Verification:")
    accuracy_diff = abs(test_acc_numpy - test_acc_sklearn)
    agreement_rate = test_comparison['agreement_rate']

    if accuracy_diff < 0.05:
        logger.info(f"  ✓ Accuracy difference < 5% ({accuracy_diff*100:.2f}%)")
    else:
        logger.info(f"  ✗ Accuracy difference >= 5% ({accuracy_diff*100:.2f}%)")

    if agreement_rate > 0.90:
        logger.info(f"  ✓ Prediction agreement > 90% ({agreement_rate*100:.2f}%)")
    else:
        logger.info(f"  ✗ Prediction agreement <= 90% ({agreement_rate*100:.2f}%)")

    if test_acc_numpy > 0.60 and test_acc_sklearn > 0.60:
        logger.info(f"  ✓ Both implementations achieve > 60% test accuracy")
    else:
        logger.info(f"  ✗ One or both implementations below 60% test accuracy")

    logger.info("\nLog files saved to output/:")
    logger.info("  - naive_bayes_numpy.log")
    logger.info("  - naive_bayes_sklearn.log")
    logger.info("  - main.log")

    logger.info("\nVisualization files saved to output/:")
    logger.info("  - numpy_feature_histograms.png")
    logger.info("  - numpy_class_priors.png")
    logger.info("  - sklearn_feature_histograms.png")
    logger.info("  - sklearn_class_priors.png")

    logger.info("\n" + "=" * 80)
    logger.info("EXECUTION COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
