"""
Logger Configuration for NumPy Naive Bayes Implementation
"""

import logging
from pathlib import Path


def get_logger(name='naive_bayes_numpy'):
    """
    Configure and return logger for NumPy implementation.

    Logs are written to output/naive_bayes_numpy.log

    Args:
        name: logger name

    Returns:
        logging.Logger: configured logger
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create output directory if it doesn't exist
        output_dir = Path(__file__).parent.parent.parent / 'output'
        output_dir.mkdir(exist_ok=True)

        # File handler
        log_file = output_dir / 'naive_bayes_numpy.log'
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)

        # Console handler (optional, for immediate feedback)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
