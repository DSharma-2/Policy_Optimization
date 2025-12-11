"""
Logging Utilities

Standardized logging setup for the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: str = 'INFO',
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Parameters
    ----------
    name : str
        Logger name
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    log_file : str, optional
        Path to log file. If None, logs to console only.
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_dict(logger: logging.Logger, d: dict, title: str = "Dictionary"):
    """
    Log a dictionary in a readable format.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    d : dict
        Dictionary to log
    title : str
        Title for the log entry
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(title)
    logger.info('=' * 70)
    for key, value in d.items():
        logger.info(f"  {key}: {value}")
    logger.info('=' * 70)


if __name__ == "__main__":
    # Example usage
    logger = setup_logger("test_logger", level="INFO")
    logger.info("Logger initialized")
    
    test_dict = {
        "model": "MLP",
        "accuracy": 0.85,
        "loss": 0.42
    }
    log_dict(logger, test_dict, "Model Results")
