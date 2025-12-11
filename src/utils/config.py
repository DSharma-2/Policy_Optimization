"""
Configuration Module

Central configuration for the LendingClub project.
"""

from pathlib import Path

# ========================================================================
# PATHS
# ========================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Model directories
MODELS_DIR = PROJECT_ROOT / 'models'

# Reports directories
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGS_DIR = REPORTS_DIR / 'figs'


# ========================================================================
# DATA SPLIT CONFIGURATION
# ========================================================================

TEMPORAL_SPLIT = {
    'train_end': '2015-12-31',
    'val_end': '2017-12-31',
    # Test starts after val_end (2018+)
}

DATE_COLUMN = 'issue_d'
TARGET_COLUMN = 'default'


# ========================================================================
# PREPROCESSING CONFIGURATION
# ========================================================================

MISSING_VALUE_CONFIG = {
    'numeric_strategy': 'median',
    'categorical_strategy': 'missing',
}


# ========================================================================
# MODEL CONFIGURATION
# ========================================================================

# Random seeds for reproducibility
RANDOM_SEED = 42

# MLP Configuration
MLP_CONFIG = {
    'hidden_dims': [512, 256, 64],
    'dropout_rates': [0.3, 0.2, 0.0],
    'batch_size': 1024,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'max_epochs': 100,
    'early_stopping_patience': 10,
}

# RL Configuration
RL_CONFIG = {
    'algorithms': ['CQL', 'IQL'],
    'n_epochs': 50,
    'batch_size': 256,
    'gamma': 0.99,
}


# ========================================================================
# EVALUATION CONFIGURATION
# ========================================================================

EVAL_METRICS = [
    'roc_auc',
    'pr_auc',
    'f1',
    'brier_score',
    'calibration',
]

# Threshold range for profit optimization
THRESHOLD_RANGE = (0.05, 0.95)
THRESHOLD_STEP = 0.01


# ========================================================================
# VISUALIZATION CONFIGURATION
# ========================================================================

PLOT_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 150,
    'style': 'whitegrid',
    'palette': 'Set2',
}


# ========================================================================
# LOGGING CONFIGURATION
# ========================================================================

LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}


if __name__ == "__main__":
    print("Configuration loaded")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
