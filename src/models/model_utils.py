"""
Model Utilities

Common utilities for model training, evaluation, and saving/loading.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns


# ========================================================================
# MODEL EVALUATION
# ========================================================================

def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation of binary classifier.
    
    Parameters
    ----------
    y_true : array
        True labels (0/1)
    y_pred_proba : array
        Predicted probabilities
    y_pred : array, optional
        Predicted labels. If None, computed from y_pred_proba with threshold
    threshold : float
        Classification threshold
    verbose : bool
        Print results
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    if y_pred is None:
        y_pred = (y_pred_proba >= threshold).astype(int)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # F1 Score
    f1 = f1_score(y_true, y_pred)
    
    # Brier Score
    brier = brier_score_loss(y_true, y_pred_proba)
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1_score': f1,
        'brier_score': brier,
        'accuracy': accuracy,
        'precision': precision_score,
        'recall': recall_score,
        'specificity': specificity,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }
    
    if verbose:
        print("=" * 70)
        print("CLASSIFICATION METRICS")
        print("=" * 70)
        print(f"ROC-AUC:       {roc_auc:.4f}")
        print(f"PR-AUC:        {pr_auc:.4f}")
        print(f"F1 Score:      {f1:.4f}")
        print(f"Brier Score:   {brier:.4f}")
        print(f"Accuracy:      {accuracy:.4f}")
        print(f"Precision:     {precision_score:.4f}")
        print(f"Recall:        {recall_score:.4f}")
        print(f"Specificity:   {specificity:.4f}")
        print()
        print(f"Confusion Matrix:")
        print(f"  TN: {tn:>8,}  |  FP: {fp:>8,}")
        print(f"  FN: {fn:>8,}  |  TP: {tp:>8,}")
        print("=" * 70)
    
    return metrics


# ========================================================================
# CALIBRATION
# ========================================================================

def compute_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve.
    
    Parameters
    ----------
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities
    n_bins : int
        Number of bins
        
    Returns
    -------
    tuple
        (true_probabilities, predicted_probabilities)
    """
    return calibration_curve(y_true, y_pred_proba, n_bins=n_bins)


# ========================================================================
# PYTORCH MODEL UTILITIES
# ========================================================================

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_pytorch_model(
    model: torch.nn.Module,
    filepath: str,
    metadata: Optional[Dict] = None
):
    """
    Save PyTorch model with metadata.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to save
    filepath : str
        Path to save file
    metadata : dict, optional
        Additional metadata to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if metadata:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, filepath)
    print(f"✅ Model saved to: {filepath}")


def load_pytorch_model(
    model: torch.nn.Module,
    filepath: str,
    device: str = 'cpu'
) -> Tuple[torch.nn.Module, Dict]:
    """
    Load PyTorch model from file.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model instance (architecture)
    filepath : str
        Path to saved model
    device : str
        Device to load model on
        
    Returns
    -------
    tuple
        (loaded_model, metadata)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    metadata = checkpoint.get('metadata', {})
    
    print(f"✅ Model loaded from: {filepath}")
    return model, metadata


# ========================================================================
# SKLEARN MODEL UTILITIES
# ========================================================================

def save_sklearn_model(
    model: Any,
    filepath: str,
    metadata: Optional[Dict] = None
):
    """
    Save sklearn model with pickle.
    
    Parameters
    ----------
    model : sklearn model
        Model to save
    filepath : str
        Path to save file
    metadata : dict, optional
        Additional metadata to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model': model,
        'model_class': model.__class__.__name__,
    }
    
    if metadata:
        save_dict['metadata'] = metadata
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f)
    
    print(f"✅ Model saved to: {filepath}")


def load_sklearn_model(filepath: str) -> Tuple[Any, Dict]:
    """
    Load sklearn model from pickle file.
    
    Parameters
    ----------
    filepath : str
        Path to saved model
        
    Returns
    -------
    tuple
        (loaded_model, metadata)
    """
    with open(filepath, 'rb') as f:
        save_dict = pickle.load(f)
    
    model = save_dict['model']
    metadata = save_dict.get('metadata', {})
    
    print(f"✅ Model loaded from: {filepath}")
    return model, metadata


# ========================================================================
# METRICS TRACKING
# ========================================================================

class MetricsTracker:
    """Track training and validation metrics over epochs."""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': {},
            'val_metrics': {},
        }
    
    def update(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        train_metrics: Optional[Dict] = None,
        val_metrics: Optional[Dict] = None
    ):
        """Update metrics for an epoch."""
        if train_loss is not None:
            self.metrics['train_loss'].append(train_loss)
        
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        
        if train_metrics:
            for key, value in train_metrics.items():
                if key not in self.metrics['train_metrics']:
                    self.metrics['train_metrics'][key] = []
                self.metrics['train_metrics'][key].append(value)
        
        if val_metrics:
            for key, value in val_metrics.items():
                if key not in self.metrics['val_metrics']:
                    self.metrics['val_metrics'][key] = []
                self.metrics['val_metrics'][key].append(value)
    
    def plot_losses(self, figsize=(10, 5)):
        """Plot training and validation losses."""
        fig, ax = plt.subplots(figsize=figsize)
        
        epochs = range(1, len(self.metrics['train_loss']) + 1)
        ax.plot(epochs, self.metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, self.metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        return fig
    
    def plot_metric(self, metric_name: str, figsize=(10, 5)):
        """Plot a specific metric over epochs."""
        fig, ax = plt.subplots(figsize=figsize)
        
        train_values = self.metrics['train_metrics'].get(metric_name, [])
        val_values = self.metrics['val_metrics'].get(metric_name, [])
        
        if train_values:
            epochs = range(1, len(train_values) + 1)
            ax.plot(epochs, train_values, 'b-', label=f'Train {metric_name}', linewidth=2)
        
        if val_values:
            epochs = range(1, len(val_values) + 1)
            ax.plot(epochs, val_values, 'r-', label=f'Val {metric_name}', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} over Epochs', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        return fig
    
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """Get the best epoch based on a metric."""
        if metric.startswith('val_'):
            metric_key = metric[4:]
            values = self.metrics['val_metrics'].get(metric_key, self.metrics.get(metric, []))
        else:
            values = self.metrics.get(metric, [])
        
        if not values:
            return 0
        
        if mode == 'min':
            return np.argmin(values) + 1
        else:
            return np.argmax(values) + 1
    
    def save(self, filepath: str):
        """Save metrics to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, filepath: str):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            self.metrics = json.load(f)


# ========================================================================
# EARLY STOPPING
# ========================================================================

class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Parameters
        ----------
        patience : int
            Number of epochs to wait for improvement
        min_delta : float
            Minimum change to qualify as improvement
        mode : str
            'min' or 'max' - whether to minimize or maximize metric
        verbose : bool
            Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, epoch: int, score: float) -> bool:
        """
        Check if training should stop.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        score : float
            Metric value to monitor
            
        Returns
        -------
        bool
            True if training should stop
        """
        if self.mode == 'min':
            score = -score
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        
        return self.early_stop


# ========================================================================
# DATA PREPARATION
# ========================================================================

def prepare_features_for_sklearn(
    df: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare features for sklearn models (handle categoricals).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names
    categorical_cols : list, optional
        List of categorical column names
        
    Returns
    -------
    tuple
        (X array, final_feature_names)
    """
    if categorical_cols is None:
        categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode categoricals
    if categorical_cols:
        df_encoded = pd.get_dummies(df[feature_cols], columns=categorical_cols, drop_first=True)
        X = df_encoded.values
        final_features = df_encoded.columns.tolist()
    else:
        X = df[feature_cols].values
        final_features = feature_cols
    
    return X, final_features


if __name__ == "__main__":
    print("Model utilities loaded")
