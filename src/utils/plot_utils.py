"""
Plotting Utilities

Common plotting functions for visualization across notebooks.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve


# Set default style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150


# ========================================================================
# ROC CURVE
# ========================================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = 'ROC Curve',
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None
) -> plt.Axes:
    """
    Plot ROC curve.
    
    Parameters
    ----------
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities
    title : str
        Plot title
    ax : matplotlib axes, optional
        Axes to plot on
    label : str, optional
        Label for the curve
        
    Returns
    -------
    matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    if label is None:
        label = f'ROC (AUC = {roc_auc:.3f})'
    else:
        label = f'{label} (AUC = {roc_auc:.3f})'
    
    ax.plot(fpr, tpr, lw=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    return ax


# ========================================================================
# PRECISION-RECALL CURVE
# ========================================================================

def plot_pr_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = 'Precision-Recall Curve',
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None
) -> plt.Axes:
    """
    Plot Precision-Recall curve.
    
    Parameters
    ----------
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities
    title : str
        Plot title
    ax : matplotlib axes, optional
        Axes to plot on
    label : str, optional
        Label for the curve
        
    Returns
    -------
    matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    if label is None:
        label = f'PR (AUC = {pr_auc:.3f})'
    else:
        label = f'{label} (AUC = {pr_auc:.3f})'
    
    ax.plot(recall, precision, lw=2, label=label)
    
    # Baseline (random classifier)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1, label=f'Baseline ({baseline:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    return ax


# ========================================================================
# CALIBRATION CURVE
# ========================================================================

def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    title: str = 'Calibration Curve',
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None
) -> plt.Axes:
    """
    Plot calibration curve.
    
    Parameters
    ----------
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities
    n_bins : int
        Number of bins
    title : str
        Plot title
    ax : matplotlib axes, optional
        Axes to plot on
    label : str, optional
        Label for the curve
        
    Returns
    -------
    matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    
    if label is None:
        label = 'Model'
    
    ax.plot(prob_pred, prob_true, marker='o', lw=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect Calibration')
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('True Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    return ax


# ========================================================================
# CONFUSION MATRIX
# ========================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = 'Confusion Matrix',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    labels : list, optional
        Class labels
    title : str
        Plot title
    ax : matplotlib axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = ['No Default', 'Default']
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    return ax


# ========================================================================
# FEATURE IMPORTANCE
# ========================================================================

def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = 'Feature Importance',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot feature importance.
    
    Parameters
    ----------
    feature_names : list
        Feature names
    importances : array
        Feature importance scores
    top_n : int
        Number of top features to show
    title : str
        Plot title
    ax : matplotlib axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Plot
    ax.barh(range(len(sorted_importances)), sorted_importances, color='steelblue')
    ax.set_yticks(range(len(sorted_importances)))
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    return ax


# ========================================================================
# DISTRIBUTION COMPARISON
# ========================================================================

def plot_distribution_comparison(
    data1: np.ndarray,
    data2: np.ndarray,
    label1: str = 'Set 1',
    label2: str = 'Set 2',
    title: str = 'Distribution Comparison',
    xlabel: str = 'Value',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot distribution comparison between two datasets.
    
    Parameters
    ----------
    data1, data2 : array
        Data arrays to compare
    label1, label2 : str
        Labels for the datasets
    title : str
        Plot title
    xlabel : str
        X-axis label
    ax : matplotlib axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(data1, bins=50, alpha=0.5, label=label1, density=True)
    ax.hist(data2, bins=50, alpha=0.5, label=label2, density=True)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    return ax


if __name__ == "__main__":
    print("Plotting utilities loaded")
