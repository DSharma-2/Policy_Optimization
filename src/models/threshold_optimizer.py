"""
Threshold Optimizer

Optimizes classification threshold based on profit maximization.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns


# ========================================================================
# PROFIT-BASED THRESHOLD OPTIMIZATION
# ========================================================================

def compute_expected_profit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loan_amnt: np.ndarray,
    int_rate: np.ndarray,
    verbose: bool = False
) -> float:
    """
    Compute expected profit for a set of predictions.
    
    Profit calculation:
    - If approve and fully paid: profit = loan_amnt * (int_rate / 100)
    - If approve and default: loss = -loan_amnt
    - If deny: profit = 0
    
    Parameters
    ----------
    y_true : array
        True labels (0 = paid, 1 = default)
    y_pred : array
        Predicted labels (0 = deny, 1 = approve)
    loan_amnt : array
        Loan amounts
    int_rate : array
        Interest rates (in percentage)
    verbose : bool
        Print detailed breakdown
        
    Returns
    -------
    float
        Total expected profit
    """
    # Approved loans
    approved_mask = (y_pred == 1)
    
    # Among approved, which ones paid and which defaulted
    approved_paid_mask = approved_mask & (y_true == 0)
    approved_default_mask = approved_mask & (y_true == 1)
    
    # Compute profits
    profit_from_paid = np.sum(loan_amnt[approved_paid_mask] * (int_rate[approved_paid_mask] / 100))
    loss_from_default = np.sum(loan_amnt[approved_default_mask])
    
    total_profit = profit_from_paid - loss_from_default
    
    if verbose:
        n_approved = approved_mask.sum()
        n_denied = (~approved_mask).sum()
        n_paid = approved_paid_mask.sum()
        n_default = approved_default_mask.sum()
        
        print(f"Total loans:       {len(y_true):>8,}")
        print(f"Approved:          {n_approved:>8,}")
        print(f"Denied:            {n_denied:>8,}")
        print(f"  - Paid:          {n_paid:>8,}")
        print(f"  - Default:       {n_default:>8,}")
        print()
        print(f"Profit from paid:  ${profit_from_paid:>12,.2f}")
        print(f"Loss from default: ${loss_from_default:>12,.2f}")
        print(f"Net profit:        ${total_profit:>12,.2f}")
    
    return total_profit


def optimize_threshold_for_profit(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    loan_amnt: np.ndarray,
    int_rate: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Tuple[float, float, pd.DataFrame]:
    """
    Find optimal classification threshold that maximizes profit.
    
    Parameters
    ----------
    y_true : array
        True labels (0 = paid, 1 = default)
    y_pred_proba : array
        Predicted default probabilities
    loan_amnt : array
        Loan amounts
    int_rate : array
        Interest rates (in percentage)
    thresholds : array, optional
        Thresholds to test. If None, uses np.linspace(0.05, 0.95, 91)
    verbose : bool
        Print results
        
    Returns
    -------
    tuple
        (optimal_threshold, max_profit, results_df)
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Predict deny (0) for high default probability
        # So we need to invert: if prob(default) >= threshold, predict deny (0)
        y_pred_approve = (y_pred_proba < threshold).astype(int)
        
        profit = compute_expected_profit(
            y_true, y_pred_approve, loan_amnt, int_rate, verbose=False
        )
        
        # Additional metrics
        n_approved = y_pred_approve.sum()
        approval_rate = n_approved / len(y_true)
        
        # Among approved, what's the default rate
        if n_approved > 0:
            approved_defaults = (y_true[y_pred_approve == 1] == 1).sum()
            default_rate_approved = approved_defaults / n_approved
        else:
            default_rate_approved = 0
        
        results.append({
            'threshold': threshold,
            'profit': profit,
            'n_approved': n_approved,
            'approval_rate': approval_rate,
            'default_rate_approved': default_rate_approved,
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold
    optimal_idx = results_df['profit'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    max_profit = results_df.loc[optimal_idx, 'profit']
    
    if verbose:
        print("=" * 70)
        print("THRESHOLD OPTIMIZATION FOR PROFIT MAXIMIZATION")
        print("=" * 70)
        print(f"Thresholds tested: {len(thresholds)}")
        print(f"Range: {thresholds.min():.3f} to {thresholds.max():.3f}")
        print()
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        print(f"Maximum profit:    ${max_profit:,.2f}")
        print(f"Approval rate:     {results_df.loc[optimal_idx, 'approval_rate']:.2%}")
        print(f"Default rate (approved): {results_df.loc[optimal_idx, 'default_rate_approved']:.2%}")
        print("=" * 70)
    
    return optimal_threshold, max_profit, results_df


def plot_threshold_optimization(
    results_df: pd.DataFrame,
    optimal_threshold: float,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Visualize threshold optimization results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from optimize_threshold_for_profit
    optimal_threshold : float
        Optimal threshold
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Profit vs Threshold
    ax = axes[0, 0]
    ax.plot(results_df['threshold'], results_df['profit'], 'b-', linewidth=2)
    ax.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, label='Optimal')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Total Profit ($)', fontsize=12)
    ax.set_title('Profit vs Threshold', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    
    # Plot 2: Approval Rate vs Threshold
    ax = axes[0, 1]
    ax.plot(results_df['threshold'], results_df['approval_rate'] * 100, 'g-', linewidth=2)
    ax.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, label='Optimal')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Approval Rate (%)', fontsize=12)
    ax.set_title('Approval Rate vs Threshold', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Default Rate (Approved) vs Threshold
    ax = axes[1, 0]
    ax.plot(results_df['threshold'], results_df['default_rate_approved'] * 100, 'r-', linewidth=2)
    ax.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, label='Optimal')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Default Rate (Approved) (%)', fontsize=12)
    ax.set_title('Default Rate Among Approved vs Threshold', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Profit vs Approval Rate
    ax = axes[1, 1]
    ax.scatter(results_df['approval_rate'] * 100, results_df['profit'], 
               c=results_df['threshold'], cmap='viridis', s=50, alpha=0.6)
    optimal_row = results_df[results_df['threshold'] == optimal_threshold].iloc[0]
    ax.scatter(optimal_row['approval_rate'] * 100, optimal_row['profit'], 
               color='red', s=200, marker='*', edgecolors='black', linewidths=2, 
               label='Optimal', zorder=10)
    ax.set_xlabel('Approval Rate (%)', fontsize=12)
    ax.set_ylabel('Total Profit ($)', fontsize=12)
    ax.set_title('Profit vs Approval Rate', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    return fig


# ========================================================================
# THRESHOLD COMPARISON
# ========================================================================

def compare_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    loan_amnt: np.ndarray,
    int_rate: np.ndarray,
    thresholds: list = [0.3, 0.5, 0.7],
    threshold_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Compare multiple fixed thresholds.
    
    Parameters
    ----------
    y_true : array
        True labels
    y_pred_proba : array
        Predicted probabilities
    loan_amnt : array
        Loan amounts
    int_rate : array
        Interest rates
    thresholds : list
        List of thresholds to compare
    threshold_names : list, optional
        Names for each threshold
        
    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    if threshold_names is None:
        threshold_names = [f"Threshold {t:.2f}" for t in thresholds]
    
    results = []
    
    for threshold, name in zip(thresholds, threshold_names):
        y_pred_approve = (y_pred_proba < threshold).astype(int)
        
        profit = compute_expected_profit(
            y_true, y_pred_approve, loan_amnt, int_rate, verbose=False
        )
        
        n_approved = y_pred_approve.sum()
        approval_rate = n_approved / len(y_true)
        
        if n_approved > 0:
            approved_defaults = (y_true[y_pred_approve == 1] == 1).sum()
            default_rate = approved_defaults / n_approved
        else:
            default_rate = 0
        
        results.append({
            'name': name,
            'threshold': threshold,
            'profit': profit,
            'approval_rate': approval_rate,
            'default_rate_approved': default_rate,
            'n_approved': n_approved,
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Threshold optimizer module loaded")
