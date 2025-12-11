"""
Preprocessing Pipeline Module

Implements the complete Phase 1 preprocessing pipeline:
1. Remove leakage columns
2. Create binary target
3. Temporal train/val/test split
4. Handle missing values
5. Encode categorical variables
6. Scale numerical features
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import pickle
from pathlib import Path


# ========================================================================
# TARGET CREATION
# ========================================================================

def create_binary_target(
    df: pd.DataFrame,
    target_col: str = 'loan_status',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create binary default target from loan_status.
    
    Default = 1 if:
        - "Charged Off"
        - "Default"
        - "Does not meet the credit policy. Status:Charged Off"
    
    Default = 0 if:
        - "Fully Paid"
        - "Does not meet the credit policy. Status:Fully Paid"
    
    Drop rows with:
        - "Current"
        - "In Grace Period"
        - "Late (16-30 days)"
        - "Late (31-120 days)"
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the loan status column
    verbose : bool
        Print information
        
    Returns
    -------
    pd.DataFrame
        Dataframe with 'default' binary target column
    """
    df_out = df.copy()
    
    if target_col not in df_out.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Define default cases (bad loans)
    default_statuses = [
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off"
    ]
    
    # Define fully paid cases (good loans)
    paid_statuses = [
        "Fully Paid",
        "Does not meet the credit policy. Status:Fully Paid"
    ]
    
    # Define censored cases (to be removed)
    censored_statuses = [
        "Current",
        "In Grace Period",
        "Late (16-30 days)",
        "Late (31-120 days)",
        "Issued",  # May exist in some versions
    ]
    
    # Count before filtering
    n_before = len(df_out)
    
    # Filter to only completed loans
    mask_completed = df_out[target_col].isin(default_statuses + paid_statuses)
    df_out = df_out[mask_completed].copy()
    
    # Create binary target
    df_out['default'] = df_out[target_col].isin(default_statuses).astype(int)
    
    if verbose:
        n_after = len(df_out)
        n_removed = n_before - n_after
        n_default = df_out['default'].sum()
        n_paid = len(df_out) - n_default
        default_rate = (n_default / n_after) * 100
        
        print("=" * 70)
        print("TARGET CREATION")
        print("=" * 70)
        print(f"Rows before filtering: {n_before:,}")
        print(f"Rows after filtering:  {n_after:,}")
        print(f"Rows removed:          {n_removed:,} (censored/incomplete loans)")
        print()
        print(f"Default = 0 (paid):    {n_paid:,} ({100 - default_rate:.2f}%)")
        print(f"Default = 1 (default): {n_default:,} ({default_rate:.2f}%)")
        print("=" * 70)
    
    # Drop original loan_status column
    df_out = df_out.drop(columns=[target_col])
    
    return df_out


# ========================================================================
# TEMPORAL SPLIT
# ========================================================================

def temporal_train_val_test_split(
    df: pd.DataFrame,
    date_col: str = 'issue_d',
    train_end: str = '2015-12-31',
    val_end: str = '2017-12-31',
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform temporal split: train (2007-2015), val (2016-2017), test (2018).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_col : str
        Name of the date column
    train_end : str
        End date for training set (inclusive)
    val_end : str
        End date for validation set (inclusive)
    verbose : bool
        Print split information
        
    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    df_split = df.copy()
    
    # Parse date column
    if not pd.api.types.is_datetime64_any_dtype(df_split[date_col]):
        df_split[date_col] = pd.to_datetime(df_split[date_col], format='%b-%Y', errors='coerce')
    
    # Convert cutoff dates
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    
    # Split
    train_mask = df_split[date_col] <= train_end_dt
    val_mask = (df_split[date_col] > train_end_dt) & (df_split[date_col] <= val_end_dt)
    test_mask = df_split[date_col] > val_end_dt
    
    train_df = df_split[train_mask].copy()
    val_df = df_split[val_mask].copy()
    test_df = df_split[test_mask].copy()
    
    if verbose:
        print("\n" + "=" * 70)
        print("TEMPORAL TRAIN/VAL/TEST SPLIT")
        print("=" * 70)
        print(f"Train: up to {train_end}        → {len(train_df):>8,} rows ({len(train_df)/len(df)*100:>5.1f}%)")
        print(f"Val:   {train_end} to {val_end} → {len(val_df):>8,} rows ({len(val_df)/len(df)*100:>5.1f}%)")
        print(f"Test:  after {val_end}          → {len(test_df):>8,} rows ({len(test_df)/len(df)*100:>5.1f}%)")
        print()
        print(f"Train default rate: {train_df['default'].mean()*100:.2f}%")
        print(f"Val default rate:   {val_df['default'].mean()*100:.2f}%")
        print(f"Test default rate:  {test_df['default'].mean()*100:.2f}%")
        print("=" * 70)
    
    return train_df, val_df, test_df


# ========================================================================
# MISSING VALUE HANDLING
# ========================================================================

def handle_missing_values(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_strategy: str = 'median',
    categorical_strategy: str = 'missing',
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Handle missing values with train-fitted imputation.
    
    For numeric columns:
        - Fill with median (or mean)
        - Create missing indicator flag
    
    For categorical columns:
        - Fill with '<MISSING>' string
    
    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Train, validation, and test dataframes
    numeric_strategy : str
        'median' or 'mean'
    categorical_strategy : str
        'missing' (adds '<MISSING>' category) or 'mode'
    verbose : bool
        Print information
        
    Returns
    -------
    tuple
        (train_df, val_df, test_df, imputation_dict)
    """
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()
    
    imputation_dict = {
        'numeric': {},
        'categorical': {},
        'missing_flags': []
    }
    
    # Identify numeric and categorical columns
    numeric_cols = train_out.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = train_out.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target if present
    if 'default' in numeric_cols:
        numeric_cols.remove('default')
    
    # Handle numeric columns
    for col in numeric_cols:
        if train_out[col].isnull().sum() > 0:
            # Compute fill value from training set
            if numeric_strategy == 'median':
                fill_value = train_out[col].median()
            else:
                fill_value = train_out[col].mean()
            
            imputation_dict['numeric'][col] = fill_value
            
            # Create missing flag
            flag_col = f'{col}_missing'
            train_out[flag_col] = train_out[col].isnull().astype(int)
            val_out[flag_col] = val_out[col].isnull().astype(int)
            test_out[flag_col] = test_out[col].isnull().astype(int)
            imputation_dict['missing_flags'].append(flag_col)
            
            # Fill missing values
            train_out[col] = train_out[col].fillna(fill_value)
            val_out[col] = val_out[col].fillna(fill_value)
            test_out[col] = test_out[col].fillna(fill_value)
    
    # Handle categorical columns
    for col in categorical_cols:
        if train_out[col].isnull().sum() > 0:
            if categorical_strategy == 'missing':
                fill_value = '<MISSING>'
            else:
                fill_value = train_out[col].mode()[0] if len(train_out[col].mode()) > 0 else '<UNKNOWN>'
            
            imputation_dict['categorical'][col] = fill_value
            
            train_out[col] = train_out[col].fillna(fill_value)
            val_out[col] = val_out[col].fillna(fill_value)
            test_out[col] = test_out[col].fillna(fill_value)
    
    if verbose:
        print("\n" + "=" * 70)
        print("MISSING VALUE HANDLING")
        print("=" * 70)
        print(f"Numeric columns imputed:     {len(imputation_dict['numeric'])}")
        print(f"Categorical columns imputed: {len(imputation_dict['categorical'])}")
        print(f"Missing flags created:       {len(imputation_dict['missing_flags'])}")
        print("=" * 70)
    
    return train_out, val_out, test_out, imputation_dict


# ========================================================================
# SAVE/LOAD FUNCTIONS
# ========================================================================

def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = 'data/processed',
    verbose: bool = True
):
    """
    Save processed datasets to pickle files.
    
    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Processed dataframes
    output_dir : str
        Output directory
    verbose : bool
        Print information
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_pickle(output_path / 'train.pkl')
    val_df.to_pickle(output_path / 'val.pkl')
    test_df.to_pickle(output_path / 'test.pkl')
    
    if verbose:
        print("\n" + "=" * 70)
        print("SAVED PROCESSED DATA")
        print("=" * 70)
        print(f"Train: {output_path / 'train.pkl'}")
        print(f"Val:   {output_path / 'val.pkl'}")
        print(f"Test:  {output_path / 'test.pkl'}")
        print("=" * 70)


def load_processed_data(
    data_dir: str = 'data/processed',
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load processed datasets from pickle files.
    
    Parameters
    ----------
    data_dir : str
        Data directory
    verbose : bool
        Print information
        
    Returns
    -------
    tuple
        (train_df, val_df, test_df)
    """
    data_path = Path(data_dir)
    
    train_df = pd.read_pickle(data_path / 'train.pkl')
    val_df = pd.read_pickle(data_path / 'val.pkl')
    test_df = pd.read_pickle(data_path / 'test.pkl')
    
    if verbose:
        print("✅ Loaded processed data")
        print(f"   Train: {len(train_df):,} rows")
        print(f"   Val:   {len(val_df):,} rows")
        print(f"   Test:  {len(test_df):,} rows")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    print("Preprocessing module ready.")
