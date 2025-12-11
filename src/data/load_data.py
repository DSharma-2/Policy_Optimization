"""
Data Loading Module

Handles loading and initial inspection of the LendingClub dataset.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import warnings


def load_lendingclub(
    path: str,
    nrows: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load the LendingClub dataset from CSV.
    
    Parameters
    ----------
    path : str
        Path to the CSV file
    nrows : int, optional
        Number of rows to load (for testing). If None, load all rows.
    verbose : bool
        Print loading information
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """
    if verbose:
        print(f"📂 Loading LendingClub data from: {path}")
        if nrows:
            print(f"   (loading first {nrows:,} rows only)")
    
    # Load with low_memory=False to prevent dtype warnings
    df = pd.read_csv(path, low_memory=False, nrows=nrows)
    
    if verbose:
        print(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    return df


def get_dataset_info(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Get comprehensive information about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Print information
        
    Returns
    -------
    dict
        Dictionary containing dataset statistics
    """
    info = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
        "n_numeric": len(df.select_dtypes(include=['number']).columns),
        "n_categorical": len(df.select_dtypes(include=['object']).columns),
        "n_datetime": len(df.select_dtypes(include=['datetime']).columns),
        "missing_cells": df.isnull().sum().sum(),
        "missing_pct": (df.isnull().sum().sum() / df.size) * 100,
    }
    
    if verbose:
        print("=" * 70)
        print("DATASET INFORMATION")
        print("=" * 70)
        print(f"Rows:              {info['n_rows']:>12,}")
        print(f"Columns:           {info['n_cols']:>12,}")
        print(f"  - Numeric:       {info['n_numeric']:>12,}")
        print(f"  - Categorical:   {info['n_categorical']:>12,}")
        print(f"  - Datetime:      {info['n_datetime']:>12,}")
        print(f"Memory:            {info['memory_mb']:>12.1f} MB")
        print(f"Missing cells:     {info['missing_cells']:>12,} ({info['missing_pct']:.1f}%)")
        print("=" * 70)
    
    return info


def check_target_distribution(
    df: pd.DataFrame,
    target_col: str = 'loan_status',
    verbose: bool = True
) -> pd.Series:
    """
    Check the distribution of the target variable (loan status).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    verbose : bool
        Print distribution
        
    Returns
    -------
    pd.Series
        Value counts of target variable
    """
    if target_col not in df.columns:
        warnings.warn(f"Column '{target_col}' not found in dataframe")
        return None
    
    counts = df[target_col].value_counts()
    pcts = df[target_col].value_counts(normalize=True) * 100
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"TARGET DISTRIBUTION: {target_col}")
        print("=" * 70)
        for status, count in counts.items():
            pct = pcts[status]
            print(f"{status:<30} {count:>10,}  ({pct:>5.2f}%)")
        print("=" * 70)
    
    return counts


def get_temporal_range(
    df: pd.DataFrame,
    date_col: str = 'issue_d',
    verbose: bool = True
) -> Tuple[str, str]:
    """
    Get the temporal range of the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_col : str
        Name of the date column
    verbose : bool
        Print temporal range
        
    Returns
    -------
    tuple
        (earliest_date, latest_date)
    """
    if date_col not in df.columns:
        warnings.warn(f"Column '{date_col}' not found in dataframe")
        return None, None
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        dates = pd.to_datetime(df[date_col], format='%b-%Y', errors='coerce')
    else:
        dates = df[date_col]
    
    earliest = dates.min()
    latest = dates.max()
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"TEMPORAL RANGE: {date_col}")
        print("=" * 70)
        print(f"Earliest: {earliest}")
        print(f"Latest:   {latest}")
        print(f"Duration: {(latest - earliest).days / 365.25:.1f} years")
        print("=" * 70)
    
    return earliest, latest


def get_missing_summary(
    df: pd.DataFrame,
    top_n: int = 20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Get summary of missing values by column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    top_n : int
        Number of top columns to show
    verbose : bool
        Print summary
        
    Returns
    -------
    pd.DataFrame
        Summary of missing values
    """
    missing = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_pct': (df.isnull().sum().values / len(df)) * 100
    })
    
    missing = missing[missing['missing_count'] > 0].sort_values(
        'missing_count', ascending=False
    )
    
    if verbose and len(missing) > 0:
        print("\n" + "=" * 70)
        print(f"TOP {min(top_n, len(missing))} COLUMNS WITH MISSING VALUES")
        print("=" * 70)
        for _, row in missing.head(top_n).iterrows():
            print(f"{row['column']:<40} {row['missing_count']:>10,}  ({row['missing_pct']:>5.1f}%)")
        print("=" * 70)
    elif verbose:
        print("\n✅ No missing values found!")
    
    return missing


def quick_profile(df: pd.DataFrame, date_col: str = 'issue_d', target_col: str = 'loan_status'):
    """
    Run a quick profiling of the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_col : str
        Name of the date column
    target_col : str
        Name of the target column
    """
    print("\n" + "🔍 " + "=" * 68)
    print("LENDINGCLUB DATASET QUICK PROFILE")
    print("=" * 70 + "\n")
    
    get_dataset_info(df, verbose=True)
    get_temporal_range(df, date_col=date_col, verbose=True)
    check_target_distribution(df, target_col=target_col, verbose=True)
    get_missing_summary(df, top_n=15, verbose=True)
    
    print("\n" + "=" * 70)
    print("✅ Quick profile complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("Data loading module ready.")
    print("Usage:")
    print("  from src.data.load_data import load_lendingclub, quick_profile")
    print("  df = load_lendingclub('data/raw/accepted_2007_to_2018Q4.csv')")
    print("  quick_profile(df)")
