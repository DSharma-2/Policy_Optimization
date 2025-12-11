"""
Feature Engineering Module

Implements feature engineering for LendingClub dataset:
1. FICO score processing
2. Employment length conversion
3. Date-based features (credit age, etc.)
4. Engineered ratio features
5. Categorical encoding
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from datetime import datetime
import warnings


# ========================================================================
# FICO SCORE FEATURES
# ========================================================================

def process_fico_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process FICO score range columns into single mean feature.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with 'fico_mean' column
    """
    df_out = df.copy()
    
    if 'fico_range_low' in df_out.columns and 'fico_range_high' in df_out.columns:
        df_out['fico_mean'] = (df_out['fico_range_low'] + df_out['fico_range_high']) / 2
        
        # Create FICO buckets for interpretability
        df_out['fico_bucket'] = pd.cut(
            df_out['fico_mean'],
            bins=[0, 580, 640, 680, 720, 850],
            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        )
    
    return df_out


# ========================================================================
# EMPLOYMENT LENGTH
# ========================================================================

def convert_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert employment length from string to numeric.
    
    Examples:
        "10+ years" -> 10
        "< 1 year" -> 0
        "2 years" -> 2
        
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with 'emp_length_numeric' column
    """
    df_out = df.copy()
    
    if 'emp_length' not in df_out.columns:
        return df_out
    
    def parse_emp_length(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().lower()
        
        if '10+' in val or '10 +' in val:
            return 10
        elif '< 1' in val or '<1' in val:
            return 0
        else:
            # Extract first number
            import re
            match = re.search(r'\d+', val)
            if match:
                return int(match.group())
        return np.nan
    
    df_out['emp_length_numeric'] = df_out['emp_length'].apply(parse_emp_length)
    
    return df_out


# ========================================================================
# DATE-BASED FEATURES
# ========================================================================

def create_credit_age_features(
    df: pd.DataFrame,
    earliest_cr_line_col: str = 'earliest_cr_line',
    issue_date_col: str = 'issue_d'
) -> pd.DataFrame:
    """
    Calculate credit history age in years.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    earliest_cr_line_col : str
        Column with earliest credit line date
    issue_date_col : str
        Column with loan issue date
        
    Returns
    -------
    pd.DataFrame
        Dataframe with 'credit_age_years' column
    """
    df_out = df.copy()
    
    if earliest_cr_line_col not in df_out.columns:
        return df_out
    
    # Parse dates
    earliest_cr = pd.to_datetime(df_out[earliest_cr_line_col], format='%b-%Y', errors='coerce')
    
    if issue_date_col in df_out.columns:
        issue_date = pd.to_datetime(df_out[issue_date_col], format='%b-%Y', errors='coerce')
    else:
        # Use current date as fallback
        issue_date = pd.Timestamp.now()
    
    # Calculate age in years
    df_out['credit_age_years'] = (issue_date - earliest_cr).dt.days / 365.25
    
    # Cap unrealistic values
    df_out['credit_age_years'] = df_out['credit_age_years'].clip(lower=0, upper=100)
    
    return df_out


def parse_issue_date(df: pd.DataFrame, date_col: str = 'issue_d') -> pd.DataFrame:
    """
    Parse issue date and extract year, month, quarter.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    date_col : str
        Date column name
        
    Returns
    -------
    pd.DataFrame
        Dataframe with date features
    """
    df_out = df.copy()
    
    if date_col not in df_out.columns:
        return df_out
    
    date_parsed = pd.to_datetime(df_out[date_col], format='%b-%Y', errors='coerce')
    
    df_out['issue_year'] = date_parsed.dt.year
    df_out['issue_month'] = date_parsed.dt.month
    df_out['issue_quarter'] = date_parsed.dt.quarter
    
    return df_out


# ========================================================================
# ENGINEERED RATIO FEATURES
# ========================================================================

def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered ratio features for risk assessment.
    
    Features:
        - loan_amnt_to_income: Loan amount / annual income (leverage)
        - income_log: log(annual_inc + 1)
        - installment_to_income: Monthly installment / monthly income
        - revol_util_clean: Revolving utilization (capped at 100)
        
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with ratio features
    """
    df_out = df.copy()
    
    # Loan amount to income ratio
    if 'loan_amnt' in df_out.columns and 'annual_inc' in df_out.columns:
        df_out['loan_amnt_to_income'] = df_out['loan_amnt'] / (df_out['annual_inc'] + 1)
        df_out['loan_amnt_to_income'] = df_out['loan_amnt_to_income'].clip(upper=10)  # Cap outliers
    
    # Log income (reduces skewness)
    if 'annual_inc' in df_out.columns:
        df_out['income_log'] = np.log1p(df_out['annual_inc'])
    
    # Installment to monthly income
    if 'installment' in df_out.columns and 'annual_inc' in df_out.columns:
        monthly_income = df_out['annual_inc'] / 12
        df_out['installment_to_income'] = df_out['installment'] / (monthly_income + 1)
        df_out['installment_to_income'] = df_out['installment_to_income'].clip(upper=1)
    
    # Clean revolving utilization
    if 'revol_util' in df_out.columns:
        df_out['revol_util_clean'] = df_out['revol_util'].clip(lower=0, upper=100)
    
    # Recent inquiry flag
    if 'inq_last_6mths' in df_out.columns:
        df_out['recent_inq_flag'] = (df_out['inq_last_6mths'] > 0).astype(int)
    
    # Delinquency flag
    if 'delinq_2yrs' in df_out.columns:
        df_out['delinq_flag'] = (df_out['delinq_2yrs'] > 0).astype(int)
    
    return df_out


# ========================================================================
# INTEREST RATE PROCESSING
# ========================================================================

def clean_interest_rate(df: pd.DataFrame, int_rate_col: str = 'int_rate') -> pd.DataFrame:
    """
    Clean interest rate column (remove % symbol if present).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    int_rate_col : str
        Interest rate column name
        
    Returns
    -------
    pd.DataFrame
        Dataframe with cleaned int_rate
    """
    df_out = df.copy()
    
    if int_rate_col not in df_out.columns:
        return df_out
    
    # If string, remove % and convert to float
    if df_out[int_rate_col].dtype == 'object':
        df_out[int_rate_col] = df_out[int_rate_col].str.replace('%', '').astype(float)
    
    return df_out


# ========================================================================
# TERM PROCESSING
# ========================================================================

def convert_term_to_months(df: pd.DataFrame, term_col: str = 'term') -> pd.DataFrame:
    """
    Convert term from string to integer months.
    
    Examples:
        " 36 months" -> 36
        " 60 months" -> 60
        
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    term_col : str
        Term column name
        
    Returns
    -------
    pd.DataFrame
        Dataframe with 'term_months' column
    """
    df_out = df.copy()
    
    if term_col not in df_out.columns:
        return df_out
    
    if df_out[term_col].dtype == 'object':
        df_out['term_months'] = df_out[term_col].str.extract(r'(\d+)').astype(float)
    else:
        df_out['term_months'] = df_out[term_col]
    
    return df_out


# ========================================================================
# MASTER FEATURE ENGINEERING PIPELINE
# ========================================================================

def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (after leakage removal)
    verbose : bool
        Print progress
        
    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features
    """
    if verbose:
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING")
        print("=" * 70)
    
    df_out = df.copy()
    n_cols_before = len(df_out.columns)
    
    # Apply transformations
    df_out = process_fico_scores(df_out)
    if verbose:
        print("✅ FICO scores processed")
    
    df_out = convert_emp_length(df_out)
    if verbose:
        print("✅ Employment length converted")
    
    df_out = create_credit_age_features(df_out)
    if verbose:
        print("✅ Credit age calculated")
    
    df_out = parse_issue_date(df_out)
    if verbose:
        print("✅ Issue date parsed")
    
    df_out = create_ratio_features(df_out)
    if verbose:
        print("✅ Ratio features created")
    
    df_out = clean_interest_rate(df_out)
    if verbose:
        print("✅ Interest rate cleaned")
    
    df_out = convert_term_to_months(df_out)
    if verbose:
        print("✅ Term converted to months")
    
    n_cols_after = len(df_out.columns)
    
    if verbose:
        print()
        print(f"Features before: {n_cols_before}")
        print(f"Features after:  {n_cols_after}")
        print(f"Features added:  {n_cols_after - n_cols_before}")
        print("=" * 70)
    
    return df_out


# ========================================================================
# FEATURE SELECTION
# ========================================================================

def get_model_features() -> Dict[str, List[str]]:
    """
    Get curated list of features for modeling.
    
    Returns
    -------
    dict
        Dictionary of feature categories
    """
    features = {
        "applicant_profile": [
            "annual_inc",
            "income_log",
            "emp_length_numeric",
            "home_ownership",
            "verification_status",
            "addr_state",
        ],
        
        "credit_history": [
            "fico_mean",
            "fico_bucket",
            "credit_age_years",
            "delinq_2yrs",
            "delinq_flag",
            "inq_last_6mths",
            "recent_inq_flag",
            "open_acc",
            "pub_rec",
            "revol_bal",
            "revol_util",
            "revol_util_clean",
            "total_acc",
        ],
        
        "loan_characteristics": [
            "loan_amnt",
            "int_rate",
            "installment",
            "term_months",
            "grade",
            "sub_grade",
            "purpose",
            "dti",
        ],
        
        "engineered_features": [
            "loan_amnt_to_income",
            "installment_to_income",
        ],
    }
    
    return features


def select_model_features(df: pd.DataFrame, feature_dict: Dict[str, List[str]] = None) -> List[str]:
    """
    Select features that exist in the dataframe for modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_dict : dict, optional
        Dictionary of features (if None, uses default)
        
    Returns
    -------
    list
        List of feature names that exist in df
    """
    if feature_dict is None:
        feature_dict = get_model_features()
    
    all_features = []
    for category, features in feature_dict.items():
        all_features.extend(features)
    
    # Filter to only existing columns
    existing_features = [f for f in all_features if f in df.columns]
    
    return existing_features


if __name__ == "__main__":
    print("Feature engineering module ready.")
    print("\nAvailable feature categories:")
    features = get_model_features()
    for category, feats in features.items():
        print(f"  {category}: {len(feats)} features")
