"""
Leakage Filter Module

This module contains critical data leakage prevention logic.
LEAKAGE COLUMNS are those that contain information AFTER the lending decision
or during the loan lifecycle, which would not be available at decision time.

⚠️ CRITICAL: Removing these columns is the most important hiring signal.
We must justify why each column is removed.
"""

from typing import List, Dict
import pandas as pd


# ========================================================================
# LEAKAGE COLUMN DEFINITIONS
# ========================================================================

# These columns contain post-decision payment information
PAYMENT_LEAKAGE_COLS = [
    "total_pymnt",           # Total amount paid by borrower (known only after payments made)
    "total_pymnt_inv",       # Total amount paid by investor portion
    "total_rec_prncp",       # Total principal received to date
    "total_rec_int",         # Total interest received to date
    "total_rec_late_fee",    # Total late fees received
    "last_pymnt_d",          # Date of last payment (future information)
    "last_pymnt_amnt",       # Amount of last payment
    "next_pymnt_d",          # Date of next payment
]

# These columns contain recovery/collection information (only known after default)
RECOVERY_LEAKAGE_COLS = [
    "recoveries",            # Post charge-off gross recovery
    "collection_recovery_fee",  # Post charge-off collection fee
]

# These columns contain outstanding balance information (changes during loan lifecycle)
OUTSTANDING_LEAKAGE_COLS = [
    "out_prncp",            # Remaining outstanding principal
    "out_prncp_inv",        # Remaining outstanding principal for investors
]

# These columns contain settlement/hardship information (known only after issues arise)
SETTLEMENT_LEAKAGE_COLS = [
    "settlement_status",     # Settlement status
    "settlement_date",       # Settlement date
    "settlement_amount",     # Settlement amount
    "settlement_percentage", # Settlement percentage
    "settlement_term",       # Settlement term
    "debt_settlement_flag",  # Whether borrower is on debt settlement plan
    "hardship_flag",         # Whether borrower is on hardship plan
    "hardship_type",         # Type of hardship
    "hardship_status",       # Hardship status
    "hardship_start_date",   # When hardship began
    "hardship_end_date",     # When hardship ended
    "hardship_loan_status",  # Loan status during hardship
]

# These columns are derived from funding (investor-side information)
FUNDING_LEAKAGE_COLS = [
    "funded_amnt",          # Amount committed by investors (may differ from loan_amnt)
    "funded_amnt_inv",      # Amount funded by investors
    "investor_funds",       # Total investor funds
]

# Credit pull dates that happen after application
CREDIT_PULL_LEAKAGE_COLS = [
    "last_credit_pull_d",   # Most recent credit inquiry (may be after origination)
]

# Target-derived columns (these ARE the outcome we're predicting)
TARGET_LEAKAGE_COLS = [
    "loan_status",          # This IS the target (will be converted to binary)
    # Note: We keep loan_status temporarily to create our target, then drop it
]


# ========================================================================
# COMBINED LEAKAGE LIST
# ========================================================================

ALL_LEAKAGE_COLS = (
    PAYMENT_LEAKAGE_COLS +
    RECOVERY_LEAKAGE_COLS +
    OUTSTANDING_LEAKAGE_COLS +
    SETTLEMENT_LEAKAGE_COLS +
    FUNDING_LEAKAGE_COLS +
    CREDIT_PULL_LEAKAGE_COLS
)


# ========================================================================
# LEAKAGE REMOVAL FUNCTIONS
# ========================================================================

def get_leakage_report(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Generate a report of which leakage columns exist in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    dict
        Dictionary mapping category -> list of existing columns
    """
    report = {
        "Payment Info": [c for c in PAYMENT_LEAKAGE_COLS if c in df.columns],
        "Recovery Info": [c for c in RECOVERY_LEAKAGE_COLS if c in df.columns],
        "Outstanding Balance": [c for c in OUTSTANDING_LEAKAGE_COLS if c in df.columns],
        "Settlement/Hardship": [c for c in SETTLEMENT_LEAKAGE_COLS if c in df.columns],
        "Funding Info": [c for c in FUNDING_LEAKAGE_COLS if c in df.columns],
        "Credit Pull Dates": [c for c in CREDIT_PULL_LEAKAGE_COLS if c in df.columns],
    }
    
    # Add total count
    total = sum(len(v) for v in report.values())
    report["TOTAL_LEAKAGE_COLS"] = total
    
    return report


def drop_leakage_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove all leakage columns from the dataset.
    
    ⚠️ CRITICAL FUNCTION: This enforces the cleanroom constraint.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        If True, print removed columns
        
    Returns
    -------
    pd.DataFrame
        Dataframe with leakage columns removed
    """
    df_clean = df.copy()
    
    cols_to_drop = [c for c in ALL_LEAKAGE_COLS if c in df_clean.columns]
    
    if verbose:
        print(f"🚫 Removing {len(cols_to_drop)} leakage columns:")
        for cat, cols in get_leakage_report(df).items():
            if cat != "TOTAL_LEAKAGE_COLS" and cols:
                print(f"  - {cat}: {len(cols)} columns")
        print()
    
    df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')
    
    return df_clean


def validate_no_leakage(df: pd.DataFrame) -> bool:
    """
    Validate that no leakage columns remain in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to check
        
    Returns
    -------
    bool
        True if no leakage detected, raises ValueError otherwise
    """
    remaining_leakage = [c for c in ALL_LEAKAGE_COLS if c in df.columns]
    
    if remaining_leakage:
        raise ValueError(
            f"❌ LEAKAGE DETECTED! Found {len(remaining_leakage)} leakage columns:\n"
            f"{remaining_leakage}"
        )
    
    print("✅ No leakage columns detected. Dataset is clean.")
    return True


# ========================================================================
# SAFE FEATURE LIST
# ========================================================================

def get_safe_features() -> Dict[str, List[str]]:
    """
    Return a curated list of SAFE features that are available at decision time.
    
    These features represent information that would be available to a lender
    BEFORE making the approval decision.
    
    Returns
    -------
    dict
        Dictionary of safe feature categories
    """
    safe_features = {
        "applicant_profile": [
            "annual_inc",           # Annual income
            "emp_length",           # Employment length
            "home_ownership",       # Home ownership status
            "verification_status",  # Income verification status
            "addr_state",          # State of residence
            "zip_code",            # Zip code (first 3 digits)
        ],
        
        "credit_history": [
            "fico_range_low",      # FICO lower bound
            "fico_range_high",     # FICO upper bound
            "earliest_cr_line",    # Date of earliest credit line
            "open_acc",            # Number of open credit lines
            "pub_rec",             # Number of derogatory public records
            "revol_bal",           # Total credit revolving balance
            "revol_util",          # Revolving line utilization rate
            "total_acc",           # Total number of credit lines
            "delinq_2yrs",         # Number of 30+ days past-due in last 2 years
            "inq_last_6mths",      # Number of inquiries in last 6 months
            "mths_since_last_delinq",      # Months since last delinquency
            "mths_since_last_record",      # Months since last public record
            "pub_rec_bankruptcies",        # Number of public record bankruptcies
            "tax_liens",                   # Number of tax liens
        ],
        
        "loan_characteristics": [
            "loan_amnt",           # Requested loan amount
            "term",                # Loan term (36 or 60 months)
            "int_rate",            # Interest rate
            "installment",         # Monthly installment
            "grade",               # LC assigned loan grade
            "sub_grade",           # LC assigned loan subgrade
            "purpose",             # Purpose of loan
            "title",               # Loan title
            "dti",                 # Debt-to-income ratio
        ],
        
        "application_info": [
            "issue_d",             # Month loan was funded (for temporal split)
            "application_type",    # Individual or joint application
            "initial_list_status", # Initial listing status (W=whole, F=fractional)
        ],
    }
    
    return safe_features


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("LEAKAGE FILTER MODULE")
    print("=" * 70)
    print(f"\nTotal leakage columns defined: {len(ALL_LEAKAGE_COLS)}")
    print("\nCategories:")
    print(f"  - Payment info: {len(PAYMENT_LEAKAGE_COLS)}")
    print(f"  - Recovery info: {len(RECOVERY_LEAKAGE_COLS)}")
    print(f"  - Outstanding balance: {len(OUTSTANDING_LEAKAGE_COLS)}")
    print(f"  - Settlement/Hardship: {len(SETTLEMENT_LEAKAGE_COLS)}")
    print(f"  - Funding info: {len(FUNDING_LEAKAGE_COLS)}")
    print(f"  - Credit pull dates: {len(CREDIT_PULL_LEAKAGE_COLS)}")
    print("\n" + "=" * 70)
