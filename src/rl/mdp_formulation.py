"""
MDP Formulation for Loan Approval

Converts loan approval problem into a Markov Decision Process (MDP) for offline RL.

Components:
- State: Applicant features (FICO, DTI, income, etc.)
- Action: {0: Reject, 1: Approve}
- Reward: Profit if paid (loan_amnt × int_rate), loss if default (-loan_amnt)
- Transition: Terminal (one-step episodic)

Author: AI Toolkit
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MDPConfig:
    """Configuration for MDP formulation."""
    state_features: list
    action_space: list = None
    reward_scale: float = 0.01  # Scale rewards to [-1, 1] range
    normalize_states: bool = True
    
    def __post_init__(self):
        if self.action_space is None:
            self.action_space = [0, 1]  # 0: Reject, 1: Approve


class LoanApprovalMDP:
    """
    Markov Decision Process formulation for loan approval.
    
    This converts the supervised learning problem into an RL problem:
    - State: Applicant features
    - Action: Approve (1) or Reject (0)
    - Reward: Financial outcome (profit or loss)
    - Policy: π(action|state) - learned approval strategy
    """
    
    def __init__(self, config: MDPConfig):
        """
        Initialize MDP formulation.
        
        Args:
            config: MDP configuration
        """
        self.config = config
        self.state_dim = len(config.state_features)
        self.action_dim = len(config.action_space)
        self.state_mean = None
        self.state_std = None
        
    def fit_normalization(self, states: np.ndarray) -> None:
        """
        Fit state normalization parameters.
        
        Args:
            states: Training states (n_samples, state_dim)
        """
        # Use nanmean and nanstd to handle NaN values
        self.state_mean = np.nanmean(states, axis=0)
        self.state_std = np.nanstd(states, axis=0)
        
        # Replace NaN in mean with 0
        self.state_mean = np.where(np.isnan(self.state_mean), 0.0, self.state_mean)
        
        # For std, replace 0 or NaN with 1 (to avoid division by zero)
        self.state_std = np.where((self.state_std == 0) | np.isnan(self.state_std), 
                                  1.0, self.state_std)
        
        print(f"✅ Fitted normalization: mean={self.state_mean[:3]}, std={self.state_std[:3]}")
        
    def normalize_states(self, states: np.ndarray) -> np.ndarray:
        """
        Normalize states using fitted parameters.
        
        Args:
            states: States to normalize (n_samples, state_dim)
            
        Returns:
            Normalized states
        """
        if not self.config.normalize_states:
            return states
            
        if self.state_mean is None:
            raise ValueError("Must call fit_normalization() first")
        
        # Normalize and replace any remaining NaN with 0
        normalized = (states - self.state_mean) / self.state_std
        normalized = np.where(np.isnan(normalized), 0.0, normalized)
        
        return normalized
        
    def compute_reward(
        self,
        action: np.ndarray,
        outcome: np.ndarray,
        loan_amnt: np.ndarray,
        int_rate: np.ndarray,
        default_penalty_multiplier: float = 5.0
    ) -> np.ndarray:
        """
        Compute reward based on action and outcome.
        
        IMPROVED Reward Structure (to fix policy collapse):
        - Action = Reject (0): reward = 0 (no profit, no loss)
        - Action = Approve (1):
          - If paid (outcome=0): reward = loan_amnt × int_rate (interest earned)
          - If default (outcome=1): reward = -default_penalty_multiplier × loan_amnt
            (heavier penalty to encourage risk-averse decisions)
        
        The default_penalty_multiplier (default=5.0) makes defaults much more costly,
        forcing CQL to learn to reject risky applicants instead of approving all.
        
        Args:
            action: Actions taken (n_samples,) - 0 or 1
            outcome: Loan outcomes (n_samples,) - 0 (paid) or 1 (default)
            loan_amnt: Loan amounts (n_samples,)
            int_rate: Interest rates (n_samples,)
            default_penalty_multiplier: Penalty multiplier for defaults (default=5.0)
            
        Returns:
            Rewards (n_samples,)
        """
        rewards = np.zeros_like(action, dtype=np.float32)
        
        # Only approved loans get rewards/penalties
        approved_mask = (action == 1)
        
        # Paid loans: profit = interest earned (loan_amnt × int_rate)
        paid_mask = approved_mask & (outcome == 0)
        rewards[paid_mask] = loan_amnt[paid_mask] * int_rate[paid_mask]
        
        # Defaulted loans: HEAVY PENALTY = -multiplier × loan_amnt
        # This encourages CQL to learn selective approval
        default_mask = approved_mask & (outcome == 1)
        rewards[default_mask] = -default_penalty_multiplier * loan_amnt[default_mask]
        
        # Scale rewards to reasonable range
        if self.config.reward_scale is not None:
            rewards = rewards * self.config.reward_scale
            
        return rewards
        
    def create_dataset(
        self,
        df: pd.DataFrame,
        include_loan_info: bool = True,
        add_synthetic_rejections: bool = True,
        rejection_rate: float = 0.2,
        default_penalty_multiplier: float = 5.0
    ) -> Dict[str, np.ndarray]:
        """
        Create offline RL dataset from dataframe.
        
        IMPROVED: Adds synthetic rejection samples to fix the "100% approval" problem.
        Since LendingClub data only has approved loans, CQL has no examples of rejections.
        We synthesize rejections for high-risk loans that defaulted.
        
        Args:
            df: Dataframe with features, default, loan_amnt, int_rate
            include_loan_info: Whether to include loan_amnt and int_rate
            add_synthetic_rejections: Add synthetic rejection samples (CRITICAL FIX)
            rejection_rate: Fraction of high-risk loans to mark as "rejected" (default=0.2)
            default_penalty_multiplier: Penalty multiplier for defaults (default=5.0)
            
        Returns:
            Dictionary with:
                - states: (n_samples, state_dim)
                - actions: (n_samples,) - includes synthetic rejections
                - rewards: (n_samples,)
                - next_states: (n_samples, state_dim) - same as states (terminal)
                - terminals: (n_samples,) - all True (one-step episodes)
                - loan_amnt: (n_samples,) - optional
                - int_rate: (n_samples,) - optional
        """
        # Extract states (features)
        state_cols = [c for c in self.config.state_features if c in df.columns]
        if len(state_cols) != len(self.config.state_features):
            missing = set(self.config.state_features) - set(state_cols)
            print(f"⚠️  Missing state features: {missing}")
            
        states = df[state_cols].values.astype(np.float32)
        
        # Historical actions (all approved in LendingClub data)
        actions = np.ones(len(df), dtype=np.int32)
        
        # Extract loan info
        loan_amnt = df['loan_amnt'].values if 'loan_amnt' in df.columns else np.ones(len(df))
        int_rate = df['int_rate'].values if 'int_rate' in df.columns else np.ones(len(df)) * 0.1
        
        # Outcomes (0: paid, 1: default)
        outcomes = df['default'].values.astype(np.int32)
        
        # CRITICAL FIX: Add synthetic rejection samples
        if add_synthetic_rejections:
            # Find high-risk loans (those that defaulted)
            default_indices = np.where(outcomes == 1)[0]
            n_to_reject = int(len(default_indices) * rejection_rate)
            
            if n_to_reject > 0:
                # Randomly select subset of defaulted loans to mark as "rejected"
                np.random.seed(42)  # For reproducibility
                reject_indices = np.random.choice(default_indices, size=n_to_reject, replace=False)
                
                # Change their actions to 0 (reject)
                actions[reject_indices] = 0
                
                print(f"   💡 Added {n_to_reject:,} synthetic rejections ({rejection_rate*100:.0f}% of defaults)")
                print(f"      This gives CQL examples of: reject risky loans → avoid losses")
        
        # Compute rewards with improved penalty
        rewards = self.compute_reward(actions, outcomes, loan_amnt, int_rate, 
                                     default_penalty_multiplier=default_penalty_multiplier)
        
        # Next states (terminal, so same as current states)
        next_states = states.copy()
        
        # Terminals (all True for episodic task)
        terminals = np.ones(len(df), dtype=bool)
        
        dataset = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'terminals': terminals
        }
        
        if include_loan_info:
            dataset['loan_amnt'] = loan_amnt
            dataset['int_rate'] = int_rate
            dataset['outcomes'] = outcomes
            
        return dataset
        
    def get_dataset_statistics(self, dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute dataset statistics.
        
        Args:
            dataset: Offline RL dataset
            
        Returns:
            Statistics dictionary
        """
        rewards = dataset['rewards']
        
        stats = {
            'n_samples': len(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'total_reward': np.sum(rewards),
            'positive_reward_pct': np.mean(rewards > 0) * 100,
            'negative_reward_pct': np.mean(rewards < 0) * 100,
            'zero_reward_pct': np.mean(rewards == 0) * 100
        }
        
        # Add outcome-specific stats if available
        if 'outcomes' in dataset:
            outcomes = dataset['outcomes']
            stats['default_rate'] = np.mean(outcomes) * 100
            stats['paid_rate'] = np.mean(1 - outcomes) * 100
            
        return stats
        
    def print_dataset_info(self, dataset: Dict[str, np.ndarray], name: str = "Dataset") -> None:
        """
        Print dataset information.
        
        Args:
            dataset: Offline RL dataset
            name: Dataset name
        """
        stats = self.get_dataset_statistics(dataset)
        
        print("=" * 70)
        print(f"{name.upper()} STATISTICS")
        print("=" * 70)
        print(f"Samples:        {stats['n_samples']:,}")
        print(f"State dim:      {dataset['states'].shape[1]}")
        print(f"Action dim:     {self.action_dim}")
        print(f"\nReward Statistics:")
        print(f"  Mean:         {stats['mean_reward']:.4f}")
        print(f"  Std:          {stats['std_reward']:.4f}")
        print(f"  Min:          {stats['min_reward']:.4f}")
        print(f"  Max:          {stats['max_reward']:.4f}")
        print(f"  Total:        {stats['total_reward']:.2f}")
        print(f"\nReward Distribution:")
        print(f"  Positive:     {stats['positive_reward_pct']:.1f}%")
        print(f"  Negative:     {stats['negative_reward_pct']:.1f}%")
        print(f"  Zero:         {stats['zero_reward_pct']:.1f}%")
        
        if 'default_rate' in stats:
            print(f"\nOutcome Distribution:")
            print(f"  Paid:         {stats['paid_rate']:.1f}%")
            print(f"  Default:      {stats['default_rate']:.1f}%")
            
        print("=" * 70)


def create_mdp_from_dataframes(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    state_features: Optional[list] = None,
    reward_scale: float = 0.01,
    normalize_states: bool = True,
    add_synthetic_rejections: bool = True,
    rejection_rate: float = 0.2,
    default_penalty_multiplier: float = 5.0
) -> Tuple[LoanApprovalMDP, Dict, Dict, Dict]:
    """
    Create MDP and datasets from dataframes.
    
    IMPROVED with synthetic rejections and better reward shaping to fix policy collapse.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        state_features: List of feature names (if None, infer from data)
        reward_scale: Reward scaling factor
        normalize_states: Whether to normalize states
        add_synthetic_rejections: Add synthetic rejection samples (CRITICAL FIX)
        rejection_rate: Fraction of defaults to mark as rejected (default=0.2)
        default_penalty_multiplier: Penalty multiplier for defaults (default=5.0)
        
    Returns:
        Tuple of (mdp, train_dataset, val_dataset, test_dataset)
    """
    # Infer state features if not provided
    if state_features is None:
        # Only include numeric columns (exclude object/string columns)
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        candidate_features = [c for c in numeric_cols 
                            if c not in ['default', 'loan_amnt', 'int_rate']]
        
        # Filter out problematic columns
        state_features = []
        excluded_nan = []
        excluded_constant = []
        
        for col in candidate_features:
            col_data = train_df[col]
            
            # Check if column is all NaN
            if col_data.isna().all():
                excluded_nan.append(col)
                continue
            
            # Check if column is constant (only 1 unique value excluding NaN)
            if col_data.nunique(dropna=True) <= 1:
                excluded_constant.append(col)
                continue
            
            # Column is valid
            state_features.append(col)
        
        print(f"Inferred {len(state_features)} valid numeric state features")
        
        if excluded_nan:
            print(f"⚠️  Excluded {len(excluded_nan)} all-NaN columns")
        if excluded_constant:
            print(f"⚠️  Excluded {len(excluded_constant)} constant columns")
        
        # Warn about non-numeric columns that were excluded
        non_numeric = [c for c in train_df.columns 
                      if c not in numeric_cols and c not in ['default', 'loan_amnt', 'int_rate']]
        if non_numeric:
            print(f"⚠️  Excluded {len(non_numeric)} non-numeric columns: {non_numeric[:5]}{'...' if len(non_numeric) > 5 else ''}")
        
    # Create MDP
    config = MDPConfig(
        state_features=state_features,
        reward_scale=reward_scale,
        normalize_states=normalize_states
    )
    mdp = LoanApprovalMDP(config)
    
    # Create datasets with improvements
    print("\n📦 Creating offline RL datasets...")
    if add_synthetic_rejections:
        print(f"   🔧 Adding synthetic rejections: {rejection_rate*100:.0f}% of defaults")
        print(f"   🔧 Default penalty multiplier: {default_penalty_multiplier}x")
    
    train_dataset = mdp.create_dataset(
        train_df, 
        add_synthetic_rejections=add_synthetic_rejections,
        rejection_rate=rejection_rate,
        default_penalty_multiplier=default_penalty_multiplier
    )
    val_dataset = mdp.create_dataset(
        val_df,
        add_synthetic_rejections=add_synthetic_rejections,
        rejection_rate=rejection_rate,
        default_penalty_multiplier=default_penalty_multiplier
    )
    test_dataset = mdp.create_dataset(
        test_df,
        add_synthetic_rejections=False,  # Keep test set pure for evaluation
        default_penalty_multiplier=default_penalty_multiplier
    )
    
    # Fit normalization on training data
    if normalize_states:
        mdp.fit_normalization(train_dataset['states'])
        train_dataset['states'] = mdp.normalize_states(train_dataset['states'])
        val_dataset['states'] = mdp.normalize_states(val_dataset['states'])
        test_dataset['states'] = mdp.normalize_states(test_dataset['states'])
        test_dataset['next_states'] = mdp.normalize_states(test_dataset['next_states'])
        
    # Print statistics
    mdp.print_dataset_info(train_dataset, "Train")
    print()
    mdp.print_dataset_info(val_dataset, "Validation")
    print()
    mdp.print_dataset_info(test_dataset, "Test")
    
    return mdp, train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Example usage
    print("MDP Formulation for Loan Approval")
    print("=" * 70)
    print("\nState: Applicant features (FICO, DTI, income, etc.)")
    print("Action: {0: Reject, 1: Approve}")
    print("Reward: Profit (loan_amnt × int_rate) or loss (-loan_amnt)")
    print("Transition: Terminal (one-step episodic)")
    print("\nThis module converts supervised learning → RL problem")
    print("=" * 70)
