"""
Policy Evaluation for Offline RL

Evaluates trained RL policies on test data with:
1. Policy performance metrics (approval rate, default rate, profit)
2. Comparison with supervised learning baseline
3. Counterfactual policy evaluation
4. Statistical significance testing

Author: AI Toolkit
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class PolicyEvaluator:
    """
    Evaluates RL policies and compares with baselines.
    
    Metrics:
    - Approval rate: % of applications approved
    - Default rate: % of approved loans that default
    - Expected profit: Total profit from approved loans
    - Profit per approved loan
    - Precision, Recall, F1 for default prediction
    """
    
    def __init__(self, reward_scale: float = 0.01):
        """
        Initialize policy evaluator.
        
        Args:
            reward_scale: Reward scaling factor (should match MDP)
        """
        self.reward_scale = reward_scale
        
    def evaluate_policy(
        self,
        actions: np.ndarray,
        outcomes: np.ndarray,
        loan_amnt: np.ndarray,
        int_rate: np.ndarray,
        policy_name: str = "Policy"
    ) -> Dict[str, float]:
        """
        Evaluate policy performance.
        
        Args:
            actions: Policy actions (n_samples,) - 0 or 1
            outcomes: True outcomes (n_samples,) - 0 (paid) or 1 (default)
            loan_amnt: Loan amounts (n_samples,)
            int_rate: Interest rates (n_samples,)
            policy_name: Name for reporting
            
        Returns:
            Dictionary of metrics
        """
        n_samples = len(actions)
        
        # Approval metrics
        approved_mask = (actions == 1)
        n_approved = np.sum(approved_mask)
        approval_rate = n_approved / n_samples
        
        # Among approved loans
        if n_approved > 0:
            approved_outcomes = outcomes[approved_mask]
            approved_loan_amnt = loan_amnt[approved_mask]
            approved_int_rate = int_rate[approved_mask]
            
            # Default rate among approved
            n_defaults = np.sum(approved_outcomes == 1)
            default_rate = n_defaults / n_approved
            
            # Profit calculation
            paid_mask = (approved_outcomes == 0)
            default_mask = (approved_outcomes == 1)
            
            profit_from_paid = np.sum(approved_loan_amnt[paid_mask] * approved_int_rate[paid_mask])
            loss_from_defaults = np.sum(approved_loan_amnt[default_mask])
            
            expected_profit = profit_from_paid - loss_from_defaults
            profit_per_loan = expected_profit / n_approved
            
            # Precision/Recall for default prediction
            # Treating "approve" as predicting "will not default"
            true_positives = np.sum(paid_mask)  # Correctly approved (paid)
            false_positives = np.sum(default_mask)  # Incorrectly approved (defaulted)
            false_negatives = np.sum(~approved_mask & (outcomes == 0))  # Rejected but would have paid
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
        else:
            default_rate = 0.0
            expected_profit = 0.0
            profit_per_loan = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            
        metrics = {
            'policy_name': policy_name,
            'n_samples': n_samples,
            'n_approved': n_approved,
            'approval_rate': approval_rate * 100,
            'default_rate': default_rate * 100,
            'expected_profit': expected_profit,
            'profit_per_loan': profit_per_loan,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
        
    def compare_policies(
        self,
        policies: Dict[str, np.ndarray],
        outcomes: np.ndarray,
        loan_amnt: np.ndarray,
        int_rate: np.ndarray,
        baseline_name: str = "Supervised"
    ) -> pd.DataFrame:
        """
        Compare multiple policies.
        
        Args:
            policies: Dictionary of {policy_name: actions}
            outcomes: True outcomes
            loan_amnt: Loan amounts
            int_rate: Interest rates
            baseline_name: Name of baseline policy
            
        Returns:
            Comparison dataframe
        """
        results = []
        
        for policy_name, actions in policies.items():
            metrics = self.evaluate_policy(actions, outcomes, loan_amnt, int_rate, policy_name)
            results.append(metrics)
            
        df = pd.DataFrame(results)
        
        # Compute improvements over baseline
        if baseline_name in policies:
            baseline_profit = df[df['policy_name'] == baseline_name]['expected_profit'].values[0]
            df['profit_improvement'] = (df['expected_profit'] - baseline_profit) / abs(baseline_profit) * 100
        else:
            df['profit_improvement'] = 0.0
            
        return df
        
    def print_comparison(self, comparison_df: pd.DataFrame) -> None:
        """
        Print policy comparison.
        
        Args:
            comparison_df: Comparison dataframe
        """
        print("\n" + "=" * 70)
        print("POLICY COMPARISON")
        print("=" * 70)
        
        for _, row in comparison_df.iterrows():
            print(f"\n{row['policy_name']}:")
            print(f"  Approval rate:      {row['approval_rate']:.2f}%")
            print(f"  Default rate:       {row['default_rate']:.2f}%")
            print(f"  Expected profit:    ${row['expected_profit']:,.0f}")
            print(f"  Profit per loan:    ${row['profit_per_loan']:,.0f}")
            print(f"  Precision:          {row['precision']:.3f}")
            print(f"  Recall:             {row['recall']:.3f}")
            print(f"  F1 Score:           {row['f1']:.3f}")
            
            if 'profit_improvement' in row:
                improvement = row['profit_improvement']
                if improvement > 0:
                    print(f"  Profit improvement: +{improvement:.1f}% 📈")
                elif improvement < 0:
                    print(f"  Profit improvement: {improvement:.1f}% 📉")
                    
        print("\n" + "=" * 70)
        
    def plot_policy_comparison(
        self,
        comparison_df: pd.DataFrame,
        savepath: Optional[str] = None
    ) -> None:
        """
        Plot policy comparison.
        
        Args:
            comparison_df: Comparison dataframe
            savepath: Path to save plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        policies = comparison_df['policy_name'].values
        
        # 1. Approval Rate
        ax = axes[0, 0]
        ax.bar(policies, comparison_df['approval_rate'], color='steelblue', alpha=0.7)
        ax.set_ylabel('Approval Rate (%)', fontsize=12)
        ax.set_title('Approval Rate', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # 2. Default Rate
        ax = axes[0, 1]
        ax.bar(policies, comparison_df['default_rate'], color='red', alpha=0.7)
        ax.set_ylabel('Default Rate (%)', fontsize=12)
        ax.set_title('Default Rate (Among Approved)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # 3. Expected Profit
        ax = axes[0, 2]
        colors = ['green' if x > 0 else 'red' for x in comparison_df['expected_profit']]
        ax.bar(policies, comparison_df['expected_profit'], color=colors, alpha=0.7)
        ax.set_ylabel('Expected Profit ($)', fontsize=12)
        ax.set_title('Expected Profit', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        ax.ticklabel_format(style='plain', axis='y')
        
        # 4. Profit per Loan
        ax = axes[1, 0]
        colors = ['green' if x > 0 else 'red' for x in comparison_df['profit_per_loan']]
        ax.bar(policies, comparison_df['profit_per_loan'], color=colors, alpha=0.7)
        ax.set_ylabel('Profit per Loan ($)', fontsize=12)
        ax.set_title('Profit per Approved Loan', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # 5. Precision/Recall/F1
        ax = axes[1, 1]
        x = np.arange(len(policies))
        width = 0.25
        ax.bar(x - width, comparison_df['precision'], width, label='Precision', alpha=0.7)
        ax.bar(x, comparison_df['recall'], width, label='Recall', alpha=0.7)
        ax.bar(x + width, comparison_df['f1'], width, label='F1', alpha=0.7)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Classification Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(policies)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # 6. Profit Improvement
        if 'profit_improvement' in comparison_df.columns:
            ax = axes[1, 2]
            colors = ['green' if x > 0 else 'red' for x in comparison_df['profit_improvement']]
            ax.bar(policies, comparison_df['profit_improvement'], color=colors, alpha=0.7)
            ax.set_ylabel('Improvement (%)', fontsize=12)
            ax.set_title('Profit Improvement vs Baseline', fontsize=14, fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax.grid(alpha=0.3, axis='y')
        else:
            axes[1, 2].axis('off')
            
        plt.tight_layout()
        
        if savepath:
            plt.savefig(savepath, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {savepath}")
            
        plt.show()
        
    def bootstrap_confidence_interval(
        self,
        actions: np.ndarray,
        outcomes: np.ndarray,
        loan_amnt: np.ndarray,
        int_rate: np.ndarray,
        metric: str = 'expected_profit',
        n_bootstrap: int = 1000,
        alpha: float = 0.05
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for policy metric.
        
        Args:
            actions: Policy actions
            outcomes: True outcomes
            loan_amnt: Loan amounts
            int_rate: Interest rates
            metric: Metric to compute ('expected_profit', 'approval_rate', etc.)
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level (e.g., 0.05 for 95% CI)
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        n_samples = len(actions)
        bootstrap_values = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Evaluate on bootstrap sample
            metrics = self.evaluate_policy(
                actions[indices],
                outcomes[indices],
                loan_amnt[indices],
                int_rate[indices]
            )
            
            bootstrap_values.append(metrics[metric])
            
        bootstrap_values = np.array(bootstrap_values)
        
        # Compute percentiles
        lower = np.percentile(bootstrap_values, alpha / 2 * 100)
        upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)
        mean = np.mean(bootstrap_values)
        
        return mean, lower, upper
        
    def statistical_test(
        self,
        actions1: np.ndarray,
        actions2: np.ndarray,
        outcomes: np.ndarray,
        loan_amnt: np.ndarray,
        int_rate: np.ndarray,
        policy1_name: str = "Policy 1",
        policy2_name: str = "Policy 2"
    ) -> Dict[str, float]:
        """
        Statistical significance test between two policies.
        
        Args:
            actions1: First policy actions
            actions2: Second policy actions
            outcomes: True outcomes
            loan_amnt: Loan amounts
            int_rate: Interest rates
            policy1_name: First policy name
            policy2_name: Second policy name
            
        Returns:
            Test results dictionary
        """
        # Compute per-sample profits for both policies
        def compute_sample_profits(actions):
            profits = np.zeros_like(actions, dtype=np.float32)
            approved_mask = (actions == 1)
            paid_mask = approved_mask & (outcomes == 0)
            default_mask = approved_mask & (outcomes == 1)
            
            profits[paid_mask] = loan_amnt[paid_mask] * int_rate[paid_mask]
            profits[default_mask] = -loan_amnt[default_mask]
            
            return profits
            
        profits1 = compute_sample_profits(actions1)
        profits2 = compute_sample_profits(actions2)
        
        # Paired t-test (same samples, different policies)
        t_stat, p_value = stats.ttest_rel(profits1, profits2)
        
        # Effect size (Cohen's d)
        diff = profits1 - profits2
        cohens_d = np.mean(diff) / np.std(diff)
        
        results = {
            'policy1': policy1_name,
            'policy2': policy2_name,
            'mean_profit1': np.mean(profits1),
            'mean_profit2': np.mean(profits2),
            'profit_diff': np.mean(diff),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }
        
        return results
        
    def print_statistical_test(self, test_results: Dict[str, float]) -> None:
        """
        Print statistical test results.
        
        Args:
            test_results: Test results dictionary
        """
        print("\n" + "=" * 70)
        print("STATISTICAL SIGNIFICANCE TEST")
        print("=" * 70)
        print(f"\nComparing: {test_results['policy1']} vs {test_results['policy2']}")
        print(f"\nMean profit ({test_results['policy1']}): ${test_results['mean_profit1']:,.2f}")
        print(f"Mean profit ({test_results['policy2']}): ${test_results['mean_profit2']:,.2f}")
        print(f"Difference: ${test_results['profit_diff']:,.2f}")
        print(f"\nt-statistic: {test_results['t_statistic']:.4f}")
        print(f"p-value: {test_results['p_value']:.4f}")
        print(f"Cohen's d: {test_results['cohens_d']:.4f}")
        
        if test_results['significant']:
            print("\n✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
        else:
            print("\n⚠️  NOT statistically significant (p >= 0.05)")
            
        print("=" * 70)


if __name__ == "__main__":
    print("Policy Evaluation for Offline RL")
    print("=" * 70)
    print("\nMetrics:")
    print("  - Approval rate: % of applications approved")
    print("  - Default rate: % of approved loans that default")
    print("  - Expected profit: Total profit from portfolio")
    print("  - Profit per loan: Average profit per approved loan")
    print("  - Precision/Recall/F1: Classification metrics")
    print("\nComparison:")
    print("  - Bootstrap confidence intervals")
    print("  - Statistical significance testing")
    print("  - Profit improvement over baseline")
    print("=" * 70)
