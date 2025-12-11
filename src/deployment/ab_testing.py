"""
A/B Testing Framework

Production A/B testing for loan approval models:
- Multi-arm bandit allocation
- Statistical significance testing
- Experiment tracking
- Performance comparison

Example:
    >>> from deployment.ab_testing import ABTestManager
    >>> 
    >>> manager = ABTestManager()
    >>> manager.add_variant('control', server_control, allocation=0.5)
    >>> manager.add_variant('treatment', server_treatment, allocation=0.5)
    >>> 
    >>> result = manager.get_variant(user_id)
    >>> manager.log_outcome(user_id, outcome, profit)
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import hashlib
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """A/B test experiment configuration."""
    name: str
    description: str
    start_date: str
    end_date: Optional[str] = None
    allocation_method: str = 'fixed'  # 'fixed', 'thompson', 'epsilon_greedy'
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    primary_metric: str = 'profit'  # 'profit', 'approval_rate', 'default_rate'
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Variant:
    """A/B test variant."""
    name: str
    model_type: str
    allocation: float  # Target allocation 0-1
    n_samples: int = 0
    n_approvals: int = 0
    n_defaults: int = 0
    total_profit: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate variant metrics."""
        if self.n_samples == 0:
            return {}
        
        return {
            'approval_rate': self.n_approvals / self.n_samples,
            'default_rate': self.n_defaults / self.n_approvals if self.n_approvals > 0 else 0,
            'avg_profit': self.total_profit / self.n_samples,
            'total_profit': self.total_profit,
            'n_samples': self.n_samples
        }


class ABTestManager:
    """
    A/B testing manager for loan approval models.
    
    Supports:
    - Multiple variants (A/B/C/... testing)
    - Traffic allocation strategies (fixed, Thompson sampling, epsilon-greedy)
    - Statistical significance testing
    - Experiment tracking and reporting
    """
    
    def __init__(
        self,
        experiment_config: Optional[ExperimentConfig] = None,
        log_dir: str = '../logs/ab_tests'
    ):
        """
        Initialize A/B test manager.
        
        Args:
            experiment_config: Experiment configuration
            log_dir: Directory for experiment logs
        """
        self.config = experiment_config or ExperimentConfig(
            name='default_experiment',
            description='Default A/B test',
            start_date=datetime.now().isoformat()
        )
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.variants: Dict[str, Variant] = {}
        self.user_assignments: Dict[str, str] = {}  # user_id -> variant_name
        self.outcomes: List[Dict] = []
        
        logger.info(f"A/B test '{self.config.name}' initialized")
    
    def add_variant(
        self,
        name: str,
        model_type: str,
        allocation: float
    ):
        """
        Add a variant to the experiment.
        
        Args:
            name: Variant name (e.g., 'control', 'treatment_a')
            model_type: Model type for this variant
            allocation: Target allocation (0-1)
        """
        if name in self.variants:
            logger.warning(f"Variant '{name}' already exists, updating allocation")
        
        self.variants[name] = Variant(
            name=name,
            model_type=model_type,
            allocation=allocation
        )
        
        logger.info(f"Added variant '{name}' (model={model_type}, allocation={allocation:.1%})")
    
    def _hash_user(self, user_id: str) -> float:
        """Hash user ID to [0, 1] for consistent assignment."""
        hash_obj = hashlib.md5(user_id.encode())
        return int(hash_obj.hexdigest(), 16) / (16**32)
    
    def _allocate_fixed(self, user_id: str) -> str:
        """Fixed allocation based on user hash."""
        hash_val = self._hash_user(user_id)
        
        cumulative = 0.0
        for variant_name, variant in self.variants.items():
            cumulative += variant.allocation
            if hash_val <= cumulative:
                return variant_name
        
        # Fallback to first variant
        return list(self.variants.keys())[0]
    
    def _allocate_thompson_sampling(self) -> str:
        """Thompson sampling allocation (multi-armed bandit)."""
        if self.config.primary_metric != 'profit':
            logger.warning("Thompson sampling best for profit metric")
        
        # Thompson sampling: sample from beta distribution
        samples = {}
        for name, variant in self.variants.items():
            if variant.n_samples < 10:
                # Exploration phase
                samples[name] = np.random.uniform(0, 1)
            else:
                # Sample from posterior (assuming beta prior)
                # For profit, normalize to [0, 1]
                metrics = variant.get_metrics()
                avg_profit = metrics.get('avg_profit', 0)
                
                # Simplified: use normalized profit as success probability
                alpha = max(1, avg_profit * 100 + 1)
                beta = max(1, (1 - avg_profit / 1000) * 100 + 1)
                samples[name] = np.random.beta(alpha, beta)
        
        return max(samples, key=samples.get)
    
    def _allocate_epsilon_greedy(self, epsilon: float = 0.1) -> str:
        """Epsilon-greedy allocation."""
        if np.random.random() < epsilon:
            # Explore: random variant
            return np.random.choice(list(self.variants.keys()))
        else:
            # Exploit: best performing variant
            best_variant = None
            best_metric = float('-inf')
            
            for name, variant in self.variants.items():
                if variant.n_samples == 0:
                    return name  # Explore unsampled variants
                
                metrics = variant.get_metrics()
                metric_val = metrics.get(self.config.primary_metric, 0)
                
                if metric_val > best_metric:
                    best_metric = metric_val
                    best_variant = name
            
            return best_variant or list(self.variants.keys())[0]
    
    def get_variant(self, user_id: str) -> str:
        """
        Get variant assignment for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Variant name
        """
        # Check if user already assigned
        if user_id in self.user_assignments:
            return self.user_assignments[user_id]
        
        # Allocate based on strategy
        if self.config.allocation_method == 'fixed':
            variant_name = self._allocate_fixed(user_id)
        elif self.config.allocation_method == 'thompson':
            variant_name = self._allocate_thompson_sampling()
        elif self.config.allocation_method == 'epsilon_greedy':
            variant_name = self._allocate_epsilon_greedy()
        else:
            logger.warning(f"Unknown allocation method: {self.config.allocation_method}")
            variant_name = list(self.variants.keys())[0]
        
        # Store assignment
        self.user_assignments[user_id] = variant_name
        
        return variant_name
    
    def log_outcome(
        self,
        user_id: str,
        action: int,
        outcome: Optional[int] = None,
        profit: Optional[float] = None
    ):
        """
        Log experiment outcome.
        
        Args:
            user_id: User identifier
            action: Approval decision (0=reject, 1=approve)
            outcome: Actual outcome (0=paid, 1=default)
            profit: Profit/loss from decision
        """
        variant_name = self.user_assignments.get(user_id)
        
        if variant_name is None:
            logger.warning(f"User {user_id} not assigned to variant")
            return
        
        variant = self.variants[variant_name]
        
        # Update variant stats
        variant.n_samples += 1
        if action == 1:
            variant.n_approvals += 1
            if outcome == 1:
                variant.n_defaults += 1
        
        if profit is not None:
            variant.total_profit += profit
        
        # Log outcome
        self.outcomes.append({
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'variant': variant_name,
            'action': action,
            'outcome': outcome,
            'profit': profit
        })
    
    def get_results(self) -> Dict[str, Any]:
        """Get experiment results."""
        results = {
            'experiment': self.config.name,
            'start_date': self.config.start_date,
            'duration_days': (datetime.now() - datetime.fromisoformat(self.config.start_date)).days,
            'total_samples': sum(v.n_samples for v in self.variants.values()),
            'variants': {}
        }
        
        for name, variant in self.variants.items():
            metrics = variant.get_metrics()
            results['variants'][name] = {
                'model_type': variant.model_type,
                'target_allocation': variant.allocation,
                'actual_allocation': variant.n_samples / results['total_samples'] if results['total_samples'] > 0 else 0,
                'metrics': metrics
            }
        
        return results
    
    def statistical_test(self, variant_a: str, variant_b: str) -> Dict[str, Any]:
        """
        Statistical significance test between two variants.
        
        Args:
            variant_a: First variant name
            variant_b: Second variant name
            
        Returns:
            Test results with p-value and confidence interval
        """
        from scipy import stats
        
        var_a = self.variants[variant_a]
        var_b = self.variants[variant_b]
        
        if var_a.n_samples < 30 or var_b.n_samples < 30:
            return {
                'error': 'Insufficient samples (need >= 30 per variant)',
                'n_samples_a': var_a.n_samples,
                'n_samples_b': var_b.n_samples
            }
        
        metrics_a = var_a.get_metrics()
        metrics_b = var_b.get_metrics()
        
        # T-test for profit difference
        # Simplified: assume normal distribution
        mean_a = metrics_a['avg_profit']
        mean_b = metrics_b['avg_profit']
        
        # Use pooled standard deviation estimate
        std_a = np.sqrt(abs(mean_a)) if mean_a != 0 else 1
        std_b = np.sqrt(abs(mean_b)) if mean_b != 0 else 1
        
        # Two-sample t-test
        t_stat = (mean_a - mean_b) / np.sqrt(std_a**2/var_a.n_samples + std_b**2/var_b.n_samples)
        df = var_a.n_samples + var_b.n_samples - 2
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((var_a.n_samples-1)*std_a**2 + (var_b.n_samples-1)*std_b**2) / df)
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval
        ci_margin = stats.t.ppf(0.975, df) * np.sqrt(std_a**2/var_a.n_samples + std_b**2/var_b.n_samples)
        
        return {
            'variant_a': variant_a,
            'variant_b': variant_b,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'difference': mean_a - mean_b,
            'relative_improvement': ((mean_a - mean_b) / mean_b * 100) if mean_b != 0 else 0,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < (1 - self.config.confidence_level),
            'cohens_d': cohens_d,
            'ci_95': (mean_a - mean_b - ci_margin, mean_a - mean_b + ci_margin)
        }
    
    def should_stop_experiment(self) -> Tuple[bool, str]:
        """
        Check if experiment should be stopped (early stopping).
        
        Returns:
            (should_stop, reason)
        """
        # Check minimum sample size
        total_samples = sum(v.n_samples for v in self.variants.values())
        if total_samples < self.config.min_sample_size:
            return False, f"Need more samples ({total_samples}/{self.config.min_sample_size})"
        
        # Check if any variant is significantly better
        variant_names = list(self.variants.keys())
        if len(variant_names) < 2:
            return False, "Need at least 2 variants"
        
        # Compare first two variants
        test_result = self.statistical_test(variant_names[0], variant_names[1])
        
        if 'error' in test_result:
            return False, test_result['error']
        
        if test_result['significant']:
            winner = variant_names[0] if test_result['difference'] > 0 else variant_names[1]
            return True, f"Variant '{winner}' significantly better (p={test_result['p_value']:.4f})"
        
        return False, "No significant difference yet"
    
    def generate_report(self) -> str:
        """Generate experiment report."""
        results = self.get_results()
        
        report = []
        report.append("="*70)
        report.append(f"A/B TEST REPORT: {self.config.name}".center(70))
        report.append("="*70)
        report.append(f"\nDescription: {self.config.description}")
        report.append(f"Start Date: {self.config.start_date}")
        report.append(f"Duration: {results['duration_days']} days")
        report.append(f"Total Samples: {results['total_samples']:,}")
        report.append(f"Primary Metric: {self.config.primary_metric}")
        
        report.append("\n--- Variant Results ---")
        for name, data in results['variants'].items():
            report.append(f"\n{name.upper()} ({data['model_type']}):")
            report.append(f"  Target Allocation: {data['target_allocation']:.1%}")
            report.append(f"  Actual Allocation: {data['actual_allocation']:.1%}")
            
            metrics = data['metrics']
            if metrics:
                report.append(f"  Samples: {metrics['n_samples']:,}")
                report.append(f"  Approval Rate: {metrics['approval_rate']:.1%}")
                report.append(f"  Default Rate: {metrics['default_rate']:.1%}")
                report.append(f"  Avg Profit: ${metrics['avg_profit']:,.2f}")
                report.append(f"  Total Profit: ${metrics['total_profit']:,.2f}")
        
        # Statistical test
        if len(self.variants) >= 2:
            variant_names = list(self.variants.keys())
            test = self.statistical_test(variant_names[0], variant_names[1])
            
            if 'error' not in test:
                report.append("\n--- Statistical Significance ---")
                report.append(f"Comparing: {test['variant_a']} vs {test['variant_b']}")
                report.append(f"Difference: ${test['difference']:,.2f} ({test['relative_improvement']:+.1f}%)")
                report.append(f"P-value: {test['p_value']:.4f}")
                report.append(f"Significant: {'YES' if test['significant'] else 'NO'}")
                report.append(f"Effect Size (Cohen's d): {test['cohens_d']:.3f}")
        
        # Recommendation
        should_stop, reason = self.should_stop_experiment()
        report.append("\n--- Recommendation ---")
        if should_stop:
            report.append(f"✅ STOP EXPERIMENT: {reason}")
        else:
            report.append(f"⏳ CONTINUE: {reason}")
        
        report.append("\n" + "="*70)
        
        return '\n'.join(report)
    
    def save_experiment(self, filename: Optional[str] = None):
        """Save experiment state."""
        if filename is None:
            filename = f"{self.config.name}_{datetime.now().strftime('%Y%m%d')}.json"
        
        filepath = self.log_dir / filename
        
        data = {
            'config': self.config.to_dict(),
            'variants': {name: var.to_dict() for name, var in self.variants.items()},
            'results': self.get_results(),
            'outcomes': self.outcomes
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Experiment saved to {filepath}")


if __name__ == '__main__':
    # Example usage
    print("="*70)
    print("A/B TESTING EXAMPLE".center(70))
    print("="*70)
    
    # Create experiment
    config = ExperimentConfig(
        name='xgboost_vs_cql',
        description='Compare XGBoost (supervised) vs CQL (RL)',
        start_date=datetime.now().isoformat(),
        allocation_method='fixed',
        primary_metric='profit'
    )
    
    manager = ABTestManager(experiment_config=config)
    
    # Add variants
    manager.add_variant('control_xgboost', model_type='xgboost', allocation=0.5)
    manager.add_variant('treatment_cql', model_type='cql', allocation=0.5)
    
    # Simulate experiment
    np.random.seed(42)
    for i in range(2000):
        user_id = f"user_{i}"
        variant = manager.get_variant(user_id)
        
        # Simulate prediction (treatment slightly better)
        if variant == 'control_xgboost':
            action = np.random.choice([0, 1], p=[0.3, 0.7])
            outcome = np.random.choice([0, 1], p=[0.85, 0.15])
            base_profit = 1200 if outcome == 0 else -10000
        else:
            action = np.random.choice([0, 1], p=[0.35, 0.65])
            outcome = np.random.choice([0, 1], p=[0.87, 0.13])
            base_profit = 1200 if outcome == 0 else -10000
        
        profit = base_profit if action == 1 else 0
        
        manager.log_outcome(user_id, action, outcome, profit)
    
    # Generate report
    print("\n" + manager.generate_report())
    
    # Save experiment
    manager.save_experiment()
    
    print("\n✅ A/B testing example complete")
