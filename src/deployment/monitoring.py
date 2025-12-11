"""
Monitoring and Logging System

Production monitoring for loan approval system:
- Performance tracking (latency, throughput, errors)
- Model metrics (accuracy, profit, approval rates)
- Drift detection (data drift, concept drift)
- Alerting and reporting

Example:
    >>> from deployment.monitoring import PerformanceMonitor
    >>> 
    >>> monitor = PerformanceMonitor()
    >>> monitor.log_prediction(features, prediction, outcome)
    >>> metrics = monitor.get_metrics()
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionLog:
    """Single prediction log entry."""
    timestamp: str
    request_id: int
    features: np.ndarray
    prediction: int  # 0=reject, 1=approve
    confidence: float
    model_type: str
    model_version: str
    latency_ms: float
    outcome: Optional[int] = None  # Actual outcome (0=paid, 1=default)
    loan_amnt: Optional[float] = None
    int_rate: Optional[float] = None
    profit: Optional[float] = None
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['features'] = self.features.tolist() if isinstance(self.features, np.ndarray) else self.features
        return d


@dataclass
class MetricsSummary:
    """Summary of model metrics over a time window."""
    start_time: str
    end_time: str
    n_predictions: int
    approval_rate: float
    avg_confidence: float
    avg_latency_ms: float
    error_rate: float
    
    # With outcomes
    n_outcomes: int = 0
    default_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    expected_profit: float = 0.0
    profit_per_loan: float = 0.0


class MetricsTracker:
    """
    Track model performance metrics over time.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Number of recent predictions to track
        """
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.outcomes = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        
        self.total_predictions = 0
        self.total_errors = 0
        
    def log_prediction(
        self, 
        prediction: int,
        confidence: float,
        latency_ms: float,
        outcome: Optional[int] = None,
        error: bool = False
    ):
        """Log a single prediction."""
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.latencies.append(latency_ms)
        self.errors.append(1 if error else 0)
        
        if outcome is not None:
            self.outcomes.append((prediction, outcome))
        
        self.total_predictions += 1
        if error:
            self.total_errors += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Compute current metrics."""
        if len(self.predictions) == 0:
            return {}
        
        metrics = {
            'total_predictions': self.total_predictions,
            'window_size': len(self.predictions),
            'approval_rate': np.mean(self.predictions) * 100,
            'avg_confidence': np.mean(self.confidences),
            'avg_latency_ms': np.mean(self.latencies),
            'p95_latency_ms': np.percentile(self.latencies, 95),
            'p99_latency_ms': np.percentile(self.latencies, 99),
            'error_rate': np.mean(self.errors) * 100,
            'total_errors': self.total_errors
        }
        
        # Add classification metrics if outcomes available
        if len(self.outcomes) > 0:
            preds, actuals = zip(*self.outcomes)
            preds = np.array(preds)
            actuals = np.array(actuals)
            
            # Precision, Recall, F1 (treating default=1 as positive)
            tp = np.sum((preds == 1) & (actuals == 1))
            fp = np.sum((preds == 1) & (actuals == 0))
            fn = np.sum((preds == 0) & (actuals == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.update({
                'n_outcomes': len(self.outcomes),
                'default_rate': np.mean(actuals[preds == 1]) * 100 if np.sum(preds == 1) > 0 else 0,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        return metrics
    
    def reset(self):
        """Reset all metrics."""
        self.predictions.clear()
        self.outcomes.clear()
        self.latencies.clear()
        self.confidences.clear()
        self.errors.clear()


class DriftDetector:
    """
    Detect data drift and concept drift.
    """
    
    def __init__(self, reference_data: Optional[np.ndarray] = None, threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference distribution (training data)
            threshold: P-value threshold for drift detection
        """
        self.reference_data = reference_data
        self.threshold = threshold
        
        if reference_data is not None:
            self.reference_mean = np.mean(reference_data, axis=0)
            self.reference_std = np.std(reference_data, axis=0)
    
    def detect_feature_drift(self, current_data: np.ndarray) -> Dict[str, Any]:
        """
        Detect feature drift using statistical tests.
        
        Args:
            current_data: Current data distribution
            
        Returns:
            Drift detection results
        """
        if self.reference_data is None:
            return {'error': 'No reference data provided'}
        
        from scipy import stats
        
        n_features = current_data.shape[1]
        drifted_features = []
        p_values = []
        
        for i in range(n_features):
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(
                self.reference_data[:, i],
                current_data[:, i]
            )
            
            p_values.append(p_value)
            
            if p_value < self.threshold:
                drifted_features.append(i)
        
        return {
            'n_features': n_features,
            'n_drifted': len(drifted_features),
            'drifted_features': drifted_features,
            'drift_detected': len(drifted_features) > 0,
            'min_p_value': min(p_values),
            'mean_p_value': np.mean(p_values)
        }
    
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect drift in prediction distribution.
        
        Args:
            reference_predictions: Historical predictions
            current_predictions: Current predictions
            
        Returns:
            Drift detection results
        """
        from scipy import stats
        
        # Compare approval rates
        ref_approval = np.mean(reference_predictions)
        curr_approval = np.mean(current_predictions)
        
        # Chi-square test
        ref_counts = np.bincount(reference_predictions, minlength=2)
        curr_counts = np.bincount(current_predictions, minlength=2)
        
        chi2, p_value = stats.chisquare(curr_counts, ref_counts)
        
        return {
            'reference_approval_rate': ref_approval,
            'current_approval_rate': curr_approval,
            'approval_rate_change': (curr_approval - ref_approval) / ref_approval * 100,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'drift_detected': p_value < self.threshold
        }


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Tracks:
    - Prediction logs
    - Performance metrics
    - Drift detection
    - Alerts
    """
    
    def __init__(
        self,
        log_dir: str = '../logs',
        metrics_window: int = 1000,
        enable_drift_detection: bool = True
    ):
        """
        Initialize performance monitor.
        
        Args:
            log_dir: Directory for log files
            metrics_window: Window size for metrics tracking
            enable_drift_detection: Enable drift detection
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_tracker = MetricsTracker(window_size=metrics_window)
        self.enable_drift = enable_drift_detection
        self.drift_detector = DriftDetector() if enable_drift else None
        
        self.logs: List[PredictionLog] = []
        self.alerts: List[Dict] = []
        
        self.start_time = datetime.now()
        
        logger.info(f"Performance monitor initialized (log_dir={log_dir})")
    
    def log_prediction(
        self,
        features: np.ndarray,
        prediction: Dict[str, Any],
        outcome: Optional[int] = None,
        loan_amnt: Optional[float] = None,
        int_rate: Optional[float] = None
    ):
        """
        Log a prediction for monitoring.
        
        Args:
            features: Input features
            prediction: Prediction dict from model server
            outcome: Actual outcome (if available)
            loan_amnt: Loan amount
            int_rate: Interest rate
        """
        # Calculate profit if outcome known
        profit = None
        if outcome is not None and loan_amnt is not None and int_rate is not None:
            if prediction['action'] == 1:  # Approved
                if outcome == 0:  # Paid
                    profit = loan_amnt * int_rate * 0.01
                else:  # Default
                    profit = -loan_amnt * 0.01
            else:  # Rejected
                profit = 0.0
        
        # Create log entry
        log_entry = PredictionLog(
            timestamp=prediction.get('timestamp', datetime.now().isoformat()),
            request_id=prediction.get('request_id', len(self.logs) + 1),
            features=features,
            prediction=prediction['action'],
            confidence=prediction['confidence'],
            model_type=prediction.get('model_type', 'unknown'),
            model_version=prediction.get('model_version', 'unknown'),
            latency_ms=prediction.get('latency_ms', 0),
            outcome=outcome,
            loan_amnt=loan_amnt,
            int_rate=int_rate,
            profit=profit
        )
        
        self.logs.append(log_entry)
        
        # Update metrics
        self.metrics_tracker.log_prediction(
            prediction=prediction['action'],
            confidence=prediction['confidence'],
            latency_ms=prediction.get('latency_ms', 0),
            outcome=outcome,
            error='error' in prediction
        )
        
        # Check for alerts
        self._check_alerts()
    
    def _check_alerts(self):
        """Check for alert conditions."""
        metrics = self.metrics_tracker.get_metrics()
        
        # High error rate
        if metrics.get('error_rate', 0) > 5.0:
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'high_error_rate',
                'message': f"Error rate: {metrics['error_rate']:.2f}%",
                'severity': 'high'
            })
        
        # High latency
        if metrics.get('p95_latency_ms', 0) > 1000:
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'high_latency',
                'message': f"P95 latency: {metrics['p95_latency_ms']:.0f}ms",
                'severity': 'medium'
            })
        
        # Unusual approval rate
        approval_rate = metrics.get('approval_rate', 0)
        if approval_rate < 20 or approval_rate > 80:
            self.alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'unusual_approval_rate',
                'message': f"Approval rate: {approval_rate:.1f}%",
                'severity': 'low'
            })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics."""
        metrics = self.metrics_tracker.get_metrics()
        
        # Add profit metrics if available
        if len(self.logs) > 0:
            recent_logs = [log for log in self.logs[-self.metrics_tracker.window_size:]
                          if log.profit is not None]
            
            if recent_logs:
                profits = [log.profit for log in recent_logs]
                metrics['total_profit'] = sum(profits)
                metrics['avg_profit_per_loan'] = np.mean(profits)
        
        metrics['uptime_hours'] = (datetime.now() - self.start_time).total_seconds() / 3600
        metrics['n_alerts'] = len(self.alerts)
        
        return metrics
    
    def get_alerts(self, last_n: Optional[int] = None) -> List[Dict]:
        """Get recent alerts."""
        if last_n:
            return self.alerts[-last_n:]
        return self.alerts
    
    def detect_drift(self, reference_data: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift in recent predictions.
        
        Args:
            reference_data: Reference data (training set)
            
        Returns:
            Drift detection results
        """
        if not self.enable_drift:
            return {'error': 'Drift detection not enabled'}
        
        if len(self.logs) < 100:
            return {'error': 'Insufficient data for drift detection'}
        
        # Get recent features
        recent_features = np.array([log.features for log in self.logs[-1000:]])
        if recent_features.ndim == 3:
            recent_features = recent_features.squeeze(1)
        
        # Detect feature drift
        self.drift_detector.reference_data = reference_data
        drift_results = self.drift_detector.detect_feature_drift(recent_features)
        
        # Detect prediction drift
        ref_predictions = np.random.binomial(1, 0.5, len(reference_data))  # Placeholder
        curr_predictions = np.array([log.prediction for log in self.logs[-1000:]])
        pred_drift = self.drift_detector.detect_prediction_drift(ref_predictions, curr_predictions)
        
        return {
            'feature_drift': drift_results,
            'prediction_drift': pred_drift
        }
    
    def save_logs(self, filename: Optional[str] = None):
        """Save logs to file."""
        if filename is None:
            filename = f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            for log in self.logs:
                f.write(json.dumps(log.to_dict()) + '\n')
        
        logger.info(f"Saved {len(self.logs)} logs to {filepath}")
    
    def generate_report(self) -> str:
        """Generate monitoring report."""
        metrics = self.get_metrics()
        alerts = self.get_alerts(last_n=10)
        
        report = []
        report.append("="*70)
        report.append("PERFORMANCE MONITORING REPORT".center(70))
        report.append("="*70)
        report.append(f"\nReport Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Uptime: {metrics.get('uptime_hours', 0):.1f} hours")
        
        report.append("\n--- Performance Metrics ---")
        report.append(f"Total Predictions: {metrics.get('total_predictions', 0):,}")
        report.append(f"Approval Rate: {metrics.get('approval_rate', 0):.1f}%")
        report.append(f"Avg Confidence: {metrics.get('avg_confidence', 0):.3f}")
        report.append(f"Avg Latency: {metrics.get('avg_latency_ms', 0):.1f}ms")
        report.append(f"P95 Latency: {metrics.get('p95_latency_ms', 0):.1f}ms")
        report.append(f"Error Rate: {metrics.get('error_rate', 0):.2f}%")
        
        if metrics.get('n_outcomes', 0) > 0:
            report.append("\n--- Model Performance ---")
            report.append(f"Outcomes Tracked: {metrics['n_outcomes']}")
            report.append(f"Default Rate: {metrics.get('default_rate', 0):.1f}%")
            report.append(f"Precision: {metrics.get('precision', 0):.3f}")
            report.append(f"Recall: {metrics.get('recall', 0):.3f}")
            report.append(f"F1 Score: {metrics.get('f1', 0):.3f}")
            report.append(f"Total Profit: ${metrics.get('total_profit', 0):,.0f}")
        
        if alerts:
            report.append(f"\n--- Recent Alerts ({len(alerts)}) ---")
            for alert in alerts[-5:]:
                report.append(f"[{alert['severity'].upper()}] {alert['message']}")
        
        report.append("\n" + "="*70)
        
        return '\n'.join(report)


if __name__ == '__main__':
    # Example usage
    print("="*70)
    print("PERFORMANCE MONITORING EXAMPLE".center(70))
    print("="*70)
    
    # Initialize monitor
    monitor = PerformanceMonitor(log_dir='../logs', metrics_window=100)
    
    # Simulate predictions
    np.random.seed(42)
    for i in range(150):
        features = np.random.randn(1, 20)
        
        # Mock prediction
        prediction = {
            'action': np.random.choice([0, 1], p=[0.4, 0.6]),
            'confidence': np.random.uniform(0.5, 0.95),
            'latency_ms': np.random.uniform(10, 100),
            'model_type': 'xgboost',
            'model_version': '1.0.0'
        }
        
        # Mock outcome (for some)
        outcome = np.random.choice([0, 1], p=[0.85, 0.15]) if i % 2 == 0 else None
        
        monitor.log_prediction(
            features=features,
            prediction=prediction,
            outcome=outcome,
            loan_amnt=10000.0,
            int_rate=0.12
        )
    
    # Get metrics
    print("\n" + monitor.generate_report())
    
    # Save logs
    monitor.save_logs()
    
    print("\n✅ Monitoring example complete")
