"""
Model Retraining Pipeline

Automated retraining system for loan approval models:
- Data drift monitoring
- Scheduled retraining
- Model validation
- A/B testing integration

Example:
    >>> from deployment.retraining import RetrainingPipeline
    >>> 
    >>> pipeline = RetrainingPipeline()
    >>> pipeline.retrain_if_needed()
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrainingConfig:
    """Configuration for model retraining pipeline."""
    model_type: str  # 'xgboost', 'mlp', 'cql', 'iql'
    retrain_schedule: str = 'weekly'  # 'daily', 'weekly', 'monthly', 'manual'
    drift_threshold: float = 0.05
    performance_threshold: float = 0.9  # Min relative performance vs best model
    min_new_samples: int = 10000
    validation_metric: str = 'profit'  # 'profit', 'f1', 'auc'
    auto_deploy: bool = False  # Auto-deploy if validation passes
    backup_models: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)


class RetrainingPipeline:
    """
    Automated model retraining pipeline.
    
    Features:
    - Scheduled retraining
    - Drift-triggered retraining
    - Model validation
    - Automatic backup and rollback
    - Integration with A/B testing
    """
    
    def __init__(
        self,
        config: RetrainingConfig,
        data_dir: str = '../data/processed',
        model_dir: str = '../models/saved',
        log_dir: str = '../logs/retraining'
    ):
        """
        Initialize retraining pipeline.
        
        Args:
            config: Retraining configuration
            data_dir: Directory for training data
            model_dir: Directory for model files
            log_dir: Directory for retraining logs
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.last_retrain_time: Optional[datetime] = None
        self.retrain_history: List[Dict] = []
        
        logger.info(f"Retraining pipeline initialized for {config.model_type}")
    
    def should_retrain(
        self,
        current_data: Optional[pd.DataFrame] = None,
        reference_data: Optional[pd.DataFrame] = None,
        current_performance: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check if model should be retrained.
        
        Args:
            current_data: Recent data
            reference_data: Training data
            current_performance: Current model performance
            
        Returns:
            (should_retrain, reason)
        """
        # Check schedule
        if self.config.retrain_schedule != 'manual':
            if self._is_retrain_scheduled():
                return True, f"Scheduled retrain ({self.config.retrain_schedule})"
        
        # Check drift
        if current_data is not None and reference_data is not None:
            drift_detected, drift_score = self._detect_drift(current_data, reference_data)
            if drift_detected:
                return True, f"Drift detected (score={drift_score:.4f})"
        
        # Check performance degradation
        if current_performance is not None:
            if current_performance < self.config.performance_threshold:
                return True, f"Performance below threshold ({current_performance:.3f} < {self.config.performance_threshold})"
        
        # Check data volume
        if current_data is not None:
            if len(current_data) >= self.config.min_new_samples:
                return True, f"Sufficient new samples ({len(current_data):,})"
        
        return False, "No retrain trigger"
    
    def _is_retrain_scheduled(self) -> bool:
        """Check if scheduled retrain is due."""
        if self.last_retrain_time is None:
            return True
        
        now = datetime.now()
        delta = now - self.last_retrain_time
        
        if self.config.retrain_schedule == 'daily':
            return delta >= timedelta(days=1)
        elif self.config.retrain_schedule == 'weekly':
            return delta >= timedelta(weeks=1)
        elif self.config.retrain_schedule == 'monthly':
            return delta >= timedelta(days=30)
        
        return False
    
    def _detect_drift(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame
    ) -> Tuple[bool, float]:
        """
        Detect data drift using statistical tests.
        
        Returns:
            (drift_detected, drift_score)
        """
        from scipy import stats
        
        # Compare distributions for each feature
        p_values = []
        
        for col in current_data.columns:
            if col in reference_data.columns:
                try:
                    # Kolmogorov-Smirnov test
                    _, p_value = stats.ks_2samp(
                        reference_data[col].dropna(),
                        current_data[col].dropna()
                    )
                    p_values.append(p_value)
                except:
                    pass
        
        if not p_values:
            return False, 1.0
        
        # Drift score: proportion of features with p < threshold
        drift_score = np.mean([p < self.config.drift_threshold for p in p_values])
        drift_detected = drift_score > 0.2  # >20% features drifted
        
        return drift_detected, drift_score
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load training, validation, and test data."""
        from data.preprocess import load_processed_data
        
        train_df, val_df, test_df = load_processed_data(str(self.data_dir))
        
        logger.info(f"Loaded data: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
        
        return train_df, val_df, test_df
    
    def train_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> Any:
        """
        Train a new model.
        
        Args:
            train_df: Training data
            val_df: Validation data
            
        Returns:
            Trained model
        """
        logger.info(f"Training {self.config.model_type} model...")
        
        X_train = train_df.drop('default', axis=1).values
        y_train = train_df['default'].values
        X_val = val_df.drop('default', axis=1).values
        y_val = val_df['default'].values
        
        if self.config.model_type == 'xgboost':
            import xgboost as xgb
            
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
        elif self.config.model_type == 'mlp':
            import torch
            from models.mlp_classifier import MLPClassifier, train_mlp
            
            model = MLPClassifier(
                input_dim=X_train.shape[1],
                hidden_dims=[128, 64, 32]
            )
            
            train_mlp(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=50,
                batch_size=256,
                lr=0.001,
                verbose=False
            )
            
        elif self.config.model_type in ['cql', 'iql']:
            from rl.mdp_formulation import create_mdp_from_dataframes
            from rl.offline_algorithms import train_offline_rl_agent
            
            # Create MDP
            test_df = val_df.copy()  # Use val as test for now
            mdp, train_ds, val_ds, test_ds = create_mdp_from_dataframes(
                train_df, val_df, test_df
            )
            
            # Train RL agent
            model = train_offline_rl_agent(
                dataset=train_ds,
                algorithm=self.config.model_type.upper(),
                n_epochs=100,
                batch_size=256,
                lr=3e-4,
                verbose=False
            )
            
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        logger.info(f"✅ Model trained")
        
        return model
    
    def validate_model(
        self,
        model: Any,
        test_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Validate trained model.
        
        Args:
            model: Trained model
            test_df: Test data
            
        Returns:
            Validation metrics
        """
        logger.info("Validating model...")
        
        X_test = test_df.drop('default', axis=1).values
        y_test = test_df['default'].values
        
        if self.config.model_type in ['xgboost', 'random_forest', 'logistic']:
            # Supervised model
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba < 0.3).astype(int)  # Approve if prob < 0.3
            
        elif self.config.model_type == 'mlp':
            import torch
            model.eval()
            with torch.no_grad():
                y_pred_proba = torch.sigmoid(model(torch.FloatTensor(X_test))).numpy()
            y_pred = (y_pred_proba < 0.3).astype(int)
            
        elif self.config.model_type in ['cql', 'iql']:
            # RL policy
            from rl.mdp_formulation import LoanApprovalMDP
            mdp = LoanApprovalMDP()
            mdp.fit_normalization(X_test)
            normalized_states = mdp.normalize_states(X_test)
            
            y_pred = model.predict(normalized_states)
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'approval_rate': np.mean(y_pred) * 100,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Add AUC if probabilities available
        if self.config.model_type in ['xgboost', 'mlp']:
            try:
                metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
            except:
                pass
        
        # Calculate profit
        if 'loan_amnt' in test_df.columns and 'int_rate' in test_df.columns:
            approved = y_pred == 1
            defaults = y_test == 1
            
            profit = 0.0
            for i in range(len(y_pred)):
                if y_pred[i] == 1:  # Approved
                    if y_test[i] == 0:  # Paid
                        profit += test_df.iloc[i]['loan_amnt'] * test_df.iloc[i]['int_rate'] * 0.01
                    else:  # Default
                        profit -= test_df.iloc[i]['loan_amnt'] * 0.01
            
            metrics['total_profit'] = profit
            metrics['profit_per_loan'] = profit / len(y_pred)
        
        logger.info(f"Validation metrics: {metrics}")
        
        return metrics
    
    def backup_model(self, model_path: Path) -> Path:
        """Create backup of current model."""
        if not model_path.exists():
            return None
        
        backup_path = model_path.parent / f"{model_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{model_path.suffix}"
        
        import shutil
        shutil.copy(model_path, backup_path)
        
        logger.info(f"Backed up model to {backup_path}")
        
        return backup_path
    
    def save_model(self, model: Any, metrics: Dict[str, float]):
        """Save trained model."""
        if self.config.model_type == 'xgboost':
            model_path = self.model_dir / 'xgboost_model.pkl'
        elif self.config.model_type == 'mlp':
            model_path = self.model_dir / 'mlp_classifier.pt'
        elif self.config.model_type in ['cql', 'iql']:
            model_path = self.model_dir / 'rl' / f'{self.config.model_type}_agent.pt'
        else:
            model_path = self.model_dir / f'{self.config.model_type}_model.pkl'
        
        # Backup existing model
        if self.config.backup_models:
            self.backup_model(model_path)
        
        # Save new model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.model_type in ['xgboost', 'random_forest', 'logistic']:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        elif self.config.model_type == 'mlp':
            import torch
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': model.fc1.in_features,
                'hidden_dims': [128, 64, 32],
                'metrics': metrics
            }, model_path)
        elif self.config.model_type in ['cql', 'iql']:
            model.save_model(str(model_path))
        
        logger.info(f"✅ Model saved to {model_path}")
        
        # Save metrics
        metrics_path = model_path.parent / f"{model_path.stem}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'model_type': self.config.model_type,
                'metrics': metrics
            }, f, indent=2)
    
    def retrain(self, force: bool = False) -> Dict[str, Any]:
        """
        Execute retraining pipeline.
        
        Args:
            force: Force retrain regardless of triggers
            
        Returns:
            Retraining results
        """
        start_time = datetime.now()
        
        logger.info("="*70)
        logger.info(f"RETRAINING PIPELINE: {self.config.model_type}")
        logger.info("="*70)
        
        # Load data
        train_df, val_df, test_df = self.load_training_data()
        
        # Check if should retrain
        if not force:
            should_retrain, reason = self.should_retrain(
                current_data=val_df.tail(1000),
                reference_data=train_df.sample(1000, random_state=42)
            )
            
            if not should_retrain:
                logger.info(f"Skipping retrain: {reason}")
                return {
                    'retrained': False,
                    'reason': reason,
                    'timestamp': start_time.isoformat()
                }
            
            logger.info(f"Retraining triggered: {reason}")
        
        # Train model
        model = self.train_model(train_df, val_df)
        
        # Validate model
        metrics = self.validate_model(model, test_df)
        
        # Save model
        self.save_model(model, metrics)
        
        # Update state
        self.last_retrain_time = datetime.now()
        duration = (self.last_retrain_time - start_time).total_seconds()
        
        result = {
            'retrained': True,
            'model_type': self.config.model_type,
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'metrics': metrics
        }
        
        self.retrain_history.append(result)
        
        logger.info(f"✅ Retraining complete ({duration:.1f}s)")
        logger.info("="*70)
        
        return result
    
    def save_state(self):
        """Save pipeline state."""
        state_path = self.log_dir / f"{self.config.model_type}_state.json"
        
        state = {
            'config': self.config.to_dict(),
            'last_retrain_time': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'history': self.retrain_history
        }
        
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"State saved to {state_path}")


if __name__ == '__main__':
    # Example usage
    print("="*70)
    print("RETRAINING PIPELINE EXAMPLE".center(70))
    print("="*70)
    
    # XGBoost retraining
    config = RetrainingConfig(
        model_type='xgboost',
        retrain_schedule='manual',
        min_new_samples=1000,
        validation_metric='f1',
        auto_deploy=False
    )
    
    pipeline = RetrainingPipeline(config)
    
    # Check if data exists
    if (Path('../data/processed/train.parquet').exists()):
        print("\n1. Checking retrain triggers...")
        train_df, val_df, test_df = pipeline.load_training_data()
        
        should_retrain, reason = pipeline.should_retrain(
            current_data=val_df.tail(1000),
            reference_data=train_df.sample(1000, random_state=42)
        )
        
        print(f"Should retrain: {should_retrain} ({reason})")
        
        if should_retrain:
            print("\n2. Executing retrain...")
            result = pipeline.retrain(force=True)
            print(f"\nRetrain result: {result}")
    else:
        print("\n⚠️  Training data not found. Run preprocessing first.")
    
    print("\n✅ Retraining pipeline example complete")
