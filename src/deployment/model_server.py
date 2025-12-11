"""
Model Serving API

Production-ready model serving for loan approval decisions.
Supports multiple model types (supervised, RL) with unified interface.

Example:
    >>> from deployment.model_server import LoanApprovalServer
    >>> 
    >>> config = ModelConfig(
    >>>     model_type='xgboost',
    >>>     model_path='models/saved/xgboost_model.pkl'
    >>> )
    >>> server = LoanApprovalServer(config)
    >>> decision = server.predict(applicant_features)
"""

import pickle
import json
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model serving."""
    model_type: str  # 'xgboost', 'mlp', 'cql', 'iql', 'ensemble'
    model_path: str
    threshold: Optional[float] = None  # For supervised models
    feature_names: Optional[List[str]] = None
    preprocessing_path: Optional[str] = None
    version: str = '1.0.0'
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ModelConfig':
        return cls(**config_dict)


class LoanApprovalServer:
    """
    Production model server for loan approval decisions.
    
    Supports:
    - Supervised models (XGBoost, MLP, RF, Logistic)
    - RL policies (CQL, IQL)
    - Ensemble predictions
    - Feature validation
    - Request logging
    """
    
    def __init__(self, config: ModelConfig, enable_logging: bool = True):
        """
        Initialize model server.
        
        Args:
            config: Model configuration
            enable_logging: Enable request/response logging
        """
        self.config = config
        self.enable_logging = enable_logging
        self.model = None
        self.preprocessing = None
        self.request_count = 0
        self.load_time = None
        
        self._load_model()
        logger.info(f"Model server initialized: {config.model_type} v{config.version}")
    
    def _load_model(self):
        """Load model from disk."""
        model_path = Path(self.config.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        start_time = datetime.now()
        
        try:
            if self.config.model_type in ['xgboost', 'random_forest', 'logistic']:
                # Load supervised models
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded {self.config.model_type} model")
                
            elif self.config.model_type == 'mlp':
                # Load PyTorch MLP
                import torch
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Reconstruct model
                from models.mlp_classifier import MLPClassifier
                self.model = MLPClassifier(
                    input_dim=checkpoint['input_dim'],
                    hidden_dims=checkpoint['hidden_dims']
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                logger.info("Loaded MLP model")
                
            elif self.config.model_type in ['cql', 'iql']:
                # Load RL policy
                from rl.offline_algorithms import OfflineRLAgent, OfflineRLConfig
                
                rl_config = OfflineRLConfig(algorithm=self.config.model_type.upper())
                self.model = OfflineRLAgent(rl_config)
                self.model.load_model(str(model_path))
                logger.info(f"Loaded {self.config.model_type.upper()} policy")
                
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
            
            # Load preprocessing if specified
            if self.config.preprocessing_path:
                with open(self.config.preprocessing_path, 'rb') as f:
                    self.preprocessing = pickle.load(f)
                logger.info("Loaded preprocessing pipeline")
            
            self.load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Model loaded in {self.load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _validate_features(self, features: Union[Dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Validate and convert input features to numpy array.
        
        Args:
            features: Input features (dict, DataFrame, or array)
            
        Returns:
            Validated numpy array
        """
        if isinstance(features, dict):
            # Convert dict to array
            if self.config.feature_names is None:
                raise ValueError("feature_names required for dict input")
            features = np.array([[features.get(f, 0) for f in self.config.feature_names]])
            
        elif isinstance(features, pd.DataFrame):
            features = features.values
            
        elif not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        return features
    
    def _preprocess(self, features: np.ndarray) -> np.ndarray:
        """Apply preprocessing if available."""
        if self.preprocessing is not None:
            return self.preprocessing.transform(features)
        return features
    
    def predict(self, features: Union[Dict, pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Make loan approval prediction.
        
        Args:
            features: Applicant features
            
        Returns:
            Prediction dict with keys:
                - decision: 'approve' or 'reject'
                - confidence: Confidence score [0, 1]
                - probability: Default probability (for supervised models)
                - action: Action index (for RL models)
                - timestamp: Prediction timestamp
                - model_type: Model type used
                - model_version: Model version
        """
        start_time = datetime.now()
        self.request_count += 1
        
        try:
            # Validate and preprocess
            features = self._validate_features(features)
            features = self._preprocess(features)
            
            # Make prediction based on model type
            if self.config.model_type in ['xgboost', 'random_forest', 'logistic']:
                # Supervised model
                proba = self.model.predict_proba(features)[0, 1]
                threshold = self.config.threshold or 0.5
                action = int(proba < threshold)  # Approve if prob < threshold
                confidence = 1 - proba if action == 1 else proba
                
                result = {
                    'decision': 'approve' if action == 1 else 'reject',
                    'confidence': float(confidence),
                    'probability': float(proba),
                    'action': action,
                    'threshold': threshold
                }
                
            elif self.config.model_type == 'mlp':
                # PyTorch MLP
                import torch
                with torch.no_grad():
                    logit = self.model(torch.FloatTensor(features))
                    proba = torch.sigmoid(logit).numpy()[0]
                
                threshold = self.config.threshold or 0.5
                action = int(proba < threshold)
                confidence = 1 - proba if action == 1 else proba
                
                result = {
                    'decision': 'approve' if action == 1 else 'reject',
                    'confidence': float(confidence),
                    'probability': float(proba),
                    'action': action,
                    'threshold': threshold
                }
                
            elif self.config.model_type in ['cql', 'iql']:
                # RL policy
                action = self.model.predict(features)[0]
                
                # Get Q-values for confidence
                try:
                    q_values = self.model.predict_value(features)[0]
                    confidence = float(abs(q_values[1] - q_values[0]))
                except:
                    confidence = 0.5  # Default if Q-values unavailable
                
                result = {
                    'decision': 'approve' if action == 1 else 'reject',
                    'confidence': confidence,
                    'action': int(action)
                }
            
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
            
            # Add metadata
            result.update({
                'timestamp': datetime.now().isoformat(),
                'model_type': self.config.model_type,
                'model_version': self.config.version,
                'latency_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'request_id': self.request_count
            })
            
            if self.enable_logging:
                logger.info(f"Request {self.request_count}: {result['decision']} "
                           f"(confidence={result['confidence']:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'decision': 'reject',  # Fail safe
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'request_id': self.request_count
            }
    
    def predict_batch(self, features_batch: Union[pd.DataFrame, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            features_batch: Batch of applicant features
            
        Returns:
            List of prediction dicts
        """
        features_batch = self._validate_features(features_batch)
        return [self.predict(features[np.newaxis, :]) for features in features_batch]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            'model_type': self.config.model_type,
            'model_version': self.config.version,
            'request_count': self.request_count,
            'load_time_seconds': self.load_time,
            'uptime_seconds': (datetime.now() - datetime.fromtimestamp(0)).total_seconds()
        }
    
    def save_config(self, path: str):
        """Save server configuration."""
        with open(path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Config saved to {path}")
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'LoanApprovalServer':
        """Load server from config file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ModelConfig.from_dict(config_dict)
        return cls(config)


class EnsembleServer(LoanApprovalServer):
    """
    Ensemble model server combining multiple models.
    """
    
    def __init__(self, configs: List[ModelConfig], weights: Optional[List[float]] = None):
        """
        Initialize ensemble server.
        
        Args:
            configs: List of model configs
            weights: Optional weights for each model (default: equal weights)
        """
        self.servers = [LoanApprovalServer(config, enable_logging=False) 
                       for config in configs]
        
        if weights is None:
            self.weights = [1.0 / len(configs)] * len(configs)
        else:
            self.weights = weights
        
        self.request_count = 0
        logger.info(f"Ensemble server initialized with {len(configs)} models")
    
    def predict(self, features: Union[Dict, pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Make ensemble prediction."""
        start_time = datetime.now()
        self.request_count += 1
        
        # Get predictions from all models
        predictions = [server.predict(features) for server in self.servers]
        
        # Weighted voting for decision
        weighted_actions = sum(pred['action'] * weight 
                              for pred, weight in zip(predictions, self.weights))
        
        action = 1 if weighted_actions >= 0.5 else 0
        confidence = abs(weighted_actions - 0.5) * 2
        
        return {
            'decision': 'approve' if action == 1 else 'reject',
            'confidence': float(confidence),
            'action': action,
            'ensemble_details': predictions,
            'weights': self.weights,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'ensemble',
            'latency_ms': (datetime.now() - start_time).total_seconds() * 1000,
            'request_id': self.request_count
        }


if __name__ == '__main__':
    # Example usage
    print("="*70)
    print("MODEL SERVER EXAMPLE".center(70))
    print("="*70)
    
    # Example 1: XGBoost model
    print("\n1. XGBoost Model Server:")
    config = ModelConfig(
        model_type='xgboost',
        model_path='../models/saved/xgboost_model.pkl',
        threshold=0.3,
        version='1.0.0'
    )
    
    # Check if model exists
    if Path(config.model_path).exists():
        server = LoanApprovalServer(config)
        
        # Mock features
        features = np.random.randn(1, 20)
        result = server.predict(features)
        print(f"Decision: {result['decision']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Stats: {server.get_stats()}")
    else:
        print("Model not found. Train models first.")
    
    # Example 2: RL policy
    print("\n2. RL Policy Server:")
    rl_config = ModelConfig(
        model_type='cql',
        model_path='../models/saved/rl/cql_agent.pt',
        version='1.0.0'
    )
    
    if Path(rl_config.model_path).exists() or Path(str(rl_config.model_path).replace('.pt', '_model.pt')).exists():
        rl_server = LoanApprovalServer(rl_config)
        result = rl_server.predict(features)
        print(f"Decision: {result['decision']}")
        print(f"Confidence: {result['confidence']:.3f}")
    else:
        print("RL model not found. Train RL policies first.")
    
    print("\n" + "="*70)
