"""
Offline RL Algorithms for Loan Approval

Implements state-of-the-art offline RL algorithms:
1. Conservative Q-Learning (CQL) - Kumar et al., 2020
2. Behavior Cloning from Demonstrations (BCQ) - Fujimoto et al., 2019
3. Implicit Q-Learning (IQL) - Kostrikov et al., 2021

All algorithms use d3rlpy library for implementation.

Author: AI Toolkit
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import json

try:
    import d3rlpy
    from d3rlpy.algos import DiscreteCQLConfig, DiscreteBCQConfig, DiscreteSACConfig
    from d3rlpy.dataset import MDPDataset, create_fifo_replay_buffer
    from d3rlpy.metrics import TDErrorEvaluator, AverageValueEstimationEvaluator
    from d3rlpy.models import VectorEncoderFactory
    D3RLPY_AVAILABLE = True
except ImportError as e:
    D3RLPY_AVAILABLE = False
    print(f"⚠️  d3rlpy import failed: {e}")
    print("Install with: pip install d3rlpy")
    # Define dummy types to avoid NameErrors
    MDPDataset = None
    DiscreteCQLConfig = None
    DiscreteBCQConfig = None
    DiscreteSACConfig = None
    TDErrorEvaluator = None
    AverageValueEstimationEvaluator = None
    VectorEncoderFactory = None


class OfflineRLConfig:
    """Configuration for offline RL algorithms."""
    
    def __init__(
        self,
        algorithm: str = "CQL",
        n_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cuda:0",
        # CQL-specific
        cql_alpha: float = 1.0,
        # BCQ-specific
        bcq_lam: float = 0.75,
        # IQL-specific
        iql_tau: float = 0.7,
        iql_beta: float = 3.0,
        # Common
        use_gpu: bool = False,
        verbose: bool = True
    ):
        self.algorithm = algorithm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device if use_gpu else "cpu:0"
        self.cql_alpha = cql_alpha
        self.bcq_lam = bcq_lam
        self.iql_tau = iql_tau
        self.iql_beta = iql_beta
        self.use_gpu = use_gpu
        self.verbose = verbose


class OfflineRLAgent:
    """
    Wrapper for offline RL algorithms using d3rlpy.
    
    Supports:
    - Conservative Q-Learning (CQL)
    - Behavior Cloning from Demonstrations (BCQ)
    - Implicit Q-Learning (IQL)
    """
    
    def __init__(self, config: OfflineRLConfig):
        """
        Initialize offline RL agent.
        
        Args:
            config: Algorithm configuration
        """
        if not D3RLPY_AVAILABLE:
            raise ImportError("d3rlpy is required. Install with: pip install d3rlpy")
            
        self.config = config
        self.algorithm = None
        self.training_history = []
        
        print(f"🤖 Initializing {config.algorithm} agent")
        print(f"   Device: {config.device}")
        print(f"   Epochs: {config.n_epochs}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Learning rate: {config.learning_rate}")
        
    def create_algorithm(self, state_dim: int, action_dim: int) -> Any:
        """
        Create offline RL algorithm.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            
        Returns:
            d3rlpy algorithm instance
        """
        algo_name = self.config.algorithm.upper()
        
        # Common configuration
        device = self.config.device
        encoder_factory = VectorEncoderFactory(hidden_units=[256, 256])
        
        if algo_name == "CQL":
            print(f"   CQL alpha: {self.config.cql_alpha}")
            config = DiscreteCQLConfig(
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                alpha=self.config.cql_alpha,
                encoder_factory=encoder_factory
            )
            algo = config.create(device=device)
            
        elif algo_name == "BCQ":
            print(f"   BCQ lambda: {self.config.bcq_lam}")
            config = DiscreteBCQConfig(
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                action_flexibility=self.config.bcq_lam,
                encoder_factory=encoder_factory
            )
            algo = config.create(device=device)
            
        elif algo_name == "IQL":
            # Note: DiscreteIQL doesn't exist in d3rlpy, using DiscreteSAC as alternative
            print(f"   Using Discrete SAC (IQL not available for discrete actions)")
            config = DiscreteSACConfig(
                actor_learning_rate=self.config.learning_rate,
                critic_learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                actor_encoder_factory=encoder_factory,
                critic_encoder_factory=encoder_factory
            )
            algo = config.create(device=device)
            
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
            
        return algo
        
    def create_mdp_dataset(self, dataset: Dict[str, np.ndarray]) -> MDPDataset:
        """
        Create d3rlpy MDPDataset from dictionary.
        
        Args:
            dataset: Dictionary with states, actions, rewards, next_states, terminals
            
        Returns:
            d3rlpy MDPDataset
        """
        # d3rlpy 2.x uses MDPDataset for offline RL
        # Prepare data in the correct format
        observations = dataset['states'].astype(np.float32)
        actions = dataset['actions'].reshape(-1).astype(np.int32)  # d3rlpy discrete actions are 1D
        rewards = dataset['rewards'].reshape(-1).astype(np.float32)
        terminals = dataset['terminals'].reshape(-1).astype(np.float32)
        
        # Create MDPDataset
        mdp_dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals
        )
        
        return mdp_dataset
        
    def train(
        self,
        train_dataset: Dict[str, np.ndarray],
        val_dataset: Optional[Dict[str, np.ndarray]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train offline RL agent.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            save_path: Path to save trained model (optional)
            
        Returns:
            Training history
        """
        print("\n" + "=" * 70)
        print(f"TRAINING {self.config.algorithm.upper()} AGENT")
        print("=" * 70)
        
        # Create d3rlpy MDPDataset
        train_mdp = self.create_mdp_dataset(train_dataset)
        
        # Infer dimensions
        state_dim = train_dataset['states'].shape[1]
        action_dim = 1  # Discrete action (0 or 1)
        
        print(f"\nDataset info:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Train samples: {len(train_dataset['states']):,}")
        if val_dataset is not None:
            print(f"  Val samples: {len(val_dataset['states']):,}")
            
        # Create algorithm
        self.algorithm = self.create_algorithm(state_dim, action_dim)
        
        # Prepare evaluators
        evaluators = {}
        if val_dataset is not None:
            val_mdp = self.create_mdp_dataset(val_dataset)
            evaluators = {
                'td_error': TDErrorEvaluator(episodes=val_mdp.episodes),
                'avg_value': AverageValueEstimationEvaluator(episodes=val_mdp.episodes)
            }
            
        # Train
        print(f"\n🚀 Training for {self.config.n_epochs} epochs...")
        print("-" * 70)
        
        try:
            # Calculate total steps for n_epochs
            n_steps_per_epoch = len(train_dataset['states']) // self.config.batch_size
            n_steps = self.config.n_epochs * n_steps_per_epoch
            
            self.algorithm.fit(
                train_mdp,
                n_steps=n_steps,
                n_steps_per_epoch=n_steps_per_epoch,
                evaluators=evaluators,
                show_progress=self.config.verbose
            )
            
            print("\n✅ Training complete!")
            
            # Save model
            if save_path is not None:
                self.save_model(save_path)
                
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            raise
            
        print("=" * 70)
        
        return {'algorithm': self.config.algorithm, 'epochs': self.config.n_epochs}
        
    def predict(self, states: np.ndarray) -> np.ndarray:
        """
        Predict actions for given states.
        
        Args:
            states: States (n_samples, state_dim)
            
        Returns:
            Actions (n_samples,)
        """
        if self.algorithm is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # d3rlpy returns 2D actions, we need 1D
        actions = self.algorithm.predict(states)
        if len(actions.shape) > 1:
            actions = actions.flatten()
            
        return actions.astype(np.int32)
        
    def predict_value(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Predict Q-values for state-action pairs.
        
        Args:
            states: States (n_samples, state_dim)
            actions: Actions (n_samples,)
            
        Returns:
            Q-values (n_samples,)
        """
        if self.algorithm is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Reshape actions for d3rlpy
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)
            
        q_values = self.algorithm.predict_value(states, actions)
        
        return q_values.flatten()
        
    def save_model(self, path: str) -> None:
        """
        Save trained model.
        
        Args:
            path: Save path
        """
        if self.algorithm is None:
            raise ValueError("No model to save")
            
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model using d3rlpy 2.x API (saves to directory)
        model_path = str(save_path).replace('.pt', '_model.d3')
        self.algorithm.save(model_path)
        
        # Save config
        config_path = str(save_path).replace('.pt', '_config.json')
        config_dict = {
            'algorithm': self.config.algorithm,
            'n_epochs': self.config.n_epochs,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'gamma': self.config.gamma
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Config saved: {config_path}")
        
    def load_model(self, path: str) -> None:
        """
        Load trained model.
        
        Args:
            path: Model path
        """
        model_path = str(path).replace('.pt', '_model.d3')
        
        # Determine algorithm type from config
        config_path = str(path).replace('.pt', '_config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        algo_name = config_dict['algorithm'].upper()
        
        # Load appropriate algorithm using d3rlpy 2.x API
        # Create empty config and load weights
        if algo_name == "CQL":
            config = DiscreteCQLConfig()
            self.algorithm = config.create(device=self.config.device)
        elif algo_name == "BCQ":
            config = DiscreteBCQConfig()
            self.algorithm = config.create(device=self.config.device)
        elif algo_name == "IQL":
            # IQL uses SAC for discrete actions
            config = DiscreteSACConfig()
            self.algorithm = config.create(device=self.config.device)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        
        # Load the saved weights
        self.algorithm.load(model_path)
            
        print(f"✅ Model loaded: {model_path}")


def train_offline_rl_agent(
    algorithm: str,
    train_dataset: Dict[str, np.ndarray],
    val_dataset: Optional[Dict[str, np.ndarray]] = None,
    n_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    use_gpu: bool = False,
    save_path: Optional[str] = None,
    **kwargs
) -> Tuple[OfflineRLAgent, Dict]:
    """
    Train offline RL agent with specified algorithm.
    
    Args:
        algorithm: Algorithm name ("CQL", "BCQ", or "IQL")
        train_dataset: Training dataset
        val_dataset: Validation dataset
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_gpu: Whether to use GPU
        save_path: Path to save model
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Tuple of (trained_agent, history)
    """
    # Create config
    config = OfflineRLConfig(
        algorithm=algorithm,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_gpu=use_gpu,
        **kwargs
    )
    
    # Create and train agent
    agent = OfflineRLAgent(config)
    history = agent.train(train_dataset, val_dataset, save_path)
    
    return agent, history


if __name__ == "__main__":
    print("Offline RL Algorithms for Loan Approval")
    print("=" * 70)
    print("\nSupported algorithms:")
    print("  1. CQL (Conservative Q-Learning) - Kumar et al., 2020")
    print("     - Penalizes Q-values for OOD actions")
    print("     - Best for general offline RL")
    print("\n  2. BCQ (Batch-Constrained Q-Learning) - Fujimoto et al., 2019")
    print("     - Learns behavioral policy")
    print("     - Constrains actions to dataset support")
    print("\n  3. IQL (Implicit Q-Learning) - Kostrikov et al., 2021")
    print("     - No explicit policy constraint")
    print("     - Often best for offline RL")
    print("=" * 70)
