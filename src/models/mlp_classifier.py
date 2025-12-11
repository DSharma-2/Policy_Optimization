"""
MLP Classifier for Loan Default Prediction

Advanced PyTorch implementation with:
- Batch normalization
- Dropout regularization
- Learning rate scheduling
- Early stopping
- Class imbalance handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


# ========================================================================
# DATASET
# ========================================================================

class LoanDataset(Dataset):
    """PyTorch Dataset for loan data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        X : array
            Features (N, D)
        y : array
            Labels (N,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ========================================================================
# MLP ARCHITECTURE
# ========================================================================

class MLPClassifier(nn.Module):
    """
    Multi-layer Perceptron for binary classification.
    
    Architecture:
        Input → Dense(512) + BN + ReLU + Dropout(0.3)
             → Dense(256) + BN + ReLU + Dropout(0.2)
             → Dense(64) + ReLU
             → Output(1)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 64],
        dropout_rates: List[float] = [0.3, 0.2, 0.0],
        use_batch_norm: bool = True
    ):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features
        hidden_dims : list
            Hidden layer dimensions
        dropout_rates : list
            Dropout rate for each layer
        use_batch_norm : bool
            Use batch normalization
        """
        super(MLPClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm and i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features (batch_size, input_dim)
            
        Returns
        -------
        torch.Tensor
            Logits (batch_size, 1)
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features
            
        Returns
        -------
        torch.Tensor
            Probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            probas = torch.sigmoid(logits)
        return probas


# ========================================================================
# TRAINER
# ========================================================================

class MLPTrainer:
    """Trainer for MLP classifier with advanced features."""
    
    def __init__(
        self,
        model: MLPClassifier,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        pos_weight: Optional[float] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        """
        Parameters
        ----------
        model : MLPClassifier
            Model to train
        device : str
            Device to use
        pos_weight : float, optional
            Weight for positive class (for imbalanced data)
        learning_rate : float
            Learning rate
        weight_decay : float
            L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss function with class weights
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight]).to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(1)
            
            # Forward pass
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item() * len(X_batch)
        
        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluate model on a dataset."""
        self.model.eval()
        total_loss = 0
        all_probas = []
        all_labels = []
        
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(1)
            
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            
            probas = torch.sigmoid(logits)
            
            total_loss += loss.item() * len(X_batch)
            all_probas.extend(probas.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader.dataset)
        all_probas = np.array(all_probas).flatten()
        all_labels = np.array(all_labels).flatten()
        
        return avg_loss, all_probas, all_labels
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data
        val_loader : DataLoader
            Validation data
        epochs : int
            Maximum number of epochs
        early_stopping_patience : int
            Patience for early stopping
        verbose : bool
            Print progress
            
        Returns
        -------
        dict
            Training history
        """
        from sklearn.metrics import roc_auc_score
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        if verbose:
            print("=" * 70)
            print("TRAINING MLP CLASSIFIER")
            print("=" * 70)
            print(f"Device: {self.device}")
            print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Epochs: {epochs}")
            print(f"Early stopping patience: {early_stopping_patience}")
            print("=" * 70)
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate
            val_loss, val_probas, val_labels = self.evaluate(val_loader)
            _, train_probas, train_labels = self.evaluate(train_loader)
            
            # Compute AUC
            train_auc = roc_auc_score(train_labels, train_probas)
            val_auc = roc_auc_score(val_labels, val_probas)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
            # Learning rate scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if verbose:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train AUC: {train_auc:.4f} | "
                      f"Val AUC: {val_auc:.4f} | "
                      f"LR: {current_lr:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\n⏹ Early stopping triggered at epoch {epoch}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if verbose:
                print(f"✅ Restored best model (Val Loss: {best_val_loss:.4f})")
        
        if verbose:
            print("=" * 70)
        
        return self.history
    
    def predict_proba(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Predict probabilities for new data.
        
        Parameters
        ----------
        X : array
            Features
        batch_size : int
            Batch size
            
        Returns
        -------
        array
            Predicted probabilities
        """
        dataset = LoanDataset(X, np.zeros(len(X)))  # Dummy labels
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_probas = []
        
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch)
                probas = torch.sigmoid(logits)
                all_probas.extend(probas.cpu().numpy())
        
        return np.array(all_probas).flatten()


# ========================================================================
# FACTORY FUNCTION
# ========================================================================

def create_mlp_classifier(
    input_dim: int,
    hidden_dims: List[int] = [512, 256, 64],
    dropout_rates: List[float] = [0.3, 0.2, 0.0],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> MLPClassifier:
    """
    Create and return an MLP classifier.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    hidden_dims : list
        Hidden layer dimensions
    dropout_rates : list
        Dropout rates
    device : str
        Device
        
    Returns
    -------
    MLPClassifier
    """
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rates=dropout_rates
    )
    
    return model.to(device)


if __name__ == "__main__":
    print("MLP classifier module loaded")
    print(f"CUDA available: {torch.cuda.is_available()}")
