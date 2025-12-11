"""
XGBoost Baseline Model

High-performance gradient boosting baseline for loan default prediction.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# ========================================================================
# XGBOOST TRAINER
# ========================================================================

class XGBoostBaseline:
    """XGBoost baseline model with hyperparameter tuning."""
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        **kwargs
    ):
        """
        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate
        subsample : float
            Subsample ratio of training instances
        colsample_bytree : float
            Subsample ratio of columns
        scale_pos_weight : float, optional
            Balancing of positive and negative weights
        random_state : int
            Random seed
        **kwargs
            Additional XGBoost parameters
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': random_state,
            'tree_method': 'hist',
            'n_jobs': -1,
        }
        
        if scale_pos_weight is not None:
            self.params['scale_pos_weight'] = scale_pos_weight
        
        self.params.update(kwargs)
        
        self.model = None
        self.feature_names = None
        self.evals_result = {}
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ):
        """
        Train XGBoost model.
        
        Parameters
        ----------
        X_train : array
            Training features
        y_train : array
            Training labels
        X_val : array, optional
            Validation features
        y_val : array, optional
            Validation labels
        feature_names : list, optional
            Feature names
        early_stopping_rounds : int
            Early stopping patience
        verbose : bool
            Print progress
        """
        self.feature_names = feature_names
        
        # Create model
        self.model = xgb.XGBClassifier(**self.params)
        
        # Prepare validation set
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Train
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set if eval_set else None,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            verbose=verbose
        )
        
        # Store evaluation results
        if hasattr(self.model, 'evals_result_'):
            self.evals_result = self.model.evals_result_
        
        if verbose:
            print(f"✅ XGBoost training complete")
            if eval_set:
                best_iteration = self.model.best_iteration
                best_score = self.model.best_score
                print(f"   Best iteration: {best_iteration}")
                print(f"   Best score: {best_score:.4f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Parameters
        ----------
        X : array
            Features
            
        Returns
        -------
        array
            Predicted probabilities for class 1
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        probas = self.model.predict_proba(X)[:, 1]
        return probas
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array
            Features
        threshold : float
            Classification threshold
            
        Returns
        -------
        array
            Predicted labels
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain',
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get feature importance.
        
        Parameters
        ----------
        importance_type : str
            Type of importance ('gain', 'weight', 'cover', 'total_gain', 'total_cover')
        top_n : int, optional
            Return top N features
            
        Returns
        -------
        pd.DataFrame
            Feature importance dataframe
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        importance_type: str = 'gain',
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot feature importance.
        
        Parameters
        ----------
        top_n : int
            Number of top features to show
        importance_type : str
            Type of importance
        figsize : tuple
            Figure size
        """
        importance_df = self.get_feature_importance(importance_type, top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.invert_yaxis()
        
        ax.set_xlabel(f'Importance ({importance_type})', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curve(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot learning curve (train and validation metrics over iterations).
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        """
        if not self.evals_result:
            raise ValueError("No evaluation results available. Train with validation set.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get results
        results = self.evals_result
        epochs = len(results['validation_0']['auc'])
        x_axis = range(0, epochs)
        
        # Plot
        ax.plot(x_axis, results['validation_0']['auc'], label='Train', linewidth=2)
        if 'validation_1' in results:
            ax.plot(x_axis, results['validation_1']['auc'], label='Val', linewidth=2)
        
        ax.set_xlabel('Boosting Iteration', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('XGBoost Learning Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig


# ========================================================================
# HYPERPARAMETER SEARCH
# ========================================================================

def xgboost_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: Dict,
    verbose: bool = True
) -> Tuple[Dict, pd.DataFrame]:
    """
    Simple grid search for XGBoost hyperparameters.
    
    Parameters
    ----------
    X_train, y_train : arrays
        Training data
    X_val, y_val : arrays
        Validation data
    param_grid : dict
        Parameter grid to search
    verbose : bool
        Print progress
        
    Returns
    -------
    tuple
        (best_params, results_df)
    """
    from itertools import product
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    results = []
    
    if verbose:
        print(f"Running grid search over {len(combinations)} combinations...")
    
    for i, params in enumerate(combinations, 1):
        if verbose:
            print(f"  [{i}/{len(combinations)}] Testing: {params}")
        
        # Train model
        model = XGBoostBaseline(**params)
        model.fit(X_train, y_train, X_val, y_val, verbose=False)
        
        # Evaluate
        val_probas = model.predict_proba(X_val)
        val_auc = roc_auc_score(y_val, val_probas)
        
        results.append({
            **params,
            'val_auc': val_auc
        })
    
    results_df = pd.DataFrame(results).sort_values('val_auc', ascending=False)
    best_params = results_df.iloc[0].to_dict()
    best_params.pop('val_auc')
    
    if verbose:
        print(f"\n✅ Grid search complete")
        print(f"Best validation AUC: {results_df.iloc[0]['val_auc']:.4f}")
        print(f"Best params: {best_params}")
    
    return best_params, results_df


if __name__ == "__main__":
    print("XGBoost baseline module loaded")
