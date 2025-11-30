"""
Training module with fold-safe preprocessing
Uses Pipeline to ensure preprocessing is fit on train fold only
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select specific features from DataFrame"""
    def __init__(self, feature_names: list):
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            # Ensure all features exist, fill missing with 0
            X_selected = pd.DataFrame(index=X.index)
            for feat in self.feature_names:
                if feat in X.columns:
                    X_selected[feat] = X[feat]
                else:
                    X_selected[feat] = 0.0
            return X_selected.values
        return X


def create_pipeline(model, use_scaler: bool = True, feature_names: list = None):
    """
    Create sklearn Pipeline with preprocessing
    
    Args:
        model: Model to use
        use_scaler: Whether to use StandardScaler
        feature_names: List of feature names (for FeatureSelector)
        
    Returns:
        Pipeline object
    """
    steps = []
    
    if feature_names:
        steps.append(('selector', FeatureSelector(feature_names)))
    
    if use_scaler:
        steps.append(('scaler', StandardScaler()))
    
    steps.append(('model', model))
    
    return Pipeline(steps)


def train_model_cv(model, X_train_val: pd.DataFrame, y_train_val: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series,
                  config: dict, model_name: str,
                  use_scaler: bool = True) -> Dict[str, Any]:
    """
    Train model with time series cross-validation (fold-safe preprocessing)
    
    Args:
        model: Model to train
        X_train_val: Training+validation features
        y_train_val: Training+validation target
        X_test: Test features
        y_test: Test target
        config: Configuration dictionary
        model_name: Name of model
        use_scaler: Whether model needs scaling
        
    Returns:
        Dictionary with results
    """
    n_splits = config.get('cv', {}).get('n_splits', 5)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    feature_names = X_train_val.columns.tolist()
    
    cv_scores = {
        'mse': [],
        'mae': [],
        'rmse': [],
        'r2': []
    }
    
    print(f"\n=== {model_name} (with fold-safe preprocessing) ===")
    
    # Cross-validation with fold-safe preprocessing
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val), 1):
        # Split data
        X_tr = X_train_val.iloc[train_idx]
        y_tr = y_train_val.iloc[train_idx]
        X_val = X_train_val.iloc[val_idx]
        y_val = y_train_val.iloc[val_idx]
        
        # Create pipeline (scaler fit on train fold only)
        pipeline = create_pipeline(model, use_scaler=use_scaler, feature_names=feature_names)
        
        # Fit on train fold
        pipeline.fit(X_tr, y_tr)
        
        # Predict on validation fold
        y_pred_val = pipeline.predict(X_val)
        
        # Calculate metrics
        from evaluate import calculate_metrics
        metrics = calculate_metrics(y_val.values, y_pred_val)
        
        cv_scores['mse'].append(metrics['MSE'])
        cv_scores['mae'].append(metrics['MAE'])
        cv_scores['rmse'].append(metrics['RMSE'])
        cv_scores['r2'].append(metrics['R2'])
        
        print(f"  Fold {fold}: MAE={metrics['MAE']:.6f}, R2={metrics['R2']:.4f}")
    
    # Average CV scores
    avg_cv_mae = np.mean(cv_scores['mae'])
    avg_cv_rmse = np.mean(cv_scores['rmse'])
    avg_cv_r2 = np.mean(cv_scores['r2'])
    
    print(f"\n  CV Average: MAE={avg_cv_mae:.6f}, RMSE={avg_cv_rmse:.6f}, R2={avg_cv_r2:.4f}")
    
    # Train final model on full train+val set
    print(f"\n  Training on full train+val set...")
    final_pipeline = create_pipeline(model, use_scaler=use_scaler, feature_names=feature_names)
    final_pipeline.fit(X_train_val, y_train_val)
    
    # Evaluate on test set
    y_pred_test = final_pipeline.predict(X_test)
    
    from evaluate import calculate_metrics
    test_metrics = calculate_metrics(y_test.values, y_pred_test)
    
    print(f"  Test Set: MAE={test_metrics['MAE']:.6f}, RMSE={test_metrics['RMSE']:.6f}, R2={test_metrics['R2']:.4f}")
    
    return {
        'cv_mae': avg_cv_mae,
        'cv_rmse': avg_cv_rmse,
        'cv_r2': avg_cv_r2,
        'test_metrics': test_metrics,
        'predictions': y_pred_test,
        'pipeline': final_pipeline,
        'model': final_pipeline.named_steps['model'] if 'model' in final_pipeline.named_steps else final_pipeline
    }


def train_tree_model_cv(model, X_train_val: pd.DataFrame, y_train_val: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series,
                       config: dict, model_name: str) -> Dict[str, Any]:
    """
    Train tree-based model (XGBoost/LightGBM) - no scaling needed
    
    Args:
        model: XGBoost or LightGBM model
        X_train_val: Training+validation features
        y_train_val: Training+validation target
        X_test: Test features
        y_test: Test target
        config: Configuration dictionary
        model_name: Name of model
        
    Returns:
        Dictionary with results
    """
    # Tree models don't need scaling, but we still use fold-safe approach
    return train_model_cv(model, X_train_val, y_train_val, X_test, y_test,
                         config, model_name, use_scaler=False)


# LSTM Model Architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length=24):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.sequence_length]
        y_val = self.y[idx+self.sequence_length-1]
        return torch.FloatTensor(X_seq), torch.FloatTensor([y_val])


def train_lstm_cv(X_train_val: pd.DataFrame, y_train_val: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series,
                 config: dict) -> Dict[str, Any]:
    """
    Train LSTM with fold-safe scaling (scaler fit on train fold only)
    
    Args:
        X_train_val: Training+validation features
        y_train_val: Training+validation target
        X_test: Test features
        y_test: Test target
        config: Configuration dictionary
        
    Returns:
        Dictionary with results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    lstm_config = config.get('models', {}).get('lstm', {})
    sequence_length = config.get('forecast', {}).get('sequence_length', 48)
    hidden_size = lstm_config.get('hidden_size', 128)
    num_layers = lstm_config.get('num_layers', 1)
    dropout = lstm_config.get('dropout', 0.3)
    batch_size = lstm_config.get('batch_size', 64)
    epochs = lstm_config.get('epochs', 50)
    lr = lstm_config.get('learning_rate', 0.001)
    weight_decay = lstm_config.get('weight_decay', 0.001)
    patience = lstm_config.get('patience', 10)
    
    input_size = X_train_val.shape[1]
    n_splits = config.get('cv', {}).get('n_splits', 5)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = {
        'mse': [],
        'mae': [],
        'rmse': [],
        'r2': []
    }
    
    print(f"\n=== LSTM (with fold-safe scaling) ===")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}\n")
    
    # Cross-validation with fold-safe scaling
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val), 1):
        print(f"\n  Fold {fold}:")
        
        # Split data
        X_tr = X_train_val.iloc[train_idx].values
        y_tr = y_train_val.iloc[train_idx].values
        X_val = X_train_val.iloc[val_idx].values
        y_val = y_train_val.iloc[val_idx].values
        
        # Fit scaler on train fold only
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        
        # Create datasets (ensure sequences don't cross boundaries)
        train_dataset = TimeSeriesDataset(X_tr_scaled, y_tr, sequence_length)
        val_dataset = TimeSeriesDataset(X_val_scaled, y_val, sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
        
        # Evaluate on validation set
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(y_batch.numpy().flatten())
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        from evaluate import calculate_metrics
        metrics = calculate_metrics(val_targets, val_preds)
        
        cv_scores['mse'].append(metrics['MSE'])
        cv_scores['mae'].append(metrics['MAE'])
        cv_scores['rmse'].append(metrics['RMSE'])
        cv_scores['r2'].append(metrics['R2'])
        
        print(f"    Val: MAE={metrics['MAE']:.6f}, R2={metrics['R2']:.4f}")
    
    # Average CV scores
    avg_cv_mae = np.mean(cv_scores['mae'])
    avg_cv_rmse = np.mean(cv_scores['rmse'])
    avg_cv_r2 = np.mean(cv_scores['r2'])
    
    print(f"\n  CV Average: MAE={avg_cv_mae:.6f}, RMSE={avg_cv_rmse:.6f}, R2={avg_cv_r2:.4f}")
    
    # Train final model on full train+val set
    print(f"\n  Training final model on full train+val set...")
    scaler_final = StandardScaler()
    X_train_val_scaled = scaler_final.fit_transform(X_train_val.values)
    y_train_val_values = y_train_val.values
    
    full_dataset = TimeSeriesDataset(X_train_val_scaled, y_train_val_values, sequence_length)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    
    final_model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(epochs):
        final_model.train()
        for X_batch, y_batch in full_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = final_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optimizer.step()
    
    # Evaluate on test set
    X_test_scaled = scaler_final.transform(X_test.values)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test.values, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    final_model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = final_model(X_batch)
            test_preds.extend(outputs.cpu().numpy().flatten())
            test_targets.extend(y_batch.numpy().flatten())
    
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    
    from evaluate import calculate_metrics
    test_metrics = calculate_metrics(test_targets, test_preds)
    
    print(f"  Test Set: MAE={test_metrics['MAE']:.6f}, RMSE={test_metrics['RMSE']:.6f}, R2={test_metrics['R2']:.4f}")
    
    return {
        'cv_mae': avg_cv_mae,
        'cv_rmse': avg_cv_rmse,
        'cv_r2': avg_cv_r2,
        'test_metrics': test_metrics,
        'predictions': test_preds,
        'targets': test_targets,
        'model': final_model,
        'scaler': scaler_final,
        'sequence_length': sequence_length,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout
    }


if __name__ == "__main__":
    # Test training
    print("Training module loaded successfully")
