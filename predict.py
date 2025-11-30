"""
Prediction module
Single inference function and artifact management
"""

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml
import warnings
warnings.filterwarnings('ignore')


class LSTMModel(nn.Module):
    """LSTM model architecture (must match training)"""
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


def load_artifacts(model_path: str, model_type: str = 'auto') -> Dict:
    """
    Load model artifacts (model, scaler, feature names, etc.)
    
    Args:
        model_path: Path to model file
        model_type: 'lightgbm', 'xgboost', 'lstm', 'linear', 'ridge', or 'auto'
        
    Returns:
        Dictionary with all artifacts
    """
    model_path = Path(model_path)
    
    if model_type == 'auto':
        # Infer from file extension
        if model_path.suffix == '.pth':
            model_type = 'lstm'
        elif model_path.suffix == '.pkl':
            # Try loading to determine type
            try:
                data = joblib.load(model_path)
                if 'model' in data:
                    if hasattr(data['model'], 'predict_proba'):
                        model_type = 'tree'  # XGBoost or LightGBM
                    else:
                        model_type = 'linear'
            except:
                model_type = 'unknown'
    
    artifacts = {}
    
    if model_type == 'lstm':
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        artifacts['model'] = checkpoint
        artifacts['model_type'] = 'lstm'
        artifacts['scaler_mean'] = checkpoint.get('scaler_mean')
        artifacts['scaler_scale'] = checkpoint.get('scaler_scale')
        artifacts['feature_names'] = checkpoint.get('feature_names', [])
        artifacts['sequence_length'] = checkpoint.get('sequence_length', 24)
        artifacts['hidden_size'] = checkpoint.get('hidden_size', 64)
        artifacts['num_layers'] = checkpoint.get('num_layers', 1)
        artifacts['dropout'] = checkpoint.get('dropout', 0.3)
        artifacts['input_size'] = checkpoint.get('input_size')
        artifacts['bidirectional'] = checkpoint.get('bidirectional', False)
        
        # Reconstruct model - check if bidirectional
        if artifacts['bidirectional']:
            # Improved LSTM with bidirectional architecture
            class ImprovedLSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout):
                    super(ImprovedLSTMModel, self).__init__()
                    self.lstm = nn.LSTM(
                        input_size, hidden_size, num_layers,
                        batch_first=True, dropout=dropout if num_layers > 1 else 0,
                        bidirectional=True
                    )
                    self.dropout = nn.Dropout(dropout)
                    self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
                    self.fc2 = nn.Linear(hidden_size, 1)
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    last_output = lstm_out[:, -1, :]
                    last_output = self.dropout(last_output)
                    hidden = torch.relu(self.fc1(last_output))
                    hidden = self.dropout(hidden)
                    output = self.fc2(hidden)
                    return output
            
            lstm_model = ImprovedLSTMModel(
                artifacts['input_size'],
                artifacts['hidden_size'],
                artifacts['num_layers'],
                artifacts['dropout']
            )
        else:
            # Standard LSTM
            lstm_model = LSTMModel(
                artifacts['input_size'],
                artifacts['hidden_size'],
                artifacts['num_layers'],
                artifacts['dropout']
            )
        
        lstm_model.load_state_dict(checkpoint['model_state_dict'])
        lstm_model.eval()
        artifacts['model_instance'] = lstm_model
        
    elif model_type in ['lightgbm', 'xgboost', 'linear', 'ridge', 'tree']:
        data = joblib.load(model_path)
        artifacts['model'] = data.get('model')
        artifacts['model_type'] = model_type
        artifacts['scaler'] = data.get('scaler')
        artifacts['feature_names'] = data.get('feature_names', [])
        artifacts['model_instance'] = artifacts['model']
    
    # Load config if available
    config_path = model_path.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            artifacts['config'] = yaml.safe_load(f)
    
    return artifacts


def save_artifacts(model, model_type: str, artifacts_dir: str, 
                  model_name: str, scaler=None, feature_names: list = None,
                  feature_schema: dict = None, training_range: tuple = None,
                  config: dict = None, **kwargs) -> str:
    """
    Save all model artifacts together
    
    Args:
        model: Trained model
        model_type: 'lightgbm', 'xgboost', 'lstm', 'linear', 'ridge'
        artifacts_dir: Directory to save artifacts
        model_name: Name for model file
        scaler: Fitted scaler (if used)
        feature_names: List of feature names
        feature_schema: Feature schema dictionary
        training_range: (start_date, end_date) tuple
        config: Configuration dictionary
        **kwargs: Additional model-specific parameters
        
    Returns:
        Path to saved model file
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    if model_type == 'lstm':
        # Save LSTM model
        model_path = artifacts_dir / f"{model_name}.pth"
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_type': 'lstm',
            'input_size': kwargs.get('input_size'),
            'hidden_size': kwargs.get('hidden_size', 128),
            'num_layers': kwargs.get('num_layers', 1),
            'dropout': kwargs.get('dropout', 0.3),
            'sequence_length': kwargs.get('sequence_length', 48),
            'bidirectional': kwargs.get('bidirectional', False),  # Support bidirectional LSTM
        }
        
        if scaler is not None:
            save_dict['scaler_mean'] = scaler.mean_
            save_dict['scaler_scale'] = scaler.scale_
        
        if feature_names:
            save_dict['feature_names'] = feature_names
        
        if feature_schema:
            save_dict['feature_schema'] = feature_schema
        
        if training_range:
            save_dict['training_range'] = training_range
        
        if config:
            save_dict['config'] = config
        
        torch.save(save_dict, model_path)
        
    else:
        # Save sklearn/tree models
        model_path = artifacts_dir / f"{model_name}.pkl"
        
        save_dict = {
            'model': model,
            'model_type': model_type,
        }
        
        if scaler:
            save_dict['scaler'] = scaler
        
        if feature_names:
            save_dict['feature_names'] = feature_names
        
        if feature_schema:
            save_dict['feature_schema'] = feature_schema
        
        if training_range:
            save_dict['training_range'] = training_range
        
        if config:
            save_dict['config'] = config
        
        joblib.dump(save_dict, model_path)
    
    # Save config separately
    if config:
        config_path = artifacts_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    
    return str(model_path)


def predict(model_path: str, features_df: pd.DataFrame, 
           history_df: Optional[pd.DataFrame] = None,
           required_history_hours: int = 72) -> float:
    """
    Single inference function: takes latest timestamp + required history → creates features → predicts
    
    Args:
        model_path: Path to saved model
        features_df: DataFrame with features (should include recent history)
        history_df: Optional historical data for feature creation
        required_history_hours: Hours of history needed for lag/rolling features
        
    Returns:
        Predicted water level
    """
    # Load artifacts
    artifacts = load_artifacts(model_path)
    model_type = artifacts.get('model_type', 'unknown')
    feature_names = artifacts.get('feature_names', [])
    
    if model_type == 'lstm':
        model = artifacts['model_instance']
        scaler_mean = artifacts['scaler_mean']
        scaler_scale = artifacts['scaler_scale']
        sequence_length = artifacts['sequence_length']
        
        # Ensure we have enough history
        if len(features_df) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} hours of history for LSTM")
        
        # Get features
        available_features = [f for f in feature_names if f in features_df.columns]
        if len(available_features) < len(feature_names) * 0.7:
            raise ValueError(f"Missing too many features. Need {len(feature_names)}, have {len(available_features)}")
        
        # Get sequence
        X = features_df[available_features].iloc[-sequence_length:].values
        
        # Fill missing features with 0
        full_X = np.zeros((sequence_length, len(feature_names)))
        for j, fname in enumerate(feature_names):
            if fname in available_features:
                idx = available_features.index(fname)
                full_X[:, j] = X[:, idx]
        
        # Scale
        X_scaled = (full_X - scaler_mean) / scaler_scale
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)
        
        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(X_tensor).item()
        
        return pred
    
    elif model_type in ['lightgbm', 'xgboost', 'linear', 'ridge', 'tree']:
        model = artifacts['model_instance']
        scaler = artifacts.get('scaler')
        
        # Get latest features
        available_features = [f for f in feature_names if f in features_df.columns]
        if len(available_features) < len(feature_names) * 0.7:
            raise ValueError(f"Missing too many features. Need {len(feature_names)}, have {len(available_features)}")
        
        X = features_df[available_features].iloc[-1:].values
        
        # Fill missing features with 0
        full_X = np.zeros((1, len(feature_names)))
        for j, fname in enumerate(feature_names):
            if fname in available_features:
                idx = available_features.index(fname)
                full_X[0, j] = X[0, idx]
        
        # Scale if needed
        if scaler:
            full_X = scaler.transform(full_X)
        
        # Predict
        pred = model.predict(full_X)[0]
        
        return pred
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_batch(model_path: str, features_df: pd.DataFrame) -> np.ndarray:
    """
    Predict for multiple timestamps
    
    Args:
        model_path: Path to saved model
        features_df: DataFrame with features for all timestamps
        
    Returns:
        Array of predictions
    """
    artifacts = load_artifacts(model_path)
    model_type = artifacts.get('model_type', 'unknown')
    feature_names = artifacts.get('feature_names', [])
    
    predictions = []
    
    if model_type == 'lstm':
        model = artifacts['model_instance']
        scaler_mean = artifacts['scaler_mean']
        scaler_scale = artifacts['scaler_scale']
        sequence_length = artifacts['sequence_length']
        
        for i in range(sequence_length - 1, len(features_df)):
            X = features_df[feature_names].iloc[i-sequence_length+1:i+1].values
            X_scaled = (X - scaler_mean) / scaler_scale
            X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)
            
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).item()
            
            predictions.append(pred)
    
    else:
        model = artifacts['model_instance']
        scaler = artifacts.get('scaler')
        
        X = features_df[feature_names].values
        if scaler:
            X = scaler.transform(X)
        
        predictions = model.predict(X)
    
    return np.array(predictions)


if __name__ == "__main__":
    print("Prediction module loaded successfully")
    print("Use predict() function for single predictions")
    print("Use load_artifacts() to load saved models")
