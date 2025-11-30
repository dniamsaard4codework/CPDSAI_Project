"""
Evaluation module
Implements baselines, metrics, and evaluation functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        sMAPE value
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                      metrics: List[str] = None) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary of metric names and values
    """
    if metrics is None:
        metrics = ['MAE', 'RMSE', 'MSE', 'MAPE', 'sMAPE', 'R2']
    
    results = {}
    
    if 'MAE' in metrics:
        results['MAE'] = mean_absolute_error(y_true, y_pred)
    
    if 'RMSE' in metrics:
        results['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if 'MSE' in metrics:
        results['MSE'] = mean_squared_error(y_true, y_pred)
    
    if 'MAPE' in metrics:
        results['MAPE'] = mape(y_true, y_pred)
    
    if 'sMAPE' in metrics:
        results['sMAPE'] = smape(y_true, y_pred)
    
    if 'R2' in metrics:
        results['R2'] = r2_score(y_true, y_pred)
    
    return results


def persistence_baseline(y: np.ndarray, horizon: int = 24) -> np.ndarray:
    """
    Persistence baseline: y_hat(t+horizon) = y(t)
    
    Args:
        y: Time series values
        horizon: Forecast horizon
        
    Returns:
        Predictions
    """
    predictions = np.full_like(y, np.nan)
    predictions[:-horizon] = y[horizon:]
    return predictions


def seasonal_naive_baseline(y: np.ndarray, season_length: int = 24) -> np.ndarray:
    """
    Seasonal naive baseline: y_hat(t+season_length) = y(t)
    
    Args:
        y: Time series values
        season_length: Season length (e.g., 24 for daily cycle)
        
    Returns:
        Predictions
    """
    predictions = np.full_like(y, np.nan)
    predictions[season_length:] = y[:-season_length]
    return predictions


def rolling_mean_baseline(y: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling mean baseline: y_hat(t) = mean(y(t-window+1), ..., y(t))
    
    Args:
        y: Time series values
        window: Rolling window size
        
    Returns:
        Predictions
    """
    return pd.Series(y).rolling(window=window, min_periods=1).mean().values


def evaluate_baselines(y_true: np.ndarray, y_series: np.ndarray, 
                      config: dict) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all baseline models
    
    Args:
        y_true: True target values
        y_series: Full time series (for baselines)
        config: Configuration dictionary
        
    Returns:
        Dictionary of baseline results
    """
    results = {}
    horizon = config.get('forecast', {}).get('horizon', 24)
    metrics = config.get('metrics', ['MAE', 'RMSE', 'MSE', 'MAPE', 'sMAPE', 'R2'])
    
    # Persistence baseline
    if config.get('baselines', {}).get('persistence', True):
        y_pred_persist = persistence_baseline(y_series, horizon)
        valid_mask = ~np.isnan(y_pred_persist) & ~np.isnan(y_true)
        if valid_mask.sum() > 0:
            results['Persistence'] = calculate_metrics(
                y_true[valid_mask], y_pred_persist[valid_mask], metrics
            )
    
    # Seasonal naive baseline
    if config.get('baselines', {}).get('seasonal_naive', True):
        y_pred_seasonal = seasonal_naive_baseline(y_series, season_length=horizon)
        valid_mask = ~np.isnan(y_pred_seasonal) & ~np.isnan(y_true)
        if valid_mask.sum() > 0:
            results['Seasonal_Naive'] = calculate_metrics(
                y_true[valid_mask], y_pred_seasonal[valid_mask], metrics
            )
    
    # Rolling mean baselines
    if config.get('baselines', {}).get('rolling_mean', True):
        windows = config.get('baselines', {}).get('rolling_mean_windows', [6, 12, 24])
        for window in windows:
            y_pred_rolling = rolling_mean_baseline(y_series, window)
            # Shift by horizon for forecast
            y_pred_rolling_shifted = np.full_like(y_pred_rolling, np.nan)
            y_pred_rolling_shifted[horizon:] = y_pred_rolling[:-horizon]
            
            valid_mask = ~np.isnan(y_pred_rolling_shifted) & ~np.isnan(y_true)
            if valid_mask.sum() > 0:
                results[f'Rolling_Mean_{window}h'] = calculate_metrics(
                    y_true[valid_mask], y_pred_rolling_shifted[valid_mask], metrics
                )
    
    return results


def error_by_period(y_true: np.ndarray, y_pred: np.ndarray,
                   timestamps: pd.DatetimeIndex) -> Dict[str, float]:
    """
    Calculate error metrics by time period (high-water/risk periods)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        timestamps: Datetime index
        
    Returns:
        Dictionary with error metrics by period
    """
    errors = np.abs(y_true - y_pred)
    
    results = {}
    
    # Error by hour of day
    hours = timestamps.hour
    for hour in range(24):
        mask = hours == hour
        if mask.sum() > 0:
            results[f'hour_{hour}'] = {
                'MAE': np.mean(errors[mask]),
                'count': mask.sum()
            }
    
    # Error by day of week
    day_of_week = timestamps.dayofweek
    for day in range(7):
        mask = day_of_week == day
        if mask.sum() > 0:
            results[f'day_{day}'] = {
                'MAE': np.mean(errors[mask]),
                'count': mask.sum()
            }
    
    # Error by month
    months = timestamps.month
    for month in range(1, 13):
        mask = months == month
        if mask.sum() > 0:
            results[f'month_{month}'] = {
                'MAE': np.mean(errors[mask]),
                'count': mask.sum()
            }
    
    # Error by water level (high vs low)
    median_level = np.median(y_true)
    high_mask = y_true >= median_level
    low_mask = y_true < median_level
    
    results['high_water'] = {
        'MAE': np.mean(errors[high_mask]) if high_mask.sum() > 0 else np.nan,
        'count': high_mask.sum()
    }
    results['low_water'] = {
        'MAE': np.mean(errors[low_mask]) if low_mask.sum() > 0 else np.nan,
        'count': low_mask.sum()
    }
    
    return results


if __name__ == "__main__":
    # Test metrics
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    
    metrics = calculate_metrics(y_true, y_pred)
    print("Metrics:", metrics)
    
    # Test baselines
    y_series = np.array([1.0, 1.5, 2.0, 1.8, 2.2, 2.5, 2.3, 2.7])
    y_pred_persist = persistence_baseline(y_series, horizon=2)
    print("Persistence:", y_pred_persist)
    
    y_pred_seasonal = seasonal_naive_baseline(y_series, season_length=2)
    print("Seasonal Naive:", y_pred_seasonal)
