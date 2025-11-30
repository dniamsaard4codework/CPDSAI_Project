"""
Data loading and quality checks module
Handles data loading, missingness analysis, outlier detection, and timezone checks
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load raw hourly data without engineered features
    
    Args:
        data_path: Path to raw data CSV file
        
    Returns:
        DataFrame with datetime index
    """
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.index.name = 'measure_datetime'
    df = df.sort_index()
    return df


def check_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate missingness report by column and time
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missingness statistics
    """
    missing_report = pd.DataFrame({
        'column': df.columns,
        'total_missing': df.isnull().sum().values,
        'pct_missing': (df.isnull().sum() / len(df) * 100).values,
        'non_null_count': df.notna().sum().values
    })
    missing_report = missing_report.sort_values('pct_missing', ascending=False)
    return missing_report


def check_timezone_consistency(df: pd.DataFrame) -> Dict:
    """
    Check timezone consistency and hourly continuity
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        Dictionary with timezone and gap information
    """
    info = {
        'timezone': str(df.index.tz) if df.index.tz else 'None (naive)',
        'has_timezone': df.index.tz is not None,
        'expected_freq': '1H',
        'actual_freq': pd.infer_freq(df.index),
        'is_regular': pd.infer_freq(df.index) == 'H'
    }
    
    # Check for gaps
    expected_range = pd.date_range(df.index.min(), df.index.max(), freq='H')
    missing_times = expected_range.difference(df.index)
    info['total_gaps'] = len(missing_times)
    info['gap_pct'] = len(missing_times) / len(expected_range) * 100 if len(expected_range) > 0 else 0
    
    if len(missing_times) > 0:
        info['gap_examples'] = missing_times[:10].tolist()
    
    return info


def detect_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 3.0) -> Dict:
    """
    Detect outliers using IQR or Z-score method
    
    Args:
        df: Input DataFrame (numeric columns only)
        method: "iqr" or "zscore"
        threshold: Threshold for outlier detection
        
    Returns:
        Dictionary with outlier information per column
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) == 0:
            continue
            
        if method == "iqr":
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        else:  # zscore
            z_scores = np.abs((values - values.mean()) / values.std())
            outlier_mask = z_scores > threshold
            outliers = df.loc[df[col].notna()].loc[outlier_mask]
        
        outlier_info[col] = {
            'count': len(outliers),
            'pct': len(outliers) / len(df) * 100,
            'min_outlier': outliers[col].min() if len(outliers) > 0 else None,
            'max_outlier': outliers[col].max() if len(outliers) > 0 else None
        }
    
    return outlier_info


def handle_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Handle missing values using specified imputation method
    
    Args:
        df: Input DataFrame
        method: "ffill", "bfill", "interpolate", or "mean"
        
    Returns:
        DataFrame with missing values filled
    """
    df_clean = df.copy()
    
    if method == "ffill":
        df_clean = df_clean.fillna(method='ffill')
    elif method == "bfill":
        df_clean = df_clean.fillna(method='bfill')
    elif method == "interpolate":
        df_clean = df_clean.interpolate(method='time')
    elif method == "mean":
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    
    # Fill any remaining NaN with forward fill
    df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    
    return df_clean


def check_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for duplicate timestamps
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with duplicate timestamp information
    """
    duplicates = df.index.duplicated(keep=False)
    if duplicates.sum() > 0:
        dup_info = df[duplicates].sort_index()
        return dup_info
    return pd.DataFrame()


def time_series_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically (no shuffling) for time series
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for test set
        
    Returns:
        train_val_df, test_df
    """
    split_idx = int((1 - test_size) * len(df))
    train_val_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_val_df, test_df


def prepare_data_for_test(df_test: pd.DataFrame, df_train: pd.DataFrame, 
                          history_hours: int = 72) -> pd.DataFrame:
    """
    Prepare test features using history from before test period
    Appends last N hours of train to test for lag/rolling computation, then cuts back
    
    Args:
        df_test: Test DataFrame
        df_train: Training DataFrame
        history_hours: Number of hours of history to append (for lag/rolling features)
        
    Returns:
        Test DataFrame with appended history (will be cut back after feature creation)
    """
    # Get last N hours from training set
    train_history = df_train.tail(history_hours).copy()
    
    # Append to test set
    df_test_with_history = pd.concat([train_history, df_test], axis=0)
    df_test_with_history = df_test_with_history.sort_index()
    
    return df_test_with_history


def data_quality_report(df: pd.DataFrame, config: dict) -> Dict:
    """
    Generate comprehensive data quality report
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        Dictionary with quality report
    """
    report = {}
    
    if config.get('data_quality', {}).get('check_missingness', True):
        report['missingness'] = check_missingness(df)
    
    if config.get('data_quality', {}).get('check_timezone', True):
        report['timezone'] = check_timezone_consistency(df)
    
    if config.get('data_quality', {}).get('check_outliers', True):
        method = config.get('data_quality', {}).get('outlier_method', 'iqr')
        threshold = config.get('data_quality', {}).get('outlier_threshold', 3.0)
        report['outliers'] = detect_outliers(df, method=method, threshold=threshold)
    
    if config.get('data_quality', {}).get('check_gaps', True):
        report['gaps'] = check_timezone_consistency(df)  # Includes gap info
    
    report['duplicates'] = check_duplicate_timestamps(df)
    
    return report


if __name__ == "__main__":
    # Test data loading
    config = load_config()
    df = load_raw_data(config['data']['raw_data_path'])
    print(f"Loaded data: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Quality check
    report = data_quality_report(df, config)
    print("\nMissingness:")
    print(report['missingness'].head())
    print("\nTimezone info:")
    print(report['timezone'])
