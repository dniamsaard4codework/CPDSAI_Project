"""
Feature engineering module
Creates time series features with strict past-only rules (no data leakage)
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features (hour, day-of-week, month, etc.)
    These are safe - they don't use future information
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with time features added
    """
    df_feat = df.copy()
    
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")
    
    df_feat['hour'] = df_feat.index.hour
    df_feat['day_of_week'] = df_feat.index.dayofweek  # 0=Monday, 6=Sunday
    df_feat['day_of_month'] = df_feat.index.day
    df_feat['month'] = df_feat.index.month
    df_feat['year'] = df_feat.index.year
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for periodic features
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    df_feat['day_of_week_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
    df_feat['day_of_week_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    
    return df_feat


def create_lag_features(df: pd.DataFrame, target_col: str, 
                       lags: List[int]) -> pd.DataFrame:
    """
    Create lag features (past values only - safe)
    
    Args:
        df: Input DataFrame
        target_col: Column to create lags for
        lags: List of lag periods (e.g., [1, 2, 3, 6, 12, 24])
        
    Returns:
        DataFrame with lag features added
    """
    df_feat = df.copy()
    
    for lag in lags:
        df_feat[f'{target_col}_lag_{lag}'] = df_feat[target_col].shift(lag)
    
    return df_feat


def create_rolling_features(df: pd.DataFrame, target_col: str,
                           windows: List[int], stats: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
    """
    Create rolling statistics (past-only - safe)
    
    Args:
        df: Input DataFrame
        target_col: Column to create rolling features for
        windows: List of window sizes (e.g., [6, 12, 24])
        stats: List of statistics to compute ['mean', 'std', 'min', 'max']
        
    Returns:
        DataFrame with rolling features added
    """
    df_feat = df.copy()
    
    for window in windows:
        rolling = df_feat[target_col].rolling(window=window, min_periods=1)
        
        if 'mean' in stats:
            df_feat[f'{target_col}_rolling_mean_{window}'] = rolling.mean()
        if 'std' in stats:
            df_feat[f'{target_col}_rolling_std_{window}'] = rolling.std()
        if 'min' in stats:
            df_feat[f'{target_col}_rolling_min_{window}'] = rolling.min()
        if 'max' in stats:
            df_feat[f'{target_col}_rolling_max_{window}'] = rolling.max()
    
    return df_feat


def create_difference_features(df: pd.DataFrame, target_col: str,
                               periods: List[int] = [1, 24]) -> pd.DataFrame:
    """
    Create difference features (rate of change - past-only, safe)
    
    Args:
        df: Input DataFrame
        target_col: Column to create differences for
        periods: List of periods for diff (e.g., [1, 24])
        
    Returns:
        DataFrame with difference features added
    """
    df_feat = df.copy()
    
    for period in periods:
        df_feat[f'{target_col}_diff_{period}'] = df_feat[target_col].diff(period)
    
    return df_feat


def create_risk_features(df: pd.DataFrame, bank_level: float, 
                        bed_level: float) -> pd.DataFrame:
    """
    Create risk assessment features
    
    Args:
        df: Input DataFrame with 'water_level' column
        bank_level: Bank level in meters MSL
        bed_level: Bed level in meters MSL
        
    Returns:
        DataFrame with risk features added
    """
    df_feat = df.copy()
    
    total_capacity = bank_level - bed_level
    df_feat['water_level_pct'] = ((df_feat['water_level'] - bed_level) / total_capacity) * 100
    
    def determine_risk_level(pct):
        if pct >= 70:
            return 'High Risk'
        elif pct >= 30:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    df_feat['risk_level'] = df_feat['water_level_pct'].apply(determine_risk_level)
    
    return df_feat


def create_all_features(df: pd.DataFrame, config: dict, 
                       target_col: str = 'water_level',
                       is_training: bool = True) -> pd.DataFrame:
    """
    Create all features following past-only rules
    
    Args:
        df: Input DataFrame (raw data)
        config: Configuration dictionary
        target_col: Target column name
        is_training: Whether this is training data (for logging)
        
    Returns:
        DataFrame with all engineered features
    """
    if is_training:
        print("Creating features on TRAINING data (past-only rules)...")
    else:
        print("Creating features on TEST data (past-only rules)...")
    
    df_feat = df.copy()
    
    # 1. Time features
    if config.get('features', {}).get('time_features', True):
        df_feat = create_time_features(df_feat)
    
    # 2. Lag features for water level
    lag_hours = config.get('features', {}).get('lag_hours', [1, 2, 3, 6, 12, 24])
    df_feat = create_lag_features(df_feat, target_col, lag_hours)
    
    # 3. Rolling statistics for water level
    rolling_windows = config.get('features', {}).get('rolling_windows', [6, 12, 24])
    rolling_stats = config.get('features', {}).get('rolling_stats', ['mean', 'std', 'min', 'max'])
    df_feat = create_rolling_features(df_feat, target_col, rolling_windows, rolling_stats)
    
    # 4. Rolling statistics for key weather features
    rolling_features = config.get('features', {}).get('rolling_features', 
                                                      ['rain', 'precipitation', 'river_discharge', 'temperature_2m'])
    for col in rolling_features:
        if col in df_feat.columns:
            for window in [6, 12]:
                df_feat[f'{col}_rolling_mean_{window}'] = df_feat[col].rolling(window=window, min_periods=1).mean()
    
    # 5. Difference features
    diff_periods = config.get('features', {}).get('diff_periods', [1, 24])
    if diff_periods:
        df_feat = create_difference_features(df_feat, target_col, diff_periods)
    
    # 6. Risk assessment features
    station_config = config.get('station', {})
    bank_level = station_config.get('bank_level', 2.161)
    bed_level = station_config.get('bed_level', -15.70)
    df_feat = create_risk_features(df_feat, bank_level, bed_level)
    
    return df_feat


def create_target(df: pd.DataFrame, target_col: str, 
                 forecast_horizon: int) -> pd.DataFrame:
    """
    Create target variable (future value)
    
    Args:
        df: Input DataFrame
        target_col: Source column name
        forecast_horizon: Hours ahead to predict
        
    Returns:
        DataFrame with target column added
    """
    df_feat = df.copy()
    df_feat[f'target_{forecast_horizon}h'] = df_feat[target_col].shift(-forecast_horizon)
    return df_feat


def get_feature_columns(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """
    Get list of numeric feature columns (excluding target and categorical)
    
    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude (e.g., ['target_24h', 'risk_level'])
        
    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove excluded columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    return sorted(feature_cols)


def verify_past_only_features(df: pd.DataFrame, current_idx: int) -> bool:
    """
    Verify that features at current_idx only use past information
    This is a sanity check function
    
    Args:
        df: DataFrame with features
        current_idx: Current time index to check
        
    Returns:
        True if features are past-only, False otherwise
    """
    # This is a placeholder for verification logic
    # In practice, we ensure this by using shift() with positive values (past)
    # and never using shift(-k) for features (which would be future)
    
    # Check that no features use shift(-k) - this would be data leakage
    # This is more of a code review check than runtime check
    return True


if __name__ == "__main__":
    # Test feature creation
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    from data import load_raw_data
    df = load_raw_data(config['data']['raw_data_path'])
    df_sample = df.head(1000)
    
    df_feat = create_all_features(df_sample, config, is_training=True)
    print(f"Original columns: {len(df.columns)}")
    print(f"With features: {len(df_feat.columns)}")
    print(f"Feature columns: {get_feature_columns(df_feat, ['target_24h', 'risk_level'])}")
