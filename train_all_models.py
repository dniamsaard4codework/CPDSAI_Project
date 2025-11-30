"""
Main training script using modular components
Demonstrates proper fold-safe preprocessing and evaluation
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)

# Import modules
from data import load_config, load_raw_data, time_series_split, prepare_data_for_test, data_quality_report
from features import create_all_features, create_target, get_feature_columns
from evaluate import evaluate_baselines, calculate_metrics, error_by_period
from train import train_model_cv, train_tree_model_cv, train_lstm_cv
from predict import save_artifacts
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb


def main():
    """Main training pipeline"""
    
    # Load configuration
    config = load_config('config.yaml')
    print("=" * 80)
    print("WATER LEVEL PREDICTION - TRAINING PIPELINE")
    print("=" * 80)
    print(f"Station: {config['station']['name']} ({config['station']['code']})")
    print(f"Forecast Horizon: {config['forecast']['horizon']} hours")
    print(f"Random Seed: {config['random_seed']}")
    print()
    
    # 1. Load raw data
    print("=" * 80)
    print("STEP 1: LOAD RAW DATA")
    print("=" * 80)
    df_raw = load_raw_data(config['data']['raw_data_path'])
    print(f"Loaded data: {df_raw.shape}")
    print(f"Date range: {df_raw.index.min()} to {df_raw.index.max()}")
    print()
    
    # 2. Data quality checks
    print("=" * 80)
    print("STEP 2: DATA QUALITY CHECKS")
    print("=" * 80)
    quality_report = data_quality_report(df_raw, config)
    
    if config.get('data_quality', {}).get('check_missingness', True):
        print("\nMissingness Report:")
        print(quality_report['missingness'].head(10))
    
    if config.get('data_quality', {}).get('check_timezone', True):
        print("\nTimezone & Gap Info:")
        tz_info = quality_report['timezone']
        print(f"  Timezone: {tz_info['timezone']}")
        print(f"  Regular frequency: {tz_info['is_regular']}")
        print(f"  Total gaps: {tz_info['total_gaps']} ({tz_info['gap_pct']:.2f}%)")
    
    print()
    
    # 3. Time series split (BEFORE feature engineering)
    print("=" * 80)
    print("STEP 3: TIME SERIES SPLIT (BEFORE FEATURE ENGINEERING)")
    print("=" * 80)
    test_size = config['split']['test_size']
    df_train_val_raw, df_test_raw = time_series_split(df_raw, test_size=test_size)
    
    print(f"Train+Val: {len(df_train_val_raw)} samples ({len(df_train_val_raw)/len(df_raw)*100:.1f}%)")
    print(f"  Date range: {df_train_val_raw.index.min()} to {df_train_val_raw.index.max()}")
    print(f"Test: {len(df_test_raw)} samples ({len(df_test_raw)/len(df_raw)*100:.1f}%)")
    print(f"  Date range: {df_test_raw.index.min()} to {df_test_raw.index.max()}")
    print()
    
    # 4. Feature engineering (separately on train and test)
    print("=" * 80)
    print("STEP 4: FEATURE ENGINEERING (SEPARATELY ON TRAIN/TEST)")
    print("=" * 80)
    
    # Create features on training set
    df_train_val_featured = create_all_features(
        df_train_val_raw, config, is_training=True
    )
    
    # Prepare test set with history from training
    history_hours = max(config['features']['lag_hours'] + config['features']['rolling_windows']) + 24
    df_test_with_history = prepare_data_for_test(
        df_test_raw, df_train_val_raw, history_hours=history_hours
    )
    
    # Create features on test set (with history appended)
    df_test_featured = create_all_features(
        df_test_with_history, config, is_training=False
    )
    
    # Cut back to test period only (after feature creation)
    df_test_featured = df_test_featured.loc[df_test_raw.index]
    
    print(f"Train+Val featured shape: {df_train_val_featured.shape}")
    print(f"Test featured shape: {df_test_featured.shape}")
    print()
    
    # 5. Create target variable
    print("=" * 80)
    print("STEP 5: CREATE TARGET VARIABLE")
    print("=" * 80)
    forecast_horizon = config['forecast']['horizon']
    target_col = f'target_{forecast_horizon}h'
    
    df_train_val_featured = create_target(
        df_train_val_featured, 'water_level', forecast_horizon
    )
    df_test_featured = create_target(
        df_test_featured, 'water_level', forecast_horizon
    )
    
    # Get feature columns
    exclude_cols = [target_col, 'risk_level']
    feature_cols = get_feature_columns(df_train_val_featured, exclude_cols)
    
    # Ensure same columns in both sets
    feature_cols = sorted(list(set(feature_cols) & set(df_test_featured.columns)))
    
    print(f"Feature columns: {len(feature_cols)}")
    print()
    
    # 6. Remove NaN rows
    print("=" * 80)
    print("STEP 6: REMOVE NaN ROWS")
    print("=" * 80)
    
    train_val_mask = ~(df_train_val_featured[feature_cols].isnull().any(axis=1) | 
                      df_train_val_featured[target_col].isnull())
    test_mask = ~(df_test_featured[feature_cols].isnull().any(axis=1) | 
                  df_test_featured[target_col].isnull())
    
    X_train_val = df_train_val_featured.loc[train_val_mask, feature_cols]
    y_train_val = df_train_val_featured.loc[train_val_mask, target_col]
    X_test = df_test_featured.loc[test_mask, feature_cols]
    y_test = df_test_featured.loc[test_mask, target_col]
    
    print(f"Train+Val: {len(X_train_val)} rows (removed {len(df_train_val_featured) - len(X_train_val)} rows)")
    print(f"Test: {len(X_test)} rows (removed {len(df_test_featured) - len(X_test)} rows)")
    print()
    
    # 7. Evaluate baselines
    print("=" * 80)
    print("STEP 7: EVALUATE BASELINES")
    print("=" * 80)
    
    baseline_results = evaluate_baselines(
        y_test.values, 
        df_test_featured['water_level'].values,
        config
    )
    
    for baseline_name, metrics in baseline_results.items():
        print(f"\n{baseline_name}:")
        for metric_name, value in metrics.items():
            if not np.isnan(value):
                print(f"  {metric_name}: {value:.6f}")
    print()
    
    # 8. Train models with fold-safe preprocessing
    print("=" * 80)
    print("STEP 8: TRAIN MODELS (WITH FOLD-SAFE PREPROCESSING)")
    print("=" * 80)
    
    all_results = {}
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_results = train_model_cv(
        lr_model, X_train_val, y_train_val, X_test, y_test,
        config, "Linear Regression", use_scaler=True
    )
    all_results['Linear Regression'] = lr_results
    
    # Ridge Regression
    ridge_alphas = config['models']['ridge']['alphas']
    ridge_model = RidgeCV(alphas=ridge_alphas, cv=TimeSeriesSplit(n_splits=5))
    ridge_results = train_model_cv(
        ridge_model, X_train_val, y_train_val, X_test, y_test,
        config, "Ridge Regression", use_scaler=True
    )
    all_results['Ridge Regression'] = ridge_results
    
    # XGBoost (no scaling)
    xgb_config = config['models']['xgboost']
    xgb_model = xgb.XGBRegressor(
        n_estimators=xgb_config['n_estimators'],
        max_depth=xgb_config['max_depth'],
        learning_rate=xgb_config['learning_rate'],
        reg_alpha=xgb_config['reg_alpha'],
        reg_lambda=xgb_config['reg_lambda'],
        subsample=xgb_config['subsample'],
        colsample_bytree=xgb_config['colsample_bytree'],
        min_child_weight=xgb_config['min_child_weight'],
        random_state=config['random_seed'],
        n_jobs=-1
    )
    xgb_results = train_tree_model_cv(
        xgb_model, X_train_val, y_train_val, X_test, y_test,
        config, "XGBoost"
    )
    all_results['XGBoost'] = xgb_results
    
    # LightGBM (no scaling)
    lgb_config = config['models']['lightgbm']
    lgb_model = lgb.LGBMRegressor(
        n_estimators=lgb_config['n_estimators'],
        max_depth=lgb_config['max_depth'],
        learning_rate=lgb_config['learning_rate'],
        reg_alpha=lgb_config['reg_alpha'],
        reg_lambda=lgb_config['reg_lambda'],
        subsample=lgb_config['subsample'],
        colsample_bytree=lgb_config['colsample_bytree'],
        min_child_samples=lgb_config['min_child_samples'],
        random_state=config['random_seed'],
        n_jobs=-1,
        verbose=-1
    )
    lgb_results = train_tree_model_cv(
        lgb_model, X_train_val, y_train_val, X_test, y_test,
        config, "LightGBM"
    )
    all_results['LightGBM'] = lgb_results
    
    # LSTM (with fold-safe scaling)
    lstm_results = train_lstm_cv(
        X_train_val, y_train_val, X_test, y_test, config
    )
    all_results['LSTM'] = lstm_results
    
    # 9. Model comparison
    print("=" * 80)
    print("STEP 9: MODEL COMPARISON")
    print("=" * 80)
    
    comparison_data = []
    for model_name, results in all_results.items():
        if 'test_metrics' in results:
            comparison_data.append({
                'Model': model_name,
                'CV MAE': results.get('cv_mae', np.nan),
                'Test MAE': results['test_metrics']['MAE'],
                'Test RMSE': results['test_metrics']['RMSE'],
                'Test R2': results['test_metrics']['R2']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model_name = comparison_df.loc[comparison_df['Test MAE'].idxmin(), 'Model']
    print(f"\n🏆 Best Model: {best_model_name}")
    print()
    
    # 10. Save best model artifacts
    print("=" * 80)
    print("STEP 10: SAVE ARTIFACTS")
    print("=" * 80)
    
    best_results = all_results[best_model_name]
    models_dir = Path(config['data']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if best_model_name == 'LSTM':
        model_path = save_artifacts(
            best_results['model'],
            'lstm',
            str(models_dir),
            'lstm_hourly_model_cv',
            scaler=best_results.get('scaler'),
            feature_names=feature_cols,
            training_range=(df_train_val_raw.index.min(), df_train_val_raw.index.max()),
            config=config,
            input_size=len(feature_cols),
            hidden_size=best_results['hidden_size'],
            num_layers=best_results['num_layers'],
            dropout=best_results['dropout'],
            sequence_length=best_results['sequence_length']
        )
    elif best_model_name in ['XGBoost', 'LightGBM']:
        model_path = save_artifacts(
            best_results['model'],
            best_model_name.lower(),
            str(models_dir),
            f'{best_model_name.lower()}_model_cv',
            scaler=None,  # Tree models don't use scaler
            feature_names=feature_cols,
            training_range=(df_train_val_raw.index.min(), df_train_val_raw.index.max()),
            config=config
        )
    else:
        # Linear/Ridge
        pipeline = best_results.get('pipeline')
        if pipeline:
            scaler = pipeline.named_steps.get('scaler')
            model = pipeline.named_steps.get('model')
            model_path = save_artifacts(
                model,
                best_model_name.lower().replace(' ', '_'),
                str(models_dir),
                f'{best_model_name.lower().replace(" ", "_")}_model_cv',
                scaler=scaler,
                feature_names=feature_cols,
                training_range=(df_train_val_raw.index.min(), df_train_val_raw.index.max()),
                config=config
            )
    
    print(f"Model saved to: {model_path}")
    print()
    
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
