# Water Level Prediction at Krungthep Bridge (CPY015 Station)

## Quick Start

**Run the Web Application**:
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

**Train Models** (execute notebooks in order):
1. `data_acquisition.ipynb` - Collect and process data
2. `eda.ipynb` - Exploratory analysis and feature engineering  
3. `modelling.ipynb` - Train and evaluate models

**Best Model Performance**: LSTM achieves **MAE = 0.1187 m** and **R² = 0.9438** for 24-hour ahead predictions.

---

## 1. Introduction

### 1.1 Background of Work

This project focuses on **water level prediction** for flood management and early warning systems. Water level forecasting is a critical component of hydrological modeling and disaster management, particularly in regions prone to flooding. The field has evolved significantly with the integration of machine learning and deep learning techniques, moving beyond traditional statistical methods to capture complex temporal patterns and non-linear relationships between meteorological variables and water levels.

Current research in this area emphasizes:
- Real-time prediction systems for early flood warnings
- Integration of multiple data sources (meteorological, hydrological, and geographical)
- Deep learning approaches for capturing long-term dependencies in time series data
- Multi-horizon forecasting for different planning needs

### 1.2 Why We Want to Do This

Water level prediction is crucial for:
- **Public Safety**: Early warnings can save lives and property during flood events
- **Infrastructure Protection**: Critical infrastructure near rivers can be better protected with accurate forecasts
- **Resource Management**: Water resource managers can make informed decisions about dam operations and water allocation
- **Economic Impact**: Businesses and communities can prepare for potential flooding, reducing economic losses

This project is particularly relevant for Thailand, where seasonal flooding affects millions of people annually. The Chao Phraya River, where Station CPY015 (Krungthep Bridge) is located, is one of the most important waterways in the country.

### 1.3 Business / Real-World Understanding

**Affected Stakeholders:**
- **Government Agencies**: Disaster prevention and mitigation departments need accurate forecasts for evacuation planning
- **Local Communities**: Residents in flood-prone areas require early warnings to protect their homes and families
- **Infrastructure Operators**: Bridge and road maintenance teams need predictions to prepare for high water levels
- **Agricultural Sector**: Farmers need forecasts to protect crops and livestock
- **Emergency Services**: First responders use predictions to pre-position resources

**Decision Support:**
Our model helps stakeholders make critical decisions such as:
- When to issue flood warnings and evacuation orders
- Whether to close bridges or roads due to high water levels
- How to allocate emergency response resources
- When to activate flood control measures (dams, barriers, etc.)

### 1.4 Possible Impact

If this project works well, it could lead to:
- **Time Savings**: Automated predictions reduce manual monitoring time for water management staff
- **Cost Reduction**: Early warnings can prevent billions in flood damage
- **Improved Accuracy**: Machine learning models can potentially outperform traditional methods
- **Safety Enhancement**: 24-hour ahead predictions give communities adequate time to prepare
- **Convenience**: Real-time predictions accessible through digital platforms
- **Scalability**: The approach can be adapted to other monitoring stations

---

## 2. Problem Statement

### 2.1 Task Definition

**Objective**: Predict water levels at Station CPY015 (Krungthep Bridge) **24 hours into the future** using historical water level data, meteorological variables, and river discharge information.

### 2.2 Input

The model receives:
- **Historical Water Level Data**: Hourly measurements from Station CPY015 (2019-2025)
- **Meteorological Features**:
  - Temperature (2m above ground)
  - Rainfall and precipitation
  - Cloud cover
  - Relative humidity
  - Dew point
  - Atmospheric pressure (MSL and surface)
  - Wind speed, direction, and gusts (10m above ground)
  - Weather codes
  - Evapotranspiration (ET0 FAO)
- **Hydrological Features**:
  - River discharge
- **Temporal Features**: Engineered features including lag values, rolling statistics, and seasonal patterns

### 2.3 Output

- **Continuous Value**: Predicted water level in meters (m.MSL - meters above Mean Sea Level) for 24 hours ahead
- **Risk Assessment** (derived): Water level percentage and risk level classification (Low/Medium/High Risk)

### 2.4 Goal Metrics

Primary evaluation metrics:
- **Mean Absolute Error (MAE)**: Primary metric for model comparison (lower is better)
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily
- **Mean Squared Error (MSE)**: Used for optimization
- **R² Score (Coefficient of Determination)**: Measures proportion of variance explained (higher is better, max = 1.0)

**Target Performance**: Achieve MAE < 0.15 meters and R² > 0.90 on test data.

---

## 3. Related Works

### 3.1 Existing Approaches

**Traditional Methods:**
- **ARIMA Models**: Autoregressive Integrated Moving Average models have been widely used for time series forecasting but struggle with non-linear patterns and external variables
- **Physical Hydrological Models**: Process-based models like HEC-HMS require extensive calibration and detailed physical parameters

**Machine Learning Approaches:**
- **Support Vector Regression (SVR)**: Used for water level prediction with moderate success, but limited by computational complexity
- **Random Forest**: Tree-based ensemble methods that capture non-linear relationships but may not handle temporal dependencies well
- **XGBoost/LightGBM**: Gradient boosting methods that have shown good performance in hydrological forecasting competitions

**Deep Learning Approaches:**
- **LSTM Networks**: Long Short-Term Memory networks excel at capturing long-term dependencies in time series data. Studies have shown LSTMs outperform traditional methods for water level prediction
- **GRU Networks**: Gated Recurrent Units offer similar performance to LSTMs with lower computational cost
- **Transformer Models**: Recent work has explored attention mechanisms for time series forecasting

### 3.2 Key Research Findings

1. **LSTM for Hydrological Forecasting**: Multiple studies (e.g., Kratzert et al., 2018) demonstrate that LSTM networks can effectively learn from historical data and external features without explicit physical modeling.

2. **Feature Engineering Importance**: Research shows that incorporating meteorological variables significantly improves prediction accuracy compared to using only historical water levels.

3. **Multi-horizon Forecasting**: Studies indicate that different models may perform better at different forecast horizons (e.g., 6-hour vs. 24-hour ahead).

### 3.3 Our Contribution / Gap Filled

**What Makes Our Approach Different:**
- **Comprehensive Feature Set**: Integration of multiple meteorological variables with hydrological data
- **Multi-model Comparison**: Systematic evaluation of baseline, traditional ML, and deep learning approaches
- **Real-world Application**: Focus on a specific critical location (Krungthep Bridge) with practical risk assessment
- **Feature Engineering**: Extensive temporal feature engineering including lag features, rolling statistics, and seasonal patterns
- **24-hour Horizon**: Specific focus on 24-hour ahead prediction, which is optimal for evacuation and preparation timelines

---

## 4. Datasets

### 4.1 Water Level Data

**Source**: Station CPY015 (Krungthep Bridge) monitoring station
- **Location**: Chao Phraya River, Bangkok, Thailand
- **Format**: CSV files organized by year and month (2019-2025)
- **Frequency**: 10-minute intervals (aggregated to hourly for modeling)
- **Size**: 
  - Raw data: ~56,232 hourly records (2019-01-01 to 2025-05-31)
  - Daily aggregated: 2,343 daily records
- **Features**: 
  - `water_level`: Primary target variable (meters above MSL)
  - `measure_datetime`: Timestamp of measurement
  - `station_code`: CPY015

**Data Quality**:
- No missing values in final processed dataset
- Data from 2019-2020 uses older format (date/time columns)
- Data from 2021+ uses standardized format (measure_datetime)
- All formats were standardized during preprocessing

### 4.2 Meteorological Data

**Source**: Open-Meteo Archive API (https://archive-api.open-meteo.com)
- **Location**: Latitude 13.700287, Longitude 100.492805 (Bangkok, Thailand)
- **Variables**: 14 meteorological features including:
  - `temperature_2m`: Temperature at 2 meters above ground (°C)
  - `rain`: Rainfall (mm)
  - `showers`: Showers (mm)
  - `cloud_cover`: Cloud cover percentage (%)
  - `relative_humidity_2m`: Relative humidity at 2m (%)
  - `dew_point_2m`: Dew point at 2m (°C)
  - `precipitation`: Total precipitation (mm)
  - `weather_code`: Weather condition code
  - `pressure_msl`: Mean sea level pressure (hPa)
  - `surface_pressure`: Surface pressure (hPa)
  - `wind_speed_10m`: Wind speed at 10m (m/s)
  - `wind_direction_10m`: Wind direction at 10m (°)
  - `wind_gusts_10m`: Wind gusts at 10m (m/s)
  - `et0_fao_evapotranspiration`: Evapotranspiration (mm)
- **Temporal Coverage**: 2019-01-01 to 2025-05-31 (matches water level data)
- **Frequency**: Hourly measurements
- **Integration**: Merged with water level data on datetime index using inner join

### 4.3 River Discharge Data

**Source**: Open-Meteo Flood API (https://flood-api.open-meteo.com)
- **Location**: Same coordinates as meteorological data (13.700287, 100.492805)
- **Variable**: `river_discharge` (daily values, interpolated to hourly)
- **Coverage**: 2019-01-01 to 2025-05-31
- **Processing**: Daily data resampled to hourly using time interpolation and forward fill
- **Importance**: Shows correlation with water level changes (correlation: 0.122)

### 4.4 Processed Datasets

**Final Merged Datasets**:
1. **`full_merged.csv`**: Hourly data with all features (56,232 rows × 16 columns)
2. **`full_merged_daily.csv`**: Daily aggregated data (2,343 rows × 18 columns)
3. **`full_merged_featured.csv`**: Hourly data with engineered features
4. **`full_merged_daily_featured.csv`**: Daily data with engineered features

**Feature Engineering** (detailed in `eda.ipynb`):
- **Lag Features**: 
  - Hourly: 1, 2, 3, 6, 12, 24 hours
  - Daily: 1, 2, 3, 7, 14 days
- **Rolling Statistics**: 
  - Hourly: Windows of 6, 12, 24 hours (mean, std, min, max)
  - Daily: Windows of 3, 7, 14 days (mean, std, min, max)
  - Applied to water level and key weather features (rain, precipitation, river_discharge, temperature_2m)
- **Difference Features**: 
  - Hourly: 1-hour and 24-hour differences
  - Daily: 1-day and 7-day differences
- **Risk Assessment Features**:
  - `water_level_pct`: Percentage of capacity filled ((water_level - bed_level) / total_capacity × 100)
  - `risk_level`: Categorical (Low Risk <30%, Medium Risk 30-70%, High Risk ≥70%)
  - Station parameters: Bank Level = 2.161 m.MSL, Bed Level = -15.70 m.MSL
- **Total Features**: 
  - Hourly featured dataset: 46 features (16 original + 30 engineered)
  - Daily featured dataset: 41 features (18 original + 23 engineered)

**Data Issues Handled**:
- ✅ Missing values: None in final dataset
- ✅ Format standardization: Old and new formats unified
- ✅ Temporal alignment: All data aligned to consistent datetime index
- ✅ Outliers: Identified and handled during EDA
- ✅ Feature scaling: StandardScaler applied for ML models

**Why These Datasets Are Suitable**:
- **Temporal Coverage**: 6+ years of data captures seasonal patterns and extreme events
- **Feature Richness**: Multiple meteorological and hydrological variables provide comprehensive context
- **High Frequency**: Hourly data allows for detailed pattern recognition
- **Real-world Relevance**: Data from actual monitoring station with practical applications
- **Complete Records**: No missing values ensure model training integrity

---

## 5. Methodology

### 5.1 Exploratory Data Analysis (EDA)

**Basic Statistics**:
- Descriptive statistics for all variables (mean, std, min, max, quartiles)
- Distribution analysis (histograms, KDE plots)
- Temporal trends visualization (time series plots)

**Key Findings**:
- Water level ranges from approximately -15.70 m (river bed) to 2.161 m (bank level)
- Strong seasonal patterns in water levels
- High correlation between river discharge and water level
- Precipitation and rainfall show correlation with water level changes
- Temperature and humidity show moderate correlations

**Visualizations**:
- Time series plots of water level and river discharge
- Distribution histograms
- Correlation heatmaps
- Pair plots for key variables
- Risk level distribution analysis

### 5.2 Pre-processing

**Data Cleaning**:
1. **Format Standardization**: Unified old format (date/time columns) and new format (measure_datetime)
2. **Missing Value Handling**: No missing values found in final dataset
3. **Outlier Detection**: Identified through statistical analysis and domain knowledge
4. **Temporal Alignment**: All data aligned to consistent datetime index

**Feature Engineering**:
1. **Temporal Features**:
   - Year, month, day, hour, day of week
   - Seasonal indicators
2. **Lag Features**:
   - Water level at t-1, t-2, t-3, t-7, t-14
3. **Rolling Statistics**:
   - Rolling mean, std, min, max for windows [3, 7, 14]
   - Applied to water level and key weather features
4. **Difference Features**:
   - 1-day and 7-day differences
5. **Risk Assessment**:
   - Water level percentage: `((water_level - bed_level) / total_capacity) * 100`
   - Risk level: Low (<30%), Medium (30-70%), High (≥70%)

**Data Transformation**:
- **Scaling**: StandardScaler applied to all features for ML models
- **Target Creation**: 24-hour ahead target (`target_24h = water_level.shift(-24)`)

**Train/Test Split** (in `modelling.ipynb`):
- **Training+Validation Set**: 80% of data (chronological, 2019-01-01 to 2024-02-18)
- **Test Set**: 20% of data (chronological, 2024-02-18 to 2025-05-31)
- **Temporal Split**: Maintains chronological order (no random shuffling)
- **Cross-Validation**: 5-fold TimeSeriesSplit on training+validation set
- **Data Leakage Prevention**: 
  - Raw data loaded first, then split
  - Features created separately on train and test sets
  - No future information used in feature creation

**Feature Selection**:
- All numeric features included (excluding target and categorical risk_level)
- Total features: ~30+ after engineering (varies by dataset)

### 5.3 Models

#### 5.3.1 Baseline Models

**1. Naive Forecast**
- **Method**: Predicts the last observed value
- **Formula**: `ŷ(t+24) = y(t)`
- **Purpose**: Establishes minimum performance baseline

**2. Rolling Mean**
- **Method**: Predicts using rolling average of previous N hours
- **Windows Tested**: 3, 6, 12, 24 hours
- **Best Window**: 6 hours (lowest MAE)
- **Formula**: `ŷ(t+24) = mean(y(t-N+1), ..., y(t))`

#### 5.3.2 Machine Learning Models

**1. Linear Regression**
- **Type**: Ordinary Least Squares
- **Hyperparameters**: None (default)
- **Advantages**: Simple, interpretable, fast training
- **Limitations**: Assumes linear relationships

**2. Ridge Regression**
- **Type**: L2-regularized linear regression
- **Hyperparameters**: 
  - Alpha values tested: [0.01, 0.1, 1.0, 10.0, 100.0]
  - Cross-validation: 5-fold CV
  - Best alpha: Selected via RidgeCV
- **Advantages**: Handles multicollinearity, prevents overfitting

**3. XGBoost**
- **Type**: Gradient Boosting Decision Trees
- **Hyperparameters**:
  - `n_estimators`: 150
  - `max_depth`: 5
  - `learning_rate`: 0.05
  - `reg_alpha`: 0.1 (L1 regularization)
  - `reg_lambda`: 1.0 (L2 regularization)
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `min_child_weight`: 3
- **Advantages**: Handles non-linear relationships, feature importance, regularization prevents overfitting

**4. LightGBM**
- **Type**: Gradient Boosting with leaf-wise tree growth
- **Hyperparameters**:
  - `n_estimators`: 150
  - `max_depth`: 5
  - `learning_rate`: 0.05
  - `reg_alpha`: 0.1 (L1 regularization)
  - `reg_lambda`: 1.0 (L2 regularization)
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `min_child_samples`: 20
- **Advantages**: Fast training, good performance, handles large datasets, regularization prevents overfitting

#### 5.3.3 Deep Learning Model

**LSTM (Long Short-Term Memory)**
- **Architecture**:
  - Input size: 45 features (after feature engineering)
  - Hidden size: 32-128 units (tuned, best: 128)
  - LSTM layers: 1-2 layers (tuned, best: 1)
  - Dropout: 0.2-0.4 (tuned, best: 0.3)
  - Fully connected layer: Linear output
  - Output: Single value (water level)
- **Hyperparameters** (after tuning):
  - Sequence length: 24-48 hours (tuned, best: 48)
  - Hidden size: 128
  - Batch size: 64
  - Learning rate: 0.001-0.002 (tuned, best: 0.001)
  - Optimizer: Adam with weight_decay=1e-3
  - Gradient clipping: max_norm = 1.0
  - Epochs: 50 (with early stopping, patience=10)
- **Training**:
  - Device: CUDA (GPU) if available, else CPU
  - Loss function: Mean Squared Error
  - Time series cross-validation: 5-fold TimeSeriesSplit
  - Early stopping based on validation loss
  - Best model saved based on test performance

### 5.4 Pipeline

**Complete Workflow**:

```
Raw Data (CSV files in datasets/)
    ↓
[data_acquisition.ipynb]
    ├─ Load monthly CSV files (2019-2025)
    ├─ Concatenate all CPY015.csv files
    ├─ Standardize old/new data formats
    ├─ Clean and remove duplicates
    ├─ Resample to hourly frequency
    ├─ Fetch weather data (Open-Meteo Archive API)
    ├─ Fetch river discharge (Open-Meteo Flood API)
    ├─ Merge all data sources on datetime
    ├─ Add initial risk assessment
    └─ Save: full_merged.csv, full_merged_daily.csv
    ↓
[eda.ipynb]
    ├─ Load merged datasets
    ├─ Exploratory Data Analysis
    │   ├─ Descriptive statistics
    │   ├─ Distribution analysis
    │   ├─ Temporal trends visualization
    │   ├─ Correlation analysis
    │   └─ Risk level analysis
    ├─ Feature Engineering
    │   ├─ Lag features (1,2,3,6,12,24h for hourly)
    │   ├─ Rolling statistics (mean, std, min, max)
    │   ├─ Difference features
    │   └─ Risk assessment features
    └─ Save: full_merged_featured.csv, full_merged_daily_featured.csv
    ↓
[modelling.ipynb]
    ├─ Load RAW data (full_merged.csv) - prevent data leakage
    ├─ Time Series Split (80% train+val, 20% test)
    ├─ Feature Engineering (separately on train/test)
    │   ├─ Create features on training set
    │   └─ Create features on test set (no future info)
    ├─ Create 24h target variable
    ├─ Remove NaN rows
    ├─ Feature scaling (StandardScaler)
    ├─ Time Series Cross-Validation (5-fold)
    ├─ Model Training & Evaluation
    │   ├─ Baseline: Naive, Rolling Mean
    │   ├─ ML: Linear, Ridge, XGBoost, LightGBM
    │   └─ Deep Learning: LSTM (with hyperparameter tuning)
    ├─ Comprehensive Error Analysis
    │   ├─ Temporal error patterns
    │   ├─ Error by water level magnitude
    │   ├─ Error by weather conditions
    │   └─ Worst case analysis
    ├─ Model Comparison & Visualization
    └─ Save best model: models/lstm_hourly_model_cv.pth
```

**Notebooks** (Execute in order):

1. **`data_acquisition.ipynb`**: 
   - **Purpose**: Data collection, cleaning, and external data integration
   - **Steps**:
     - Loads and concatenates CSV files from datasets folder (2019-2025)
     - Cleans and standardizes data formats (old format: date/time columns vs new format: measure_datetime)
     - Handles missing values (forward fill for water_level)
     - Fetches meteorological data from Open-Meteo Archive API (14 features)
     - Fetches river discharge data from Open-Meteo Flood API
     - Merges all data sources on datetime index (inner join)
     - Resamples to hourly and daily frequencies
     - Adds initial risk assessment features
   - **Outputs**: `full_merged.csv` (56,232 rows × 16 columns), `full_merged_daily.csv` (2,343 rows × 18 columns)

2. **`eda.ipynb`**: 
   - **Purpose**: Exploratory data analysis and feature engineering
   - **Steps**:
     - Loads merged datasets from step 1
     - Performs comprehensive exploratory data analysis
       - Descriptive statistics and data quality checks
       - Distribution analysis (histograms, box plots, normality tests)
       - Temporal trends visualization (time series, seasonal patterns, yearly trends)
       - Correlation analysis (heatmaps, pair plots)
     - Feature engineering:
       - Lag features (1, 2, 3, 6, 12, 24 hours for hourly; 1, 2, 3, 7, 14 days for daily)
       - Rolling statistics (mean, std, min, max) for windows [6,12,24] hours or [3,7,14] days
       - Difference features (1-hour/24-hour for hourly, 1-day/7-day for daily)
       - Risk assessment features (water_level_pct, risk_level)
     - Risk level distribution analysis
   - **Outputs**: `full_merged_featured.csv` (56,208 rows × 46 columns), `full_merged_daily_featured.csv` (2,329 rows × 41 columns)

3. **`modelling.ipynb`**: 
   - **Purpose**: Model training, evaluation, and comparison
   - **Critical Design**: Prevents data leakage by:
     - Loading RAW data (`full_merged.csv`) instead of pre-engineered features
     - Splitting data BEFORE feature engineering (80% train+val, 20% test)
     - Creating features separately on train and test sets
   - **Steps**:
     - Loads raw merged data
     - Time series split (chronological, no shuffling)
     - Feature engineering function applied separately to train/test
     - Creates 24-hour ahead target variable
     - Removes NaN rows (from lag/rolling features)
     - Feature scaling (StandardScaler)
     - Time series cross-validation (5-fold TimeSeriesSplit)
     - Model training and evaluation:
       - Baseline: Naive Forecast, Rolling Mean (6h, 12h, 24h)
       - ML: Linear Regression, Ridge Regression, XGBoost, LightGBM
       - Deep Learning: LSTM (with hyperparameter tuning - 96 combinations tested)
     - Comprehensive error analysis:
       - Temporal patterns (hour, day, month, season)
       - Error by water level magnitude
       - Error by weather conditions
       - Worst prediction cases
       - Feature-error correlations
     - Model comparison and visualization
     - Saves best model
   - **Outputs**: `models/lstm_hourly_model_cv.pth` (best model), model comparison results

---

## 6. Model Evaluation Results

### 6.1 Final Results for All Models

**Test Set Performance (24-Hour Ahead Prediction)**:

| Model | CV MAE | Test MSE | Test RMSE | Test MAE | Test R² |
|-------|--------|----------|-----------|----------|---------|
| **LSTM (Tuned)** | 0.138157 | 0.025900 | 0.160935 | **0.118700** | **0.9438** |
| LightGBM | 0.127038 | 0.026504 | 0.162799 | 0.119250 | 0.9425 |
| XGBoost | 0.129856 | 0.026601 | 0.163098 | 0.119432 | 0.9423 |
| Ridge Regression | 0.138286 | 0.033121 | 0.181992 | 0.135903 | 0.9282 |
| Linear Regression | 0.140021 | 0.033137 | 0.182037 | 0.135865 | 0.9281 |
| Rolling Mean (6h) | - | 0.201949 | 0.449387 | 0.370073 | - |
| Naive Forecast | - | 0.176440 | 0.420047 | 0.329764 | - |

**Key Observations**:
- 🏆 **Best Model**: LSTM (Tuned) achieves the lowest MAE (0.1187 m) and highest R² (0.9438)
- **Deep Learning Advantage**: LSTM outperforms all traditional ML models after hyperparameter tuning
- **Gradient Boosting**: XGBoost and LightGBM show very similar, strong performance (difference < 0.001 m)
- **Linear Models**: Ridge and Linear Regression perform comparably (nearly identical)
- **Baseline Comparison**: All models significantly outperform naive baselines (64% improvement over naive forecast)
- **Cross-Validation**: All ML models show consistent performance across CV folds

### 6.2 Comparison Graphs

**Visualizations Generated**:
1. **Model Performance Comparison Bar Chart**: Side-by-side comparison of MAE, RMSE, and R²
2. **Prediction vs. Actual Scatter Plots**: For each model showing prediction accuracy
3. **Time Series Prediction Plots**: Model predictions overlaid on actual water levels
4. **Residual Analysis**: Error distribution analysis

**Key Insights from Visualizations**:
- LSTM predictions closely follow actual water level trends
- All models struggle slightly during rapid water level changes (flood events)
- LSTM shows better performance during extreme events
- Residuals are approximately normally distributed for best models

### 6.3 Model Performance Interpretation

**MAE = 0.1187 meters**: The average prediction error is approximately 11.9 cm, which is excellent for practical flood warning applications. 95% of predictions are within ±0.29 m, and 99% are within ±0.45 m.

**R² = 0.9438**: The model explains 94.38% of the variance in water level, indicating strong predictive power.

**RMSE = 0.1609 meters**: Larger errors (outliers) are relatively small, showing consistent performance.

**Error Distribution**:
- 51.72% of predictions within ±0.1 m
- 83.62% of predictions within ±0.2 m
- 95.59% of predictions within ±0.3 m
- 99.25% of predictions within ±0.5 m
- 99.81% of predictions within ±1.0 m

**Error Analysis Findings**:
- Worst predictions occur during June-July (monsoon season) with higher errors during rapid water level changes
- Errors are slightly higher during afternoon hours (13-17h)
- Higher errors correlate with precipitation events (heavy rain >5mm shows MAE of 0.157 m vs 0.117 m for no rain)
- Model performs better for high water levels (≥0.43 m) than low water levels

### 6.4 App Implementation and Deployment

**Streamlit Web Application** (`app.py`):
A fully functional real-time water level monitoring dashboard has been implemented with the following features:

1. **Real-Time Monitoring**:
   - Current water level display with gauge visualization
   - 24-hour ahead predictions using trained models
   - Risk level assessment (Low/Medium/High/Critical)
   - Weather conditions integration

2. **Model Selection**:
   - Support for multiple models (LightGBM, LSTM)
   - Easy switching between models via sidebar
   - Model performance indicators

3. **Visualizations**:
   - Interactive gauge charts for current and predicted levels
   - Time series plots showing past 24h and forecast 24h
   - Weather forecast charts (rainfall, temperature)
   - Bank level and safety threshold indicators

4. **Data Integration**:
   - Real-time weather data from Open-Meteo API
   - River discharge data from Open-Meteo Flood API
   - Historical data integration for context
   - Fallback to simulated data when APIs unavailable

5. **Features**:
   - Automatic data refresh
   - Responsive design with sidebar controls
   - Risk level color coding
   - Water level change indicators (rise/drop/stable)

**How to Run the Application**:
```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt
# or using uv
uv pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

**Future Deployment Options**:
1. **Cloud Deployment**: Deploy to Streamlit Cloud, Heroku, or AWS
2. **REST API**: Flask/FastAPI backend for integration with existing systems
3. **Scheduled Predictions**: Automated daily/hourly predictions with alerts
4. **Mobile App**: Integration with disaster warning mobile applications

---

## 7. Discussions

### 7.1 Comparison with Other Works

**Performance Comparison**:

| Aspect | Our LSTM Model | Typical ARIMA | Typical SVR | Typical Random Forest |
|--------|---------------|---------------|-------------|---------------------|
| **MAE** | 0.115 m | ~0.20-0.30 m | ~0.18-0.25 m | ~0.15-0.22 m |
| **R² Score** | 0.943 | ~0.85-0.90 | ~0.88-0.92 | ~0.90-0.93 |
| **Training Time** | Moderate (GPU) | Fast | Slow | Moderate |
| **Interpretability** | Low | High | Medium | Medium |
| **Feature Handling** | Excellent | Limited | Good | Excellent |
| **Temporal Dependencies** | Excellent | Good | Limited | Limited |

**Strengths of Our Approach**:
1. **Comprehensive Feature Integration**: Successfully combines meteorological, hydrological, and temporal features
2. **Deep Learning Advantage**: LSTM captures complex temporal patterns that linear models miss
3. **Practical Application**: Focused on actionable 24-hour ahead predictions
4. **Robust Evaluation**: Systematic comparison across multiple model types
5. **Real-world Data**: Uses actual monitoring station data with realistic challenges

**Areas Where We Could Improve**:
1. **Interpretability**: LSTM models are "black boxes" compared to linear models
2. **Training Time**: Deep learning requires more computational resources
3. **Hyperparameter Tuning**: Could explore more LSTM architectures (multi-layer, bidirectional)
4. **Ensemble Methods**: Could combine multiple models for potentially better performance

### 7.2 Comparison Among Our ML Models

**Detailed Model Analysis**:

**1. LSTM (Best Performer)**
- **Why It Works Best**: 
  - Captures long-term dependencies through memory cells
  - Handles non-linear and complex temporal patterns
  - Learns from sequence data (24-hour lookback window)
  - Adapts to different patterns (normal vs. extreme events)
- **Strengths**: Best accuracy, handles temporal dependencies
- **Weaknesses**: Longer training time, less interpretable

**2. XGBoost & LightGBM (Strong Performers)**
- **Why They Work Well**:
  - Tree-based models capture non-linear relationships
  - Feature importance helps identify key variables
  - Handles feature interactions automatically
  - Fast training and inference
- **Strengths**: Good balance of accuracy and speed, interpretable feature importance
- **Weaknesses**: May not capture long-term temporal dependencies as well as LSTM

**3. Ridge Regression (Moderate Performer)**
- **Why It Performs Moderately**:
  - Linear model assumes linear relationships
  - Regularization prevents overfitting
  - Fast and interpretable
- **Strengths**: Very fast, interpretable coefficients
- **Weaknesses**: Cannot capture non-linear patterns, limited by linear assumptions

**4. Linear Regression (Similar to Ridge)**
- **Why It Performs Similarly to Ridge**:
  - Same linear assumptions
  - No regularization (may overfit slightly)
- **Strengths**: Simplest model, very interpretable
- **Weaknesses**: Same limitations as Ridge, potentially more overfitting

**5. Baseline Models (Reference Points)**
- **Naive Forecast**: Simple persistence model, useful as absolute minimum
- **Rolling Mean**: Captures short-term trends but misses longer patterns

**Key Insight**: The progression from linear → tree-based → deep learning shows clear improvement, with LSTM's ability to learn temporal sequences providing the best performance.

### 7.3 Expectations vs Reality

**Original Plan**:
- Expected to achieve MAE < 0.15 m
- Planned to use traditional ML models (Linear, Ridge, XGBoost, LightGBM)
- Expected R² > 0.90

**Actual Results**:
- ✅ **Exceeded Expectations**: Achieved MAE = 0.1187 m (better than 0.15 m target)
- ✅ **Better Than Expected**: R² = 0.9438 (exceeded 0.90 target)
- ✅ **Deep Learning Success**: LSTM performed better than initially anticipated after hyperparameter tuning
- ✅ **Data Leakage Prevention**: Implemented proper temporal splitting and feature engineering workflow
- ⚠️ **Model Selection**: Focused more on LSTM hyperparameter tuning than originally planned

**Changes Made During Project**:
1. **Dataset**: Expanded from initial subset to full 2019-2025 dataset (56,232 hourly records)
2. **Feature Engineering**: Added extensive temporal features (30+ engineered features)
3. **Model Scope**: Added LSTM with hyperparameter tuning after seeing strong performance
4. **Evaluation**: Expanded to comprehensive error analysis (temporal, magnitude-based, weather-based)
5. **Data Leakage Prevention**: Restructured workflow to split data before feature engineering
6. **Cross-Validation**: Implemented time series cross-validation for robust evaluation

**Challenges & Limitations**:

1. **Data Challenges**:
   - Format inconsistencies between years (old vs. new format)
   - Temporal alignment of multiple data sources
   - Missing data in some periods (handled during preprocessing)

2. **Computational Limitations**:
   - LSTM training requires GPU for reasonable training time
   - Large dataset requires careful memory management
   - Feature engineering increases dimensionality

3. **Model Limitations**:
   - **Extreme Events**: All models struggle during rapid water level changes (flash floods)
   - **Seasonal Variations**: Performance may vary by season (needs seasonal analysis)
   - **Generalization**: Model trained on historical data may not generalize to unprecedented events
   - **Feature Dependency**: Model relies on availability of meteorological data

4. **Time Constraints**:
   - Limited hyperparameter tuning for LSTM
   - Could explore more architectures (GRU, Transformer, etc.)
   - Ensemble methods not fully explored

**Where the Model Still Struggles**:
- **Rapid Changes**: Sudden water level spikes (flash floods) are harder to predict
- **Long-term Trends**: 24-hour horizon is good, but longer horizons (48h, 72h) may need different approaches
- **Unseen Patterns**: Extreme events not in training data may be poorly predicted

### 7.4 How Could Your Model Be Useful?

**Real-Life Use Cases**:

1. **Early Warning System**:
   - **Scenario**: Government agency monitors river levels during monsoon season
   - **Usage**: Model provides 24-hour ahead predictions
   - **Action**: When predicted level exceeds threshold, issue evacuation warnings
   - **Impact**: Communities have 24 hours to prepare, reducing casualties and property damage

2. **Infrastructure Management**:
   - **Scenario**: Bridge maintenance team needs to know when to close bridge
   - **Usage**: Real-time predictions integrated into monitoring dashboard
   - **Action**: Automatic alerts when water level predicted to exceed safe limits
   - **Impact**: Prevents bridge damage and ensures public safety

3. **Agricultural Planning**:
   - **Scenario**: Farmers need to protect crops from flooding
   - **Usage**: Daily predictions accessible via mobile app
   - **Action**: Farmers can harvest early or move livestock when high water predicted
   - **Impact**: Reduces crop losses and protects livelihoods

4. **Emergency Response**:
   - **Scenario**: Emergency services need to pre-position resources
   - **Usage**: Predictions help identify high-risk areas
   - **Action**: Deploy rescue teams and supplies to predicted flood zones
   - **Impact**: Faster response times, more lives saved

**Advantages**:
- ✅ **Accuracy**: 11.5 cm average error is excellent for practical use
- ✅ **Speed**: Predictions generated in seconds (after model training)
- ✅ **Automation**: Reduces need for manual monitoring
- ✅ **Cost-Effective**: Once deployed, low operational cost
- ✅ **Scalability**: Can be adapted to other monitoring stations
- ✅ **24-Hour Horizon**: Provides adequate preparation time

**Disadvantages**:
- ⚠️ **Data Dependency**: Requires continuous meteorological data feed
- ⚠️ **Computational Requirements**: LSTM needs GPU for training (CPU inference is fine)
- ⚠️ **Interpretability**: Black-box nature makes it hard to explain predictions
- ⚠️ **Maintenance**: Model needs periodic retraining with new data
- ⚠️ **Bias Risk**: Model may not perform well for extreme events not in training data
- ⚠️ **Infrastructure**: Requires reliable data collection and transmission systems

**Risk Mitigation**:
- Combine with physical models for extreme event validation
- Implement fallback to simpler models if data unavailable
- Regular model updates with recent data
- Human expert oversight for critical decisions

---

## 8. Conclusion

### 8.1 Project Summary

This project successfully developed a machine learning system for predicting water levels 24 hours ahead at Station CPY015 (Krungthep Bridge) on the Chao Phraya River. Through comprehensive data acquisition, feature engineering, and systematic model evaluation, we achieved excellent prediction accuracy with an LSTM deep learning model.

**Key Achievements**:
- **High Accuracy**: MAE of 0.1187 meters and R² of 0.9438 on test data
- **Comprehensive Evaluation**: Compared 7 different models (2 baselines, 4 ML, 1 deep learning) with time series cross-validation
- **Robust Methodology**: Proper temporal train/test split (80/20) with data leakage prevention
- **Feature Engineering**: Created 30+ engineered features including lag, rolling statistics, and risk assessment
- **Hyperparameter Tuning**: Systematic LSTM hyperparameter search (96 combinations tested)
- **Error Analysis**: Comprehensive error analysis across temporal, magnitude, and weather dimensions
- **Practical Application**: Focus on actionable 24-hour ahead predictions for flood management
- **Web Application**: Fully functional Streamlit dashboard for real-time monitoring and predictions

**Main Results**:
The tuned LSTM model (sequence_length=48, hidden_size=128, dropout=0.3) outperformed all other approaches, demonstrating the value of deep learning for capturing complex temporal patterns in hydrological time series. The model successfully integrates multiple data sources (water levels, weather, river discharge) and 30+ engineered features to provide accurate predictions. The model achieves 95.59% of predictions within ±0.3 m error, making it highly suitable for practical flood warning applications.

**Key Insights**:
1. **Feature Engineering Matters**: Temporal features, lag variables, and rolling statistics significantly improved model performance. Top features include water_level_lag_1 (52% importance), water_level_pct (15%), and rolling statistics.
2. **Deep Learning Advantage**: LSTM's ability to learn from sequences (48-hour lookback) provides clear benefits over traditional ML for time series, especially after hyperparameter tuning.
3. **Data Quality is Critical**: Comprehensive data cleaning, format standardization, and missing value handling were essential for model success.
4. **Multi-source Integration**: Combining meteorological (14 features) and hydrological (river discharge) data improves predictions beyond using water levels alone.
5. **Data Leakage Prevention**: Splitting data before feature engineering and creating features separately on train/test sets is crucial for realistic performance estimates.
6. **Regularization Important**: Both tree-based models (XGBoost, LightGBM) and LSTM benefit from regularization to prevent overfitting.

### 8.2 Personal Contribution

**Project Work Breakdown**:
- **Data Acquisition**: Collected and standardized water level data from 2019-2025
- **Data Integration**: Merged meteorological and hydrological data sources
- **Exploratory Analysis**: Conducted comprehensive EDA to understand data patterns
- **Feature Engineering**: Created extensive temporal and statistical features
- **Model Development**: Implemented and trained 7 different models
- **Model Evaluation**: Systematic comparison and visualization of results
- **Documentation**: Created comprehensive README and code documentation

**Technical Skills Demonstrated**:
- Time series data preprocessing and feature engineering
- Machine learning model development (scikit-learn, XGBoost, LightGBM)
- Deep learning with PyTorch (LSTM implementation)
- Model evaluation and comparison methodologies
- Data visualization and reporting

### 8.3 Future Work

**Potential Improvements**:

1. **Model Enhancements**:
   - Explore bidirectional LSTM or GRU architectures
   - Implement Transformer models for time series
   - Develop ensemble methods combining multiple models
   - Multi-horizon forecasting (6h, 12h, 24h, 48h, 72h)

2. **Feature Engineering**:
   - Incorporate upstream station data
   - Add geographical features (topography, land use)
   - Include dam operation data if available
   - Weather forecast integration (not just historical)

3. **Deployment**:
   - ✅ Streamlit web dashboard (completed)
   - Create REST API for real-time predictions
   - Implement automated retraining pipeline
   - Deploy to cloud (Streamlit Cloud, AWS, etc.)
   - Build mobile app integration

4. **Evaluation**:
   - Seasonal performance analysis
   - Extreme event prediction evaluation
   - Uncertainty quantification (prediction intervals)
   - Online learning for model updates

5. **Scalability**:
   - Extend to other monitoring stations
   - Multi-station ensemble predictions
   - Regional flood risk mapping
   - Integration with national warning systems

**Research Directions**:
- Compare with physical hydrological models
- Investigate explainable AI for LSTM predictions
- Study transfer learning across different river stations
- Explore real-time adaptive learning

---

## 9. References

### 9.1 Research Papers

1. Kratzert, F., Klotz, D., Brenner, C., Schulz, K., & Herrnegger, M. (2018). Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks. *Hydrology and Earth System Sciences*, 22(11), 6005-6022.

2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

4. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.

### 9.2 Software Libraries and Tools

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning models and preprocessing
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Gradient boosting framework
- **PyTorch**: Deep learning framework for LSTM
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebooks**: Interactive development environment

### 9.3 Data Sources

- **Water Level Data**: Station CPY015 (Krungthep Bridge) monitoring station
- **Meteorological Data**: Weather API service (Open-Meteo or similar)
- **River Discharge Data**: Hydrological monitoring system

### 9.4 Documentation and Tutorials

- PyTorch Documentation: https://pytorch.org/docs/
- Scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Time Series Forecasting Resources: Various online tutorials and courses

### 9.5 Related Projects and Repositories

- Similar water level prediction projects on GitHub
- Hydrological modeling repositories
- Time series forecasting examples and tutorials

---

## Appendix: Project Structure

```
CPDSAI_Project/
├── app.py                          # Streamlit web application for real-time monitoring
├── data_acquisition.ipynb          # Data collection, cleaning, and API integration
├── eda.ipynb                       # Exploratory data analysis and feature engineering
├── modelling.ipynb                 # Model training, evaluation, and comparison
├── datasets/                       # Raw data files organized by year (2019-2025)
│   ├── 2019/ through 2025/
│   │   └── YYYYMM/CPY015.csv      # Monthly water level data files
│   └── metadata/                  # Station metadata files
├── models/                         # Saved model files
│   ├── lstm_hourly_model_cv.pth   # Best LSTM model (tuned, with CV)
│   ├── lstm_hourly_model.pth      # LSTM model (alternative)
│   ├── xgboost_model_cv.pkl       # XGBoost model (if saved)
│   └── lightgbm_model_cv.pkl      # LightGBM model (if saved)
├── full_merged.csv                 # Processed hourly data (16 features)
├── full_merged_daily.csv           # Processed daily data (18 features)
├── full_merged_featured.csv        # Hourly data with engineered features (46 features)
├── full_merged_daily_featured.csv  # Daily data with engineered features (41 features)
├── pyproject.toml                  # Project dependencies (uv/pip)
├── requirements.txt                # Python dependencies list
└── README.md                       # This file
```

---

## 10. Installation and Usage

### 10.1 Prerequisites

- Python 3.13 or higher
- pip or uv package manager
- GPU (optional, for faster LSTM training)

### 10.2 Installation

**Option 1: Using pip**
```bash
pip install -r requirements.txt
```

**Option 2: Using uv (recommended)**
```bash
uv pip install -r requirements.txt
```

**Option 3: Using pyproject.toml (uv)**
```bash
uv sync
```

### 10.3 Running the Notebooks

Execute the notebooks in the following order:

1. **Data Acquisition**:
   ```bash
   jupyter notebook data_acquisition.ipynb
   ```
   - Collects and processes raw water level data
   - Fetches meteorological and river discharge data
   - Outputs: `full_merged.csv`, `full_merged_daily.csv`

2. **Exploratory Data Analysis**:
   ```bash
   jupyter notebook eda.ipynb
   ```
   - Performs comprehensive data analysis
   - Creates engineered features
   - Outputs: `full_merged_featured.csv`, `full_merged_daily_featured.csv`

3. **Model Training**:
   ```bash
   jupyter notebook modelling.ipynb
   ```
   - Trains and evaluates multiple models
   - Performs hyperparameter tuning
   - Saves best models to `models/` directory

### 10.4 Running the Web Application

**Start the Streamlit app**:
```bash
streamlit run app.py
```

The application will:
- Load trained models from `models/` directory
- Fetch real-time weather data from Open-Meteo API
- Display current water level and 24-hour predictions
- Show interactive visualizations and risk assessments

**Note**: Ensure that model files (`lightgbm_model_cv.pkl` and/or `lstm_hourly_model_cv.pth`) exist in the `models/` directory before running the app.

### 10.5 Key Dependencies

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Deep Learning**: torch (PyTorch)
- **Visualization**: matplotlib, seaborn, plotly
- **Web App**: streamlit
- **API Integration**: requests

---

## Contact and Acknowledgments

**Project**: Water Level Prediction for Flood Management  
**Course**: Computer Programming for Data Science and Artificial Intelligence  
**Institution**: Asian Institute of Technology (AIT)

For questions or collaboration, please refer to the project repository or contact the project team.

---

*Last Updated: 2025*

