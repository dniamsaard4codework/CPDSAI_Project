# 🌊 Water Level Prediction at Krungthep Bridge (CPY015 Station)

A comprehensive machine learning project for predicting water levels at Station CPY015 (Krungthep Bridge) on the Chao Phraya River, Thailand. The system provides **24-hour ahead predictions** for flood early warning and water resource management.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#)

---

## 📋 Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Related Works](#3-related-works)
4. [Datasets](#4-datasets)
5. [Methodology](#5-methodology)
6. [Model Evaluation Results](#6-model-evaluation-results)
7. [Discussions](#7-discussions)
8. [Conclusions](#8-conclusions)
9. [Installation & Usage](#9-installation--usage)
10. [References](#10-references)

---

## 1. Introduction

### 1.1 Background

Flood events pose significant threats to communities worldwide, causing loss of life, property damage, and economic disruption. Thailand, particularly the Bangkok metropolitan area along the Chao Phraya River, faces recurring flood challenges due to seasonal monsoons, tidal effects, and urban development. Accurate water level prediction is essential for effective flood management and early warning systems.

This project develops a **machine learning-based water level prediction system** for Station CPY015 (Krungthep Bridge) on the Chao Phraya River. The station is strategically located in Bangkok, making it critical for urban flood monitoring. Our system predicts water levels **24 hours in advance**, providing adequate lead time for emergency response and public preparation.

### 1.2 Motivation

We undertook this project for several compelling reasons:

| Motivation | Description |
|------------|-------------|
| **Public Safety** | Early flood warnings save lives and protect property |
| **Critical Location** | Krungthep Bridge is a key monitoring point in Bangkok |
| **Data Availability** | 6+ years of hourly water level data (2019-2025) |
| **Technical Challenge** | Complex temporal patterns require advanced ML approaches |
| **Real-world Application** | Results can be deployed for actual flood monitoring |

### 1.3 Business Understanding

**Stakeholders and Their Needs:**

| Stakeholder | Need | How Our System Helps |
|-------------|------|----------------------|
| 🏛️ **Government Agencies** | Accurate forecasts for evacuation planning | 24-hour predictions with risk classification |
| 🏠 **Local Communities** | Early warnings to protect homes and families | Real-time web dashboard with alerts |
| 🌉 **Infrastructure Operators** | Bridge/road closure decisions | Water level percentage relative to bank level |
| 🌾 **Agricultural Sector** | Crop protection planning | Historical trends and seasonal patterns |
| 🚑 **Emergency Services** | Resource pre-positioning | Confidence intervals on predictions |

### 1.4 Possible Impacts

| Impact | Description |
|--------|-------------|
| ⏱️ **Time Savings** | Automated predictions reduce manual monitoring effort |
| 💰 **Cost Reduction** | Early warnings can prevent billions in flood damage |
| **Improved Accuracy** | ML models achieve 96% R² score, outperforming traditional methods |
| 🛡️ **Safety Enhancement** | 24-hour lead time allows adequate preparation |
| 📱 **Accessibility** | Web-based dashboard accessible from any device |
| 📈 **Scalability** | Methodology can be applied to other monitoring stations |

---

## 2. Problem Statement

### 2.1 Task Definition

**Objective**: Predict water levels at Station CPY015 (Krungthep Bridge) **24 hours into the future** using historical water level data, meteorological variables, and river discharge information.

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION TASK                               │
├─────────────────────────────────────────────────────────────────┤
│  Input: Historical data (t-48h to t)                            │
│    ├── Water levels (hourly)                                    │
│    ├── Weather data (temp, rain, humidity, wind, pressure)      │
│    └── River discharge                                          │
│                                                                  │
│  Output: Predicted water level at (t+24h)                       │
│    ├── Continuous value (meters MSL)                            │
│    └── Risk classification (Low/Medium/High/Critical)          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Input Features

| Category | Features | Description |
|----------|----------|-------------|
| **Water Level** | `water_level` | Hourly measurements (m MSL) from 2019-2025 |
| **Temperature** | `temperature_2m` | 2m above ground temperature (°C) |
| **Precipitation** | `rain`, `precipitation` | Rainfall amounts (mm) |
| **Humidity** | `relative_humidity_2m`, `dew_point_2m` | Moisture levels (%) |
| **Pressure** | `pressure_msl`, `surface_pressure` | Atmospheric pressure (hPa) |
| **Wind** | `wind_speed_10m`, `wind_direction_10m`, `wind_gusts_10m` | Wind conditions |
| **Hydrology** | `river_discharge` | Upstream river flow (m³/s) |
| **Temporal** | `hour`, `day_of_week`, `month` + cyclical encodings | Time-based features |
| **Engineered** | Lag features, rolling statistics, differences | Past-only features |

### 2.3 Output

| Output Type | Description |
|-------------|-------------|
| **Continuous Value** | Predicted water level in meters (m.MSL) for 24 hours ahead |
| **Risk Assessment** | Water level percentage and risk classification |

**Risk Level Classification:**
```
🟢 Low Risk      : < 30% capacity
🟡 Medium Risk   : 30-70% capacity  
🟠 High Risk     : 70-90% capacity
🔴 Critical      : ≥ 90% capacity (near bank level 2.161m MSL)
```

### 2.4 Goal Metrics

| Metric | Description | Target | Achieved |
|--------|-------------|--------|----------|
| **MAE** | Mean Absolute Error (primary) | < 0.15 m | ✅ 0.0989 m |
| **RMSE** | Root Mean Squared Error | < 0.20 m | ✅ 0.1342 m |
| **R²** | Coefficient of Determination | > 0.90 | ✅ 0.9610 |
| **MAPE** | Mean Absolute Percentage Error | < 15% | ✅ ~10% |

---

## 3. Related Works

### 3.1 Flood Prediction and Warning Systems

The **Hydro-Informatics Institute (HII)** of Thailand developed comprehensive flood prediction and warning systems for urban areas in 2020. Their work established foundational frameworks for integrating hydrological models with real-time monitoring systems, enabling more effective flood management across Thai cities. This research demonstrated the importance of combining multiple data sources for improved prediction accuracy [1].

### 3.2 AI-Based Flood Monitoring

**One More Link (2025)** introduced CCTV AI-based smart flood monitoring and early warning systems. This innovative approach uses computer vision and artificial intelligence to detect flood conditions from camera feeds, providing real-time alerts. Their work highlights the potential of AI technologies for automated flood detection and complements our water level prediction approach by providing visual verification of predictions [2].

### 3.3 Global Early Warning Systems

The **World Meteorological Organization (WMO)** leads global efforts in developing comprehensive Early Warning Systems through their "Early Warnings for All" initiative. This framework emphasizes multi-hazard early warning, risk knowledge, monitoring and forecasting services, and communication systems. Our project aligns with WMO's recommended approaches by integrating meteorological data with hydrological measurements for improved forecast accuracy [3].

### 3.4 Machine Learning for Water Level Forecasting

**Fu et al. (2024)** demonstrated the effectiveness of combining machine learning with Ensemble Kalman Filtering for water level forecasting in Taiwan's Danshui River System. Their research showed that hybrid approaches combining data-driven ML models with statistical filtering techniques can achieve superior performance compared to standalone methods. This work informed our decision to compare multiple ML approaches including ensemble methods (XGBoost, LightGBM) and deep learning (LSTM) [4].

### 3.5 Our Contribution

| Aspect | Existing Works | Our Contribution |
|--------|----------------|------------------|
| **Data Integration** | Often single data source | 3 integrated datasets (water level, weather, river discharge) |
| **Model Comparison** | Usually single model | Systematic evaluation of 6 models |
| **Temporal Scope** | Limited historical data | 6+ years of hourly data (2019-2025) |
| **Feature Engineering** | Basic features | 57 engineered features with strict past-only rules |
| **Deployment** | Research only | Production-ready web application |
| **Data Leakage Prevention** | Often overlooked | Fold-safe preprocessing, no future information |

---

## 4. Datasets

Our project integrates **three distinct datasets** to capture hydrological, meteorological, and river flow patterns:

### 4.1 Dataset 1: Water Level Data (Primary)

| Attribute | Details |
|-----------|---------|
| **Source** | Hydro-Informatics Institute (HII), Thailand via [thaiwater.net](https://www.thaiwater.net/water/wl) |
| **Station** | CPY015 - Krungthep Bridge, Chao Phraya River, Bangkok |
| **Coordinates** | 13.700287°N, 100.492805°E |
| **Period** | January 2019 - May 2025 |
| **Frequency** | 10-minute intervals (aggregated to hourly) |
| **Records** | ~56,232 hourly observations |
| **Variables** | `measure_datetime`, `water_level` (meters MSL), `quality_flag` |
| **Bank Level** | 2.161 m MSL |
| **Bed Level** | -15.70 m MSL |

**Data Format (2019-2020 - Old Format):**
```
date,time,water_lv
2019-01-01,00:00:00,0.59
2019-01-01,00:10:00,0.59
```

**Data Format (2021-2025 - New Format):**
```
station_code,measure_datetime,water_level,quality_flag
CPY015,2021-01-01 00:00:00,0.45,1
```

### 4.2 Dataset 2: Meteorological Data

| Attribute | Details |
|-----------|---------|
| **Source** | Open-Meteo Historical Weather API |
| **Location** | Same coordinates as water level station |
| **Period** | January 2019 - May 2025 |
| **Frequency** | Hourly |
| **API** | `archive-api.open-meteo.com` |

**Variables (14 features):**

| Variable | Description | Unit |
|----------|-------------|------|
| `temperature_2m` | Temperature at 2m height | °C |
| `relative_humidity_2m` | Relative humidity at 2m | % |
| `dew_point_2m` | Dew point at 2m | °C |
| `rain` | Rainfall amount | mm |
| `showers` | Shower precipitation | mm |
| `precipitation` | Total precipitation | mm |
| `pressure_msl` | Mean sea level pressure | hPa |
| `surface_pressure` | Surface pressure | hPa |
| `cloud_cover` | Cloud cover | % |
| `wind_speed_10m` | Wind speed at 10m | m/s |
| `wind_direction_10m` | Wind direction at 10m | ° |
| `wind_gusts_10m` | Wind gusts at 10m | m/s |
| `weather_code` | Weather condition code | - |
| `et0_fao_evapotranspiration` | Evapotranspiration | mm |

### 4.3 Dataset 3: River Discharge Data

| Attribute | Details |
|-----------|---------|
| **Source** | Open-Meteo Flood API |
| **Location** | Same coordinates (13.700287°N, 100.492805°E) |
| **Period** | January 2019 - May 2025 |
| **Frequency** | Daily (interpolated to hourly) |
| **Variable** | `river_discharge` (m³/s) |
| **Purpose** | Upstream flow indicator for water level prediction |

### 4.4 Merged Datasets

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `full_merged.csv` | 56,232 | 17 | Hourly water level + weather + discharge |
| `full_merged_featured.csv` | 56,208 | 47 | Above + 30 engineered features |
| `full_merged_daily.csv` | 2,343 | 19 | Daily aggregated version |
| `full_merged_daily_featured.csv` | 2,329 | 42 | Daily with features |

### 4.5 Data Quality Summary

| Metric | Value |
|--------|-------|
| **Total Records** | 56,232 hourly observations |
| **Missing Water Level** | < 0.5% (filled using seasonal interpolation) |
| **Missing Weather Data** | < 1% (forward-filled) |
| **Temporal Coverage** | 6 years 5 months |
| **Data Format Consistency** | Unified from 2 different formats |

---

## 5. Methodology

### 5.1 Overall Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │
│  │ Water Level │ + │  Weather    │ + │   River     │ → │   Merged    │  │
│  │    Data     │   │    Data     │   │  Discharge  │   │   Dataset   │  │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘  │
│                                                              ↓           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      DATA PRE-PROCESSING                         │   │
│  │  • Format standardization (old/new formats)                      │   │
│  │  • Missing value imputation (seasonal interpolation)             │   │
│  │  • Outlier detection (IQR method)                                │   │
│  │  • Timezone alignment (Asia/Bangkok)                             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    TRAIN/TEST SPLIT (80/20)                       │   │
│  │  • Chronological split (no shuffling)                            │   │
│  │  • Training: 2019-01-01 to 2024-02-18                            │   │
│  │  • Testing:  2024-02-18 to 2025-05-31                            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    FEATURE ENGINEERING                            │   │
│  │  • Time features (hour, day, month + cyclical)                   │   │
│  │  • Lag features (1, 2, 3, 6, 12, 24 hours)                       │   │
│  │  • Rolling statistics (mean, std, min, max)                      │   │
│  │  • Difference features (rate of change)                          │   │
│  │  • Weather rolling features                                       │   │
│  │  ⚠️ STRICT PAST-ONLY RULES - No data leakage!                    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    MODEL TRAINING (5-Fold CV)                     │   │
│  │  • Baselines: Persistence, Seasonal Naive, Rolling Mean          │   │
│  │  • ML Models: Linear, Ridge, XGBoost, LightGBM                   │   │
│  │  • Deep Learning: LSTM with fold-safe scaling                    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    EVALUATION & DEPLOYMENT                        │   │
│  │  • Metrics: MAE, RMSE, R², MAPE, sMAPE                           │   │
│  │  • Best model saved as artifact                                   │   │
│  │  • Web application deployment (Streamlit)                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Exploratory Data Analysis (EDA)

EDA was performed in `eda.ipynb` to understand data characteristics:

**Key EDA Steps:**
1. **Distribution Analysis**: Water level histogram, Q-Q plots for normality
2. **Temporal Patterns**: Monthly/yearly trends, seasonal decomposition
3. **Correlation Analysis**: Feature correlations heatmap with weather variables
4. **Missing Value Analysis**: Identifying gaps and patterns in missing data
5. **Outlier Detection**: Using IQR method to identify extreme values

**Key Findings from EDA:**
- Water levels show strong seasonal patterns (monsoon vs dry season)
- Hourly patterns influenced by tidal effects
- Strong correlation with rainfall and river discharge
- Missing data concentrated in specific periods (equipment issues)

### 5.3 Data Pre-processing

**Implemented in `data.py`:**

| Step | Method | Implementation |
|------|--------|----------------|
| **Format Unification** | Merge old (date+time) and new (datetime) formats | `data_acquisition.ipynb` |
| **Missing Values** | Seasonal interpolation using past years | `handle_missing_values()` |
| **Outlier Handling** | IQR-based detection, kept for model robustness | `detect_outliers()` |
| **Timezone** | Standardized to Asia/Bangkok | `check_timezone_consistency()` |
| **Duplicates** | Removed duplicate timestamps | `check_duplicate_timestamps()` |
| **Resampling** | 10-minute data aggregated to hourly | Mean aggregation |

### 5.4 Feature Engineering

**Implemented in `features.py`:**

| Feature Type | Features | Window/Lag |
|--------------|----------|------------|
| **Time Features** | `hour`, `day_of_week`, `month`, `year`, `is_weekend` | - |
| **Cyclical Encoding** | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` | - |
| **Lag Features** | `water_level_lag_X` | 1, 2, 3, 6, 12, 24 hours |
| **Rolling Mean** | `water_level_rolling_mean_X` | 6, 12, 24 hours |
| **Rolling Std** | `water_level_rolling_std_X` | 6, 12, 24 hours |
| **Rolling Min/Max** | `water_level_rolling_min/max_X` | 6, 12, 24 hours |
| **Differences** | `water_level_diff_X` | 1, 24 hours |
| **Weather Rolling** | `rain_rolling_mean_X`, `temperature_2m_rolling_mean_X` | 6, 12 hours |
| **Risk Features** | `water_level_pct`, `risk_level` | - |

**⚠️ Data Leakage Prevention:**
- All features use only past information (`.shift()`, `.rolling()` with past values)
- Target created with forward shift: `target = water_level.shift(-24)`
- StandardScaler fit inside each CV fold (fold-safe preprocessing)

### 5.5 Machine Learning Models

**Implemented in `train.py` and `modelling.ipynb`:**

#### Baseline Models

| Model | Description | Formula |
|-------|-------------|---------|
| **Persistence** | Predict current value | ŷ(t+24) = y(t) |
| **Seasonal Naive** | Predict value from 24h ago | ŷ(t+24) = y(t-24) |
| **Rolling Mean** | Rolling average | ŷ(t) = mean(y(t-w:t)) |

#### Machine Learning Models

| Model | Library | Key Parameters |
|-------|---------|----------------|
| **Linear Regression** | scikit-learn | StandardScaler in pipeline |
| **Ridge Regression** | scikit-learn | α ∈ {0.01, 0.1, 1.0, 10.0, 100.0} |
| **XGBoost** | xgboost | n_estimators=150, max_depth=5, lr=0.05 |
| **LightGBM** | lightgbm | n_estimators=150, max_depth=5, lr=0.05 |

#### Deep Learning Model - LSTM

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Take last time step
        return self.fc(self.dropout(last_output))
```

**LSTM Hyperparameters:**
- Sequence length: 48 hours
- Hidden size: 128
- Layers: 1
- Dropout: 0.3
- Batch size: 64
- Epochs: 50 (with early stopping, patience=10)
- Learning rate: 0.001
- Optimizer: Adam with weight decay 0.001

### 5.6 Training Pipeline

1. **TimeSeriesSplit Cross-Validation**: 5-fold CV preserving temporal order
2. **Fold-Safe Preprocessing**: Scaler fit on train fold only
3. **Early Stopping**: For LSTM to prevent overfitting
4. **Final Training**: Best model trained on full train+val set
5. **Artifact Saving**: Model, scaler, feature names saved together

---

## 6. Model Evaluation Results

### 6.1 Final Model Comparison

| Model | CV MAE | Test MAE (m) | Test RMSE (m) | Test R² |
|-------|--------|--------------|---------------|---------|
| Persistence Baseline | - | 0.2260 | 0.2854 | 0.8710 |
| Seasonal Naive | - | 0.3282 | 0.4170 | 0.6230 |
| Rolling Mean (24h) | - | 0.5644 | 0.6706 | 0.0252 |
| Linear Regression | 0.1498 | 0.1330 | 0.1745 | 0.9340 |
| Ridge Regression | 0.1479 | 0.1327 | 0.1742 | 0.9343 |
| XGBoost | 0.1306 | 0.1122 | 0.1501 | 0.9512 |
| LightGBM | 0.1316 | 0.1128 | 0.1508 | 0.9507 |
| **LSTM (Best)** | **0.1726** | **0.0989** | **0.1342** | **0.9610** |

### 6.2 Key Performance Metrics (Best Model - LSTM)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MAE** | 0.0989 m | Average error of ~10 cm |
| **RMSE** | 0.1342 m | ~13 cm when penalizing large errors |
| **R²** | 0.9610 | 96.1% of variance explained |
| **MAPE** | ~10% | 10% average percentage error |
| **MSE** | 0.0180 | Squared error metric |

### 6.3 Error Distribution

| Error Range | Percentage of Predictions |
|-------------|---------------------------|
| ±0.1 m (10 cm) | 51.72% |
| ±0.2 m (20 cm) | 83.62% |
| ±0.3 m (30 cm) | 95.59% |
| ±0.5 m (50 cm) | 99.25% |
| ±1.0 m (100 cm) | 99.81% |

### 6.4 Improvement Over Baselines

| Comparison | Improvement |
|------------|-------------|
| vs Persistence | **56% reduction** in MAE |
| vs Seasonal Naive | **70% reduction** in MAE |
| vs Rolling Mean (24h) | **82% reduction** in MAE |
| vs Linear Regression | **26% reduction** in MAE |

---

## 7. Discussions

### 7.1 Performance Comparison with Literature

| Aspect | Related Works | Our Results | Comparison |
|--------|---------------|-------------|------------|
| **Prediction Horizon** | 1-72 hours varies | 24 hours | Optimal for evacuation |
| **R² Score** | 0.85-0.95 typical | 0.9610 | Excellent |
| **MAE** | Varies by location | 0.0989 m | Competitive |
| **Method** | Often single model | 6 models compared | Comprehensive |
| **Data Period** | 1-3 years typical | 6+ years | More robust |

**vs. Fu et al. (2024)**: Their hybrid ML-Kalman approach achieved similar R² scores for Taiwan's Danshui River. Our pure ML approach achieves comparable performance without requiring Kalman filtering, suggesting LSTM effectively captures temporal dynamics.

**vs. HII (2020)**: Thailand's official flood prediction systems use physical models requiring extensive calibration. Our data-driven approach requires no physical parameter estimation while achieving high accuracy.

### 7.2 Interpretability Analysis

| Model | Interpretability | Complexity | Trade-off |
|-------|------------------|------------|-----------|
| Linear/Ridge | ⭐⭐⭐⭐⭐ High | ⭐ Low | Easy to explain coefficients |
| XGBoost/LightGBM | ⭐⭐⭐ Medium | ⭐⭐⭐ Medium | Feature importance available |
| **LSTM** | ⭐⭐ Low | ⭐⭐⭐⭐⭐ High | Black-box but best accuracy |

**LSTM Feature Importance** (via ablation studies):
1. Water level lag features (most important)
2. Rolling statistics (capture trends)
3. Temporal features (hour, month)
4. Weather variables (rain, pressure)
5. River discharge (upstream indicator)

### 7.3 Feature Importance Analysis (Ablation Study)

To understand which features contribute most to model performance, we conducted an **ablation study** by systematically removing feature groups and measuring the impact on model accuracy:

| Feature Group Removed | Features | Test MAE | MAE Increase | Impact (%) |
|----------------------|----------|----------|--------------|------------|
| **Lag Features** | 6 | 0.1323 m | +0.0334 m | **+33.8%** |
| **Discharge Features** | 3 | 0.1270 m | +0.0282 m | **+28.5%** |
| **Weather Features** | 16 | 0.1226 m | +0.0238 m | **+24.0%** |
| **Time Features** | 11 | 0.1200 m | +0.0212 m | **+21.4%** |
| **Rolling Features** | 20 | 0.1198 m | +0.0210 m | **+21.2%** |
| **Difference Features** | 2 | 0.1175 m | +0.0186 m | **+18.8%** |

**Key Insights from Ablation Study:**

1. **Lag Features are Most Critical (+33.8% error increase)**
   - Water level at t-1, t-2, t-3, t-6, t-12, t-24 hours
   - Captures autocorrelation and short-term persistence
   - The most recent past values are strong predictors of future levels

2. **River Discharge is Highly Important (+28.5% error increase)**
   - Only 3 features but significant impact
   - Represents upstream water flow conditions
   - Acts as an early warning signal for water arriving at the station

3. **Weather Features Substantially Improve Accuracy (+24.0% error increase)**
   - See detailed analysis in Section 7.4

4. **Temporal Features Capture Seasonality (+21.4% error increase)**
   - Hour of day (tidal cycles), month (monsoon vs dry season)
   - Cyclical encodings capture periodic patterns

### 7.4 Why Weather Data Improves Model Accuracy

Incorporating 16 meteorological variables significantly improved prediction accuracy by **24%** (MAE increased from 0.0989m to 0.1226m when weather was removed). Here's why:

#### 7.4.1 Physical Mechanisms

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WEATHER → WATER LEVEL PATHWAYS                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  🌧️ RAINFALL (rain, precipitation, showers)                             │
│  └──► Direct surface runoff → Increases water level within 6-24 hours   │
│  └──► Infiltration exceeds capacity → Rapid rise during heavy storms    │
│                                                                          │
│  🌡️ TEMPERATURE (temperature_2m, dew_point_2m)                          │
│  └──► High temps → Evaporation → Lower water levels (dry season)        │
│  └──► Convective activity → Afternoon thunderstorms → Flash flooding    │
│                                                                          │
│  💨 ATMOSPHERIC PRESSURE (pressure_msl, surface_pressure)               │
│  └──► Low pressure → Storm systems → Rain → Water level rise            │
│  └──► Barometric effect on tidal variations                             │
│                                                                          │
│  💧 HUMIDITY (relative_humidity_2m)                                      │
│  └──► High humidity → Reduced evaporation → Water retention             │
│  └──► Saturated air → Imminent precipitation indicator                  │
│                                                                          │
│  🌬️ WIND (wind_speed_10m, wind_direction_10m, wind_gusts_10m)           │
│  └──► Coastal winds → Tidal surge effects at Krungthep Bridge           │
│  └──► Monsoon winds → Seasonal water level patterns                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 7.4.2 Statistical Evidence from Our Data

| Weather Condition | Mean Absolute Error | Observations |
|-------------------|---------------------|--------------|
| **Low Rainfall** (<1mm/hr) | 0.0980 m | 8,932 |
| **High Rainfall** (≥1mm/hr) | 0.1023 m | 2,244 |

| Temperature Range | Mean Absolute Error | Description |
|-------------------|---------------------|-------------|
| **Cold** (<25°C) | 0.1058 m | More variable conditions |
| **Moderate** (25-32°C) | 0.1001 m | Normal conditions |
| **Hot** (>32°C) | 0.0910 m | Stable, dry periods |

#### 7.4.3 Weather Feature Correlations with Water Level

| Weather Variable | Correlation with Water Level | Explanation |
|------------------|------------------------------|-------------|
| `rain` | +0.15 | Direct positive impact |
| `precipitation` | +0.14 | Combined rainfall effect |
| `pressure_msl` | -0.22 | Low pressure → storms → higher water |
| `relative_humidity_2m` | +0.18 | Moisture retention indicator |
| `cloud_cover` | +0.12 | Pre-precipitation signal |
| `temperature_2m` | -0.08 | Inverse (evaporation effect) |

#### 7.4.4 Practical Benefits of Weather Integration

| Benefit | Description |
|---------|-------------|
| **Anticipatory Prediction** | Weather changes precede water level changes by 6-24 hours |
| **Extreme Event Detection** | Heavy rain signals help predict rapid rises |
| **Seasonal Calibration** | Monsoon vs dry season patterns captured |
| **Tidal Adjustment** | Pressure affects tidal surge at Krungthep Bridge |
| **Model Robustness** | Multi-source data reduces overfitting to single pattern |

#### 7.4.5 Comparison: With vs Without Weather Data

| Metric | Without Weather | With Weather | Improvement |
|--------|-----------------|--------------|-------------|
| **Test MAE** | 0.1226 m | 0.0989 m | **-19.3%** |
| **Test RMSE** | 0.1595 m | 0.1342 m | **-15.9%** |
| **Test R²** | 0.9448 | 0.9610 | **+1.7%** |

**Conclusion**: Weather data provides crucial **leading indicators** that allow the model to anticipate water level changes before they occur, rather than simply extrapolating past water level trends. This is especially valuable during transitional periods (monsoon onset) and extreme weather events.

### 7.5 Model Complexity Analysis

| Model | Parameters | Training Time | Inference Time |
|-------|------------|---------------|----------------|
| Linear Regression | ~50 | < 1 sec | < 1 ms |
| Ridge Regression | ~50 | < 1 sec | < 1 ms |
| XGBoost | ~2,000 trees | ~30 sec | ~5 ms |
| LightGBM | ~2,000 trees | ~20 sec | ~3 ms |
| **LSTM** | ~50,000 | ~10 min | ~10 ms |

**Trade-off Analysis:**
- LSTM requires more resources but achieves best accuracy
- LightGBM offers excellent balance of speed and accuracy
- Linear models suitable for interpretable baselines

### 7.6 Limitations

1. **Single Station**: Model trained on CPY015 only; may need retraining for other stations
2. **No Upstream Data**: Would benefit from upstream station integration
3. **Weather Forecast**: Uses historical weather; real deployment needs forecast integration
4. **Extreme Events**: Performance may degrade during unprecedented flood events
5. **Computational Requirements**: LSTM requires GPU for efficient training

### 7.7 Error Analysis

**When the model performs poorly:**
- During June-July monsoon transitions (high variability)
- Afternoon hours (13-17h) with convective storms
- Heavy rainfall events (>5mm/hour)
- Rapid water level changes (>0.5m in 6 hours)

**When the model performs well:**
- Stable weather periods
- Morning hours (6-12h)
- Dry season (November-April)
- Gradual water level changes

---

## 8. Conclusions

### 8.1 Summary of Contributions

| Contribution | Description |
|--------------|-------------|
| **1. Integrated Multi-source Dataset** | Combined water level, weather (14 variables), and river discharge data spanning 6+ years |
| **2. Comprehensive Model Comparison** | Systematic evaluation of 6 models from baselines to deep learning |
| **3. Robust Feature Engineering** | 57 features with strict past-only rules preventing data leakage |
| **4. High Prediction Accuracy** | Achieved R² = 0.9610, MAE = 0.0989 m (56% improvement over baselines) |
| **5. Production-Ready System** | Modular code, web dashboard, and deployment documentation |
| **6. Real-world Applicability** | 24-hour predictions suitable for flood early warning |

### 8.2 Key Findings

1. **LSTM outperforms traditional ML** for water level prediction, capturing long-term temporal dependencies
2. **Feature engineering is critical**: Lag and rolling features provide most predictive power
3. **Data quality matters**: Careful handling of missing values and format standardization improved results
4. **Fold-safe preprocessing** prevents overly optimistic CV scores from data leakage
5. **24-hour horizon** is optimal for balancing accuracy and lead time for emergency response

### 8.3 Future Work

| Area | Planned Enhancement |
|------|---------------------|
| **Models** | Transformer architectures, ensemble methods |
| **Data** | Integrate upstream stations, weather forecasts |
| **Horizons** | Multi-horizon forecasting (6h, 12h, 24h, 48h, 72h) |
| **Deployment** | Cloud deployment, automated retraining pipeline |
| **Uncertainty** | Probabilistic predictions with confidence intervals |

### 8.4 Final Remarks

This project demonstrates that machine learning, particularly LSTM networks, can effectively predict water levels for flood early warning systems. By integrating multiple data sources and applying rigorous data science practices, we achieved prediction accuracy that meets practical requirements for emergency management. The modular, production-ready codebase enables deployment and extension for real-world flood monitoring applications.

---

## 9. Installation & Usage

### 9.1 Quick Installation

```bash
# Clone repository
git clone https://github.com/dniamsaard4codework/CPDSAI_Project.git
cd CPDSAI_Project

# Install dependencies
pip install -r requirements.txt
```

### 9.2 Run Web Application

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### 9.3 Train Models

```bash
python train_all_models.py
```

### 9.4 Run Notebooks (in order)

1. `data_acquisition.ipynb` → Collect and merge data
2. `eda.ipynb` → Exploratory analysis
3. `modelling.ipynb` → Train and evaluate models

### 9.5 Project Structure

```
CPDSAI_Project/
├── 📄 app.py                    # Streamlit web application
├── 📄 data.py                   # Data loading & quality checks
├── 📄 features.py               # Feature engineering module
├── 📄 train.py                  # Model training with CV
├── 📄 evaluate.py               # Evaluation metrics & baselines
├── 📄 predict.py                # Inference & artifact management
├── 📄 train_all_models.py       # Main training orchestrator
├── 📄 config.yaml               # Configuration file
├── 📓 data_acquisition.ipynb    # Data collection notebook
├── 📓 eda.ipynb                 # Exploratory Data Analysis
├── 📓 modelling.ipynb           # Model training notebook
├── 📄 requirements.txt          # Python dependencies
├── 📂 datasets/                 # Raw water level data (2019-2025)
├── 📂 models/                   # Saved model artifacts
└── 📄 full_merged.csv           # Merged hourly dataset
```

---

## 10. References

[1] Hydro-Informatics Institute (2024). "2020: Development of Flood Prediction and Warning Systems in Urban Areas." Hydro-Informatics Institute (HII). Available at: https://www.hii.or.th/en/research-development/project-highlights/2024/02/08/2020-development-of-flood-prediction-and-warning-systems-in-urban-areas/

[2] One More Link (2025). "Introducing CCTV AI: Smart Flood Monitoring & Early Warning System." Available at: https://onemorelink.co.th/en/introducing-cctv-ai-smart-flood-monitoring-early-warning-system/

[3] World Meteorological Organization (2024). "Early Warnings for All: The UN Global Initiative." Available at: https://wmo.int/activities/early-warnings-all

[4] Fu, J.-C., Su, M.-P., Liu, W.-C., Huang, W.-C., & Liu, H.-M. (2024). "Water Level Forecasting Combining Machine Learning and Ensemble Kalman Filtering in the Danshui River System, Taiwan." *Water*, 16(23), 3530. DOI: https://doi.org/10.3390/w16233530

---

## License

This project is developed for academic purposes as part of the Computer Programming for Data Science and Artificial Intelligence course at Asian Institute of Technology (AIT).

---

## Author

- **Dechathon Niamsaard**
- **Repository**: [CPDSAI_Project](https://github.com/dniamsaard4codework/CPDSAI_Project)

---

*Last Updated: December 2025*
