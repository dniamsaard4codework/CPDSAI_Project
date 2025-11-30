"""
Water Level Prediction Web Application
Station CPY015 - Krungthep Bridge, Chao Phraya River

Real-time water level monitoring and 24-hour ahead predictions
using trained machine learning models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# Optional PyTorch import for LSTM
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Water Level Monitoring - CPY015",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Water Level Monitoring System - Station CPY015",
    },
)

# Custom CSS for Nothing.tech Inspired Design
st.markdown(
    """
<style>
    /* Nothing.tech Design System - Pure Monochrome + Red Accent */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

    :root {
        --nothing-black: #000000;
        --nothing-dark: #0A0A0A;
        --nothing-gray-dark: #1A1A1A;
        --nothing-gray: #333333;
        --nothing-gray-light: #666666;
        --nothing-gray-lighter: #888888;
        --nothing-white: #FFFFFF;
        --nothing-red: #D71921;
        --nothing-red-dim: rgba(215, 25, 33, 0.15);
    }

    /* Global Reset */
    html, body, [class*="css"], .stApp {
        font-family: 'Space Mono', monospace !important;
        background-color: var(--nothing-black) !important;
        color: var(--nothing-white);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Container */
    .main .block-container {
        background-color: var(--nothing-black);
        padding: 1rem 2rem 2rem 2rem;
        max-width: 1400px;
    }

    /* Dot Grid Background */
    .stApp > div:first-child {
        background-image: radial-gradient(circle, var(--nothing-gray) 1px, transparent 1px);
        background-size: 24px 24px;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6, p, span, div {
        font-family: 'Space Mono', monospace !important;
    }

    h1, h2, h3 {
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
    }

    /* Nothing Style Cards */
    .nothing-card {
        background: var(--nothing-dark);
        border: 1px solid var(--nothing-gray);
        padding: 1.5rem;
        position: relative;
        transition: all 0.15s ease;
    }

    .nothing-card:hover {
        border-color: var(--nothing-white);
        box-shadow: 4px 4px 0 var(--nothing-red);
        transform: translate(-2px, -2px);
    }

    .nothing-card::after {
        content: '';
        position: absolute;
        top: 8px;
        right: 8px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--nothing-red);
    }

    /* Header Container */
    .nothing-header {
        border: 1px solid var(--nothing-gray);
        padding: 2.5rem;
        margin-bottom: 2rem;
        background: var(--nothing-black);
        position: relative;
    }

    .nothing-header::before {
        content: '( )';
        position: absolute;
        top: 1rem;
        right: 1.5rem;
        font-size: 1.5rem;
        color: var(--nothing-red);
        font-weight: 700;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--nothing-black) !important;
        border-right: 1px solid var(--nothing-gray);
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: var(--nothing-black) !important;
    }

    /* Input Fields */
    .stNumberInput > div > div > input {
        background-color: var(--nothing-dark) !important;
        border: 1px solid var(--nothing-gray) !important;
        color: var(--nothing-white) !important;
        font-family: 'Space Mono', monospace !important;
    }

    .stNumberInput > div > div > input:focus {
        border-color: var(--nothing-red) !important;
        box-shadow: 0 0 0 1px var(--nothing-red) !important;
    }

    /* Buttons */
    .stButton > button {
        background: transparent !important;
        color: var(--nothing-white) !important;
        border: 1px solid var(--nothing-white) !important;
        border-radius: 0 !important;
        padding: 0.75rem 2rem !important;
        font-family: 'Space Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        font-weight: 700 !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        background: var(--nothing-red) !important;
        border-color: var(--nothing-red) !important;
        color: var(--nothing-white) !important;
        box-shadow: 4px 4px 0 var(--nothing-white) !important;
        transform: translate(-2px, -2px) !important;
    }

    /* Metric Display */
    .nothing-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px dashed var(--nothing-gray);
    }

    .nothing-metric-label {
        color: var(--nothing-gray-lighter);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .nothing-metric-value {
        color: var(--nothing-white);
        font-size: 1.1rem;
        font-weight: 700;
    }

    /* Section Headers */
    .nothing-section {
        background: var(--nothing-black);
        border: 1px solid var(--nothing-gray);
        padding: 1rem 1.5rem;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .nothing-section::before {
        content: '//';
        color: var(--nothing-red);
        font-weight: 700;
    }

    .nothing-section::after {
        content: '//';
        color: var(--nothing-red);
        font-weight: 700;
    }

    /* Status Indicators */
    .nothing-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        border: 1px solid var(--nothing-gray);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .nothing-status::before {
        content: '';
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--nothing-red);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Risk Level Styles */
    .risk-critical { border-color: var(--nothing-red) !important; color: var(--nothing-red) !important; }
    .risk-high { border-color: #FF6B00 !important; color: #FF6B00 !important; }
    .risk-medium { border-color: #FFB800 !important; color: #FFB800 !important; }
    .risk-low { border-color: #00FF88 !important; color: #00FF88 !important; }

    /* Dividers */
    hr {
        border: none;
        border-top: 1px dashed var(--nothing-gray);
        margin: 1.5rem 0;
    }

    /* Sidebar Styling */
    .nothing-sidebar-header {
        background: var(--nothing-black);
        border: 1px solid var(--nothing-gray);
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .nothing-sidebar-header span {
        color: var(--nothing-white);
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 3px;
    }

    .nothing-sidebar-section {
        border-top: 1px solid var(--nothing-gray);
        border-bottom: 1px solid var(--nothing-gray);
        padding: 0.5rem 0;
        margin: 1rem 0 0.5rem 0;
        text-align: center;
    }

    .nothing-sidebar-section span {
        color: var(--nothing-gray-light);
        font-size: 0.65rem;
        letter-spacing: 2px;
        font-weight: 600;
    }

    .nothing-time-display {
        border: 1px solid var(--nothing-gray);
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }

    .nothing-time-label {
        font-size: 0.6rem;
        color: var(--nothing-gray-light);
        letter-spacing: 2px;
        margin-bottom: 0.3rem;
    }

    .nothing-time-date {
        font-size: 0.85rem;
        color: var(--nothing-gray-light);
        margin-bottom: 0.2rem;
    }

    .nothing-time-hour {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--nothing-red);
        line-height: 1;
    }

    .nothing-time-zone {
        font-size: 0.55rem;
        color: var(--nothing-gray);
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    .nothing-info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
    }

    .nothing-info-item {
        border: 1px dotted var(--nothing-gray);
        padding: 0.5rem;
        text-align: center;
    }

    .nothing-info-label {
        display: block;
        font-size: 0.55rem;
        color: var(--nothing-gray);
        letter-spacing: 1px;
        margin-bottom: 0.2rem;
    }

    .nothing-info-value {
        display: block;
        font-size: 0.8rem;
        color: var(--nothing-white);
        font-weight: 600;
    }

    .nothing-model-status {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        border: 1px solid var(--nothing-gray);
    }

    .nothing-model-indicator {
        width: 8px;
        height: 8px;
        background: var(--nothing-red);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    .nothing-model-info {
        display: flex;
        flex-direction: column;
    }

    .nothing-model-name {
        font-size: 0.9rem;
        color: var(--nothing-white);
        font-weight: 700;
        letter-spacing: 1px;
    }

    .nothing-model-type {
        font-size: 0.55rem;
        color: var(--nothing-gray);
        letter-spacing: 1px;
    }

    /* Footer */
    .nothing-footer {
        border: 1px solid var(--nothing-gray);
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        background: var(--nothing-dark);
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--nothing-black);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--nothing-gray);
        border: 1px solid var(--nothing-gray-dark);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--nothing-red);
    }

    /* Remove default Streamlit spacing */
    .element-container {
        margin: 0 !important;
    }

    /* Plotly Chart Styling */
    .js-plotly-plot {
        border: 1px solid var(--nothing-gray) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Constants
BANK_LEVEL = 2.161  # meters MSL
BED_LEVEL = -15.70  # meters MSL
TOTAL_CAPACITY = BANK_LEVEL - BED_LEVEL
STATION_LAT = 13.700287
STATION_LON = 100.492805


# LSTM Model Architecture (must match training)
if TORCH_AVAILABLE:

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_output = lstm_out[:, -1, :]
            last_output = self.dropout(last_output)
            output = self.fc(last_output)
            return output


def calculate_risk_level(water_level):
    """Calculate risk level based on water level percentage"""
    water_level_pct = ((water_level - BED_LEVEL) / TOTAL_CAPACITY) * 100
    if water_level_pct >= 90:
        return "🔴 Critical", "critical", water_level_pct
    elif water_level_pct >= 70:
        return "🟠 High Risk", "high", water_level_pct
    elif water_level_pct >= 50:
        return "🟡 Medium Risk", "medium", water_level_pct
    else:
        return "🟢 Low Risk", "low", water_level_pct


@st.cache_data(ttl=1800)
def fetch_current_weather():
    """Fetch current and forecast weather data from Open-Meteo API"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": STATION_LAT,
            "longitude": STATION_LON,
            "hourly": [
                "temperature_2m",
                "rain",
                "relative_humidity_2m",
                "precipitation",
                "pressure_msl",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
                "cloud_cover",
                "showers",
                "weather_code",
            ],
            "past_days": 3,
            "forecast_days": 3,
            "timezone": "Asia/Bangkok",
        }

        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

        # Add missing columns with defaults
        for col in ["dew_point_2m", "wind_gusts_10m", "et0_fao_evapotranspiration"]:
            if col not in df.columns:
                df[col] = 0.0

        return df
    except Exception:
        return None


def get_manual_water_level_input():
    """
    Get water level inputs manually from user via Streamlit sidebar
    Station: สถานีสะพานกรุงเทพ (Krungthep Bridge Station)
    
    Returns dict with water level values at different time points
    """
    # Check session state for manual water level inputs
    if "water_level_now" in st.session_state:
        return {
            "water_level_now": st.session_state.get("water_level_now", 0.5),
            "water_level_1h": st.session_state.get("water_level_1h", 0.5),
            "water_level_6h": st.session_state.get("water_level_6h", 0.5),
            "water_level_12h": st.session_state.get("water_level_12h", 0.5),
            "water_level_24h": st.session_state.get("water_level_24h", 0.5),
            "measure_datetime": datetime.now(),
            "source": "Manual Input",
            "status": "success",
        }
    return None


@st.cache_data(ttl=1800)
def fetch_river_discharge():
    """Fetch river discharge data from Open-Meteo Flood API"""
    try:
        url = "https://flood-api.open-meteo.com/v1/flood"
        params = {
            "latitude": STATION_LAT,
            "longitude": STATION_LON,
            "daily": "river_discharge",
            "past_days": 7,
            "forecast_days": 7,
            "timezone": "Asia/Bangkok",
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data["daily"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

        # Resample to hourly
        hourly_idx = pd.date_range(
            df.index.min(), df.index.max() + timedelta(hours=23), freq="h"
        )
        df_hourly = df.reindex(hourly_idx).interpolate(method="time").ffill().bfill()
        df_hourly.index.name = "time"
        return df_hourly
    except Exception:
        return None


def generate_weather_data(hours=72):
    """Generate weather data when API is unavailable"""
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    idx = pd.date_range(start=now - timedelta(hours=48), periods=hours, freq="h")

    np.random.seed(int(now.timestamp()) % 2**31)
    hour_of_day = np.array([t.hour for t in idx])
    month = now.month

    # Temperature: daily cycle
    base_temp = 28 + 4 * np.sin(2 * np.pi * (month - 4) / 12)
    daily_variation = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    temperature = base_temp + daily_variation + np.random.normal(0, 1, hours)

    # Rainfall
    is_monsoon = 5 <= month <= 10
    rain_prob = 0.25 if is_monsoon else 0.08
    rain = np.where(
        np.random.random(hours) < rain_prob, np.random.exponential(2, hours), 0
    )

    humidity = (
        70
        + 15 * np.sin(2 * np.pi * (hour_of_day + 6) / 24)
        + np.random.normal(0, 5, hours)
    )
    humidity = np.clip(humidity, 40, 100)

    base_discharge = 1500 if is_monsoon else 800
    discharge = base_discharge + np.cumsum(rain) * 30 + np.random.normal(0, 100, hours)
    discharge = np.clip(discharge, 300, 4000)

    df = pd.DataFrame(
        {
            "temperature_2m": temperature,
            "rain": rain,
            "relative_humidity_2m": humidity,
            "precipitation": rain * 1.2,
            "pressure_msl": 1013 + np.random.normal(0, 3, hours),
            "surface_pressure": 1011 + np.random.normal(0, 3, hours),
            "wind_speed_10m": 5 + np.random.exponential(3, hours),
            "wind_direction_10m": np.random.uniform(0, 360, hours),
            "cloud_cover": np.clip(
                30 + 40 * (rain > 0) + np.random.normal(0, 20, hours), 0, 100
            ),
            "showers": rain * 0.3,
            "weather_code": np.where(rain > 1, 61, np.where(rain > 0, 51, 0)),
            "river_discharge": discharge,
        },
        index=idx,
    )

    return df


@st.cache_resource
def load_models():
    """Load trained LSTM model"""
    models = {}

    # Load LSTM model (if PyTorch available)
    if TORCH_AVAILABLE:
        try:
            checkpoint = torch.load(
                "models/lstm_hourly_model_cv.pth",
                map_location="cpu",
                weights_only=False,
            )
            input_size = checkpoint["input_size"]
            hidden_size = checkpoint.get("hidden_size", 64)
            num_layers = checkpoint.get("num_layers", 1)
            dropout = checkpoint.get("dropout", 0.3)

            lstm_model = LSTMModel(input_size, hidden_size, num_layers, dropout)
            lstm_model.load_state_dict(checkpoint["model_state_dict"])
            lstm_model.eval()

            models["LSTM"] = {
                "type": "lstm",
                "model": lstm_model,
                "scaler_mean": checkpoint["scaler_mean"],
                "scaler_scale": checkpoint["scaler_scale"],
                "feature_names": checkpoint["feature_names"],
                "sequence_length": checkpoint.get("sequence_length", 24),
            }
        except Exception as e:
            st.sidebar.error(f"⚠️ LSTM not loaded: {e}")
    else:
        st.error(
            "PyTorch is not available. Please install torch to use the LSTM model."
        )

    return models


def create_features(df):
    """
    Create features for prediction
    Uses new modular features module if available, falls back to old method
    """
    try:
        # Try using new modular features function
        from features import create_all_features
        import yaml

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Use new feature creation (includes time features)
        df_featured = create_all_features(df, config, is_training=False)
        return df_featured
    except (ImportError, FileNotFoundError):
        # Fall back to old method if new one not available
        pass

    # Fallback: Original feature creation method
    df_featured = df.copy()

    for lag in [1, 2, 3, 6, 12, 24]:
        df_featured[f"water_level_lag_{lag}"] = df_featured["water_level"].shift(lag)

    for window in [6, 12, 24]:
        df_featured[f"water_level_rolling_mean_{window}"] = (
            df_featured["water_level"].rolling(window=window).mean()
        )
        df_featured[f"water_level_rolling_std_{window}"] = (
            df_featured["water_level"].rolling(window=window).std()
        )
        df_featured[f"water_level_rolling_min_{window}"] = (
            df_featured["water_level"].rolling(window=window).min()
        )
        df_featured[f"water_level_rolling_max_{window}"] = (
            df_featured["water_level"].rolling(window=window).max()
        )

    for col in ["rain", "precipitation", "river_discharge", "temperature_2m"]:
        if col in df_featured.columns:
            df_featured[f"{col}_rolling_mean_6"] = (
                df_featured[col].rolling(window=6).mean()
            )
            df_featured[f"{col}_rolling_mean_12"] = (
                df_featured[col].rolling(window=12).mean()
            )

    df_featured["water_level_diff_1"] = df_featured["water_level"].diff(1)
    df_featured["water_level_diff_24"] = df_featured["water_level"].diff(24)
    df_featured["water_level_pct"] = (
        (df_featured["water_level"] - BED_LEVEL) / TOTAL_CAPACITY
    ) * 100

    return df_featured


def simulate_water_levels(model_info, weather_df, initial_level):
    """Simulate water levels using LSTM model"""
    feature_names = model_info["feature_names"]

    df = weather_df.copy()

    # Initialize with tidal pattern
    hour_of_day = np.array([t.hour for t in df.index])
    tidal = 0.25 * np.sin(2 * np.pi * hour_of_day / 12.42)
    rain_effect = (
        np.convolve(df["rain"].values, np.exp(-np.arange(24) / 6), mode="same") * 0.04
    )
    base_level = (
        initial_level + tidal + rain_effect + np.random.normal(0, 0.03, len(df))
    )
    df["water_level"] = np.clip(base_level, -1.5, 2.5)

    df_featured = create_features(df)

    if model_info["type"] == "lstm" and TORCH_AVAILABLE:
        model = model_info["model"]
        scaler_mean = model_info["scaler_mean"]
        scaler_scale = model_info["scaler_scale"]
        seq_length = model_info["sequence_length"]

        for i in range(max(48, seq_length), len(df)):
            df_temp = df_featured.iloc[: i + 1].dropna()
            if len(df_temp) < seq_length:
                continue

            available_features = [f for f in feature_names if f in df_temp.columns]
            if len(available_features) < len(feature_names) * 0.7:
                continue

            try:
                X = df_temp[available_features].iloc[-seq_length:].values
                full_X = np.zeros((seq_length, len(feature_names)))
                for j, fname in enumerate(feature_names):
                    if fname in available_features:
                        idx = available_features.index(fname)
                        full_X[:, j] = X[:, idx] if idx < X.shape[1] else 0

                X_scaled = (full_X - scaler_mean) / scaler_scale
                X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)

                with torch.no_grad():
                    pred = model(X_tensor).item()

                df.iloc[i, df.columns.get_loc("water_level")] = (
                    0.7 * pred + 0.3 * df["water_level"].iloc[i]
                )
                df_featured = create_features(df)
            except Exception:
                continue

    return df


def predict_future(model_info, features_df):
    """
    Predict water level 24 hours ahead using LSTM model
    """
    try:
        # Try using new modular predict function
        from predict import predict

        # Determine model path from model_info
        model_type = model_info.get("type", "unknown")
        if model_type == "lstm":
            model_path = "models/lstm_hourly_model_cv.pth"
            try:
                # Use new predict function
                prediction = predict(model_path, features_df)
                return prediction
            except Exception as e:
                # Fall back to old method if new one fails
                st.warning(f"Using fallback prediction method: {e}")
    except ImportError:
        # Fall back to old method if predict module not available
        pass

    # Fallback: Original prediction method
    feature_names = model_info["feature_names"]
    available_features = [f for f in feature_names if f in features_df.columns]

    if model_info["type"] == "lstm" and TORCH_AVAILABLE:
        model = model_info["model"]
        scaler_mean = model_info["scaler_mean"]
        scaler_scale = model_info["scaler_scale"]
        seq_length = model_info["sequence_length"]

        if len(features_df) < seq_length:
            return features_df["water_level"].iloc[-1]

        X = features_df[available_features].iloc[-seq_length:].values
        full_X = np.zeros((seq_length, len(feature_names)))
        for j, fname in enumerate(feature_names):
            if fname in available_features:
                idx = available_features.index(fname)
                full_X[:, j] = X[:, idx] if idx < X.shape[1] else 0

        X_scaled = (full_X - scaler_mean) / scaler_scale
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)

        with torch.no_grad():
            return model(X_tensor).item()

    return features_df["water_level"].iloc[-1]


def create_gauge(value, title, min_val=-2, max_val=3):
    """Create gauge chart with Nothing.tech theme"""
    # Monochrome with Red accent
    if value >= BANK_LEVEL:
        bar_color = "#D71921"  # Nothing Red
    else:
        bar_color = "#FFFFFF"  # White

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": title.upper(),
                "font": {"size": 12, "color": "#666666", "family": "Space Mono"},
            },
            number={
                "suffix": " m",
                "font": {"size": 32, "color": "#FFFFFF", "family": "Space Mono"},
            },
            delta={
                "reference": BANK_LEVEL,
                "position": "top",
                "font": {"size": 11, "color": "#666666", "family": "Space Mono"},
            },
            gauge={
                "axis": {
                    "range": [min_val, max_val],
                    "tickwidth": 1,
                    "tickcolor": "#444444",
                    "tickfont": {
                        "size": 9,
                        "color": "#666666",
                        "family": "Space Mono",
                    },
                },
                "bar": {"color": bar_color, "line": {"color": "#000", "width": 2}},
                "bgcolor": "#0A0A0A",
                "borderwidth": 1,
                "bordercolor": "#333333",
                "steps": [
                    {"range": [min_val, 0], "color": "#0A0A0A"},
                    {"range": [0, BANK_LEVEL], "color": "#1A1A1A"},
                    {"range": [BANK_LEVEL, max_val], "color": "rgba(215,25,33,0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#D71921", "width": 3},
                    "thickness": 1,
                    "value": BANK_LEVEL,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=15, r=15, t=35, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono"),
    )
    return fig


def present(
    current_water_level,
    prediction_24h,
    current_data,
    df_with_real,
    current_idx,
    now,
    data_source,
    measure_time,
    selected_model_name,
):
    """
    Present all UI components - Simplified Nothing.tech style
    """
    pred_time = now + timedelta(hours=24)

    # Main metrics row - compact
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")

    with col1:
        st.markdown("<div class='nothing-card'><div style='color:#666;font-size:0.7rem;letter-spacing:2px;margin-bottom:0.3rem;'>CURRENT</div>", unsafe_allow_html=True)
        fig1 = create_gauge(current_water_level, "")
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

        risk_label, risk_class, risk_pct = calculate_risk_level(current_water_level)
        risk_styles = {
            "critical": "border-color: #D71921; color: #D71921;",
            "high": "border-color: #FF6B00; color: #FF6B00;",
            "medium": "border-color: #FFB800; color: #FFB800;",
            "low": "border-color: #00FF88; color: #00FF88;",
        }
        st.markdown(
            f"""
        <div style='border: 1px solid; {risk_styles.get(risk_class, "")} padding: 0.5rem; text-align: center; 
                    font-weight: 700; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;'>
            {risk_label.split(" ")[0]} // {risk_pct:.1f}%
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("<div class='nothing-card'><div style='color:#666;font-size:0.7rem;letter-spacing:2px;margin-bottom:0.3rem;'>24H FORECAST</div>", unsafe_allow_html=True)

        fig2 = create_gauge(prediction_24h, "")
        st.plotly_chart(
            fig2, use_container_width=True, config={"displayModeBar": False}
        )

        change = prediction_24h - current_water_level
        if change > 0.1:
            change_style = "border-color: #D71921; color: #D71921;"
            change_icon = "↑"
            change_text = f"RISE +{change:.2f}M"
        elif change < -0.1:
            change_style = "border-color: #00BFFF; color: #00BFFF;"
            change_icon = "↓"
            change_text = f"DROP {change:.2f}M"
        else:
            change_style = "border-color: #00FF88; color: #00FF88;"
            change_icon = "→"
            change_text = f"STABLE {change:+.2f}M"

        st.markdown(
            f"""
        <div style='border: 1px solid; {change_style} padding: 0.5rem; text-align: center; 
                    font-weight: 700; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;'>
            {change_icon} {change_text}
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown("<div class='nothing-card'><div style='color:#666;font-size:0.7rem;letter-spacing:2px;margin-bottom:0.5rem;'>WEATHER</div>", unsafe_allow_html=True)

        # Weather metrics - Nothing.tech style
        metrics_data = [
            ("TEMP", f"{current_data['temperature_2m']:.1f}°C"),
            ("RAIN", f"{current_data['rain']:.1f}mm"),
            ("HUMID", f"{current_data['relative_humidity_2m']:.0f}%"),
            ("FLOW", f"{current_data['river_discharge']:.0f}m³/s"),
        ]

        for label, value in metrics_data:
            st.markdown(
                f"""
            <div class="nothing-metric">
                <span class="nothing-metric-label">{label}</span>
                <span class="nothing-metric-value">{value}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Water Level Timeline Section
    st.markdown(
        """
    <div class="nothing-section" style="justify-content: center;">
        <span style="color: #FFF; font-weight: 700; font-size: 1.1rem; letter-spacing: 2px;">WATER LEVEL TIMELINE</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Get all water level inputs for display
    wl_now = st.session_state.get("water_level_now", current_water_level)
    wl_1h = st.session_state.get("water_level_1h", current_water_level)
    wl_6h = st.session_state.get("water_level_6h", current_water_level)
    wl_12h = st.session_state.get("water_level_12h", current_water_level)
    wl_24h = st.session_state.get("water_level_24h", current_water_level)

    # Create comprehensive water level chart
    # Past 24h from interpolation
    past_start = max(0, current_idx - 24)
    df_past = df_with_real.iloc[past_start : current_idx + 1]

    # Future prediction line (smooth transition from current to predicted)
    future_times = pd.date_range(now, periods=25, freq="h")
    future_levels = np.linspace(current_water_level, prediction_24h, 25)
    # Add slight variation to make it more realistic
    future_levels += np.sin(np.linspace(0, 4 * np.pi, 25)) * 0.03

    fig = go.Figure()

    # Past data (interpolated from manual inputs)
    fig.add_trace(
        go.Scatter(
            x=df_past.index,
            y=df_past["water_level"],
            mode="lines",
            name="PAST 24H",
            line=dict(color="#FFFFFF", width=2),
            fill="tozeroy",
            fillcolor="rgba(255, 255, 255, 0.05)",
        )
    )

    # Manual input points (key data points)
    input_times = [
        now,
        now - timedelta(hours=1),
        now - timedelta(hours=6),
        now - timedelta(hours=12),
        now - timedelta(hours=24),
    ]
    input_levels = [wl_now, wl_1h, wl_6h, wl_12h, wl_24h]
    input_labels = ["NOW", "-1H", "-6H", "-12H", "-24H"]

    fig.add_trace(
        go.Scatter(
            x=input_times,
            y=input_levels,
            mode="markers+text",
            name="INPUT DATA",
            marker=dict(
                color="#FFFFFF",
                size=12,
                symbol="diamond",
                line=dict(color="#D71921", width=2),
            ),
            text=input_labels,
            textposition="top center",
            textfont=dict(color="#888", size=10, family="Space Mono"),
        )
    )

    # Future prediction
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=future_levels,
            mode="lines",
            name="FORECAST +24H",
            line=dict(color="#D71921", width=2, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(215, 25, 33, 0.08)",
        )
    )

    # Current point (highlighted)
    fig.add_trace(
        go.Scatter(
            x=[now],
            y=[current_water_level],
            mode="markers",
            name="CURRENT",
            marker=dict(color="#FFFFFF", size=10, line=dict(color="#D71921", width=2)),
        )
    )

    # 24h prediction point
    fig.add_trace(
        go.Scatter(
            x=[pred_time],
            y=[prediction_24h],
            mode="markers",
            name="24H PREDICTION",
            marker=dict(
                color="#D71921",
                size=10,
                symbol="star",
                line=dict(color="#FFFFFF", width=2),
            ),
        )
    )

    fig.add_hline(
        y=BANK_LEVEL,
        line_dash="dash",
        line_color="#D71921",
        line_width=2,
        annotation_text=f"BANK LEVEL {BANK_LEVEL}M",
        annotation_position="right",
        annotation_font_color="#D71921",
        annotation_font_family="Space Mono",
        annotation_font_size=10,
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="#444",
        line_width=1,
        annotation_text="SEA LEVEL",
        annotation_position="right",
        annotation_font_color="#666",
        annotation_font_family="Space Mono",
        annotation_font_size=10,
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="LEVEL (m)",
        height=400,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#888", size=10, family="Space Mono"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=50, r=30, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono", size=11, color="#FFF"),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="#333",
            tickfont=dict(color="#666"),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="#333",
            tickfont=dict(color="#666"),
            zeroline=False,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Weather Section
    st.markdown(
        """
    <div class="nothing-section" style="margin-top: 2rem; justify-content: center;">
        <span style="color: #FFF; font-weight: 700; font-size: 1.1rem; letter-spacing: 2px;">WEATHER DATA</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_w1, col_w2 = st.columns(2)

    df_weather_display = df_with_real.iloc[past_start:]

    with col_w1:
        fig_rain = go.Figure()
        fig_rain.add_trace(
            go.Bar(
                x=df_weather_display.index,
                y=df_weather_display["rain"],
                marker=dict(
                    color=df_weather_display["rain"],
                    colorscale=[
                        [0, "#1A1A1A"],
                        [0.3, "#444"],
                        [0.6, "#888"],
                        [1, "#D71921"],
                    ],
                    showscale=False,
                ),
                name="RAINFALL",
            )
        )
        fig_rain.update_layout(
            title=dict(
                text="RAINFALL",
                font=dict(size=12, color="#888", family="Space Mono"),
                x=0.5,
            ),
            height=220,
            margin=dict(l=40, r=20, t=40, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Mono", color="#666", size=10),
            xaxis=dict(
                showgrid=False, 
                linecolor="#333", 
                tickfont=dict(color="#444")
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor="rgba(255,255,255,0.03)", 
                linecolor="#333",
                tickfont=dict(color="#444"),
                title=dict(text="mm/h", font=dict(color="#555", size=10)),
            ),
        )
        st.plotly_chart(
            fig_rain, use_container_width=True, config={"displayModeBar": False}
        )

    with col_w2:
        fig_temp = go.Figure()
        fig_temp.add_trace(
            go.Scatter(
                x=df_weather_display.index,
                y=df_weather_display["temperature_2m"],
                mode="lines",
                name="TEMPERATURE",
                line=dict(color="#D71921", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(215, 25, 33, 0.08)",
            )
        )
        fig_temp.update_layout(
            title=dict(
                text="TEMPERATURE",
                font=dict(size=12, color="#888", family="Space Mono"),
                x=0.5,
            ),
            height=220,
            margin=dict(l=40, r=20, t=40, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Mono", color="#666", size=10),
            xaxis=dict(
                showgrid=False, 
                linecolor="#333", 
                tickfont=dict(color="#444")
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor="rgba(255,255,255,0.03)", 
                linecolor="#333",
                tickfont=dict(color="#444"),
                title=dict(text="°C", font=dict(color="#555", size=10)),
            ),
        )
        st.plotly_chart(
            fig_temp, use_container_width=True, config={"displayModeBar": False}
        )

    # Footer - simplified
    st.markdown(
        """
    <div style="text-align: center; padding: 2rem 0; margin-top: 2rem; border-top: 1px dashed #222;">
        <span style="color: #444; font-size: 0.7rem; letter-spacing: 2px;">
            LSTM · R²=0.94 · MAE=0.12m · 
            <a href="https://www.thaiwater.net/water/wl" target="_blank" style="color: #666;">THAIWATER.NET ↗</a> · 
            CPDSAI @ AIT
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )


def setup_controls(models):
    """
    Setup controls on main page (simplified layout)
    """
    # Default to LSTM
    selected_model_name = "LSTM"
    selected_model = models.get("LSTM")

    if not selected_model:
        st.error("LSTM model not found!")
        return None, None, datetime.now()

    # Current time
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    # Initialize session state for water levels if not exists
    default_level = 0.5
    if "water_level_now" not in st.session_state:
        st.session_state.water_level_now = default_level
        st.session_state.water_level_1h = default_level
        st.session_state.water_level_6h = default_level
        st.session_state.water_level_12h = default_level
        st.session_state.water_level_24h = default_level

    # Header
    st.markdown(
        f"""
    <div class="nothing-header">
        <div style="font-size: 2.2rem; font-weight: 700; letter-spacing: -1px; line-height: 1;">
            WATER<span style="color: #D71921;">.</span>LEVEL
        </div>
        <div style="color: #666; font-size: 0.8rem; letter-spacing: 2px; margin-top: 0.3rem;">
            CPY015 · KRUNGTHEP BRIDGE · {now.strftime("%Y.%m.%d %H:%M")}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Control bar - all inputs in one row
    st.markdown(
        """
    <div class="nothing-section" style="justify-content: center; margin-bottom: 0.5rem;">
        <span style="color: #888; font-size: 0.8rem; letter-spacing: 2px;">WATER LEVEL INPUT (m.MSL)</span>
        <span style="color: #666; font-size: 0.7rem; margin-left: 1rem;">
            from <a href="https://www.thaiwater.net/water/wl" target="_blank" style="color: #D71921;">thaiwater.net ↗</a>
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )
    
    # All inputs in one row
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 0.8])
    
    with c1:
        st.session_state.water_level_now = st.number_input(
            "NOW",
            min_value=-15.0,
            max_value=3.0,
            value=st.session_state.water_level_now,
            step=0.01,
            format="%.2f",
            key="input_now",
        )
    with c2:
        st.session_state.water_level_1h = st.number_input(
            "-1H",
            min_value=-15.0,
            max_value=3.0,
            value=st.session_state.water_level_1h,
            step=0.01,
            format="%.2f",
            key="input_1h",
        )
    with c3:
        st.session_state.water_level_6h = st.number_input(
            "-6H",
            min_value=-15.0,
            max_value=3.0,
            value=st.session_state.water_level_6h,
            step=0.01,
            format="%.2f",
            key="input_6h",
        )
    with c4:
        st.session_state.water_level_12h = st.number_input(
            "-12H",
            min_value=-15.0,
            max_value=3.0,
            value=st.session_state.water_level_12h,
            step=0.01,
            format="%.2f",
            key="input_12h",
        )
    with c5:
        st.session_state.water_level_24h = st.number_input(
            "-24H",
            min_value=-15.0,
            max_value=3.0,
            value=st.session_state.water_level_24h,
            step=0.01,
            format="%.2f",
            key="input_24h",
        )
    with c6:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("⟳", use_container_width=True, help="Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    return selected_model_name, selected_model, now


def fetch_and_process_data(selected_model_name, selected_model, now):
    """
    Fetch and process all data needed for predictions
    """
    # Get water level from manual input (session state)
    manual_water_data = get_manual_water_level_input()

    # Fetch weather data
    weather_df = fetch_current_weather()
    discharge_df = fetch_river_discharge()

    if weather_df is None:
        st.warning("⚠️ Could not fetch weather API. Using simulated data.")
        weather_df = generate_weather_data(hours=120)

    if discharge_df is not None and "river_discharge" in discharge_df.columns:
        weather_df = weather_df.join(
            discharge_df[["river_discharge"]], how="left", rsuffix="_api"
        )
        if "river_discharge_api" in weather_df.columns:
            weather_df["river_discharge"] = weather_df["river_discharge_api"].fillna(
                weather_df.get("river_discharge", 1000)
            )
            weather_df = weather_df.drop(columns=["river_discharge_api"])

    if "river_discharge" not in weather_df.columns:
        weather_df["river_discharge"] = 1200 + np.random.normal(0, 100, len(weather_df))

    # Get current water level from manual input
    if (
        manual_water_data
        and manual_water_data["status"] == "success"
    ):
        current_water_level = manual_water_data["water_level_now"]
        measure_time = manual_water_data["measure_datetime"]
        data_source = "✏️ Manual Input"
        
        # Get all water level inputs
        wl_now = manual_water_data["water_level_now"]
        wl_1h = manual_water_data["water_level_1h"]
        wl_6h = manual_water_data["water_level_6h"]
        wl_12h = manual_water_data["water_level_12h"]
        wl_24h = manual_water_data["water_level_24h"]
    else:
        # Use default values from session state
        current_water_level = st.session_state.get("water_level_now", 0.5)
        wl_now = current_water_level
        wl_1h = st.session_state.get("water_level_1h", 0.5)
        wl_6h = st.session_state.get("water_level_6h", 0.5)
        wl_12h = st.session_state.get("water_level_12h", 0.5)
        wl_24h = st.session_state.get("water_level_24h", 0.5)
        measure_time = now
        data_source = "✏️ Manual Input"

    # Add current water level to weather dataframe for predictions
    weather_df["water_level"] = current_water_level

    # Create time series with current level
    df_with_real = weather_df.copy()

    # Create past 24 hours water levels using interpolation from manual inputs
    # Key points: now (0h), 1h ago, 6h ago, 12h ago, 24h ago
    key_hours = [0, 1, 6, 12, 24]
    key_levels = [wl_now, wl_1h, wl_6h, wl_12h, wl_24h]
    
    # Interpolate for all 24 hours
    all_hours = np.arange(0, 25)  # 0 to 24 hours ago
    interpolated_levels = np.interp(all_hours, key_hours, key_levels)
    
    # Create past 24h time index
    past_24h_idx = pd.date_range(end=now, periods=25, freq="h")
    
    # Assign interpolated water levels (reverse order: oldest first)
    for i, idx in enumerate(past_24h_idx):
        hours_ago = 24 - i  # 24h ago to now
        if idx in df_with_real.index:
            df_with_real.loc[idx, "water_level"] = interpolated_levels[hours_ago]

    # Set current water level at current time
    if now in df_with_real.index:
        df_with_real.loc[now, "water_level"] = current_water_level
    else:
        # Add current time if not in index
        df_with_real.loc[now] = df_with_real.iloc[-1]
        df_with_real.loc[now, "water_level"] = current_water_level
        df_with_real = df_with_real.sort_index()

    # Forward fill water level for future predictions
    df_with_real["water_level"] = df_with_real["water_level"].ffill()

    # Get current data for display
    current_idx = df_with_real.index.get_indexer([now], method="nearest")[0]
    if current_idx < 0 or current_idx >= len(df_with_real):
        current_idx = len(df_with_real) - 1

    current_data = df_with_real.iloc[current_idx]

    # Predict 24 hours ahead using selected model
    df_featured = create_features(df_with_real.iloc[: current_idx + 1]).dropna()
    if len(df_featured) > 0:
        prediction_24h = predict_future(selected_model, df_featured)
    else:
        prediction_24h = current_water_level

    return (
        current_water_level,
        prediction_24h,
        current_data,
        df_with_real,
        current_idx,
        data_source,
        measure_time,
    )


def main():
    # Load models
    models = load_models()
    if not models:
        st.error(
            "❌ No models available. Please ensure model files exist in 'models/' directory."
        )
        return

    # Setup controls on main page
    selected_model_name, selected_model, now = setup_controls(models)

    # Fetch and process data
    with st.spinner("Loading data..."):
        (
            current_water_level,
            prediction_24h,
            current_data,
            df_with_real,
            current_idx,
            data_source,
            measure_time,
        ) = fetch_and_process_data(selected_model_name, selected_model, now)

    # Present the data
    present(
        current_water_level,
        prediction_24h,
        current_data,
        df_with_real,
        current_idx,
        now,
        data_source,
        measure_time,
        selected_model_name,
    )


if __name__ == "__main__":
    main()
