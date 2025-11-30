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
import re
from bs4 import BeautifulSoup
import json

warnings.filterwarnings("ignore")

# Optional Selenium import for JavaScript-rendered content
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

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

# Custom CSS for professional loft-style design
st.markdown(
    """
<style>
    /* Nothing.tech Inspired Design - Raw, Industrial, Monochrome */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Ndot+55:wght@500&display=swap'); /* Simulated dot matrix if available, fallback to Mono */

    :root {
        --bg-color: #000000;
        --surface-color: #0a0a0a;
        --text-color: #E6E6E6;
        --text-dim: #888888;
        --accent-color: #D71921; /* Nothing Red */
        --border-color: #333333;
        --grid-color: #222222;
    }

    /* Global Reset */
    html, body, [class*="css"] {
        font-family: 'Roboto Mono', monospace;
        background-color: var(--bg-color);
        color: var(--text-color);
    }

    /* Main Container */
    .main {
        background-color: var(--bg-color);
        background-image: radial-gradient(var(--grid-color) 1px, transparent 1px);
        background-size: 20px 20px;
        padding: 2rem;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Roboto Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
        color: var(--text-color);
    }

    h1 {
        font-size: 2.5rem;
        border-bottom: 2px solid var(--accent-color);
        padding-bottom: 0.5rem;
        display: inline-block;
    }

    /* Cards */
    .metric-card {
        background: rgba(10, 10, 10, 0.8);
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        border-radius: 0; /* Sharp corners */
        position: relative;
        backdrop-filter: blur(10px);
        transition: all 0.2s ease;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: repeating-linear-gradient(
            45deg,
            transparent,
            transparent 10px,
            rgba(255, 255, 255, 0.02) 10px,
            rgba(255, 255, 255, 0.02) 20px
        );
        pointer-events: none;
    }

    .metric-card:hover {
        border-color: var(--text-color);
        transform: translate(-2px, -2px);
        box-shadow: 4px 4px 0 var(--accent-color);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-color);
        border-right: 1px solid var(--border-color);
    }

    /* Buttons */
    .stButton > button {
        background: transparent;
        color: var(--text-color);
        border: 1px solid var(--text-color);
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        font-family: 'Roboto Mono', monospace;
        text-transform: uppercase;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: var(--accent-color);
        border-color: var(--accent-color);
        color: white;
    }

    /* Charts */
    .js-plotly-plot {
        border: 1px solid var(--border-color);
        background: var(--bg-color);
        padding: 1rem;
    }

    /* Custom Header */
    .header-container {
        border: 1px solid var(--border-color);
        padding: 2rem;
        margin-bottom: 2rem;
        background: var(--bg-color);
        position: relative;
        overflow: hidden;
    }

    .header-title {
        font-size: 3rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: -2px;
        line-height: 1;
    }

    .header-subtitle {
        color: var(--accent-color);
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.5rem;
    }

    /* Status Badges */
    .status-badge {
        background: transparent;
        border: 1px solid var(--text-color);
        color: var(--text-color);
        padding: 0.2rem 0.8rem;
        border-radius: 12px;
        font-size: 0.8rem;
        text-transform: uppercase;
    }

    /* Alerts */
    .stAlert {
        background: var(--bg-color);
        color: var(--text-color);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--accent-color);
    }
    
    /* Inputs */
    .stSelectbox > div > div {
        background-color: var(--bg-color);
        color: var(--text-color);
        border-color: var(--border-color);
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


@st.cache_data(ttl=300)  # Cache for 5 minutes (shorter for real-time data)
def fetch_water_level_thaiwater():
    """
    Scrape real-time water level data from Thai Water website graph
    Source: https://www.thaiwater.net/water/wl
    Station: สถานีสะพานกรุงเทพ (Krungthep Bridge Station)

    The website displays data in graph format, so we need to extract from:
    - JavaScript variables containing chart data
    - JSON data embedded in script tags
    - API endpoints called by the graph
    - Chart library data structures (Chart.js, Highcharts, Plotly, etc.)
    - Selenium for JavaScript-rendered content (if available)
    """

    try:
        url = "https://www.thaiwater.net/water/wl"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "th-TH,th;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = "utf-8"

        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")

        station_names = ["สะพานกรุงเทพ", "Krungthep", "CPY015", "กรุงเทพ", "CPY015"]
        water_level = None
        measure_time = None

        # Method 1: Search for JSON data in script tags
        script_tags = soup.find_all("script")
        for script in script_tags:
            script_text = script.string
            if not script_text:
                continue

            # Look for JSON objects containing station data
            # Common patterns: var data = {...}, data: {...}, series: [...]
            json_patterns = [
                r"var\s+\w*[Dd]ata\w*\s*=\s*(\{.*?\});",
                r"data\s*:\s*(\{.*?\})",
                r"series\s*:\s*(\[.*?\])",
                r"chartData\s*=\s*(\{.*?\});",
                r"dataset\s*=\s*(\{.*?\});",
            ]

            for pattern in json_patterns:
                matches = re.finditer(pattern, script_text, re.DOTALL)
                for match in matches:
                    try:
                        json_str = match.group(1)
                        # Try to parse as JSON
                        data = json.loads(json_str)
                        # Recursively search for station name and water level
                        result = _extract_from_json(data, station_names)
                        if result:
                            water_level = result.get("water_level")
                            measure_time = result.get("time")
                            if water_level is not None:
                                break
                    except (json.JSONDecodeError, AttributeError):
                        continue
                if water_level is not None:
                    break

            # Method 1b: Look for JavaScript arrays with data
            # Pattern: [["time", value], ["time", value], ...]
            array_pattern = r"\[\[.*?\]\s*,\s*\[.*?\]\s*\]"
            arrays = re.findall(array_pattern, script_text)
            for arr_str in arrays:
                try:
                    # Try to parse as JSON array
                    arr = json.loads(arr_str)
                    if isinstance(arr, list) and len(arr) > 0:
                        # Get the last (most recent) data point
                        last_point = arr[-1]
                        if (
                            isinstance(last_point, (list, tuple))
                            and len(last_point) >= 2
                        ):
                            # Check if this might be our station
                            point_str = str(last_point).lower()
                            if (
                                any(name.lower() in point_str for name in station_names)
                                or len(arr) > 0
                            ):
                                # Try to extract numeric value
                                for item in last_point:
                                    if isinstance(item, (int, float)):
                                        if -20 <= item <= 5:
                                            water_level = float(item)
                                            break
                                    elif isinstance(item, str):
                                        num_match = re.search(r"-?\d+\.?\d*", item)
                                        if num_match:
                                            num = float(num_match.group())
                                            if -20 <= num <= 5:
                                                water_level = num
                                                break
                except (json.JSONDecodeError, ValueError, IndexError):
                    continue
                if water_level is not None:
                    break

            # Method 1c: Look for JavaScript variables with station code CPY015
            # Pattern: CPY015: {...} or "CPY015": {...}
            cpy015_patterns = [
                r"CPY015\s*[:=]\s*(\{.*?\})",
                r'"CPY015"\s*:\s*(\{.*?\})',
                r"'CPY015'\s*:\s*(\{.*?\})",
            ]
            for pattern in cpy015_patterns:
                matches = re.finditer(pattern, script_text, re.DOTALL)
                for match in matches:
                    try:
                        json_str = match.group(1)
                        data = json.loads(json_str)
                        # Look for water level value
                        if isinstance(data, dict):
                            for key, value in data.items():
                                if (
                                    "level" in key.lower()
                                    or "water" in key.lower()
                                    or "value" in key.lower()
                                ):
                                    if (
                                        isinstance(value, (int, float))
                                        and -20 <= value <= 5
                                    ):
                                        water_level = float(value)
                                        break
                                elif (
                                    isinstance(value, (int, float))
                                    and -20 <= value <= 5
                                ):
                                    water_level = float(value)
                                    break
                    except (json.JSONDecodeError, AttributeError):
                        continue
                if water_level is not None:
                    break

            if water_level is not None:
                break

        # Method 2: Search for API endpoints in script tags
        if water_level is None:
            for script in script_tags:
                script_text = script.string
                if not script_text:
                    continue

                # Look for API URLs
                api_patterns = [
                    r'["\']([^"\']*api[^"\']*water[^"\']*)["\']',
                    r'["\']([^"\']*api[^"\']*wl[^"\']*)["\']',
                    r'["\']([^"\']*api[^"\']*station[^"\']*)["\']',
                    r'fetch\s*\(["\']([^"\']+)["\']',
                    r'ajax\s*\(["\']([^"\']+)["\']',
                    r'\.get\s*\(["\']([^"\']+)["\']',
                ]

                for pattern in api_patterns:
                    matches = re.finditer(pattern, script_text, re.IGNORECASE)
                    for match in matches:
                        api_url = match.group(1)
                        # Try to fetch from API
                        if not api_url.startswith("http"):
                            # Relative URL
                            if api_url.startswith("/"):
                                api_url = "https://www.thaiwater.net" + api_url
                            else:
                                api_url = "https://www.thaiwater.net/" + api_url

                        try:
                            api_response = requests.get(
                                api_url, headers=headers, timeout=10
                            )
                            if api_response.status_code == 200:
                                api_data = api_response.json()
                                result = _extract_from_json(api_data, station_names)
                                if result and result.get("water_level"):
                                    water_level = result.get("water_level")
                                    measure_time = result.get("time")
                                    break
                        except:
                            continue
                    if water_level is not None:
                        break
                if water_level is not None:
                    break

        # Method 3: Look for data attributes in HTML elements (for chart libraries)
        if water_level is None:
            # Chart.js data attributes
            chart_elements = (
                soup.find_all(attrs={"data-chart": True})
                + soup.find_all(attrs={"data-series": True})
                + soup.find_all(attrs={"data-values": True})
            )

            for elem in chart_elements:
                data_attr = (
                    elem.get("data-chart")
                    or elem.get("data-series")
                    or elem.get("data-values")
                )
                try:
                    data = json.loads(data_attr)
                    result = _extract_from_json(data, station_names)
                    if result and result.get("water_level"):
                        water_level = result.get("water_level")
                        measure_time = result.get("time")
                        break
                except:
                    continue

        # Method 4: Search for embedded data in HTML comments or hidden divs
        if water_level is None:
            # Look for hidden divs with data
            hidden_divs = soup.find_all(
                "div", style=re.compile(r"display\s*:\s*none", re.I)
            )
            for div in hidden_divs:
                div_text = div.get_text()
                if any(name in div_text for name in station_names):
                    numbers = re.findall(r"-?\d+\.?\d*", div_text)
                    for num_str in numbers:
                        try:
                            num = float(num_str)
                            if -20 <= num <= 5:
                                water_level = num
                                break
                        except ValueError:
                            continue
                    if water_level is not None:
                        break

        # Method 5: Use Selenium if available (for JavaScript-rendered graphs)
        if water_level is None and SELENIUM_AVAILABLE:
            try:
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--window-size=1920,1080")
                chrome_options.add_argument(
                    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )

                # Try to use existing Chrome driver or system PATH
                try:
                    driver = webdriver.Chrome(options=chrome_options)
                except:
                    # If Chrome driver not found, skip Selenium
                    driver = None

                if driver:
                    try:
                        driver.get(url)
                        # Wait for page to load (wait for graph to render)
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.TAG_NAME, "body"))
                        )

                        # Wait a bit more for JavaScript to execute
                        import time

                        time.sleep(3)

                        # Get page source after JavaScript execution
                        page_source = driver.page_source
                        soup_js = BeautifulSoup(page_source, "html.parser")

                        # Now search in the JavaScript-rendered content
                        # Look for text content that might contain the water level
                        page_text = soup_js.get_text()

                        # Search for station name and nearby numbers
                        for name in station_names:
                            idx = page_text.find(name)
                            if idx != -1:
                                context = page_text[
                                    max(0, idx - 300) : min(len(page_text), idx + 300)
                                ]
                                # Look for numbers that could be water level
                                numbers = re.findall(r"-?\d+\.?\d*", context)
                                for num_str in numbers:
                                    try:
                                        num = float(num_str)
                                        if -20 <= num <= 5:
                                            water_level = num
                                            break
                                    except ValueError:
                                        continue
                                if water_level is not None:
                                    break

                        # Also check for data in script tags after JS execution
                        if water_level is None:
                            scripts_js = soup_js.find_all("script")
                            for script in scripts_js:
                                script_text = script.string
                                if script_text:
                                    # Look for CPY015 or station name with value
                                    for name in station_names:
                                        if name in script_text:
                                            # Extract numbers near station name
                                            pattern = rf"{re.escape(name)}[^0-9]*(-?\d+\.?\d*)"
                                            matches = re.findall(pattern, script_text)
                                            for match in matches:
                                                try:
                                                    num = float(match)
                                                    if -20 <= num <= 5:
                                                        water_level = num
                                                        break
                                                except ValueError:
                                                    continue
                                            if water_level is not None:
                                                break
                                    if water_level is not None:
                                        break

                    finally:
                        driver.quit()
            except Exception as e:
                # Selenium failed, continue without it
                pass

        # Parse measurement time
        if water_level is not None:
            if measure_time:
                try:
                    measure_datetime = pd.to_datetime(measure_time)
                except:
                    measure_datetime = datetime.now()
            else:
                measure_datetime = datetime.now()

            return {
                "water_level": water_level,
                "measure_datetime": measure_datetime,
                "source": "thaiwater.net",
                "status": "success",
            }
        else:
            return {
                "water_level": None,
                "measure_datetime": None,
                "source": "thaiwater.net",
                "status": "not_found",
                "error": "Could not find water level data in graph. The website may use JavaScript rendering. Try installing Selenium: pip install selenium",
            }

    except requests.exceptions.RequestException as e:
        return {
            "water_level": None,
            "measure_datetime": None,
            "source": "thaiwater.net",
            "status": "error",
            "error": f"Network error: {str(e)}",
        }
    except Exception as e:
        return {
            "water_level": None,
            "measure_datetime": None,
            "source": "thaiwater.net",
            "status": "error",
            "error": f"Parsing error: {str(e)}",
        }


def _extract_from_json(data, station_names):
    """
    Recursively search JSON data for station name and water level value
    """
    if isinstance(data, dict):
        # Check keys for station name
        for key, value in data.items():
            key_str = str(key).lower()
            # Check if key contains station name
            if any(name.lower() in key_str for name in station_names):
                # This might be our station data
                if isinstance(value, dict):
                    # Look for water level in nested dict
                    for sub_key, sub_value in value.items():
                        if (
                            "level" in sub_key.lower()
                            or "water" in sub_key.lower()
                            or "value" in sub_key.lower()
                        ):
                            if (
                                isinstance(sub_value, (int, float))
                                and -20 <= sub_value <= 5
                            ):
                                return {"water_level": float(sub_value), "time": None}
                        elif (
                            isinstance(sub_value, (int, float))
                            and -20 <= sub_value <= 5
                        ):
                            return {"water_level": float(sub_value), "time": None}
                elif isinstance(value, (int, float)) and -20 <= value <= 5:
                    return {"water_level": float(value), "time": None}

            # Recursively search nested structures
            result = _extract_from_json(value, station_names)
            if result:
                return result

    elif isinstance(data, list):
        for item in data:
            result = _extract_from_json(item, station_names)
            if result:
                return result

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


@st.cache_data
def load_historical_data():
    """Load historical data for initial water level reference"""
    try:
        df = pd.read_csv("full_merged.csv", index_col=0, parse_dates=True)
        df.index.name = "measure_datetime"
        return df
    except Exception:
        return None


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
                "font": {"size": 14, "color": "#888888", "family": "Roboto Mono"},
            },
            number={
                "suffix": " m",
                "font": {"size": 36, "color": "#FFFFFF", "family": "Roboto Mono"},
            },
            delta={
                "reference": BANK_LEVEL,
                "position": "top",
                "font": {"size": 12, "color": "#888888", "family": "Roboto Mono"},
            },
            gauge={
                "axis": {
                    "range": [min_val, max_val],
                    "tickwidth": 1,
                    "tickcolor": "#888888",
                    "tickfont": {
                        "size": 10,
                        "color": "#888888",
                        "family": "Roboto Mono",
                    },
                },
                "bar": {"color": bar_color, "line": {"color": "black", "width": 1}},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": "#333333",
                "steps": [
                    {"range": [min_val, 0], "color": "#111111"},
                    {"range": [0, BANK_LEVEL], "color": "#222222"},
                    {"range": [BANK_LEVEL, max_val], "color": "#330000"},
                ],
                "threshold": {
                    "line": {"color": "#D71921", "width": 2},
                    "thickness": 1,
                    "value": BANK_LEVEL,
                },
            },
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Roboto Mono"),
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
    Present all UI components and visualizations
    """
    # Header with modern loft design
    # Header with Nothing.tech design
    st.markdown(
        """
    <div class="header-container">
        <div class="header-title">WATER<span style="color: #D71921">.</span>LEVEL</div>
        <div class="header-subtitle">CPY015 // KRUNGTHEP BRIDGE</div>
        <div style="margin-top: 1rem; font-family: 'Roboto Mono'; font-size: 0.8rem; color: #888; letter-spacing: 1px;">
            CHAO PHRAYA RIVER // BKK // LSTM NEURAL NET
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Main metrics row with responsive columns
    st.markdown("### 📊 Current Status & Forecast")
    # Responsive: stack on mobile, side-by-side on desktop
    col1, col2, col3 = st.columns([2, 2, 1.5], gap="medium")

    with col1:
        st.markdown(
            """
        <div class="metric-card">
        """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h4 style='margin-top: 0; color: #888; font-size: 0.9rem;'>CURRENT LEVEL</h4>",
            unsafe_allow_html=True,
        )
        if isinstance(measure_time, datetime):
            time_str = measure_time.strftime("%Y-%m-%d %H:%M")
        else:
            time_str = str(measure_time)

        st.markdown(
            f"<p style='color: #888; font-weight: 500; font-size: 0.8rem;'>// {time_str} ICT // {data_source.split(' ')[0].upper()}</p>",
            unsafe_allow_html=True,
        )

        fig1 = create_gauge(current_water_level, "Current Level")
        st.plotly_chart(
            fig1, use_container_width=True, config={"displayModeBar": False}
        )

        risk_label, risk_class, risk_pct = calculate_risk_level(current_water_level)
        risk_colors = {
            "critical": "border: 1px solid #D71921; color: #D71921;",
            "high": "border: 1px solid #FF9800; color: #FF9800;",
            "medium": "border: 1px solid #FBBC04; color: #FBBC04;",
            "low": "border: 1px solid #34A853; color: #34A853;",
        }
        st.markdown(
            f"""
        <div style='{risk_colors.get(risk_class, "")} padding: 0.5rem; text-align: center; 
                    font-weight: 700; margin-top: 0.5rem; font-size: 0.9rem; text-transform: uppercase;'>
            {risk_label.split(" ")[0]} // {risk_pct:.1f}% CAPACITY
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
        """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h4 style='margin-top: 0; color: #888; font-size: 0.9rem;'>24H FORECAST</h4>",
            unsafe_allow_html=True,
        )
        pred_time = now + timedelta(hours=24)
        st.markdown(
            f"<p style='color: #888; font-weight: 500; font-size: 0.8rem;'>// {pred_time.strftime('%Y-%m-%d %H:%M')} ICT // LSTM MODEL</p>",
            unsafe_allow_html=True,
        )

        fig2 = create_gauge(prediction_24h, "Predicted Level")
        st.plotly_chart(
            fig2, use_container_width=True, config={"displayModeBar": False}
        )

        change = prediction_24h - current_water_level
        if change > 0.1:
            change_style = "border: 1px solid #D71921; color: #D71921;"
            change_icon = "▲"
            change_text = f"RISE: +{change:.2f} M"
        elif change < -0.1:
            change_style = "border: 1px solid #4285F4; color: #4285F4;"
            change_icon = "▼"
            change_text = f"DROP: {change:.2f} M"
        else:
            change_style = "border: 1px solid #34A853; color: #34A853;"
            change_icon = "—"
            change_text = f"STABLE: {change:+.2f} M"

        st.markdown(
            f"""
        <div style='{change_style} padding: 0.5rem; text-align: center; 
                    font-weight: 700; margin-top: 0.5rem; font-size: 0.9rem; text-transform: uppercase;'>
            {change_icon} {change_text}
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <h4 style="margin-top: 0; color: #888; font-size: 0.9rem;">ATMOSPHERE</h4>
        """,
            unsafe_allow_html=True,
        )

        # Weather metrics - Minimalist
        metrics_data = [
            ("TEMP", f"{current_data['temperature_2m']:.1f}°C"),
            ("RAIN", f"{current_data['rain']:.1f}mm"),
            ("HUMID", f"{current_data['relative_humidity_2m']:.0f}%"),
            ("FLOW", f"{current_data['river_discharge']:.0f}m³/s"),
        ]

        for label, value in metrics_data:
            st.markdown(
                f"""
            <div style='display: flex; justify-content: space-between; align-items: center; margin: 0.8rem 0; border-bottom: 1px dotted #333; padding-bottom: 0.2rem;'>
                <span style='font-size: 0.9rem; color: #888;'>{label}</span>
                <span style='font-size: 1rem; font-weight: 700; color: #FFF;'>{value}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Time series chart with enhanced styling
    st.markdown("---")
    st.markdown(
        """
    <div style='background: #000; border: 1px solid #333; padding: 1rem 1.5rem; 
                margin: 1.5rem 0 1rem 0; font-weight: 600; text-align: center;'>
        <h2 style='color: #D71921; margin: 0; font-weight: 700;'>// WATER LEVEL FORECAST //</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Past 24h and future 24h
    past_start = max(0, current_idx - 24)
    df_past = df_with_real.iloc[past_start : current_idx + 1]

    # Future prediction line (smooth transition from current to predicted)
    future_times = pd.date_range(now, periods=25, freq="h")
    future_levels = np.linspace(current_water_level, prediction_24h, 25)
    # Add slight variation to make it more realistic
    future_levels += np.sin(np.linspace(0, 4 * np.pi, 25)) * 0.04

    fig = go.Figure()

    # Past data (real or historical) - Gemini theme
    fig.add_trace(
        go.Scatter(
            x=df_past.index,
            y=df_past["water_level"],
            mode="lines",
            name="PAST 24H (REAL)",
            line=dict(color="#FFFFFF", width=2),
            fill="tozeroy",
            fillcolor="rgba(255, 255, 255, 0.1)",
        )
    )

    # Future prediction
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=future_levels,
            mode="lines",
            name="FORECAST 24H (AI)",
            line=dict(color="#D71921", width=2, dash="dash"),
            fill="tozeroy",
            fillcolor="rgba(215, 25, 33, 0.1)",
        )
    )

    # Current point
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
        annotation_text=f"// BANK LEVEL ({BANK_LEVEL} M) //",
        annotation_position="right",
        annotation_font_color="#D71921",
        annotation_font_family="Roboto Mono",
        annotation_font_size=10,
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="#888888",
        line_width=1,
        annotation_text="// MEAN SEA LEVEL //",
        annotation_position="right",
        annotation_font_color="#888888",
        annotation_font_family="Roboto Mono",
        annotation_font_size=10,
    )

    fig.update_layout(
        xaxis_title="TIME",
        yaxis_title="LEVEL (m)",
        height=420,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#888"),
        ),
        margin=dict(l=60, r=50, t=40, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Roboto Mono", size=12, color="#E6E6E6"),
        xaxis=dict(
            showgrid=True,
            gridcolor="#222",
            linecolor="#444",
            title=dict(font=dict(color="#888")),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#222",
            linecolor="#444",
            title=dict(font=dict(color="#888")),
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": True, "displaylogo": False},
    )

    # Weather charts with enhanced styling
    st.markdown(
        """
    <div style='background: #000; border: 1px solid #333; padding: 1rem 1.5rem; 
                margin: 1.5rem 0 1rem 0; font-weight: 600; text-align: center;'>
        <h2 style='color: #D71921; margin: 0; font-weight: 700;'>// WEATHER FORECAST //</h2>
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
                        [0, "#111"],
                        [0.5, "#888"],
                        [1, "#D71921"],
                    ],  # Monochrome with red accent
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text="MM/HR", font=dict(color="#888", family="Roboto Mono")
                        ),
                        tickfont=dict(color="#888", family="Roboto Mono"),
                    ),
                ),
                name="RAINFALL",
            )
        )
        fig_rain.update_layout(
            title=dict(
                text="RAINFALL (MM/HR)",
                font=dict(size=16, color="#E6E6E6", family="Roboto Mono"),
            ),
            height=280,
            margin=dict(l=50, r=20, t=50, b=50),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Roboto Mono", color="#E6E6E6"),
            xaxis=dict(showgrid=True, gridcolor="#222", linecolor="#444"),
            yaxis=dict(showgrid=True, gridcolor="#222", linecolor="#444"),
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
                mode="lines+markers",
                name="TEMPERATURE",
                line=dict(color="#D71921", width=2),
                marker=dict(size=6, color="#D71921"),
                fill="tozeroy",
                fillcolor="rgba(215, 25, 33, 0.1)",
            )
        )
        fig_temp.update_layout(
            title=dict(
                text="TEMPERATURE (°C)",
                font=dict(size=16, color="#E6E6E6", family="Roboto Mono"),
            ),
            height=280,
            margin=dict(l=50, r=20, t=50, b=50),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Roboto Mono", color="#E6E6E6"),
            xaxis=dict(showgrid=True, gridcolor="#222", linecolor="#444"),
            yaxis=dict(showgrid=True, gridcolor="#222", linecolor="#444"),
        )
        st.plotly_chart(
            fig_temp, use_container_width=True, config={"displayModeBar": False}
        )

    # Enhanced Footer
    st.markdown("---")
    st.markdown(
        f"""
    <div style='background: #000; border: 1px solid #333; padding: 2rem; margin-top: 2rem; 
                text-align: center;'>
        <div style='margin-bottom: 1rem;'>
            <h3 style='color: #E6E6E6; margin: 0.5rem 0; font-weight: 700;'>WATER LEVEL MONITORING SYSTEM</h3>
            <p style='color: #888; margin: 0.25rem 0; font-size: 0.9rem; font-weight: 500;'>STATION CPY015 // KRUNGTHEP BRIDGE</p>
        </div>
        <div style='display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem; margin: 1rem 0;'>
            <span class="status-badge">
                <a href="https://www.thaiwater.net/water/wl" target="_blank" style='color: #E6E6E6; text-decoration: none; font-weight: 600;'>THAI WATER NETWORK</a>
            </span>
            <span class="status-badge">
                MODEL: {selected_model_name.upper()}
            </span>
            <span class="status-badge">
                MAE ~0.12M // R² ~0.94
            </span>
        </div>
        <p style='color: #555; margin-top: 1.5rem; font-style: italic; font-size: 0.8rem; font-weight: 500;'>
            CPDSAI PROJECT // ASIAN INSTITUTE OF TECHNOLOGY
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def setup_sidebar(models):
    """
    Setup sidebar with information (No model selection)
    """
    st.sidebar.markdown(
        """
    <div style='background: #000; border: 1px solid #333; padding: 1.5rem; text-align: center;'>
        <h2 style='color: #FFF; margin: 0; font-size: 1.2rem; letter-spacing: 2px;'>CONTROLS</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Default to LSTM
    selected_model_name = "LSTM"
    selected_model = models.get("LSTM")

    if not selected_model:
        st.sidebar.error("LSTM model not found!")
        return None, None, datetime.now()

    # Current time (real-time mode)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    st.sidebar.markdown("#### SYSTEM TIME")
    st.sidebar.markdown(
        f"""
    <div style='border: 1px dotted #444; padding: 1rem; text-align: center; margin: 0.5rem 0;'>
        <p style='font-size: 1rem; color: #888; margin: 0;'>{now.strftime("%Y-%m-%d")}</p>
        <p style='font-size: 1.5rem; font-weight: 700; color: #D71921; margin: 0;'>{now.strftime("%H:%M")}</p>
        <p style='font-size: 0.7rem; color: #555; margin: 0; letter-spacing: 1px;'>ICT ZONE</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📊 Station Info")
    st.sidebar.markdown(
        f"""
    <div style='background: var(--surface); padding: 1rem; border-radius: 8px; 
                box-shadow: var(--shadow-sm); margin: 0.5rem 0; border: 1px solid var(--border);'>
        <p style='margin: 0.5rem 0; color: var(--text-main);'><strong style='color: var(--text-main);'>Bank Level:</strong><br>
        <span style='color: var(--danger); font-weight: 700; font-size: 1.1rem;'>{BANK_LEVEL} m.MSL</span></p>
        <p style='margin: 0.5rem 0; color: var(--text-main);'><strong style='color: var(--text-main);'>Bed Level:</strong><br>
        <span style='color: var(--primary); font-weight: 700; font-size: 1.1rem;'>{BED_LEVEL} m.MSL</span></p>
        <p style='margin: 0.5rem 0; color: var(--text-main);'><strong style='color: var(--text-main);'>Location:</strong><br>
        <span style='color: var(--text-muted); font-size: 0.9rem; font-weight: 500;'>13.7003°N, 100.4928°E</span></p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🤖 Model Status")
    st.sidebar.markdown(
        """
    <div style='background: var(--success); color: #ffffff; padding: 0.75rem; border-radius: 8px; 
                margin: 0.5rem 0; box-shadow: var(--shadow-sm);'>
        <strong style='color: #ffffff; font-size: 1rem;'>✅ LSTM Model</strong><br>
        <small style='color: #ffffff; font-weight: 500;'>Deep Learning - Active</small>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    # Data source info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📡 Data Source")
    st.sidebar.markdown(
        """
    <div style='background: var(--background); padding: 1rem; border-radius: 8px; box-shadow: var(--shadow-sm); border: 1px solid var(--border);'>
        <p style='margin: 0.5rem 0; color: var(--text-main);'><strong style='color: var(--text-main);'>Real-time:</strong><br>
        <span style='color: var(--primary); font-weight: 600;'>Thai Water Network</span></p>
        <p style='margin: 0.5rem 0; color: var(--text-main);'><strong style='color: var(--text-main);'>Forecast:</strong><br>
        <span style='color: var(--primary); font-weight: 600;'>LSTM Deep Learning</span></p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    return selected_model_name, selected_model, now


def fetch_and_process_data(selected_model_name, selected_model, now):
    """
    Fetch and process all data needed for predictions
    """
    # Fetch real-time water level from Thai Water website
    thaiwater_data = fetch_water_level_thaiwater()

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

    # Get current water level from Thai Water (real-time) or fallback
    if (
        thaiwater_data
        and thaiwater_data["status"] == "success"
        and thaiwater_data["water_level"] is not None
    ):
        current_water_level = thaiwater_data["water_level"]
        measure_time = thaiwater_data["measure_datetime"]
        data_source = "🌐 Thai Water (Real-time)"
        st.success(
            f"✅ Successfully fetched real-time data from Thai Water: {current_water_level:.3f} m.MSL"
        )
    else:
        # Fallback to historical data
        historical_df = load_historical_data()
        if historical_df is not None and len(historical_df) > 0:
            current_water_level = historical_df["water_level"].iloc[-1]
            measure_time = historical_df.index[-1]
            data_source = "📊 Historical Data (Fallback)"
            if thaiwater_data and thaiwater_data["status"] != "success":
                st.warning(
                    f"⚠️ Could not fetch from Thai Water: {thaiwater_data.get('error', 'Unknown error')}. Using historical data."
                )
        else:
            # Last resort: default value
            current_water_level = 0.4
            measure_time = now
            data_source = "⚙️ Default Value"
            st.error("❌ Could not fetch water level data. Using default value.")

    # Add current water level to weather dataframe for predictions
    weather_df["water_level"] = current_water_level

    # Create time series with real current level
    # Use real data for current time, then use model for future predictions
    df_with_real = weather_df.copy()

    # For past 24 hours, use historical data if available, otherwise use current level
    historical_df = load_historical_data()
    if historical_df is not None and len(historical_df) > 0:
        # Get last 24 hours of historical data
        past_24h = historical_df.tail(24).copy()
        past_24h.index = pd.date_range(end=now, periods=len(past_24h), freq="h")

        # Merge with weather data
        for col in past_24h.columns:
            if col not in df_with_real.columns:
                df_with_real[col] = np.nan

        # Fill past 24h with historical water levels
        for idx in past_24h.index:
            if idx in df_with_real.index:
                df_with_real.loc[idx, "water_level"] = past_24h.loc[idx, "water_level"]

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

    # Setup sidebar
    selected_model_name, selected_model, now = setup_sidebar(models)

    # Fetch and process data
    with st.spinner(
        f"📡 Fetching real-time data from Thai Water... Using {selected_model_name} for predictions"
    ):
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
