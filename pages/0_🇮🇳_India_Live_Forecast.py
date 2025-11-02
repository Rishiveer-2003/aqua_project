import json
import os

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from geopy.geocoders import Nominatim
import openmeteo_requests
import requests_cache
from retry_requests import retry

# --- Page Configuration ---
st.set_page_config(
    page_title="India Live Forecast",
    page_icon="🇮🇳",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    paths = {
        'LightGBM': os.path.join(BASE_DIR, 'lgbm_model.pkl'),
        'RandomForest': os.path.join(BASE_DIR, 'rf_model.pkl'),
        'XGBoost': os.path.join(BASE_DIR, 'xgboost_model.pkl'),
        'SVR': os.path.join(BASE_DIR, 'svr_model.pkl'),
        'KNN': os.path.join(BASE_DIR, 'knn_model.pkl'),
    }
    for name, path in paths.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception:
                pass
    return models

@st.cache_resource(show_spinner=False)
def load_feature_columns():
    path = os.path.join(BASE_DIR, 'feature_columns.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def get_coords(city_name: str):
    geolocator = Nominatim(user_agent="aqua_flood_app")
    try:
        location = geolocator.geocode(city_name)
        if location:
            return float(location.latitude), float(location.longitude)
    except Exception as e:
        st.error(f"Geocoding error: {e}")
    return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def get_rainfall_forecast(lat: float, lon: float):
    """Return tomorrow's total precipitation (mm) using Open-Meteo daily precipitation_sum."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=session)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum",
        "forecast_days": 2,
    }
    responses = client.weather_api(url, params=params)
    response = responses
    daily = response.Daily()
    precip = daily.Variables(0).ValuesAsNumpy()
    return float(precip[1])

# Map rainfall (mm) to MonsoonIntensity feature scale [0..16]
def map_rainfall_to_intensity(rainfall_mm: float) -> int:
    if rainfall_mm <= 10: return 2
    elif rainfall_mm <= 25: return 4
    elif rainfall_mm <= 50: return 6
    elif rainfall_mm <= 75: return 8
    elif rainfall_mm <= 100: return 10
    elif rainfall_mm <= 150: return 12
    elif rainfall_mm <= 200: return 14
    else: return 16

# === START: NEW CITY_PROFILES DICTIONARY ===
CITY_PROFILES = {
    "Mumbai": {
        'TopographyDrainage': 4, 'RiverManagement': 5, 'Deforestation': 12, 'Urbanization': 17,
        'ClimateChange': 13, 'DamsQuality': 9, 'Siltation': 14, 'AgriculturalPractices': 2,
        'Encroachments': 16, 'IneffectiveDisasterPreparedness': 11, 'DrainageSystems': 5,
        'CoastalVulnerability': 17, 'Landslides': 7, 'Watersheds': 6,
        'DeterioratingWaterQuality': 13, 'PopulationScore': 18, 'WetlandLoss': 15,
        'InadequatePlanning': 14, 'PoliticalFactors': 10
    },
    "Kolkata": {
        'TopographyDrainage': 3, 'RiverManagement': 6, 'Deforestation': 11, 'Urbanization': 16,
        'ClimateChange': 14, 'DamsQuality': 8, 'Siltation': 15, 'AgriculturalPractices': 4,
        'Encroachments': 15, 'IneffectiveDisasterPreparedness': 12, 'DrainageSystems': 6,
        'CoastalVulnerability': 16, 'Landslides': 2, 'Watersheds': 5,
        'DeterioratingWaterQuality': 14, 'PopulationScore': 17, 'WetlandLoss': 16,
        'InadequatePlanning': 13, 'PoliticalFactors': 11
    },
    "Chennai": {
        'TopographyDrainage': 5, 'RiverManagement': 7, 'Deforestation': 10, 'Urbanization': 15,
        'ClimateChange': 15, 'DamsQuality': 10, 'Siltation': 13, 'AgriculturalPractices': 5,
        'Encroachments': 14, 'IneffectiveDisasterPreparedness': 10, 'DrainageSystems': 7,
        'CoastalVulnerability': 18, 'Landslides': 3, 'Watersheds': 7,
        'DeterioratingWaterQuality': 12, 'PopulationScore': 16, 'WetlandLoss': 14,
        'InadequatePlanning': 12, 'PoliticalFactors': 9
    },
    "Surat": {
        'TopographyDrainage': 6, 'RiverManagement': 8, 'Deforestation': 9, 'Urbanization': 14,
        'ClimateChange': 12, 'DamsQuality': 11, 'Siltation': 12, 'AgriculturalPractices': 7,
        'Encroachments': 13, 'IneffectiveDisasterPreparedness': 13, 'DrainageSystems': 9,
        'CoastalVulnerability': 15, 'Landslides': 4, 'Watersheds': 8,
        'DeterioratingWaterQuality': 10, 'PopulationScore': 15, 'WetlandLoss': 12,
        'InadequatePlanning': 10, 'PoliticalFactors': 8
    },
    "Patna": {
        'TopographyDrainage': 3, 'RiverManagement': 4, 'Deforestation': 13, 'Urbanization': 13,
        'ClimateChange': 13, 'DamsQuality': 7, 'Siltation': 16, 'AgriculturalPractices': 10,
        'Encroachments': 17, 'IneffectiveDisasterPreparedness': 9, 'DrainageSystems': 4,
        'CoastalVulnerability': 2, 'Landslides': 2, 'Watersheds': 4,
        'DeterioratingWaterQuality': 15, 'PopulationScore': 16, 'WetlandLoss': 13,
        'InadequatePlanning': 15, 'PoliticalFactors': 12
    },
    "Guwahati": {
        'TopographyDrainage': 5, 'RiverManagement': 5, 'Deforestation': 14, 'Urbanization': 12,
        'ClimateChange': 14, 'DamsQuality': 8, 'Siltation': 15, 'AgriculturalPractices': 8,
        'Encroachments': 15, 'IneffectiveDisasterPreparedness': 8, 'DrainageSystems': 5,
        'CoastalVulnerability': 1, 'Landslides': 12, 'Watersheds': 5,
        'DeterioratingWaterQuality': 13, 'PopulationScore': 14, 'WetlandLoss': 14,
        'InadequatePlanning': 14, 'PoliticalFactors': 11
    },
    "Delhi": {
        'TopographyDrainage': 7, 'RiverManagement': 7, 'Deforestation': 15, 'Urbanization': 18,
        'ClimateChange': 14, 'DamsQuality': 10, 'Siltation': 13, 'AgriculturalPractices': 6,
        'Encroachments': 16, 'IneffectiveDisasterPreparedness': 12, 'DrainageSystems': 8,
        'CoastalVulnerability': 1, 'Landslides': 3, 'Watersheds': 7,
        'DeterioratingWaterQuality': 16, 'PopulationScore': 19, 'WetlandLoss': 17,
        'InadequatePlanning': 13, 'PoliticalFactors': 13
    },
    "Bengaluru": {
        'TopographyDrainage': 12, 'RiverManagement': 9, 'Deforestation': 16, 'Urbanization': 17,
        'ClimateChange': 11, 'DamsQuality': 12, 'Siltation': 10, 'AgriculturalPractices': 4,
        'Encroachments': 18, 'IneffectiveDisasterPreparedness': 14, 'DrainageSystems': 7,
        'CoastalVulnerability': 1, 'Landslides': 2, 'Watersheds': 9,
        'DeterioratingWaterQuality': 11, 'PopulationScore': 17, 'WetlandLoss': 18,
        'InadequatePlanning': 16, 'PoliticalFactors': 9
    },
    "Hyderabad": {
        'TopographyDrainage': 10, 'RiverManagement': 8, 'Deforestation': 13, 'Urbanization': 16,
        'ClimateChange': 12, 'DamsQuality': 11, 'Siltation': 11, 'AgriculturalPractices': 7,
        'Encroachments': 15, 'IneffectiveDisasterPreparedness': 13, 'DrainageSystems': 8,
        'CoastalVulnerability': 1, 'Landslides': 5, 'Watersheds': 8,
        'DeterioratingWaterQuality': 12, 'PopulationScore': 16, 'WetlandLoss': 16,
        'InadequatePlanning': 11, 'PoliticalFactors': 10
    },
    "Ahmedabad": {
        'TopographyDrainage': 8, 'RiverManagement': 10, 'Deforestation': 8, 'Urbanization': 15,
        'ClimateChange': 11, 'DamsQuality': 13, 'Siltation': 9, 'AgriculturalPractices': 9,
        'Encroachments': 12, 'IneffectiveDisasterPreparedness': 14, 'DrainageSystems': 10,
        'CoastalVulnerability': 3, 'Landslides': 1, 'Watersheds': 10,
        'DeterioratingWaterQuality': 10, 'PopulationScore': 16, 'WetlandLoss': 11,
        'InadequatePlanning': 9, 'PoliticalFactors': 7
    },
    "Pune": {
        'TopographyDrainage': 14, 'RiverManagement': 11, 'Deforestation': 11, 'Urbanization': 16,
        'ClimateChange': 10, 'DamsQuality': 14, 'Siltation': 8, 'AgriculturalPractices': 6,
        'Encroachments': 14, 'IneffectiveDisasterPreparedness': 15, 'DrainageSystems': 9,
        'CoastalVulnerability': 1, 'Landslides': 10, 'Watersheds': 12,
        'DeterioratingWaterQuality': 9, 'PopulationScore': 15, 'WetlandLoss': 13,
        'InadequatePlanning': 10, 'PoliticalFactors': 8
    },
    "Jaipur": {
        'TopographyDrainage': 13, 'RiverManagement': 7, 'Deforestation': 7, 'Urbanization': 13,
        'ClimateChange': 10, 'DamsQuality': 9, 'Siltation': 7, 'AgriculturalPractices': 11,
        'Encroachments': 11, 'IneffectiveDisasterPreparedness': 11, 'DrainageSystems': 7,
        'CoastalVulnerability': 1, 'Landslides': 4, 'Watersheds': 9,
        'DeterioratingWaterQuality': 8, 'PopulationScore': 14, 'WetlandLoss': 10,
        'InadequatePlanning': 12, 'PoliticalFactors': 9
    },
    "Lucknow": {
        'TopographyDrainage': 6, 'RiverManagement': 6, 'Deforestation': 9, 'Urbanization': 14,
        'ClimateChange': 11, 'DamsQuality': 8, 'Siltation': 12, 'AgriculturalPractices': 12,
        'Encroachments': 13, 'IneffectiveDisasterPreparedness': 10, 'DrainageSystems': 6,
        'CoastalVulnerability': 1, 'Landslides': 1, 'Watersheds': 6,
        'DeterioratingWaterQuality': 11, 'PopulationScore': 15, 'WetlandLoss': 12,
        'InadequatePlanning': 13, 'PoliticalFactors': 11
    },
    "Kanpur": {
        'TopographyDrainage': 5, 'RiverManagement': 5, 'Deforestation': 10, 'Urbanization': 15,
        'ClimateChange': 12, 'DamsQuality': 7, 'Siltation': 14, 'AgriculturalPractices': 13,
        'Encroachments': 14, 'IneffectiveDisasterPreparedness': 9, 'DrainageSystems': 5,
        'CoastalVulnerability': 1, 'Landslides': 1, 'Watersheds': 5,
        'DeterioratingWaterQuality': 17, 'PopulationScore': 16, 'WetlandLoss': 13,
        'InadequatePlanning': 14, 'PoliticalFactors': 12
    },
    "Nagpur": {
        'TopographyDrainage': 11, 'RiverManagement': 9, 'Deforestation': 8, 'Urbanization': 13,
        'ClimateChange': 10, 'DamsQuality': 12, 'Siltation': 7, 'AgriculturalPractices': 10,
        'Encroachments': 10, 'IneffectiveDisasterPreparedness': 12, 'DrainageSystems': 9,
        'CoastalVulnerability': 1, 'Landslides': 3, 'Watersheds': 11,
        'DeterioratingWaterQuality': 9, 'PopulationScore': 14, 'WetlandLoss': 9,
        'InadequatePlanning': 8, 'PoliticalFactors': 7
    },
    "Indore": {
        'TopographyDrainage': 13, 'RiverManagement': 8, 'Deforestation': 7, 'Urbanization': 14,
        'ClimateChange': 9, 'DamsQuality': 11, 'Siltation': 6, 'AgriculturalPractices': 11,
        'Encroachments': 9, 'IneffectiveDisasterPreparedness': 13, 'DrainageSystems': 10,
        'CoastalVulnerability': 1, 'Landslides': 2, 'Watersheds': 10,
        'DeterioratingWaterQuality': 8, 'PopulationScore': 13, 'WetlandLoss': 8,
        'InadequatePlanning': 7, 'PoliticalFactors': 6
    },
    "Thane": {
        'TopographyDrainage': 6, 'RiverManagement': 7, 'Deforestation': 11, 'Urbanization': 16,
        'ClimateChange': 13, 'DamsQuality': 10, 'Siltation': 12, 'AgriculturalPractices': 3,
        'Encroachments': 15, 'IneffectiveDisasterPreparedness': 11, 'DrainageSystems': 7,
        'CoastalVulnerability': 16, 'Landslides': 8, 'Watersheds': 7,
        'DeterioratingWaterQuality': 12, 'PopulationScore': 17, 'WetlandLoss': 14,
        'InadequatePlanning': 13, 'PoliticalFactors': 9
    },
    "Bhopal": {
        'TopographyDrainage': 12, 'RiverManagement': 10, 'Deforestation': 6, 'Urbanization': 12,
        'ClimateChange': 10, 'DamsQuality': 13, 'Siltation': 5, 'AgriculturalPractices': 9,
        'Encroachments': 8, 'IneffectiveDisasterPreparedness': 12, 'DrainageSystems': 11,
        'CoastalVulnerability': 1, 'Landslides': 2, 'Watersheds': 13,
        'DeterioratingWaterQuality': 7, 'PopulationScore': 13, 'WetlandLoss': 7,
        'InadequatePlanning': 8, 'PoliticalFactors': 7
    },
    "Visakhapatnam": {
        'TopographyDrainage': 7, 'RiverManagement': 9, 'Deforestation': 9, 'Urbanization': 14,
        'ClimateChange': 14, 'DamsQuality': 11, 'Siltation': 10, 'AgriculturalPractices': 6,
        'Encroachments': 12, 'IneffectiveDisasterPreparedness': 12, 'DrainageSystems': 8,
        'CoastalVulnerability': 18, 'Landslides': 11, 'Watersheds': 8,
        'DeterioratingWaterQuality': 11, 'PopulationScore': 15, 'WetlandLoss': 11,
        'InadequatePlanning': 11, 'PoliticalFactors': 8
    },
    "Vadodara": {
        'TopographyDrainage': 9, 'RiverManagement': 9, 'Deforestation': 7, 'Urbanization': 13,
        'ClimateChange': 11, 'DamsQuality': 12, 'Siltation': 8, 'AgriculturalPractices': 10,
        'Encroachments': 11, 'IneffectiveDisasterPreparedness': 13, 'DrainageSystems': 9,
        'CoastalVulnerability': 2, 'Landslides': 1, 'Watersheds': 9,
        'DeterioratingWaterQuality': 9, 'PopulationScore': 14, 'WetlandLoss': 10,
        'InadequatePlanning': 9, 'PoliticalFactors': 7
    },
    "Ghaziabad": {
        'TopographyDrainage': 6, 'RiverManagement': 5, 'Deforestation': 11, 'Urbanization': 17,
        'ClimateChange': 13, 'DamsQuality': 8, 'Siltation': 13, 'AgriculturalPractices': 8,
        'Encroachments': 15, 'IneffectiveDisasterPreparedness': 10, 'DrainageSystems': 6,
        'CoastalVulnerability': 1, 'Landslides': 2, 'Watersheds': 6,
        'DeterioratingWaterQuality': 15, 'PopulationScore': 18, 'WetlandLoss': 14,
        'InadequatePlanning': 14, 'PoliticalFactors': 12
    },
    "Ludhiana": {
        'TopographyDrainage': 8, 'RiverManagement': 7, 'Deforestation': 6, 'Urbanization': 15,
        'ClimateChange': 10, 'DamsQuality': 9, 'Siltation': 9, 'AgriculturalPractices': 14,
        'Encroachments': 12, 'IneffectiveDisasterPreparedness': 11, 'DrainageSystems': 7,
        'CoastalVulnerability': 1, 'Landslides': 1, 'Watersheds': 8,
        'DeterioratingWaterQuality': 12, 'PopulationScore': 16, 'WetlandLoss': 9,
        'InadequatePlanning': 11, 'PoliticalFactors': 10
    },
    "Agra": {
        'TopographyDrainage': 7, 'RiverManagement': 4, 'Deforestation': 8, 'Urbanization': 14,
        'ClimateChange': 12, 'DamsQuality': 6, 'Siltation': 13, 'AgriculturalPractices': 11,
        'Encroachments': 13, 'IneffectiveDisasterPreparedness': 9, 'DrainageSystems': 6,
        'CoastalVulnerability': 1, 'Landslides': 2, 'Watersheds': 5,
        'DeterioratingWaterQuality': 16, 'PopulationScore': 15, 'WetlandLoss': 11,
        'InadequatePlanning': 13, 'PoliticalFactors': 11
    },
    "Nashik": {
        'TopographyDrainage': 13, 'RiverManagement': 10, 'Deforestation': 9, 'Urbanization': 13,
        'ClimateChange': 10, 'DamsQuality': 13, 'Siltation': 7, 'AgriculturalPractices': 9,
        'Encroachments': 11, 'IneffectiveDisasterPreparedness': 12, 'DrainageSystems': 9,
        'CoastalVulnerability': 1, 'Landslides': 9, 'Watersheds': 11,
        'DeterioratingWaterQuality': 8, 'PopulationScore': 14, 'WetlandLoss': 10,
        'InadequatePlanning': 9, 'PoliticalFactors': 8
    },
    "Faridabad": {
        'TopographyDrainage': 7, 'RiverManagement': 6, 'Deforestation': 10, 'Urbanization': 16,
        'ClimateChange': 13, 'DamsQuality': 8, 'Siltation': 12, 'AgriculturalPractices': 7,
        'Encroachments': 14, 'IneffectiveDisasterPreparedness': 10, 'DrainageSystems': 7,
        'CoastalVulnerability': 1, 'Landslides': 3, 'Watersheds': 6,
        'DeterioratingWaterQuality': 14, 'PopulationScore': 17, 'WetlandLoss': 13,
        'InadequatePlanning': 13, 'PoliticalFactors': 11
    },
    "Meerut": {
        'TopographyDrainage': 8, 'RiverManagement': 6, 'Deforestation': 8, 'Urbanization': 14,
        'ClimateChange': 11, 'DamsQuality': 7, 'Siltation': 11, 'AgriculturalPractices': 13,
        'Encroachments': 12, 'IneffectiveDisasterPreparedness': 9, 'DrainageSystems': 6,
        'CoastalVulnerability': 1, 'Landslides': 1, 'Watersheds': 7,
        'DeterioratingWaterQuality': 13, 'PopulationScore': 15, 'WetlandLoss': 10,
        'InadequatePlanning': 12, 'PoliticalFactors': 10
    },
    "Rajkot": {
        'TopographyDrainage': 10, 'RiverManagement': 8, 'Deforestation': 6, 'Urbanization': 13,
        'ClimateChange': 10, 'DamsQuality': 11, 'Siltation': 7, 'AgriculturalPractices': 12,
        'Encroachments': 10, 'IneffectiveDisasterPreparedness': 12, 'DrainageSystems': 9,
        'CoastalVulnerability': 4, 'Landslides': 2, 'Watersheds': 9,
        'DeterioratingWaterQuality': 9, 'PopulationScore': 14, 'WetlandLoss': 9,
        'InadequatePlanning': 8, 'PoliticalFactors': 7
    },
    "Varanasi": {
        'TopographyDrainage': 4, 'RiverManagement': 3, 'Deforestation': 11, 'Urbanization': 14,
        'ClimateChange': 13, 'DamsQuality': 6, 'Siltation': 17, 'AgriculturalPractices': 11,
        'Encroachments': 18, 'IneffectiveDisasterPreparedness': 8, 'DrainageSystems': 4,
        'CoastalVulnerability': 1, 'Landslides': 2, 'Watersheds': 3,
        'DeterioratingWaterQuality': 18, 'PopulationScore': 17, 'WetlandLoss': 15,
        'InadequatePlanning': 16, 'PoliticalFactors': 13
    },
    "Srinagar": {
        'TopographyDrainage': 9, 'RiverManagement': 8, 'Deforestation': 10, 'Urbanization': 11,
        'ClimateChange': 15, 'DamsQuality': 10, 'Siltation': 12, 'AgriculturalPractices': 7,
        'Encroachments': 13, 'IneffectiveDisasterPreparedness': 7, 'DrainageSystems': 7,
        'CoastalVulnerability': 1, 'Landslides': 14, 'Watersheds': 10,
        'DeterioratingWaterQuality': 11, 'PopulationScore': 12, 'WetlandLoss': 12,
        'InadequatePlanning': 13, 'PoliticalFactors': 14
    },
    "Aurangabad": {
        'TopographyDrainage': 12, 'RiverManagement': 7, 'Deforestation': 8, 'Urbanization': 12,
        'ClimateChange': 9, 'DamsQuality': 10, 'Siltation': 8, 'AgriculturalPractices': 10,
        'Encroachments': 9, 'IneffectiveDisasterPreparedness': 11, 'DrainageSystems': 8,
        'CoastalVulnerability': 1, 'Landslides': 5, 'Watersheds': 9,
        'DeterioratingWaterQuality': 10, 'PopulationScore': 13, 'WetlandLoss': 9,
        'InadequatePlanning': 10, 'PoliticalFactors': 8
    }
}
# === END: NEW CITY_PROFILES DICTIONARY ===

models = load_models()
feature_columns = load_feature_columns()

st.title('🇮🇳 India Live Flood Forecast')
st.write('Select a city to fetch tomorrow\'s rainfall via Open‑Meteo, map it to Monsoon Intensity,\nthen simulate a neighborhood heatmap using the city\'s risk profile and our ML ensemble.')

# Sidebar controls
st.sidebar.header('Forecast Controls')
city_selection = st.sidebar.selectbox("Choose a city:", list(CITY_PROFILES.keys()))
grid_size = st.sidebar.slider('Grid Size (N x N)', min_value=10, max_value=35, value=15, step=5)

if st.button(f'Get Forecast for {city_selection}', type='primary'):
    lat, lon = get_coords(city_selection)
    if lat is None:
        st.error('Could not geocode the selected city.')
        st.stop()

    with st.spinner(f'Fetching Open‑Meteo rainfall for {city_selection}...'):
        rainfall_mm = get_rainfall_forecast(lat, lon)

    st.success(f"Tomorrow rainfall in {city_selection}: {rainfall_mm:.1f} mm (Open‑Meteo)")
    monsoon_intensity = map_rainfall_to_intensity(rainfall_mm)

    # Build grid around city center
    lat_range = np.linspace(lat - 0.1, lat + 0.1, grid_size)
    lon_range = np.linspace(lon - 0.1, lon + 0.1, grid_size)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    grid_df = pd.DataFrame({'latitude': lat_grid.ravel(), 'longitude': lon_grid.ravel()})

    # Assemble prediction dataframe using city profile as baseline + MonsoonIntensity
    prediction_data = grid_df.copy()
    base = CITY_PROFILES[city_selection]
    for feature, value in base.items():
        prediction_data[feature] = value
    prediction_data['MonsoonIntensity'] = monsoon_intensity

    # Add slight intra-city variation to make map realistic
    rng = np.random.default_rng(42)
    vary_features = [
        'TopographyDrainage', 'DrainageSystems', 'PopulationScore', 'Urbanization', 'Encroachments'
    ]
    variation_factor = 0.15
    for feature in vary_features:
        if feature in prediction_data.columns and feature in base:
            noise = rng.normal(0.0, max(1.0, base[feature]) * variation_factor, len(prediction_data))
            prediction_data[feature] = np.clip(prediction_data[feature] + noise, 0, 20)

    # Ensure feature alignment
    model_input = prediction_data.copy()
    for col in feature_columns:
        if col not in model_input.columns:
            model_input[col] = 0
    model_input = model_input[feature_columns]

    # Predict with ensemble
    def predict_prob(mdl, X):
        if hasattr(mdl, 'predict_proba'):
            return mdl.predict_proba(X)[:, 1]
        vals = mdl.predict(X)
        return np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)

    preds_list = []
    for name, mdl in models.items():
        try:
            preds_list.append(predict_prob(mdl, model_input))
        except Exception:
            pass

    if not preds_list:
        st.error('No models available to generate predictions.')
        st.stop()

    ensemble_preds = np.mean(np.vstack(preds_list), axis=0)
    prediction_data['risk'] = ensemble_preds
    overall_risk = float(np.mean(ensemble_preds))

    st.header(f'Flood Risk Prediction for {city_selection}')
    if overall_risk >= 0.55:
        st.error(f"HIGH OVERALL RISK ({overall_risk:.1%})")
    elif overall_risk >= 0.45:
        st.warning(f"MODERATE OVERALL RISK ({overall_risk:.1%})")
    else:
        st.success(f"LOW OVERALL RISK ({overall_risk:.1%})")

    # Heatmap
    st.subheader("Simulated Risk Distribution Heatmap")
    layer = pdk.Layer(
        'HeatmapLayer',
        data=prediction_data,
        get_position='[longitude, latitude]',
        get_weight='risk',
        opacity=0.8,
        aggregation=pdk.types.String('MEAN'),
    )

    view_state = pdk.ViewState(
        latitude=float(prediction_data['latitude'].mean()),
        longitude=float(prediction_data['longitude'].mean()),
        zoom=10,
        pitch=45,
    )

    deck = pdk.Deck(
        map_style=None,  # no Mapbox key required
        initial_view_state=view_state,
        layers=[layer],
        tooltip={
            'html': '<b>Risk:</b> {risk}',
            'style': {'color': 'white'}
        },
    )
    st.pydeck_chart(deck)

    # Optional: allow download
    st.download_button(
        label='Download predictions CSV',
        data=prediction_data.to_csv(index=False).encode('utf-8'),
        file_name=f'{city_selection}_risk_grid.csv',
        mime='text/csv'
    )
