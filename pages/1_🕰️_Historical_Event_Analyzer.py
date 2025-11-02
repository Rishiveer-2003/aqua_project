import json
import os
import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
import openmeteo_requests
import requests_cache
from retry_requests import retry

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Historical Flood Risk Analysis",
    page_icon="🕰️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# --- MODEL AND DATA LOADING ---
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

models = load_models()
feature_columns = load_feature_columns()

# --- CITY PROFILES (copied from India Live page) ---
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
    },
    "Goa": {
        'TopographyDrainage': 8, 'RiverManagement': 9, 'Deforestation': 12, 'Urbanization': 11,
        'ClimateChange': 14, 'DamsQuality': 9, 'Siltation': 10, 'AgriculturalPractices': 7,
        'Encroachments': 12, 'IneffectiveDisasterPreparedness': 10, 'DrainageSystems': 8,
        'CoastalVulnerability': 18, 'Landslides': 10, 'Watersheds': 8,
        'DeterioratingWaterQuality': 11, 'PopulationScore': 12, 'WetlandLoss': 13,
        'InadequatePlanning': 11, 'PoliticalFactors': 9
    },
    "Navi Mumbai": {
        'TopographyDrainage': 5, 'RiverManagement': 7, 'Deforestation': 11, 'Urbanization': 16,
        'ClimateChange': 13, 'DamsQuality': 10, 'Siltation': 12, 'AgriculturalPractices': 3,
        'Encroachments': 13, 'IneffectiveDisasterPreparedness': 10, 'DrainageSystems': 8,
        'CoastalVulnerability': 17, 'Landslides': 6, 'Watersheds': 7,
        'DeterioratingWaterQuality': 12, 'PopulationScore': 17, 'WetlandLoss': 14,
        'InadequatePlanning': 10, 'PoliticalFactors': 9
    }
}

# --- HELPERS ---
@st.cache_data(show_spinner=False)
def get_coords(city_name: str):
    geolocator = Nominatim(user_agent="aqua_historical_app")
    try:
        location = geolocator.geocode(city_name)
        if location:
            return float(location.latitude), float(location.longitude)
    except Exception:
        pass
    return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def get_historical_rainfall(lat: float, lon: float, date: datetime.date):
    """Fetch historical daily rainfall (mm) for a specific date using Open-Meteo Archive API."""
    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        session = retry(cache_session, retries=5, backoff_factor=0.2)
        client = openmeteo_requests.Client(session=session)

        url = "https://archive-api.open-meteo.com/v1/archive"
        ds = date.strftime('%Y-%m-%d')
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": ds,
            "end_date": ds,
            "daily": "precipitation_sum",
            "timezone": "auto",
        }

        responses = client.weather_api(url, params=params)
        response = responses[0] if isinstance(responses, (list, tuple)) else responses
        daily = response.Daily()
        precip = daily.Variables(0).ValuesAsNumpy()
        if getattr(precip, 'size', 0) > 0:
            return float(precip[0])
        return None
    except Exception as e:
        st.error(f"Historical weather fetch failed: {e}")
        return None

def map_rainfall_to_intensity(rainfall_mm: float) -> int:
    if rainfall_mm <= 10: return 2
    elif rainfall_mm <= 25: return 4
    elif rainfall_mm <= 50: return 6
    elif rainfall_mm <= 75: return 8
    elif rainfall_mm <= 100: return 10
    elif rainfall_mm <= 150: return 12
    elif rainfall_mm <= 200: return 14
    else: return 16

def predict_prob(mdl, X: pd.DataFrame):
    if hasattr(mdl, 'predict_proba'):
        vals = mdl.predict_proba(X)[:, 1]
    else:
        vals = mdl.predict(X)
    arr = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)
    return arr

# --- USER INTERFACE ---
st.title('🕰️ Historical Event Analyzer')
st.write("Select a city and a past date to analyze what the flood risk prediction would have been based on the recorded rainfall for that day.")

# Sidebar controls
st.sidebar.header("Analysis Controls")
_cities = list(CITY_PROFILES.keys())
city_selection = st.sidebar.selectbox("Choose a city", _cities, index=0)

# Open-Meteo historical data typically has ~5 day delay
max_date = datetime.date.today() - datetime.timedelta(days=5)
selected_date = st.sidebar.date_input(
    "Select a date for analysis",
    value=max_date,
    min_value=datetime.date(1940, 1, 1),
    max_value=max_date
)

model_options = ['Ensemble (Average)'] + list(models.keys()) if len(models) > 1 else list(models.keys())
model_choice = st.sidebar.selectbox("Select Model for Analysis", model_options)

if st.button(f"Analyze Risk for {city_selection} on {selected_date.strftime('%Y-%m-%d')}", use_container_width=True):
    lat, lon = get_coords(city_selection)
    if lat is None:
        st.error('Could not geocode the selected city.')
        st.stop()

    with st.spinner(f'Fetching historical rainfall for {city_selection}...'):
        rainfall_mm = get_historical_rainfall(lat, lon, selected_date)

    if rainfall_mm is None:
        st.error('No rainfall data available for the selected date. Please try another date.')
        st.stop()

    st.success(f"Recorded rainfall on {selected_date.strftime('%Y-%m-%d')}: {rainfall_mm:.2f} mm (Open‑Meteo Archive)")

    # Build model input from city profile + derived MonsoonIntensity
    monsoon_intensity = map_rainfall_to_intensity(rainfall_mm)
    base = CITY_PROFILES[city_selection].copy()
    base['MonsoonIntensity'] = monsoon_intensity
    input_df = pd.DataFrame([base])

    # Align to feature columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    # Predict
    if model_choice == 'Ensemble (Average)' and len(models) > 1:
        preds = []
        for name, mdl in models.items():
            try:
                preds.append(predict_prob(mdl, input_df))
            except Exception:
                pass
        if not preds:
            st.error('No models available to generate predictions.')
            st.stop()
        final_arr = np.mean(np.vstack(preds), axis=0)
    else:
        if model_choice not in models:
            st.error('Selected model is not available.')
            st.stop()
        final_arr = predict_prob(models[model_choice], input_df)

    final_proba = float(np.clip(final_arr[0], 0.0, 1.0))

    # Display
    st.header('Retrospective Flood Risk Analysis')
    st.metric("Predicted Risk", f"{final_proba:.1%}")

    if final_proba < 0.45:
        st.success("CONCLUSION: LOW RISK")
    elif final_proba < 0.55:
        st.warning("CONCLUSION: MODERATE RISK")
    else:
        st.error("CONCLUSION: HIGH RISK")

    st.info("This prediction is based on the city's static risk profile combined with the actual rainfall recorded on the selected date.")
