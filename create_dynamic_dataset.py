"""
Dynamic Flood Dataset Generator
================================
This script generates a time-series flood risk dataset by:
1. Fetching historical rainfall data for all cities in CITY_PROFILES
2. Creating 7-day rolling window features (rain_lag_1 through rain_lag_7)
3. Using the ensemble of trained models to generate proxy flood probabilities
4. Applying baseline calibration to isolate rainfall-induced risk

Output: dynamic_flood_data.csv suitable for LSTM training
"""

import json
import os
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests_cache
from geopy.geocoders import Nominatim
from retry_requests import retry
import openmeteo_requests

# ==============================================================================
# SECTION 1: HELPER FUNCTIONS PORTED FROM app.py
# ==============================================================================

BASE_DIR = os.path.dirname(__file__)

def load_models():
    """Load all 5 trained ML models from .pkl files"""
    models = {}
    model_files = {
        'LightGBM': 'lgbm_model.pkl',
        'RandomForest': 'rf_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'SVR': 'svr_model.pkl',
        'KNN': 'knn_model.pkl',
    }
    
    for name, filename in model_files.items():
        filepath = os.path.join(BASE_DIR, filename)
        if os.path.exists(filepath):
            try:
                models[name] = joblib.load(filepath)
                print(f"✓ Successfully loaded {name} from {filename}")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
        else:
            print(f"✗ Warning: Model file {filename} not found.")
    
    return models

def load_feature_columns():
    """Load the feature column order from JSON"""
    filepath = os.path.join(BASE_DIR, 'feature_columns.json')
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            feature_columns = json.load(f)
            print(f"✓ Loaded {len(feature_columns)} feature columns from feature_columns.json")
            return feature_columns
    except FileNotFoundError:
        print("✗ FATAL: feature_columns.json not found.")
        return None
    except Exception as e:
        print(f"✗ Error loading feature_columns.json: {e}")
        return None

def get_coords(city_name: str):
    """Geocode a city name to latitude/longitude using Nominatim"""
    geolocator = Nominatim(user_agent="aqua_project_dataset_generator")
    try:
        location = geolocator.geocode(city_name)
        if location:
            return float(location.latitude), float(location.longitude)
    except Exception as e:
        print(f"✗ Geocoding error for {city_name}: {e}")
    return None, None

def map_rainfall_to_intensity(rainfall_mm: float) -> int:
    """Map rainfall amount (mm) to MonsoonIntensity feature scale [0..16]"""
    if rainfall_mm <= 10: return 2
    elif rainfall_mm <= 25: return 4
    elif rainfall_mm <= 50: return 6
    elif rainfall_mm <= 75: return 8
    elif rainfall_mm <= 100: return 10
    elif rainfall_mm <= 150: return 12
    elif rainfall_mm <= 200: return 14
    else: return 16

# === CITY_PROFILES: 32 Indian cities with static risk factors ===
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

# ==============================================================================
# SECTION 2: NEW HISTORICAL RAINFALL FETCH FUNCTION
# ==============================================================================

def get_historical_rainfall_range(lat, lon, start_date, end_date):
    """
    Fetches daily rainfall data for a given lat/lon and date range.
    Uses Open-Meteo Archive API (1940-present).
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with columns ['date', 'rainfall'] or None if error
    """
    try:
        # Use a cache to avoid re-downloading data
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "precipitation_sum",
            "timezone": "auto"
        }
        
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0] if isinstance(responses, list) else responses
        
        daily = response.Daily()
        daily_precipitation_sum = daily.Variables(0).ValuesAsNumpy()
        
        dates = pd.to_datetime(daily.Time(), unit="s")
        
        if len(dates) == 0:
            print(f"  ✗ No data returned for coordinates ({lat}, {lon})")
            return None
        
        return pd.DataFrame({'date': dates, 'rainfall': daily_precipitation_sum})
    
    except Exception as e:
        print(f"  ✗ Error in get_historical_rainfall_range: {e}")
        return None

# ==============================================================================
# SECTION 3: MAIN SCRIPT LOGIC
# ==============================================================================

def predict_prob(model, X):
    """
    Helper function to get probability predictions from a model.
    Uses predict_proba if available, otherwise clips predict() output to [0,1].
    """
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    vals = model.predict(X)
    return np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)

if __name__ == "__main__":
    print("=" * 80)
    print("DYNAMIC FLOOD DATASET GENERATOR")
    print("=" * 80)
    print()
    
    # Load models and feature schema
    print("STEP 1: Loading ML models and feature schema...")
    print("-" * 80)
    models = load_models()
    feature_columns = load_feature_columns()
    
    if not models or not feature_columns:
        print("\n✗ FATAL ERROR: Could not load models or feature columns. Exiting.")
        exit(1)
    
    print(f"\n✓ Successfully loaded {len(models)} models.")
    print(f"✓ Feature columns: {len(feature_columns)} features")
    print()
    
    # Set date range for the past year
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print("STEP 2: Setting date range...")
    print("-" * 80)
    print(f"Start Date: {start_date}")
    print(f"End Date:   {end_date}")
    print(f"Duration:   365 days")
    print()
    
    # Process each city
    print("STEP 3: Processing cities...")
    print("-" * 80)
    print(f"Total cities to process: {len(CITY_PROFILES)}")
    print()
    
    all_city_data = []
    
    for idx, (city, profile) in enumerate(CITY_PROFILES.items(), 1):
        print(f"[{idx}/{len(CITY_PROFILES)}] Processing {city}...")
        
        # Get coordinates
        lat, lon = get_coords(city)
        if not lat or not lon:
            print(f"  ✗ Could not get coordinates for {city}. Skipping.")
            print()
            continue
        print(f"  ✓ Coordinates: ({lat:.4f}, {lon:.4f})")
        
        # Fetch historical rainfall
        rainfall_df = get_historical_rainfall_range(lat, lon, start_date, end_date)
        if rainfall_df is None or rainfall_df.empty:
            print(f"  ✗ Could not get rainfall data for {city}. Skipping.")
            print()
            continue
        print(f"  ✓ Fetched {len(rainfall_df)} days of rainfall data")
        
        # --- Time-Series Feature Engineering ---
        # Create rolling features for the past 7 days of rainfall
        print("  → Creating 7-day lag features...")
        for i in range(1, 8):
            rainfall_df[f'rain_lag_{i}'] = rainfall_df['rainfall'].shift(i)
        
        # Drop rows with NaN values (the first 7 days)
        rainfall_df.dropna(inplace=True)
        
        if rainfall_df.empty:
            print(f"  ✗ No data left for {city} after lag creation. Skipping.")
            print()
            continue
        print(f"  ✓ Valid data rows after lag creation: {len(rainfall_df)}")
        
        # --- Proxy Target Generation ---
        # Use the ensemble to "score" the flood risk for each day
        print("  → Generating ensemble predictions...")
        daily_predictions = []
        
        for index, row in rainfall_df.iterrows():
            # 1. Map rainfall to the 'MonsoonIntensity' feature
            monsoon_intensity = map_rainfall_to_intensity(row['rainfall'])
            
            # 2. Create the base input data from the city's static profile
            input_data = profile.copy()
            
            # 3. Create the "baseline" (0 rain) input DataFrame
            baseline_input = input_data.copy()
            baseline_input['MonsoonIntensity'] = 0  # 0 rain intensity
            # Ensure column order matches training!
            baseline_df = pd.DataFrame([baseline_input])[feature_columns]
            
            # 4. Create the "actual" (today's rain) input DataFrame
            actual_input = input_data.copy()
            actual_input['MonsoonIntensity'] = monsoon_intensity  # Today's rain intensity
            # Ensure column order matches training!
            actual_df = pd.DataFrame([actual_input])[feature_columns]
            
            # 5. Get ensemble predictions for both
            base_preds = [predict_prob(model, baseline_df)[0] for model in models.values()]
            baseline_risk = np.mean(base_preds)
            
            actual_preds = [predict_prob(model, actual_df)[0] for model in models.values()]
            actual_risk = np.mean(actual_preds)
            
            # 6. Calibrate: The "proxy" risk is the *additional* risk from the rain
            calibrated_risk = max(0.0, actual_risk - baseline_risk)
            daily_predictions.append(calibrated_risk)
        
        rainfall_df['FloodProbability_Proxy'] = daily_predictions
        rainfall_df['city'] = city
        all_city_data.append(rainfall_df)
        
        print(f"  ✓ Generated {len(daily_predictions)} proxy predictions")
        print(f"  ✓ Finished processing {city}")
        print()
        
        # Pause to be respectful to the API
        time.sleep(2)
    
    # --- Final Output ---
    print("=" * 80)
    print("STEP 4: Saving dataset...")
    print("-" * 80)
    
    if not all_city_data:
        print("✗ No data was processed. Exiting.")
        exit(1)
    
    final_df = pd.concat(all_city_data, ignore_index=True)
    output_filename = 'dynamic_flood_data.csv'
    final_df.to_csv(output_filename, index=False)
    
    print(f"✓ Dynamic dataset created successfully: '{output_filename}'")
    print(f"✓ Total rows generated: {len(final_df):,}")
    print(f"✓ Total cities processed: {len(all_city_data)}")
    print(f"✓ Date range: {start_date} to {end_date}")
    print()
    
    # Display sample statistics
    print("Dataset Statistics:")
    print("-" * 80)
    print(final_df[['rainfall', 'FloodProbability_Proxy']].describe())
    print()
    
    print("Columns in output file:")
    print("-" * 80)
    print(list(final_df.columns))
    print()
    
    print("=" * 80)
    print("✓ DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("1. Inspect the output file: dynamic_flood_data.csv")
    print("2. Use the 7-day lag features (rain_lag_1 to rain_lag_7) as LSTM inputs")
    print("3. Use FloodProbability_Proxy as the target variable")
    print("4. Train an LSTM model to capture temporal rainfall patterns")
    print()
