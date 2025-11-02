import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import shap
import openmeteo_requests
import requests_cache
from retry_requests import retry
from geopy.geocoders import Nominatim

# --- Page Configuration ---
st.set_page_config(
    page_title="Project AQUA: Flood Prediction Engine",
    layout="wide"
)

# --- Constants ---
FEATURE_RANGES = {
    'MonsoonIntensity': (0, 16, 8), 'TopographyDrainage': (0, 18, 9),
    'RiverManagement': (0, 16, 8), 'Deforestation': (0, 17, 8),
    'Urbanization': (0, 17, 9), 'ClimateChange': (0, 17, 8),
    'DamsQuality': (0, 16, 8), 'Siltation': (0, 16, 8),
    'AgriculturalPractices': (0, 16, 8), 'Encroachments': (0, 18, 9),
    'IneffectiveDisasterPreparedness': (0, 16, 8), 'DrainageSystems': (0, 17, 9),
    'CoastalVulnerability': (0, 17, 8), 'Landslides': (0, 16, 8),
    'Watersheds': (0, 16, 8), 'DeterioratingWaterQuality': (0, 17, 8),
    'PopulationScore': (0, 18, 9), 'WetlandLoss': (0, 20, 10),
    'InadequatePlanning': (0, 16, 8), 'PoliticalFactors': (0, 16, 8)
}

BASE_DIR = os.path.dirname(__file__)
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
    # The client can return a list of response objects; select the first
    response = responses[0] if isinstance(responses, (list, tuple)) else responses
    daily = response.Daily()
    precip = daily.Variables(0).ValuesAsNumpy()
    # Index 1 is tomorrow when present; fallback to first value
    val = precip[1] if getattr(precip, 'shape', None) and precip.shape[0] > 1 else precip[0]
    return float(val)

# Use st.cache_resource to load models only once
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
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # Fallback to current feature set order if json missing
    return list(FEATURE_RANGES.keys())

models = load_models()
feature_columns = load_feature_columns()

# --- Web App Interface ---
st.title('🌊 Project AQUA: Flood Prediction Engine')
st.write(
    """
    Welcome to the AQUA Flood Prediction Engine. This tool uses an ensemble of machine learning models to
    forecast flood probability. Adjust the sliders in the sidebar to create a hypothetical scenario and see the
    predicted flood risk.
    """
)

# --- Sidebar for User Input ---
st.sidebar.header('Adjust Scenario Features')

# Optional: quick live weather (tomorrow rainfall) helper
with st.sidebar.expander('Live Weather (optional)'):
    city_q = st.text_input('City name', placeholder='e.g., Mumbai')
    if st.button('Get Tomorrow Rainfall', key='btn_weather'):
        if not city_q.strip():
            st.warning('Please enter a city name.')
        else:
            latlon = get_coords(city_q.strip())
            if latlon[0] is None:
                st.error('Could not geocode that city. Try another one.')
            else:
                rain = get_rainfall_forecast(latlon[0], latlon[1])
                st.metric('Tomorrow Rainfall (mm)', f"{rain:.1f} mm")
                st.caption('Source: Open‑Meteo daily precipitation_sum')

def user_input_features():
    data = {}
    for feature, (min_val, max_val, default_val) in FEATURE_RANGES.items():
        # Display label formatting
        display_name = feature.replace('_', ' ').title()
        data[feature] = st.sidebar.slider(display_name, min_val, max_val, default_val)

    # Create single-row DataFrame
    df = pd.DataFrame([data])

    # Align to the training feature set: add missing as 0, drop extras, reorder
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df

input_df = user_input_features()

# --- Main Panel for Displaying Results ---
st.header('Prediction Results')
st.write('Current Input Parameters:')
st.dataframe(input_df, width='stretch')

if st.button('Predict Flood Risk', width='stretch'):
    # --- Prediction Logic (USP #1: Ensemble) ---
    # Get probability predictions from each model (supports classifier or regressor)
    def predict_probability(model, X):
        if hasattr(model, 'predict_proba'):
            return float(model.predict_proba(X)[:, 1].item())
        # Regressor path: predict and clip to [0,1]
        val = float(model.predict(X).item())
        return min(max(val, 0.0), 1.0)

    probs = []
    labels = []
    for name, model in models.items():
        proba = predict_probability(model, input_df)
        probs.append(proba)
        labels.append((f"{name} Prediction", proba))

    # Calculate the ensemble (average) probability
    ensemble_proba = sum(probs) / len(probs)

    # Display the results
    cols = st.columns(min(5, max(2, len(labels))))
    for i, (label, val) in enumerate(labels[:len(cols)]):
        cols[i].metric(label, f"{val:.2%}")
    st.metric("Final Ensemble Prediction", f"{ensemble_proba:.2%}", delta_color="off")

    st.progress(float(ensemble_proba))

    # --- Explainability (USP #2: XAI with SHAP) ---
    st.header('Why Did the Model Make This Prediction?')
    st.write(
        """
        This chart explains the prediction using SHAP (SHapley Additive exPlanations). It shows which factors had the
        biggest impact on the final score.
        - Red bars represent features that increased the flood risk.
        - Blue bars represent features that decreased the flood risk.
        The length of the bar shows the magnitude of the impact.
        """
    )

    try:
        # Prefer LightGBM, then XGBoost, then RandomForest for SHAP
        model_for_shap = models.get('LightGBM') or models.get('XGBoost') or models.get('RandomForest')
        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer.shap_values(input_df)

        # Binary classification can return list [neg_class, pos_class] or array; handle both
        expected_value = explainer.expected_value
        if isinstance(shap_values, list):
            # pick positive class (index 1 when available)
            shap_arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            if isinstance(expected_value, (list, tuple)):
                exp_val = expected_value[1] if len(expected_value) > 1 else expected_value[0]
            else:
                exp_val = expected_value
        else:
            shap_arr = shap_values
            if isinstance(expected_value, (list, tuple)):
                exp_val = expected_value[0]
            else:
                exp_val = expected_value

        # Use the first (and only) row
        shap_row = shap_arr[0]

        # Force plot via matplotlib backend for a single prediction
        fig = plt.figure(figsize=(10, 2.5))
        shap.force_plot(exp_val, shap_row, input_df.iloc[0], matplotlib=True, show=False)
        st.pyplot(fig, bbox_inches='tight', clear_figure=True)

    except Exception as e:
        st.info(f"SHAP explanation could not be generated: {e}")
        # Fallback: show feature values as a bar chart
        st.bar_chart(input_df.T.rename(columns={input_df.index[0]: 'Value'}))
