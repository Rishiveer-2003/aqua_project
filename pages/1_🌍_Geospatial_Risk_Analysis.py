import json
import os

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from geopy.geocoders import Nominatim

# --- Page Configuration ---
st.set_page_config(
    page_title="Geospatial Risk Analysis",
    page_icon="🌍",
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

# Feature ranges consistent with app.py
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

models = load_models()
feature_columns = load_feature_columns()

st.title('🌍 Geospatial Risk Analysis')
st.write(
    """
This page demonstrates how varying a single risk factor can create hotspots of vulnerability across a simulated city grid.
Use the controls to set baseline values and vary one feature from West → East to see a dynamic heatmap.
    """
)

@st.cache_data(show_spinner=False)
def create_grid(grid_size: int, center_lat: float, center_lon: float):
    # Build a grid around provided center
    lat_range = np.linspace(center_lat - 0.5, center_lat + 0.5, grid_size)
    lon_range = np.linspace(center_lon - 0.5, center_lon + 0.5, grid_size)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    return pd.DataFrame({'latitude': lat_grid.ravel(), 'longitude': lon_grid.ravel()})

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

# Sidebar controls
st.sidebar.header('Geospatial Scenario Controls')

grid_size = st.sidebar.slider('Grid Size (N x N)', min_value=10, max_value=50, value=25, step=5)
map_style_opt = st.sidebar.selectbox('Map layer', ['Heatmap', 'Hexagon', 'Scatter'], index=0)
model_options = ['Ensemble (average)'] + list(models.keys()) if len(models) > 1 else list(models.keys())
chosen_model = st.sidebar.selectbox('Model to use', model_options)
city = st.sidebar.text_input('Center grid by city (optional)', placeholder='e.g., Mumbai')
if city.strip():
    lat, lon = get_coords(city.strip())
    if lat is None:
        st.info('Using default center (India) because the city could not be geocoded.')
        lat, lon = 22.5, 82.5
else:
    lat, lon = 22.5, 82.5

grid_df = create_grid(grid_size, lat, lon)

# Pick a feature to vary across the map; limit to features known by model and with defined ranges
vary_options = [f for f in feature_columns if f in FEATURE_RANGES]
feature_to_vary = st.sidebar.selectbox('Feature to Vary Across Map', options=vary_options, index=vary_options.index('TopographyDrainage') if 'TopographyDrainage' in vary_options else 0)

st.sidebar.subheader('Baseline for Other Features')
baseline_values = {}
for f in vary_options:
    if f == feature_to_vary:
        continue
    min_val, max_val, default_val = FEATURE_RANGES[f]
    baseline_values[f] = st.sidebar.slider(f.replace('_',' ').title(), min_val, max_val, default_val)

if st.button('Generate Risk Heatmap', type='primary'):
    # Build prediction dataframe
    data = pd.DataFrame(index=range(len(grid_df)))
    # Apply baseline for all except the varying feature
    for f in feature_columns:
        if f == feature_to_vary:
            continue
        if f in baseline_values:
            data[f] = baseline_values[f]
        else:
            # Fallback to default from FEATURE_RANGES or zero
            if f in FEATURE_RANGES:
                data[f] = FEATURE_RANGES[f][2]
            else:
                data[f] = 0

    # Create a west->east gradient for the selected feature
    min_v, max_v, _ = FEATURE_RANGES[feature_to_vary]
    # Normalize longitudes to 0..1, then scale to feature range
    long_norm = (grid_df['longitude'] - grid_df['longitude'].min()) / max(1e-9, (grid_df['longitude'].max() - grid_df['longitude'].min()))
    data[feature_to_vary] = (min_v + (max_v - min_v) * long_norm).astype(float)

    # Ensure all required columns exist and in correct order
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0
    model_input = data[feature_columns]

    # Predict and clip to [0,1]
    def predict_prob(mdl, X):
        if hasattr(mdl, 'predict_proba'):
            return mdl.predict_proba(X)[:, 1]
        vals = mdl.predict(X)
        return np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)

    if chosen_model == 'Ensemble (average)' and len(models) > 1:
        preds_list = []
        for name, mdl in models.items():
            try:
                preds_list.append(predict_prob(mdl, model_input))
            except Exception:
                pass
        if preds_list:
            preds = np.mean(np.vstack(preds_list), axis=0)
        else:
            st.error('No models available to generate predictions.')
            st.stop()
    else:
        if chosen_model not in models:
            st.error('Selected model is not available.')
            st.stop()
        preds = predict_prob(models[chosen_model], model_input)

    prediction_data = grid_df.copy()
    prediction_data['flood_probability'] = preds

    st.header('Flood Risk Heatmap')
    st.caption(f"Varying: {feature_to_vary.replace('_',' ').title()} (west→east gradient). Other features held at baseline.")

    # Map layer
    if map_style_opt == 'Heatmap':
        map_layer = pdk.Layer(
            'HeatmapLayer',
            data=prediction_data,
            get_position='[longitude, latitude]',
            get_weight='flood_probability',
            aggregation=pdk.types.String('MEAN'),
            opacity=0.9,
        )
    elif map_style_opt == 'Hexagon':
        map_layer = pdk.Layer(
            'HexagonLayer',
            data=prediction_data,
            get_position='[longitude, latitude]',
            elevation_scale=50,
            elevation_range=[0, 3000],
            extruded=True,
            radius=200,
            get_elevation='flood_probability * 3000',
            get_color='[flood_probability * 255, (1-flood_probability) * 255, 120]'
        )
    else:
        map_layer = pdk.Layer(
            'ScatterplotLayer',
            data=prediction_data,
            get_position='[longitude, latitude]',
            get_fill_color='[flood_probability * 255, (1-flood_probability) * 255, 120, 200]',
            get_radius=120,
            pickable=True,
        )

    view_state = pdk.ViewState(
        latitude=float(prediction_data['latitude'].mean()),
        longitude=float(prediction_data['longitude'].mean()),
        zoom=7,
        pitch=45,
    )

    deck = pdk.Deck(
        map_style=None,  # Avoid Mapbox key requirement; renders without basemap
        initial_view_state=view_state,
        layers=[map_layer],
        tooltip={
            'html': '<b>Flood Probability:</b> {flood_probability}',
            'style': {'color': 'white'}
        },
    )

    st.pydeck_chart(deck)

    st.info('Tip: Switch the map layer in the sidebar. Supply a MAPBOX_API_KEY for prettier base maps (optional).')
