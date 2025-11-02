import json
import os

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Geospatial Risk Analysis",
    page_icon="🌍",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

@st.cache_resource(show_spinner=False)
def load_model():
    lgbm_path = os.path.join(BASE_DIR, 'lgbm_model.pkl')
    rf_path = os.path.join(BASE_DIR, 'rf_model.pkl')
    model = None
    if os.path.exists(lgbm_path):
        model = joblib.load(lgbm_path)
    else:
        model = joblib.load(rf_path)
    return model

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

model = load_model()
feature_columns = load_feature_columns()

st.title('🌍 Geospatial Risk Analysis')
st.write(
    """
This page demonstrates how varying a single risk factor can create hotspots of vulnerability across a simulated city grid.
Use the controls to set baseline values and vary one feature from West → East to see a dynamic heatmap.
    """
)

@st.cache_data(show_spinner=False)
def create_grid(grid_size: int):
    # Simulated grid centered in India for neutrality
    center_lat, center_lon = 22.5, 82.5
    lat_range = np.linspace(center_lat - 0.5, center_lat + 0.5, grid_size)
    lon_range = np.linspace(center_lon - 0.5, center_lon + 0.5, grid_size)
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    return pd.DataFrame({'latitude': lat_grid.ravel(), 'longitude': lon_grid.ravel()})

# Sidebar controls
st.sidebar.header('Geospatial Scenario Controls')

grid_size = st.sidebar.slider('Grid Size (N x N)', min_value=10, max_value=50, value=25, step=5)
grid_df = create_grid(grid_size)

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
    preds = model.predict(model_input)
    preds = np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)

    prediction_data = grid_df.copy()
    prediction_data['flood_probability'] = preds

    st.header('Flood Risk Heatmap')
    st.caption(f"Varying: {feature_to_vary.replace('_',' ').title()} (west→east gradient). Other features held at baseline.")

    # Heatmap layer
    heatmap_layer = pdk.Layer(
        'HeatmapLayer',
        data=prediction_data,
        get_position='[longitude, latitude]',
        get_weight='flood_probability',
        aggregation=pdk.types.String('MEAN'),
        opacity=0.9,
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
        layers=[heatmap_layer],
        tooltip={
            'html': '<b>Flood Probability:</b> {flood_probability}',
            'style': {'color': 'white'}
        },
    )

    st.pydeck_chart(deck)

    st.info('Tip: In a real deployment, you can supply a MAPBOX_API_KEY for prettier base maps.')
