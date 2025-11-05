# pages/4_⚡_Dynamic_LSTM_Analyzer.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.geocoders import Nominatim
import openmeteo_requests
import requests_cache
from retry_requests import retry
import datetime
import json
import os

# Try to import PyTorch, show error if not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.error("⚠️ PyTorch is not installed. The LSTM model cannot be loaded.")
    st.info("""
    **To use this page, PyTorch must be installed.**
    
    If you're running this locally:
    ```
    pip install torch
    ```
    
    If on Streamlit Cloud, please wait for the dependencies to install (may take a few minutes).
    The app will automatically reload once installation is complete.
    """)
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Dynamic LSTM Analyzer",
    page_icon="⚡",
    layout="wide"
)

# ==============================================================================
# 1. DEFINE PYTORCH MODEL CLASS
#    (MUST be identical to the class in train_lstm_pytorch.py)
# ==============================================================================
class FloodLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(FloodLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, output_size)
        
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        # Take the output from the last timestep
        out = lstm_out[:, -1, :]
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# ==============================================================================
# 2. LOAD LSTM MODEL AND SCALERS
# ==============================================================================
@st.cache_resource
def load_lstm_system():
    """Loads the trained PyTorch model and scalers, downloads from GitHub release if missing."""
    import os
    import urllib.request
    
    # GitHub Release URLs for model files
    MODEL_URLS = {
        'lstm_model_best.pth': 'https://github.com/Rishiveer-2003/aqua_project/releases/download/v1.0/lstm_model_best.pth',
        'lstm_scaler_X.pkl': 'https://github.com/Rishiveer-2003/aqua_project/releases/download/v1.0/lstm_scaler_X.pkl',
        'lstm_scaler_y.pkl': 'https://github.com/Rishiveer-2003/aqua_project/releases/download/v1.0/lstm_scaler_y.pkl'
    }
    
    try:
        # Check and download missing files
        for filename, url in MODEL_URLS.items():
            if not os.path.exists(filename):
                st.info(f"📥 Downloading {filename} from GitHub Release...")
                try:
                    urllib.request.urlretrieve(url, filename)
                    st.success(f"✅ Downloaded {filename}")
                except Exception as download_err:
                    st.warning(f"⚠️ Could not download {filename}: {download_err}")
                    st.info("Model files should be uploaded as a GitHub Release or the model needs retraining.")
                    return None, None, None, None
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model
        model = FloodLSTM(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
        model.load_state_dict(torch.load('lstm_model_best.pth', map_location=device, weights_only=True))
        model.eval()  # Set model to evaluation mode
        
        # Load Scalers
        scaler_X = joblib.load('lstm_scaler_X.pkl')
        scaler_y = joblib.load('lstm_scaler_y.pkl')
        
        return model, scaler_X, scaler_y, device
    except FileNotFoundError as e:
        st.error(f"❌ Error loading model files: {e}")
        st.info("Please ensure model files are available as a GitHub Release at v1.0 tag.")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
        return None, None, None, None

model, scaler_X, scaler_y, device = load_lstm_system()

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

# Complete 32-city dictionary
CITY_PROFILES = {
    "Mumbai": {"TopographyDrainage": 4, "RiverManagement": 5, "Deforestation": 12, "Urbanization": 17, "ClimateChange": 13, "DamsQuality": 9, "Siltation": 14, "AgriculturalPractices": 2, "Encroachments": 16, "IneffectiveDisasterPreparedness": 11, "DrainageSystems": 5, "CoastalVulnerability": 17, "Landslides": 7, "Watersheds": 6, "DeterioratingWaterQuality": 13, "PopulationScore": 18, "WetlandLoss": 15, "InadequatePlanning": 14, "PoliticalFactors": 10},
    "Kolkata": {"TopographyDrainage": 3, "RiverManagement": 6, "Deforestation": 11, "Urbanization": 16, "ClimateChange": 14, "DamsQuality": 8, "Siltation": 15, "AgriculturalPractices": 4, "Encroachments": 15, "IneffectiveDisasterPreparedness": 12, "DrainageSystems": 6, "CoastalVulnerability": 16, "Landslides": 2, "Watersheds": 5, "DeterioratingWaterQuality": 14, "PopulationScore": 17, "WetlandLoss": 16, "InadequatePlanning": 13, "PoliticalFactors": 11},
    "Chennai": {"TopographyDrainage": 5, "RiverManagement": 7, "Deforestation": 10, "Urbanization": 15, "ClimateChange": 15, "DamsQuality": 10, "Siltation": 13, "AgriculturalPractices": 5, "Encroachments": 14, "IneffectiveDisasterPreparedness": 10, "DrainageSystems": 7, "CoastalVulnerability": 18, "Landslides": 3, "Watersheds": 7, "DeterioratingWaterQuality": 12, "PopulationScore": 16, "WetlandLoss": 14, "InadequatePlanning": 12, "PoliticalFactors": 9},
    "Surat": {"TopographyDrainage": 6, "RiverManagement": 8, "Deforestation": 9, "Urbanization": 14, "ClimateChange": 12, "DamsQuality": 11, "Siltation": 12, "AgriculturalPractices": 7, "Encroachments": 13, "IneffectiveDisasterPreparedness": 13, "DrainageSystems": 9, "CoastalVulnerability": 15, "Landslides": 4, "Watersheds": 8, "DeterioratingWaterQuality": 10, "PopulationScore": 15, "WetlandLoss": 12, "InadequatePlanning": 10, "PoliticalFactors": 8},
    "Patna": {"TopographyDrainage": 3, "RiverManagement": 4, "Deforestation": 13, "Urbanization": 13, "ClimateChange": 13, "DamsQuality": 7, "Siltation": 16, "AgriculturalPractices": 10, "Encroachments": 17, "IneffectiveDisasterPreparedness": 9, "DrainageSystems": 4, "CoastalVulnerability": 2, "Landslides": 2, "Watersheds": 4, "DeterioratingWaterQuality": 15, "PopulationScore": 16, "WetlandLoss": 13, "InadequatePlanning": 15, "PoliticalFactors": 12},
    "Guwahati": {"TopographyDrainage": 5, "RiverManagement": 5, "Deforestation": 14, "Urbanization": 12, "ClimateChange": 14, "DamsQuality": 8, "Siltation": 15, "AgriculturalPractices": 8, "Encroachments": 15, "IneffectiveDisasterPreparedness": 8, "DrainageSystems": 5, "CoastalVulnerability": 1, "Landslides": 12, "Watersheds": 5, "DeterioratingWaterQuality": 13, "PopulationScore": 14, "WetlandLoss": 14, "InadequatePlanning": 14, "PoliticalFactors": 11},
    "Delhi": {"TopographyDrainage": 7, "RiverManagement": 7, "Deforestation": 15, "Urbanization": 18, "ClimateChange": 14, "DamsQuality": 10, "Siltation": 13, "AgriculturalPractices": 6, "Encroachments": 16, "IneffectiveDisasterPreparedness": 12, "DrainageSystems": 8, "CoastalVulnerability": 1, "Landslides": 3, "Watersheds": 7, "DeterioratingWaterQuality": 16, "PopulationScore": 19, "WetlandLoss": 17, "InadequatePlanning": 13, "PoliticalFactors": 13},
    "Bengaluru": {"TopographyDrainage": 12, "RiverManagement": 9, "Deforestation": 16, "Urbanization": 17, "ClimateChange": 11, "DamsQuality": 12, "Siltation": 10, "AgriculturalPractices": 4, "Encroachments": 18, "IneffectiveDisasterPreparedness": 14, "DrainageSystems": 7, "CoastalVulnerability": 1, "Landslides": 2, "Watersheds": 9, "DeterioratingWaterQuality": 11, "PopulationScore": 17, "WetlandLoss": 18, "InadequatePlanning": 16, "PoliticalFactors": 9},
    "Hyderabad": {"TopographyDrainage": 10, "RiverManagement": 8, "Deforestation": 13, "Urbanization": 16, "ClimateChange": 12, "DamsQuality": 11, "Siltation": 11, "AgriculturalPractices": 7, "Encroachments": 15, "IneffectiveDisasterPreparedness": 13, "DrainageSystems": 8, "CoastalVulnerability": 1, "Landslides": 5, "Watersheds": 8, "DeterioratingWaterQuality": 12, "PopulationScore": 16, "WetlandLoss": 16, "InadequatePlanning": 11, "PoliticalFactors": 10},
    "Ahmedabad": {"TopographyDrainage": 8, "RiverManagement": 10, "Deforestation": 8, "Urbanization": 15, "ClimateChange": 11, "DamsQuality": 13, "Siltation": 9, "AgriculturalPractices": 9, "Encroachments": 12, "IneffectiveDisasterPreparedness": 14, "DrainageSystems": 10, "CoastalVulnerability": 3, "Landslides": 1, "Watersheds": 10, "DeterioratingWaterQuality": 10, "PopulationScore": 16, "WetlandLoss": 11, "InadequatePlanning": 9, "PoliticalFactors": 7},
    "Pune": {"TopographyDrainage": 14, "RiverManagement": 11, "Deforestation": 11, "Urbanization": 16, "ClimateChange": 10, "DamsQuality": 14, "Siltation": 8, "AgriculturalPractices": 6, "Encroachments": 14, "IneffectiveDisasterPreparedness": 15, "DrainageSystems": 9, "CoastalVulnerability": 1, "Landslides": 10, "Watersheds": 12, "DeterioratingWaterQuality": 9, "PopulationScore": 15, "WetlandLoss": 13, "InadequatePlanning": 10, "PoliticalFactors": 8},
    "Jaipur": {"TopographyDrainage": 13, "RiverManagement": 7, "Deforestation": 7, "Urbanization": 13, "ClimateChange": 10, "DamsQuality": 9, "Siltation": 7, "AgriculturalPractices": 11, "Encroachments": 11, "IneffectiveDisasterPreparedness": 11, "DrainageSystems": 7, "CoastalVulnerability": 1, "Landslides": 4, "Watersheds": 9, "DeterioratingWaterQuality": 8, "PopulationScore": 14, "WetlandLoss": 10, "InadequatePlanning": 12, "PoliticalFactors": 9},
    "Lucknow": {"TopographyDrainage": 6, "RiverManagement": 6, "Deforestation": 9, "Urbanization": 14, "ClimateChange": 11, "DamsQuality": 8, "Siltation": 12, "AgriculturalPractices": 12, "Encroachments": 13, "IneffectiveDisasterPreparedness": 10, "DrainageSystems": 6, "CoastalVulnerability": 1, "Landslides": 1, "Watersheds": 6, "DeterioratingWaterQuality": 11, "PopulationScore": 15, "WetlandLoss": 12, "InadequatePlanning": 13, "PoliticalFactors": 11},
    "Kanpur": {"TopographyDrainage": 5, "RiverManagement": 5, "Deforestation": 10, "Urbanization": 15, "ClimateChange": 12, "DamsQuality": 7, "Siltation": 14, "AgriculturalPractices": 13, "Encroachments": 14, "IneffectiveDisasterPreparedness": 9, "DrainageSystems": 5, "CoastalVulnerability": 1, "Landslides": 1, "Watersheds": 5, "DeterioratingWaterQuality": 17, "PopulationScore": 16, "WetlandLoss": 13, "InadequatePlanning": 14, "PoliticalFactors": 12},
    "Nagpur": {"TopographyDrainage": 11, "RiverManagement": 9, "Deforestation": 8, "Urbanization": 13, "ClimateChange": 10, "DamsQuality": 12, "Siltation": 7, "AgriculturalPractices": 10, "Encroachments": 10, "IneffectiveDisasterPreparedness": 12, "DrainageSystems": 9, "CoastalVulnerability": 1, "Landslides": 3, "Watersheds": 11, "DeterioratingWaterQuality": 9, "PopulationScore": 14, "WetlandLoss": 9, "InadequatePlanning": 8, "PoliticalFactors": 7},
    "Indore": {"TopographyDrainage": 13, "RiverManagement": 8, "Deforestation": 7, "Urbanization": 14, "ClimateChange": 9, "DamsQuality": 11, "Siltation": 6, "AgriculturalPractices": 11, "Encroachments": 9, "IneffectiveDisasterPreparedness": 13, "DrainageSystems": 10, "CoastalVulnerability": 1, "Landslides": 2, "Watersheds": 10, "DeterioratingWaterQuality": 8, "PopulationScore": 13, "WetlandLoss": 8, "InadequatePlanning": 7, "PoliticalFactors": 6},
    "Thane": {"TopographyDrainage": 6, "RiverManagement": 7, "Deforestation": 11, "Urbanization": 16, "ClimateChange": 13, "DamsQuality": 10, "Siltation": 12, "AgriculturalPractices": 3, "Encroachments": 15, "IneffectiveDisasterPreparedness": 11, "DrainageSystems": 7, "CoastalVulnerability": 16, "Landslides": 8, "Watersheds": 7, "DeterioratingWaterQuality": 12, "PopulationScore": 17, "WetlandLoss": 14, "InadequatePlanning": 13, "PoliticalFactors": 9},
    "Bhopal": {"TopographyDrainage": 12, "RiverManagement": 10, "Deforestation": 6, "Urbanization": 12, "ClimateChange": 10, "DamsQuality": 13, "Siltation": 5, "AgriculturalPractices": 9, "Encroachments": 8, "IneffectiveDisasterPreparedness": 12, "DrainageSystems": 11, "CoastalVulnerability": 1, "Landslides": 2, "Watersheds": 13, "DeterioratingWaterQuality": 7, "PopulationScore": 13, "WetlandLoss": 7, "InadequatePlanning": 8, "PoliticalFactors": 7},
    "Visakhapatnam": {"TopographyDrainage": 7, "RiverManagement": 9, "Deforestation": 9, "Urbanization": 14, "ClimateChange": 14, "DamsQuality": 11, "Siltation": 10, "AgriculturalPractices": 6, "Encroachments": 12, "IneffectiveDisasterPreparedness": 12, "DrainageSystems": 8, "CoastalVulnerability": 18, "Landslides": 11, "Watersheds": 8, "DeterioratingWaterQuality": 11, "PopulationScore": 15, "WetlandLoss": 11, "InadequatePlanning": 11, "PoliticalFactors": 8},
    "Vadodara": {"TopographyDrainage": 9, "RiverManagement": 9, "Deforestation": 7, "Urbanization": 13, "ClimateChange": 11, "DamsQuality": 12, "Siltation": 8, "AgriculturalPractices": 10, "Encroachments": 11, "IneffectiveDisasterPreparedness": 13, "DrainageSystems": 9, "CoastalVulnerability": 2, "Landslides": 1, "Watersheds": 9, "DeterioratingWaterQuality": 9, "PopulationScore": 14, "WetlandLoss": 10, "InadequatePlanning": 9, "PoliticalFactors": 7},
    "Ghaziabad": {"TopographyDrainage": 6, "RiverManagement": 5, "Deforestation": 11, "Urbanization": 17, "ClimateChange": 13, "DamsQuality": 8, "Siltation": 13, "AgriculturalPractices": 8, "Encroachments": 15, "IneffectiveDisasterPreparedness": 10, "DrainageSystems": 6, "CoastalVulnerability": 1, "Landslides": 2, "Watersheds": 6, "DeterioratingWaterQuality": 15, "PopulationScore": 18, "WetlandLoss": 14, "InadequatePlanning": 14, "PoliticalFactors": 12},
    "Ludhiana": {"TopographyDrainage": 8, "RiverManagement": 7, "Deforestation": 6, "Urbanization": 15, "ClimateChange": 10, "DamsQuality": 9, "Siltation": 9, "AgriculturalPractices": 14, "Encroachments": 12, "IneffectiveDisasterPreparedness": 11, "DrainageSystems": 7, "CoastalVulnerability": 1, "Landslides": 1, "Watersheds": 8, "DeterioratingWaterQuality": 12, "PopulationScore": 16, "WetlandLoss": 9, "InadequatePlanning": 11, "PoliticalFactors": 10},
    "Agra": {"TopographyDrainage": 7, "RiverManagement": 4, "Deforestation": 8, "Urbanization": 14, "ClimateChange": 12, "DamsQuality": 6, "Siltation": 13, "AgriculturalPractices": 11, "Encroachments": 13, "IneffectiveDisasterPreparedness": 9, "DrainageSystems": 6, "CoastalVulnerability": 1, "Landslides": 2, "Watersheds": 5, "DeterioratingWaterQuality": 16, "PopulationScore": 15, "WetlandLoss": 11, "InadequatePlanning": 13, "PoliticalFactors": 11},
    "Nashik": {"TopographyDrainage": 13, "RiverManagement": 10, "Deforestation": 9, "Urbanization": 13, "ClimateChange": 10, "DamsQuality": 13, "Siltation": 7, "AgriculturalPractices": 9, "Encroachments": 11, "IneffectiveDisasterPreparedness": 12, "DrainageSystems": 9, "CoastalVulnerability": 1, "Landslides": 9, "Watersheds": 11, "DeterioratingWaterQuality": 8, "PopulationScore": 14, "WetlandLoss": 10, "InadequatePlanning": 9, "PoliticalFactors": 8},
    "Faridabad": {"TopographyDrainage": 7, "RiverManagement": 6, "Deforestation": 10, "Urbanization": 16, "ClimateChange": 13, "DamsQuality": 8, "Siltation": 12, "AgriculturalPractices": 7, "Encroachments": 14, "IneffectiveDisasterPreparedness": 10, "DrainageSystems": 7, "CoastalVulnerability": 1, "Landslides": 3, "Watersheds": 6, "DeterioratingWaterQuality": 14, "PopulationScore": 17, "WetlandLoss": 13, "InadequatePlanning": 13, "PoliticalFactors": 11},
    "Meerut": {"TopographyDrainage": 8, "RiverManagement": 6, "Deforestation": 8, "Urbanization": 14, "ClimateChange": 11, "DamsQuality": 7, "Siltation": 11, "AgriculturalPractices": 13, "Encroachments": 12, "IneffectiveDisasterPreparedness": 9, "DrainageSystems": 6, "CoastalVulnerability": 1, "Landslides": 1, "Watersheds": 7, "DeterioratingWaterQuality": 13, "PopulationScore": 15, "WetlandLoss": 10, "InadequatePlanning": 12, "PoliticalFactors": 10},
    "Rajkot": {"TopographyDrainage": 10, "RiverManagement": 8, "Deforestation": 6, "Urbanization": 13, "ClimateChange": 10, "DamsQuality": 11, "Siltation": 7, "AgriculturalPractices": 12, "Encroachments": 10, "IneffectiveDisasterPreparedness": 12, "DrainageSystems": 9, "CoastalVulnerability": 4, "Landslides": 2, "Watersheds": 9, "DeterioratingWaterQuality": 9, "PopulationScore": 14, "WetlandLoss": 9, "InadequatePlanning": 8, "PoliticalFactors": 7},
    "Varanasi": {"TopographyDrainage": 4, "RiverManagement": 3, "Deforestation": 11, "Urbanization": 14, "ClimateChange": 13, "DamsQuality": 6, "Siltation": 17, "AgriculturalPractices": 11, "Encroachments": 18, "IneffectiveDisasterPreparedness": 8, "DrainageSystems": 4, "CoastalVulnerability": 1, "Landslides": 2, "Watersheds": 3, "DeterioratingWaterQuality": 18, "PopulationScore": 17, "WetlandLoss": 15, "InadequatePlanning": 16, "PoliticalFactors": 13},
    "Srinagar": {"TopographyDrainage": 9, "RiverManagement": 8, "Deforestation": 10, "Urbanization": 11, "ClimateChange": 15, "DamsQuality": 10, "Siltation": 12, "AgriculturalPractices": 7, "Encroachments": 13, "IneffectiveDisasterPreparedness": 7, "DrainageSystems": 7, "CoastalVulnerability": 1, "Landslides": 14, "Watersheds": 10, "DeterioratingWaterQuality": 11, "PopulationScore": 12, "WetlandLoss": 12, "InadequatePlanning": 13, "PoliticalFactors": 14},
    "Aurangabad": {"TopographyDrainage": 12, "RiverManagement": 7, "Deforestation": 8, "Urbanization": 12, "ClimateChange": 9, "DamsQuality": 10, "Siltation": 8, "AgriculturalPractices": 10, "Encroachments": 9, "IneffectiveDisasterPreparedness": 11, "DrainageSystems": 8, "CoastalVulnerability": 1, "Landslides": 5, "Watersheds": 9, "DeterioratingWaterQuality": 10, "PopulationScore": 13, "WetlandLoss": 9, "InadequatePlanning": 10, "PoliticalFactors": 8},
    "Goa": {"TopographyDrainage": 8, "RiverManagement": 9, "Deforestation": 12, "Urbanization": 11, "ClimateChange": 14, "DamsQuality": 9, "Siltation": 10, "AgriculturalPractices": 7, "Encroachments": 12, "IneffectiveDisasterPreparedness": 10, "DrainageSystems": 8, "CoastalVulnerability": 18, "Landslides": 10, "Watersheds": 8, "DeterioratingWaterQuality": 11, "PopulationScore": 12, "WetlandLoss": 13, "InadequatePlanning": 11, "PoliticalFactors": 9},
    "Navi Mumbai": {"TopographyDrainage": 5, "RiverManagement": 7, "Deforestation": 11, "Urbanization": 16, "ClimateChange": 13, "DamsQuality": 10, "Siltation": 12, "AgriculturalPractices": 3, "Encroachments": 13, "IneffectiveDisasterPreparedness": 10, "DrainageSystems": 8, "CoastalVulnerability": 17, "Landslides": 6, "Watersheds": 7, "DeterioratingWaterQuality": 12, "PopulationScore": 17, "WetlandLoss": 14, "InadequatePlanning": 10, "PoliticalFactors": 9}
}

@st.cache_data
def get_coords(city_name):
    """Get coordinates for a city using hardcoded values"""
    HARDCODED_COORDS = {
        "Mumbai": (19.0760, 72.8777), "Kolkata": (22.5726, 88.3639),
        "Chennai": (13.0827, 80.2707), "Surat": (21.1702, 72.8311),
        "Patna": (25.5941, 85.1376), "Guwahati": (26.1445, 91.7362),
        "Delhi": (28.7041, 77.1025), "Bengaluru": (12.9716, 77.5946),
        "Hyderabad": (17.3850, 78.4867), "Ahmedabad": (23.0225, 72.5714),
        "Pune": (18.5204, 73.8567), "Jaipur": (26.9124, 75.7873),
        "Lucknow": (26.8467, 80.9462), "Kanpur": (26.4499, 80.3319),
        "Nagpur": (21.1458, 79.0882), "Indore": (22.7196, 75.8577),
        "Thane": (19.2183, 72.9781), "Bhopal": (23.2599, 77.4126),
        "Visakhapatnam": (17.6868, 83.2185), "Vadodara": (22.3072, 73.1812),
        "Ghaziabad": (28.6692, 77.4538), "Ludhiana": (30.9010, 75.8573),
        "Agra": (27.1767, 78.0081), "Nashik": (19.9975, 73.7898),
        "Faridabad": (28.4089, 77.3178), "Meerut": (28.9845, 77.7064),
        "Rajkot": (22.3039, 70.8022), "Varanasi": (25.3176, 82.9739),
        "Srinagar": (34.0837, 74.7973), "Aurangabad": (19.8762, 75.3433),
        "Goa": (15.2993, 74.1240), "Navi Mumbai": (19.0330, 73.0297)
    }
    if city_name in HARDCODED_COORDS:
        return HARDCODED_COORDS[city_name]
    
    # Fallback to geocoding
    try:
        geolocator = Nominatim(user_agent="aqua_lstm_app", timeout=10)
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None
    return None, None

@st.cache_data(ttl=3600)
def get_historical_7_day_rainfall(lat, lon, end_date):
    """
    Fetches the 7-day rainfall history *ending on* the selected date.
    Returns array of 7 rainfall values in chronological order (oldest to newest).
    """
    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        
        # We need 6 days *before* the end_date, plus the end_date itself
        start_date = end_date - datetime.timedelta(days=6)
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "daily": "precipitation_sum",
            "timezone": "auto"
        }
        
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0] if isinstance(responses, (list, tuple)) else responses
        
        daily = response.Daily()
        daily_precipitation_sum = daily.Variables(0).ValuesAsNumpy()
        
        if daily_precipitation_sum.size == 7:
            return daily_precipitation_sum
        else:
            st.error(f"Error: API returned {daily_precipitation_sum.size} days, expected 7.")
            return None

    except Exception as e:
        st.error(f"An error occurred while fetching historical weather data: {e}")
        return None

# ==============================================================================
# 4. USER INTERFACE
# ==============================================================================

st.title("⚡ Dynamic LSTM Flood Risk Analyzer")
st.markdown("""
This page uses a **deep learning LSTM model** trained on 32 Indian cities to predict flood risk based on the **7-day rainfall pattern**, not just a single day's forecast.

**Model Performance:** ~99.77% accuracy on test data
""")

if not all([model, scaler_X, scaler_y, device]):
    st.error("🚨 LSTM Model system not loaded. Please check logs and ensure model files are present.")
else:
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("⚙️ Dynamic Analysis Controls")
    
    # Use searchable city dropdown
    city_search_term = st.sidebar.text_input("🔍 Search for a city:", "")
    filtered_cities = [city for city in CITY_PROFILES if city_search_term.lower() in city.lower()]
    if not filtered_cities:
        filtered_cities = list(CITY_PROFILES.keys())
    
    city_selection = st.sidebar.selectbox(
        "Choose a city:",
        filtered_cities,
        index=0,
        help="Select from 32 pre-analyzed Indian cities"
    )
    
    # Date picker (similar to Historical Analyzer)
    max_date = datetime.date.today() - datetime.timedelta(days=5) # Open-Meteo archive delay
    selected_date = st.sidebar.date_input(
        "📅 Select an 'as of' date for analysis",
        value=max_date,
        min_value=datetime.date(1940, 1, 1),
        max_value=max_date,
        help="The model will predict the risk for this date based on the 7-day rainfall *ending* on this date."
    )
    
    # Info box
    st.sidebar.info("""
    **How it works:**
    1. Fetches 7 days of historical rainfall ending on your selected date
    2. Feeds the temporal pattern into the LSTM neural network
    3. Outputs a flood probability prediction
    
    This model learned from 3,590 historical data points across all 32 cities.
    """)
    
    # --- MAIN PANEL ---
    if st.sidebar.button(f'🚀 **Analyze Dynamic Risk**', type="primary", use_container_width=True):
        lat, lon = get_coords(city_selection)
        
        if lat is not None and lon is not None:
            with st.spinner(f'📡 Fetching 7-day rainfall history for {city_selection}...'):
                rainfall_7_days = get_historical_7_day_rainfall(lat, lon, selected_date)

            if rainfall_7_days is not None:
                # Create two columns for display
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader(f"📊 7-Day Rainfall Pattern")
                    st.caption(f"Ending on {selected_date.strftime('%B %d, %Y')}")
                    
                    # Create a dataframe for the chart
                    date_range = [selected_date - datetime.timedelta(days=i) for i in range(6, -1, -1)]
                    chart_data = pd.DataFrame({
                        'Date': date_range,
                        'Rainfall (mm)': rainfall_7_days
                    })
                    
                    # Display bar chart
                    st.bar_chart(chart_data.set_index('Date'))
                    
                    # Display data table
                    with st.expander("📋 View detailed rainfall data"):
                        chart_data['Date'] = chart_data['Date'].dt.strftime('%Y-%m-%d')
                        st.dataframe(chart_data, use_container_width=True)
                        st.caption(f"Total 7-day rainfall: {rainfall_7_days.sum():.2f} mm")

                # --- PREDICTION LOGIC ---
                try:
                    # 1. Reshape data for scaler (7,) -> (1, 7)
                    input_data = rainfall_7_days.reshape(1, -1)
                    
                    # 2. Scale features
                    scaled_input = scaler_X.transform(input_data)
                    
                    # 3. Reshape for LSTM (1, 7) -> (1, 7, 1) [batch_size, timesteps, features]
                    reshaped_input = scaled_input.reshape(1, 7, 1)
                    
                    # 4. Convert to PyTorch tensor
                    tensor_input = torch.tensor(reshaped_input, dtype=torch.float32).to(device)
                    
                    # 5. Get prediction
                    with torch.no_grad(): # Disable gradient calculation for inference
                        scaled_output_tensor = model(tensor_input)
                    
                    # 6. Move to CPU and convert to numpy
                    scaled_output = scaled_output_tensor.cpu().numpy()
                    
                    # 7. Inverse transform the prediction
                    final_prediction = scaler_y.inverse_transform(scaled_output)
                    
                    # Extract the single probability value
                    final_proba = final_prediction[0][0]
                    
                    # --- DISPLAY RESULTS ---
                    with col2:
                        st.subheader(f'🎯 Prediction Results')
                        st.caption(f"For {city_selection} on {selected_date.strftime('%B %d, %Y')}")
                        
                        # Large metric display
                        st.metric(
                            "LSTM Predicted Flood Risk", 
                            f"{final_proba:.2%}",
                            help="Based on 7-day rainfall temporal pattern"
                        )
                        
                        # Risk classification with color coding
                        if final_proba < 0.01: # 1%
                            st.success("✅ **RISK LEVEL: LOW**")
                            st.markdown("""
                            The 7-day rainfall pattern indicates minimal flood risk. 
                            Normal conditions expected.
                            """)
                        elif final_proba < 0.05: # 5%
                            st.warning("⚠️ **RISK LEVEL: MODERATE**")
                            st.markdown("""
                            The rainfall pattern shows elevated risk. 
                            Monitor local conditions and stay alert.
                            """)
                        else:
                            st.error("🚨 **RISK LEVEL: HIGH**")
                            st.markdown("""
                            **Significant flood risk detected!** 
                            The 7-day rainfall pattern indicates dangerous conditions.
                            Take precautionary measures immediately.
                            """)
                        
                        # Model info
                        with st.expander("ℹ️ About this prediction"):
                            st.markdown(f"""
                            **Model Type:** LSTM Neural Network  
                            **Architecture:** 2-layer LSTM (50 units each) + Dense layers  
                            **Training Accuracy:** 99.73%  
                            **Testing Accuracy:** 99.77%  
                            **Training Data:** 3,590 historical records from 32 cities  
                            **Device:** {device}
                            
                            This prediction is based **only** on the 7-day rainfall pattern. 
                            It does not consider the city's static infrastructure features.
                            
                            The LSTM model learned temporal relationships between rainfall 
                            sequences and flood probabilities across multiple Indian cities.
                            """)
                
                except Exception as e:
                    st.error(f"❌ An error occurred during the prediction phase: {e}")
                    import traceback
                    with st.expander("🐛 Debug Information"):
                        st.code(traceback.format_exc())
                    
            else:
                st.error("❌ Could not retrieve 7-day historical weather data for this date.")
                st.info("""
                Possible reasons:
                - The date is too far in the past (API limitations)
                - Network connection issues
                - API service temporarily unavailable
                
                Try selecting a more recent date (within the last 5 years).
                """)
        else:
            st.error(f"❌ Could not find coordinates for {city_selection}.")
    else:
        # Show placeholder when no analysis has been run
        st.info("👈 Select a city and date from the sidebar, then click **Analyze Dynamic Risk** to begin.")
        
        # Show some example use cases
        st.subheader("💡 Example Use Cases")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📍 Historical Analysis**
            
            Analyze past flood events to understand what rainfall patterns preceded them.
            """)
        
        with col2:
            st.markdown("""
            **📊 Model Comparison**
            
            Compare LSTM predictions with the static ensemble model results from other pages.
            """)
        
        with col3:
            st.markdown("""
            **🔬 Pattern Recognition**
            
            Discover how different 7-day rainfall distributions affect flood probability.
            """)
