# Dynamic Dataset Generation Guide

## Overview

This guide walks you through executing the `create_dynamic_dataset.py` script to generate a time-series flood risk dataset suitable for LSTM model training.

## What This Script Does

The script performs the following operations:

1. **Fetches Historical Rainfall**: Downloads 365 days of historical rainfall data for all 32 Indian cities using the Open-Meteo Archive API
2. **Creates Time-Series Features**: Generates 7-day rolling window features (rain_lag_1 through rain_lag_7)
3. **Generates Proxy Targets**: Uses your existing 5-model ensemble to calculate calibrated flood probability scores
4. **Applies Baseline Calibration**: Isolates rainfall-induced risk from static city characteristics
5. **Saves Dataset**: Outputs a single CSV file ready for LSTM training

## Prerequisites

### 1. Verify Python Environment

```powershell
python --version
# Should be Python 3.8 or higher
```

### 2. Check Required Files

Ensure the following files exist in your project directory:

- ✅ `lgbm_model.pkl`
- ✅ `rf_model.pkl`
- ✅ `xgboost_model.pkl`
- ✅ `svr_model.pkl`
- ✅ `knn_model.pkl`
- ✅ `feature_columns.json`

### 3. Install Dependencies (if needed)

```powershell
pip install -r requirements.txt
```

All required packages are already in your `requirements.txt`:
- pandas
- numpy
- joblib
- scikit-learn
- lightgbm
- xgboost
- geopy
- openmeteo-requests
- requests-cache
- retry-requests

## Execution Steps

### Step 1: Navigate to Project Directory

```powershell
cd C:\Users\rishi\Desktop\aqua_project
```

### Step 2: Run the Script

```powershell
python create_dynamic_dataset.py
```

### Step 3: Monitor Progress

You'll see output like this:

```
================================================================================
DYNAMIC FLOOD DATASET GENERATOR
================================================================================

STEP 1: Loading ML models and feature schema...
--------------------------------------------------------------------------------
✓ Successfully loaded LightGBM from lgbm_model.pkl
✓ Successfully loaded RandomForest from rf_model.pkl
✓ Successfully loaded XGBoost from xgboost_model.pkl
✓ Successfully loaded SVR from svr_model.pkl
✓ Successfully loaded KNN from knn_model.pkl

✓ Successfully loaded 5 models.
✓ Feature columns: 20 features

STEP 2: Setting date range...
--------------------------------------------------------------------------------
Start Date: 2024-11-04
End Date:   2025-11-04
Duration:   365 days

STEP 3: Processing cities...
--------------------------------------------------------------------------------
Total cities to process: 32

[1/32] Processing Mumbai...
  ✓ Coordinates: (19.0760, 72.8777)
  ✓ Fetched 365 days of rainfall data
  → Creating 7-day lag features...
  ✓ Valid data rows after lag creation: 358
  → Generating ensemble predictions...
  ✓ Generated 358 proxy predictions
  ✓ Finished processing Mumbai

[2/32] Processing Kolkata...
...
```

## Expected Runtime

- **Per City**: ~10-15 seconds (includes 2-second API rate limit pause)
- **Total Time**: ~6-8 minutes for all 32 cities
- **Network**: Requires stable internet connection

## Output File

### Filename
`dynamic_flood_data.csv`

### Expected Size
~11,500 rows (358 days × 32 cities)

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Date of the observation |
| `rainfall` | float | Daily rainfall in mm (today) |
| `rain_lag_1` | float | Rainfall from 1 day ago (mm) |
| `rain_lag_2` | float | Rainfall from 2 days ago (mm) |
| `rain_lag_3` | float | Rainfall from 3 days ago (mm) |
| `rain_lag_4` | float | Rainfall from 4 days ago (mm) |
| `rain_lag_5` | float | Rainfall from 5 days ago (mm) |
| `rain_lag_6` | float | Rainfall from 6 days ago (mm) |
| `rain_lag_7` | float | Rainfall from 7 days ago (mm) |
| `FloodProbability_Proxy` | float | Calibrated flood risk (0.0-1.0) |
| `city` | string | City name |

### Sample Data

```csv
date,rainfall,rain_lag_1,rain_lag_2,...,FloodProbability_Proxy,city
2024-11-12,15.3,8.2,0.0,...,0.045,Mumbai
2024-11-13,42.7,15.3,8.2,...,0.123,Mumbai
```

## Validation Checklist

After the script completes, verify:

- [ ] Output file `dynamic_flood_data.csv` exists
- [ ] File size is reasonable (~2-5 MB)
- [ ] All 32 cities are present (check unique city count)
- [ ] No missing values in lag features (except possibly first 7 days)
- [ ] `FloodProbability_Proxy` values are in range [0.0, 1.0]

### Quick Validation Commands

```python
import pandas as pd

# Load the generated dataset
df = pd.read_csv('dynamic_flood_data.csv')

# Check basic stats
print(f"Total rows: {len(df):,}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Cities: {df['city'].nunique()}")
print(f"City list: {sorted(df['city'].unique())}")

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check flood probability distribution
print("\nFlood Probability Statistics:")
print(df['FloodProbability_Proxy'].describe())

# Verify lag features
print("\nRainfall lag features:")
print(df[['rainfall', 'rain_lag_1', 'rain_lag_2', 'rain_lag_3']].head(10))
```

## Troubleshooting

### Issue: "Could not get coordinates for [City]"

**Cause**: Geocoding service (Nominatim) failed or rate-limited

**Solution**: 
- Check internet connection
- Wait 1-2 minutes and re-run the script (it uses caching)
- The script will skip problematic cities and continue

### Issue: "No data returned for coordinates"

**Cause**: Open-Meteo API doesn't have data for that location/date range

**Solution**: 
- This is rare; the script will skip and continue
- Check if the city coordinates are valid

### Issue: "FATAL ERROR: Could not load models"

**Cause**: Model .pkl files or feature_columns.json missing

**Solution**:
```powershell
# Verify all files exist
Get-ChildItem *.pkl
Get-ChildItem feature_columns.json
```

### Issue: Script runs very slowly

**Cause**: Network latency or API throttling

**Solution**: 
- The script includes 2-second delays between cities (respectful to API)
- This is intentional and cannot be reduced
- Total runtime: 6-8 minutes is normal

### Issue: "Module not found" errors

**Cause**: Missing dependencies

**Solution**:
```powershell
pip install --upgrade -r requirements.txt
```

## Next Steps After Generation

### 1. Inspect the Dataset

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dynamic_flood_data.csv')

# Plot rainfall vs flood probability for a specific city
mumbai = df[df['city'] == 'Mumbai']
plt.scatter(mumbai['rainfall'], mumbai['FloodProbability_Proxy'])
plt.xlabel('Rainfall (mm)')
plt.ylabel('Flood Probability')
plt.title('Mumbai: Rainfall vs Flood Risk')
plt.show()
```

### 2. Prepare for LSTM Training

The dataset is now ready for LSTM model training:

- **Input features**: `rain_lag_1` through `rain_lag_7` (7-day sequence)
- **Target variable**: `FloodProbability_Proxy`
- **Additional features**: You can add static city features if needed

### 3. LSTM Architecture Suggestion

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(7, 1)),  # 7 timesteps, 1 feature
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Output: flood probability
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## Benefits of This Dataset

1. **Temporal Dependencies**: Captures 7-day rainfall history, allowing LSTM to learn cumulative effects
2. **Calibrated Targets**: Flood probabilities are rainfall-aware (baseline-calibrated)
3. **Multi-City**: 32 cities provide diverse geographic and climatic conditions
4. **Real Weather Data**: Uses actual historical rainfall from Open-Meteo Archive
5. **Phase 2 Ready**: Perfect foundation for the LSTM enhancement mentioned in your project reports

## API Rate Limits

**Open-Meteo Archive API**:
- Free tier: Unlimited requests
- Cache enabled: Subsequent runs reuse downloaded data
- Respectful delays: 2 seconds between cities

**Nominatim Geocoding**:
- Rate limit: 1 request/second
- Cache enabled: City coordinates cached after first lookup

## File Locations

- **Script**: `create_dynamic_dataset.py`
- **Output**: `dynamic_flood_data.csv`
- **Cache**: `.cache/` directory (auto-created)
- **Models**: `*.pkl` files (required)
- **Config**: `feature_columns.json` (required)

## Support

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Verify all prerequisites are met
3. Check the console output for specific error messages
4. Ensure stable internet connection

## Customization Options

You can modify the script to:

### Change Date Range

Edit line ~475:
```python
# Current: 365 days (1 year)
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

# Option 1: Last 2 years
start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

# Option 2: Specific date range
start_date = "2023-01-01"
end_date = "2023-12-31"
```

### Change Lag Window

Edit line ~525 to create more/fewer lag features:
```python
# Current: 7-day lag
for i in range(1, 8):
    rainfall_df[f'rain_lag_{i}'] = rainfall_df['rainfall'].shift(i)

# Option: 14-day lag
for i in range(1, 15):
    rainfall_df[f'rain_lag_{i}'] = rainfall_df['rainfall'].shift(i)
```

### Add Selected Cities Only

Edit the main loop to process specific cities:
```python
selected_cities = ["Mumbai", "Delhi", "Chennai", "Kolkata"]
for idx, (city, profile) in enumerate(CITY_PROFILES.items(), 1):
    if city not in selected_cities:
        continue
    # ... rest of processing
```

---

**Generated**: November 4, 2025  
**Project**: AQUA - AI-based Quantum Urban Analytics  
**Purpose**: LSTM Time-Series Model Training Dataset
