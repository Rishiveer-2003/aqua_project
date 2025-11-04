# Quick Start: Dynamic Dataset Generation

## TL;DR - Fast Track

```powershell
# 1. Navigate to project directory
cd C:\Users\rishi\Desktop\aqua_project

# 2. Run pre-flight check (optional but recommended)
python preflight_check.py

# 3. Generate dataset
python create_dynamic_dataset.py

# 4. Verify output
python -c "import pandas as pd; df = pd.read_csv('dynamic_flood_data.csv'); print(f'Rows: {len(df):,}'); print(f'Cities: {df['city'].nunique()}'); print(df.head())"
```

**Expected time**: ~6-8 minutes  
**Expected output**: `dynamic_flood_data.csv` with ~11,500 rows

---

## What Gets Created

### 1. `create_dynamic_dataset.py` (Main Script)
- Fetches 1 year of historical rainfall for 32 cities
- Creates 7-day lag features (rain_lag_1 to rain_lag_7)
- Generates calibrated flood probability targets using your 5-model ensemble
- Outputs: `dynamic_flood_data.csv`

### 2. `preflight_check.py` (Validation Script)
- Checks Python version
- Verifies all dependencies installed
- Confirms model files exist
- Tests network connectivity
- Validates model loading

### 3. `DATASET_GENERATION_GUIDE.md` (Complete Documentation)
- Full technical details
- Troubleshooting guide
- Customization options
- Next steps for LSTM training

---

## Step-by-Step Execution

### Step 1: Pre-Flight Check (Recommended)

```powershell
python preflight_check.py
```

**Expected output**:
```
================================================================================
PRE-FLIGHT CHECK FOR DATASET GENERATION
================================================================================

--------------------------------------------------------------------------------
Python Version Check
--------------------------------------------------------------------------------
Python Version: 3.12.0
✓ Python version is compatible (3.8+)

--------------------------------------------------------------------------------
Dependency Check
--------------------------------------------------------------------------------
✓ pandas                    version: 2.1.0
✓ numpy                     version: 1.24.0
✓ joblib                    version: 1.3.0
...

✓ ALL CHECKS PASSED!

You are ready to run the dataset generation script:

    python create_dynamic_dataset.py
```

**If checks fail**: Follow the error messages and fix issues before proceeding.

---

### Step 2: Generate Dataset

```powershell
python create_dynamic_dataset.py
```

**Console output** (abbreviated):
```
================================================================================
DYNAMIC FLOOD DATASET GENERATOR
================================================================================

STEP 1: Loading ML models and feature schema...
✓ Successfully loaded LightGBM from lgbm_model.pkl
✓ Successfully loaded RandomForest from rf_model.pkl
...
✓ Successfully loaded 5 models.

STEP 2: Setting date range...
Start Date: 2024-11-04
End Date:   2025-11-04

STEP 3: Processing cities...
[1/32] Processing Mumbai...
  ✓ Coordinates: (19.0760, 72.8777)
  ✓ Fetched 365 days of rainfall data
  ✓ Generated 358 proxy predictions
  ✓ Finished processing Mumbai

[2/32] Processing Kolkata...
...

STEP 4: Saving dataset...
✓ Dynamic dataset created successfully: 'dynamic_flood_data.csv'
✓ Total rows generated: 11,456
✓ Total cities processed: 32

✓ DATASET GENERATION COMPLETE!
```

**Duration**: 6-8 minutes (includes 2-second delays between cities for API rate limiting)

---

### Step 3: Verify Output

```powershell
# Quick verification
python -c "import pandas as pd; df = pd.read_csv('dynamic_flood_data.csv'); print(f'Total rows: {len(df):,}'); print(f'Date range: {df['date'].min()} to {df['date'].max()}'); print(f'Cities: {df['city'].nunique()}'); print('\nSample:'); print(df.head())"
```

**Expected output**:
```
Total rows: 11,456
Date range: 2024-11-12 to 2025-11-04
Cities: 32

Sample:
        date  rainfall  rain_lag_1  rain_lag_2  ...  FloodProbability_Proxy       city
0 2024-11-12      15.3         8.2         0.0  ...                  0.0450     Mumbai
1 2024-11-13      42.7        15.3         8.2  ...                  0.1230     Mumbai
2 2024-11-14       0.0        42.7        15.3  ...                  0.0980     Mumbai
...
```

---

## Understanding the Output

### File: `dynamic_flood_data.csv`

#### Core Columns
- **`date`**: Date of observation (YYYY-MM-DD)
- **`rainfall`**: Today's rainfall in mm
- **`rain_lag_1` to `rain_lag_7`**: Rainfall from 1-7 days ago (mm)
- **`FloodProbability_Proxy`**: Calibrated flood risk score (0.0-1.0)
- **`city`**: City name

#### Key Statistics
```python
import pandas as pd
df = pd.read_csv('dynamic_flood_data.csv')

# Rainfall distribution
print(df['rainfall'].describe())
#   mean: ~5-15mm
#   max: ~200mm (extreme events)

# Flood probability distribution  
print(df['FloodProbability_Proxy'].describe())
#   mean: ~0.05-0.15 (5-15% average risk)
#   max: ~0.8-1.0 (high-risk events)
```

---

## What Makes This Dataset Special

### 1. **Time-Series Features**
- 7-day rolling window captures cumulative rainfall effects
- Models can learn patterns like "3 days of moderate rain = high risk"

### 2. **Baseline Calibration**
- Proxy targets are rainfall-aware
- Zero rainfall → near-zero risk
- High rainfall → proportional risk increase

### 3. **Ensemble-Based Targets**
- Uses all 5 trained models (LightGBM, RF, XGBoost, SVR, KNN)
- More robust than single-model predictions

### 4. **Real Historical Data**
- Open-Meteo Archive API (1940-present)
- Actual weather observations, not synthetic data

### 5. **Multi-City Coverage**
- 32 cities with diverse climates
- Coastal (Mumbai, Chennai) to inland (Delhi, Bhopal)
- Different risk profiles and rainfall patterns

---

## Next Step: Train LSTM Model

### Basic LSTM Architecture

```python
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('dynamic_flood_data.csv')

# Prepare features (7-day lag sequence)
feature_cols = [f'rain_lag_{i}' for i in range(1, 8)]
X = df[feature_cols].values
y = df['FloodProbability_Proxy'].values

# Reshape for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], 7, 1))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(7, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['mae']
)

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.4f}")

# Save model
model.save('lstm_flood_model.h5')
```

### Expected Performance
- **MAE**: ~0.05-0.10 (5-10% error)
- **Improvement over current**: 10-15% (as per your project report)
- **Best for**: Multi-day rainfall events with cumulative effects

---

## File Summary

| File | Purpose | Size |
|------|---------|------|
| `create_dynamic_dataset.py` | Main generation script | ~20 KB |
| `preflight_check.py` | Validation script | ~8 KB |
| `DATASET_GENERATION_GUIDE.md` | Full documentation | ~15 KB |
| `QUICK_START.md` | This file | ~5 KB |
| `dynamic_flood_data.csv` | Output dataset | ~2-5 MB |

---

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Module not found | `pip install -r requirements.txt` |
| Model file missing | Verify all .pkl files in directory |
| Network error | Check internet connection |
| Geocoding fails | Wait 1-2 minutes, re-run (uses cache) |
| API rate limit | Script includes delays, just wait |

---

## Support & References

- **Full Guide**: See `DATASET_GENERATION_GUIDE.md`
- **Project Report**: See `PROJECT_STATUS_REPORT.md`
- **Calibration Details**: See `CALIBRATION_UPDATE.md`

---

## Credits

- **Open-Meteo**: Historical rainfall data (https://open-meteo.com)
- **Nominatim**: Geocoding service (OpenStreetMap)
- **Project AQUA**: AI-based Quantum Urban Analytics flood prediction system

---

**Created**: November 4, 2025  
**Version**: 1.0  
**Status**: Production Ready ✅
