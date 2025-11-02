# Project AQUA: Flood Prediction Engine

This project trains two ML models (LightGBM and RandomForest) on a generalized flood dataset (`flood.csv`) and serves an interactive Streamlit app to predict flood probability and explain predictions with SHAP.

## Setup (Windows PowerShell)

1. Optional: Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

## Train Models

Make sure `flood.csv` is in this folder (same as `train_models.py`). Then run:

```powershell
python train_models.py
```

This will create `rf_model.pkl` and (if LightGBM is available) `lgbm_model.pkl`, plus `feature_columns.json`.

## Run the App

After training completes:

```powershell
streamlit run app.py
```

Your browser will open the interactive app. Adjust sliders, click "Predict Flood Risk", and view both the ensemble prediction and SHAP explanation.

### Multi‑page navigation
This app now includes additional pages under `pages/`:
- Geospatial Risk Analysis: Generates a simulated city grid and renders a flood risk heatmap.
- Model Performance: Displays metrics and a SHAP global importance summary.

Use the sidebar page selector in Streamlit to navigate.

## Notes
- The app aligns user inputs to the training feature set via `feature_columns.json`.
- If LightGBM isn’t available on your system, training and the app still work with RandomForest.
- SHAP explanation prefers LightGBM when available; otherwise, it uses RandomForest.
