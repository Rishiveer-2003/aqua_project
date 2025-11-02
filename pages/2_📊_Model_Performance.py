import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- Page Configuration ---
st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
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

@st.cache_data(show_spinner=False)
def load_data():
    csv_path = os.path.join(BASE_DIR, 'flood.csv')
    df = pd.read_csv(csv_path)
    # Align with training logic: drop id, target; create X/y
    target_col = 'FloodProbability'
    drop_cols = [c for c in ['id', target_col] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    # Basic NA handling
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 0)
    return X, y

models = load_models()
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.title('📊 Model Performance and Interpretability')

st.header('Evaluation Metrics')
cols = st.columns(2)

metrics_rows = []
for name, mdl in models.items():
    preds = mdl.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)
    metrics_rows.append((name, rmse, r2))

if metrics_rows:
    # Show top (best R2) model prominently
    best = sorted(metrics_rows, key=lambda r: r[2], reverse=True)[0]
    cols[0].metric(f"Best Model (R²)", best[0])
    cols[1].metric("R² | RMSE", f"{best[2]:.4f} | {best[1]:.4f}")

    st.write('Detailed:')
    for name, rmse, r2 in metrics_rows:
        st.write(f"- {name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
else:
    st.warning('No models found to evaluate.')

st.header('Global Feature Importance (SHAP)')
st.write(
    """
This SHAP summary plot shows which features are most influential across the dataset for the best tree-based model.
    """
)

# Choose a tree model for SHAP (LightGBM preferred)
model_for_shap = models.get('LightGBM') or models.get('XGBoost') or models.get('RandomForest')
if model_for_shap is not None:
    with st.spinner('Computing SHAP values...'):
        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer.shap_values(X_test)
        fig = plt.figure()
        try:
            shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
        except Exception:
            # Some model outputs use list for classes; handle generically
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[0], X_test, plot_type='bar', show=False)
            else:
                shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
        st.pyplot(fig, bbox_inches='tight', clear_figure=True)
else:
    st.info('A tree-based model is required to render SHAP global importance.')
