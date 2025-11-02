# Project AQUA - Model Training Script
# Trains two models (LightGBM, RandomForest) on flood.csv and saves them.

import json
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def main():
    print("--- Starting Model Training ---")

    # 1. Load the Dataset
    print("Step 1/5: Loading data...")
    csv_path = os.path.join(os.path.dirname(__file__), 'flood.csv')
    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find dataset at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # 2. Prepare the Data for Training
    print("Step 2/5: Preparing data...")
    target_col = 'FloodProbability'
    if target_col not in df.columns:
        print(f"ERROR: Target column '{target_col}' not found in dataset. Columns: {list(df.columns)}")
        sys.exit(1)

    # Drop ID if present and the target
    drop_cols = [c for c in ['id', target_col] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Basic NA handling: fill numeric columns with median, others with mode
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() <= 20 else None
    )
    print(f"Data prepared: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # 3. Train Multiple Models (USP #1: Ensemble Modeling)
    print("Step 3/5: Training models...")

    # Model A: LightGBM Regressor
    lgbm = None
    if LIGHTGBM_AVAILABLE:
        try:
            print("  - Training LightGBM Regressor...")
            lgbm = LGBMRegressor(random_state=42)
            lgbm.fit(X_train, y_train)
        except Exception as e:
            print(f"  ! LightGBM training failed: {e}")
            lgbm = None
    else:
        print("  ! LightGBM not available; skipping LightGBM model.")

    # Model B: Random Forest Regressor
    print("  - Training Random Forest Regressor...")
    # Reduce model size to keep artifact under GitHub's 100MB limit
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)

    # Model C: XGBoost Regressor (optional)
    xgb = None
    if XGBOOST_AVAILABLE:
        try:
            print("  - Training XGBoost Regressor...")
            xgb = XGBRegressor(random_state=42, n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, tree_method='hist', objective='reg:squarederror', eval_metric='rmse')
            xgb.fit(X_train, y_train)
        except Exception as e:
            print(f"  ! XGBoost training failed: {e}")
            xgb = None
    else:
        print("  ! XGBoost not available; skipping XGBoost model.")

    # Model D: Support Vector Regressor
    print("  - Training SVR...")
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train)

    # Model E: K-Neighbors Regressor
    print("  - Training KNN Regressor...")
    knn = KNeighborsRegressor(n_neighbors=7)
    knn.fit(X_train, y_train)

    print("Models trained successfully.")

    # 4. Evaluate Model Performance (regression metrics)
    print("Step 4/5: Evaluating models...")
    def print_regression_report(model_name: str, model):
        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = r2_score(y_test, preds)
        print(f"  - {model_name} RMSE: {rmse:.4f} | R^2: {r2:.4f}")

    if lgbm is not None:
        print_regression_report("LightGBM", lgbm)
    else:
        print("  - LightGBM: skipped")

    print_regression_report("Random Forest", rf)
    if xgb is not None:
        print_regression_report("XGBoost", xgb)
    else:
        print("  - XGBoost: skipped")
    print_regression_report("SVR", svr)
    print_regression_report("KNN", knn)

    # 5. Save the Trained Models + Feature Columns
    print("Step 5/5: Saving models...")
    out_dir = os.path.dirname(__file__)
    if lgbm is not None:
        joblib.dump(lgbm, os.path.join(out_dir, 'lgbm_model.pkl'), compress=9)
    joblib.dump(rf, os.path.join(out_dir, 'rf_model.pkl'), compress=9)
    if xgb is not None:
        joblib.dump(xgb, os.path.join(out_dir, 'xgboost_model.pkl'), compress=9)
    joblib.dump(svr, os.path.join(out_dir, 'svr_model.pkl'), compress=9)
    joblib.dump(knn, os.path.join(out_dir, 'knn_model.pkl'), compress=9)

    feature_cols_path = os.path.join(out_dir, 'feature_columns.json')
    with open(feature_cols_path, 'w', encoding='utf-8') as f:
        json.dump(list(X.columns), f, indent=2)

    print("\n--- Process Complete ---")
    saved = ["rf_model.pkl", "svr_model.pkl", "knn_model.pkl"]
    if lgbm is not None:
        saved.append("lgbm_model.pkl")
    if xgb is not None:
        saved.append("xgboost_model.pkl")
    print("Models saved:", ", ".join(saved))
    print("Feature columns saved as 'feature_columns.json'")


if __name__ == "__main__":
    main()
