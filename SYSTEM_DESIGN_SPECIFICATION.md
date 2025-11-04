# Project AQUA: System Design Specification
**For Visual Diagram Creation**

This document provides a detailed textual description of the system architecture, components, data flows, and interactions. Use this specification to create visual system design diagrams.

---

## 1. HIGH-LEVEL SYSTEM ARCHITECTURE

### 1.1 Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│              (Streamlit Web Application)                     │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                        │
│        (Business Logic, ML Inference, Data Processing)       │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                             │
│      (Models, Datasets, APIs, Cache)                        │
└─────────────────────────────────────────────────────────────┘
```

**Tier Descriptions:**
- **Presentation Layer:** User interface, visualization, user interactions
- **Application Layer:** ML model loading, prediction logic, feature engineering, SHAP explainability
- **Data Layer:** Stored models (.pkl files), CSV dataset, external weather APIs, caching mechanisms

---

## 2. COMPONENT BREAKDOWN

### 2.1 Presentation Layer Components

**A. Streamlit Web Application**
```
Main App (app.py - Homepage)
├── Sidebar Navigation
│   ├── Page Selector
│   └── Settings/Filters
├── India Live Flood Forecast Module
│   ├── City Search Dropdown (32 cities)
│   ├── Model Selection (Ensemble/Individual)
│   ├── Map Layer Toggle (Heatmap/Hexagon/Scatter)
│   ├── Pydeck 3D Map Visualization
│   └── Risk Prediction Display
└── Real-time Weather Integration

Page 1: Historical Event Analyzer (pages/1_🕰️_Historical_Event_Analyzer.py)
├── City Selector
├── Date Picker (1940 - present, with 5-day delay)
├── Model Selection
├── Historical Rainfall Display
└── Risk Conclusion (LOW/MODERATE/HIGH)

Page 2: Model Performance Dashboard (pages/2_📊_Model_Performance.py)
├── Performance Metrics Table (RMSE, R²)
├── Model Comparison Charts
├── SHAP Global Feature Importance Plots
└── Model Selection Interface

Page 3: Interactive Risk Calculator (pages/3_🔬_Risk_Calculator.py)
├── 20 Feature Sliders (Sidebar)
├── Model Selection (Ensemble/Single)
├── Real-time Prediction Display
├── SHAP Force Plot (Explainability)
└── Optional Live Weather Helper
```

### 2.2 Application Layer Components

**B. ML Inference Engine**
```
Model Manager
├── Model Loader (joblib deserialization)
├── Feature Alignment (feature_columns.json)
├── Prediction Router
│   ├── Ensemble Mode (average of 5 models)
│   └── Single Model Mode (user-selected)
├── Classifier/Regressor Handler
└── Output Formatter (probability scaling)

Feature Engineering Module
├── Rainfall-to-Intensity Mapper (mm → MonsoonIntensity [0-16])
├── Grid-based Feature Generator (intra-city variation)
├── Live Weather Feature Extractor
└── User Input Feature Validator
```

**C. Explainability Module (SHAP)**
```
SHAP Engine
├── TreeExplainer Initialization (gradient boosting models)
├── Global Feature Importance Calculator
│   ├── SHAP Summary Plots
│   └── Feature Ranking
├── Local Feature Importance Calculator
│   └── SHAP Force Plots (single prediction)
└── Visualization Generator (matplotlib)
```

**D. Data Processing Module**
```
Preprocessing Pipeline (train_models.py)
├── Data Loader (CSV reader)
├── Missing Value Imputer
│   ├── Numeric: Median fill
│   └── Categorical: Mode fill
├── Feature Validator (20 features required)
├── Train-Test Splitter (80/20, stratified, seed=42)
└── Feature Saver (feature_columns.json)
```

### 2.3 Data Layer Components

**E. Persistent Storage**
```
Local File System
├── Models/
│   ├── lgbm_model.pkl (LightGBM serialized model)
│   ├── rf_model.pkl (Random Forest serialized model)
│   ├── xgboost_model.pkl (XGBoost serialized model)
│   ├── svr_model.pkl (SVR serialized model)
│   └── knn_model.pkl (KNN serialized model)
├── Data/
│   ├── flood.csv (50,001 samples, 21 columns)
│   └── feature_columns.json (feature ordering schema)
└── Cache/
    └── .cache/requests_cache.sqlite (HTTP cache)
```

**F. External APIs**
```
Open-Meteo Forecast API
├── Endpoint: https://api.open-meteo.com/v1/forecast
├── Parameters: latitude, longitude, daily=precipitation_sum
├── Response: Tomorrow's rainfall in mm
└── Caching: requests-cache with retry logic

Open-Meteo Archive API
├── Endpoint: https://archive-api.open-meteo.com/v1/archive
├── Parameters: latitude, longitude, start_date, end_date, daily=precipitation_sum
├── Response: Historical daily rainfall (1940-present)
└── Data Delay: 5-day processing delay constraint
```

**G. In-Memory Cache**
```
Streamlit Cache
├── @st.cache_resource
│   └── Model Loading (lgbm_model, rf_model, xgb_model, svr_model, knn_model)
└── @st.cache_data
    ├── Rainfall Forecast Fetching (TTL-based)
    └── Historical Rainfall Fetching (TTL-based)
```

---

## 3. DATA FLOW DIAGRAMS (Textual Descriptions)

### 3.1 Training Pipeline Data Flow

```
[flood.csv] 
    ↓ (load 50,001 samples)
[Data Loader]
    ↓ (validate 21 columns)
[Preprocessing Module]
    ↓ (median/mode imputation)
[Clean Dataset (no nulls)]
    ↓ (extract 20 features + 1 target)
[Feature Extractor]
    ↓ (save feature order)
[feature_columns.json]
    ↓ (split 80/20, stratified)
[Train/Test Split]
    ├─→ [Training Set: 40,000 samples] ──┐
    └─→ [Testing Set: 10,000 samples]    │
                                          ↓
                    ┌─────────────────────┴─────────────────────┐
                    │       5 Model Training (Parallel)          │
                    ├────────────────────────────────────────────┤
                    │ LGBMRegressor(random_state=42)             │
                    │ RandomForestRegressor(n_estimators=50)     │
                    │ XGBRegressor(n_estimators=200, ...)        │
                    │ SVR(kernel='rbf', C=1.0, epsilon=0.1)      │
                    │ KNeighborsRegressor(n_neighbors=7)         │
                    └────────────────────────────────────────────┘
                                          ↓
                    ┌─────────────────────┴─────────────────────┐
                    │         Model Evaluation                   │
                    │  (RMSE, R² on 10,000 test samples)        │
                    └────────────────────────────────────────────┘
                                          ↓
                    ┌─────────────────────┴─────────────────────┐
                    │    Model Serialization (joblib)           │
                    │    compress=9 for GitHub deployment       │
                    └────────────────────────────────────────────┘
                                          ↓
            [5 .pkl files saved to disk] + [Performance Metrics Logged]
```

### 3.2 Live Forecast Data Flow (Homepage)

```
[User Opens Homepage]
    ↓
[Streamlit UI Loads]
    ↓
[Load 5 Models from Cache (@st.cache_resource)]
    ↓
[User Selects City from Dropdown]
    ↓
[Retrieve City Profile (lat, lon, baseline risk factors)]
    ↓
[Call Open-Meteo Forecast API (lat, lon)]
    ↓ (returns tomorrow's rainfall_mm)
[Rainfall Fetched (@st.cache_data, 1-hour TTL)]
    ↓
[Map rainfall_mm to MonsoonIntensity [0-16]]
    ↓
[Generate 100-point Grid Around City Center]
    ↓ (add intra-city variation)
[Create Feature Matrix (100 rows × 20 features)]
    ↓
[User Selects Model (Ensemble or Single)]
    ↓
    ├─→ [Ensemble Mode] ──→ [Predict with all 5 models] ──→ [Average predictions]
    └─→ [Single Model Mode] ──→ [Predict with selected model]
    ↓
[100 Flood Probabilities (0-1)]
    ↓
[User Selects Map Layer (Heatmap/Hexagon/Scatter)]
    ↓
[Pydeck Visualization Renders]
    ↓
[Display: Interactive 3D Map + Risk Statistics]
```

### 3.3 Historical Analysis Data Flow

```
[User Opens Historical Event Analyzer Page]
    ↓
[Load 5 Models from Cache]
    ↓
[User Selects City + Date (date picker with 5-day delay constraint)]
    ↓
[Retrieve City Profile (lat, lon, baseline risk factors)]
    ↓
[Call Open-Meteo Archive API (lat, lon, selected_date)]
    ↓ (returns historical daily rainfall_mm for that date)
[Historical Rainfall Fetched (@st.cache_data)]
    ↓
[Map rainfall_mm to MonsoonIntensity [0-16]]
    ↓
[Construct Feature Vector (1 row × 20 features)]
    ↓
[User Selects Model]
    ↓
[Model Predicts Flood Probability]
    ↓
[Risk Classification Logic]
    ├─→ prob < 0.3 → LOW RISK
    ├─→ 0.3 ≤ prob < 0.7 → MODERATE RISK
    └─→ prob ≥ 0.7 → HIGH RISK
    ↓
[Display: Rainfall (mm) + Flood Probability + Risk Conclusion]
```

### 3.4 Interactive Calculator Data Flow

```
[User Opens Risk Calculator Page]
    ↓
[Load 5 Models from Cache]
    ↓
[Display 20 Slider Inputs (Sidebar)]
    ├─→ MonsoonIntensity [0-16]
    ├─→ TopographyDrainage [0-20]
    ├─→ RiverManagement [0-20]
    ├─→ ... (17 more features)
    └─→ PoliticalFactors [0-20]
    ↓
[User Adjusts Sliders (Real-time)]
    ↓
[Construct Feature Vector from Slider Values]
    ↓
[User Selects Model (Ensemble or Single)]
    ↓
[Model Predicts Flood Probability]
    ↓
[Display: Probability + Risk Gauge]
    ↓
[SHAP Force Plot Generation]
    ├─→ Load TreeExplainer for selected model
    ├─→ Compute SHAP values for feature vector
    └─→ Generate force plot (red=increase risk, blue=decrease risk)
    ↓
[Display: Explainability Visualization]
    ↓
[Optional: Live Weather Helper]
    └─→ Geocode user location → Fetch current weather → Auto-fill MonsoonIntensity
```

### 3.5 Model Performance Dashboard Data Flow

```
[User Opens Model Performance Page]
    ↓
[Load 5 Models from Cache]
    ↓
[Load Test Dataset (10,000 samples)]
    ↓
[Compute Predictions for Each Model]
    ├─→ LightGBM predictions
    ├─→ Random Forest predictions
    ├─→ XGBoost predictions
    ├─→ SVR predictions
    └─→ KNN predictions
    ↓
[Calculate Metrics (RMSE, R²) per Model]
    ↓
[Display: Performance Comparison Table]
    ↓
[User Selects Model for SHAP Analysis]
    ↓
[Initialize TreeExplainer with Selected Model]
    ↓
[Compute Global SHAP Values (sample of test set)]
    ↓
[Generate SHAP Summary Plot (bar chart)]
    ├─→ Feature names on Y-axis
    ├─→ Mean |SHAP value| on X-axis
    └─→ Ranked by importance
    ↓
[Display: Global Feature Importance Visualization]
```

---

## 4. SEQUENCE DIAGRAMS (Textual Descriptions)

### 4.1 Live Forecast Prediction Sequence

```
Actor: User
System Components: Streamlit UI, Model Manager, Feature Engineer, Open-Meteo API, Pydeck

1. User → Streamlit UI: Open homepage
2. Streamlit UI → Model Manager: Load 5 models (@st.cache_resource)
3. Model Manager → Streamlit UI: Return 5 loaded models
4. Streamlit UI → User: Display city dropdown (32 cities)
5. User → Streamlit UI: Select "Mumbai"
6. Streamlit UI → Feature Engineer: Retrieve Mumbai profile (lat=19.0760, lon=72.8777)
7. Feature Engineer → Open-Meteo API: GET /forecast?lat=19.0760&lon=72.8777&daily=precipitation_sum
8. Open-Meteo API → Feature Engineer: Return {"precipitation_sum": [45.2]} (tomorrow)
9. Feature Engineer → Feature Engineer: Map 45.2mm → MonsoonIntensity=12
10. Feature Engineer → Feature Engineer: Generate 100-point grid (±0.05° lat/lon variation)
11. Feature Engineer → Model Manager: Feature matrix (100×20)
12. User → Streamlit UI: Select "Ensemble" mode
13. Model Manager → Model Manager: Predict with all 5 models, average results
14. Model Manager → Streamlit UI: Return 100 probabilities [0.62, 0.58, ..., 0.71]
15. Streamlit UI → Pydeck: Render HeatmapLayer with probabilities
16. Pydeck → User: Display interactive 3D heatmap
17. Streamlit UI → User: Show statistics (Mean: 0.65, Max: 0.78, Min: 0.52)
```

### 4.2 Historical Analysis Sequence

```
Actor: User
System Components: Streamlit UI, Model Manager, Feature Engineer, Archive API

1. User → Streamlit UI: Navigate to "Historical Event Analyzer"
2. Streamlit UI → Model Manager: Load 5 models
3. Streamlit UI → User: Display city dropdown + date picker
4. User → Streamlit UI: Select "Chennai" + Date "2015-11-30"
5. Streamlit UI → Feature Engineer: Retrieve Chennai profile (lat=13.0827, lon=80.2707)
6. Feature Engineer → Archive API: GET /archive?lat=13.0827&lon=80.2707&start_date=2015-11-30&end_date=2015-11-30
7. Archive API → Feature Engineer: Return {"precipitation_sum": [286.0]} (historical rainfall)
8. Feature Engineer → Feature Engineer: Map 286.0mm → MonsoonIntensity=16 (max)
9. Feature Engineer → Model Manager: Feature vector (1×20) with Chennai baseline + intensity=16
10. User → Streamlit UI: Select "XGBoost" model
11. Model Manager → Model Manager: XGBoost.predict(feature_vector)
12. Model Manager → Streamlit UI: Return probability = 0.89
13. Streamlit UI → Streamlit UI: Classify 0.89 → HIGH RISK
14. Streamlit UI → User: Display "Rainfall: 286.0mm | Probability: 0.89 | HIGH RISK ⚠️"
```

### 4.3 SHAP Explanation Sequence

```
Actor: User
System Components: Streamlit UI, Model Manager, SHAP Engine, Matplotlib

1. User → Streamlit UI: Navigate to "Risk Calculator"
2. Streamlit UI → Model Manager: Load 5 models
3. Streamlit UI → User: Display 20 sliders
4. User → Streamlit UI: Adjust sliders (e.g., MonsoonIntensity=14, Deforestation=18)
5. Streamlit UI → Feature Engineer: Construct feature vector from slider values
6. User → Streamlit UI: Select "LightGBM" model
7. Model Manager → Model Manager: LightGBM.predict(feature_vector)
8. Model Manager → Streamlit UI: Return probability = 0.73
9. Streamlit UI → User: Display "Flood Probability: 73%"
10. User → Streamlit UI: Request SHAP explanation (auto-triggered)
11. Streamlit UI → SHAP Engine: Initialize TreeExplainer(lgbm_model)
12. SHAP Engine → SHAP Engine: Compute SHAP values for feature_vector
13. SHAP Engine → SHAP Engine: shap_values = [0.12, -0.05, 0.18, ...] (20 values)
14. SHAP Engine → Matplotlib: Generate force plot
    ├─→ Base value: 0.50 (global mean)
    ├─→ Red arrows: positive SHAP (increase risk) - MonsoonIntensity(+0.12), Deforestation(+0.18)
    └─→ Blue arrows: negative SHAP (decrease risk) - DrainageSystems(-0.05)
15. Matplotlib → Streamlit UI: Return force plot image
16. Streamlit UI → User: Display explainability visualization
```

---

## 5. COMPONENT INTERACTION MATRIX

### 5.1 Component Dependencies

```
Component A → Component B (Dependency Relationship)

Streamlit UI → Model Manager (loads models, requests predictions)
Streamlit UI → Feature Engineer (feature construction, preprocessing)
Streamlit UI → SHAP Engine (explainability requests)
Streamlit UI → Pydeck (visualization rendering)
Streamlit UI → Open-Meteo APIs (weather data fetching)

Model Manager → File System (model loading from .pkl files)
Model Manager → feature_columns.json (feature alignment)
Model Manager → SHAP Engine (provides model for TreeExplainer)

Feature Engineer → Open-Meteo Forecast API (live rainfall data)
Feature Engineer → Open-Meteo Archive API (historical rainfall data)
Feature Engineer → City Profiles (baseline risk factors)

SHAP Engine → Model Manager (requires loaded model)
SHAP Engine → Matplotlib (visualization generation)

Training Pipeline → flood.csv (dataset loading)
Training Pipeline → File System (model saving)
Training Pipeline → feature_columns.json (feature schema saving)

Cache Layer → All API Calls (caching responses)
Cache Layer → Model Loading (caching loaded models)
```

### 5.2 Data Exchange Formats

```
Component A ←→ Component B: Data Format

Streamlit UI ←→ Model Manager: 
    → Feature matrix (numpy array: N×20)
    ← Predictions (numpy array: N×1)

Streamlit UI ←→ Open-Meteo APIs:
    → HTTP GET request (lat, lon, date parameters)
    ← JSON response {"precipitation_sum": [float]}

Model Manager ←→ File System:
    → joblib.load() call
    ← Scikit-learn model object (LGBMRegressor, etc.)

Feature Engineer ←→ City Profiles:
    → City name (string)
    ← Dictionary {lat, lon, MonsoonIntensity, TopographyDrainage, ...}

SHAP Engine ←→ Model Manager:
    → Model object + feature matrix
    ← SHAP values (numpy array: N×20)

Training Pipeline ←→ flood.csv:
    → pandas.read_csv() call
    ← DataFrame (50,001 rows × 21 columns)
```

---

## 6. DEPLOYMENT ARCHITECTURE

### 6.1 Current Deployment (Local)

```
User's Machine
├── Python 3.12 Runtime
├── Streamlit Server (localhost:8501)
├── Local File System
│   ├── app.py + pages/
│   ├── train_models.py
│   ├── flood.csv
│   ├── 5 × .pkl model files
│   └── feature_columns.json
└── Internet Connection (for Open-Meteo APIs)
```

### 6.2 Potential Cloud Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER BROWSER                         │
└─────────────────┬───────────────────────────────────────┘
                  │ HTTPS
                  ↓
┌─────────────────────────────────────────────────────────┐
│            LOAD BALANCER / CDN                          │
│         (e.g., Cloudflare, AWS CloudFront)              │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────────┐
│          WEB SERVER (Streamlit Cloud / EC2)             │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Streamlit Application Container                │   │
│  │  - app.py + pages/                              │   │
│  │  - Python 3.12 environment                      │   │
│  │  - Dependencies (requirements.txt)              │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────┬──────────────┬────────────────────────┘
                  │              │
                  ↓              ↓
    ┌─────────────────────┐  ┌──────────────────────┐
    │  OBJECT STORAGE     │  │  EXTERNAL APIS       │
    │  (AWS S3 / Azure)   │  │  - Open-Meteo        │
    │  - Models (.pkl)    │  │  - Geocoding         │
    │  - flood.csv        │  │                      │
    │  - feature_columns  │  │                      │
    └─────────────────────┘  └──────────────────────┘
                  │
                  ↓
    ┌─────────────────────────┐
    │   CACHE LAYER           │
    │   (Redis / Memcached)   │
    │   - API responses       │
    │   - Model predictions   │
    └─────────────────────────┘
```

**Deployment Options:**
1. **Streamlit Cloud:** Direct GitHub integration, free tier available
2. **AWS EC2 + S3:** Full control, scalable compute
3. **Google Cloud Run:** Containerized deployment, auto-scaling
4. **Azure App Service:** Managed PaaS, easy deployment
5. **Heroku:** Quick deployment, limited free tier

---

## 7. SECURITY & PERFORMANCE CONSIDERATIONS

### 7.1 Security Architecture

```
Security Layer: API Rate Limiting
├── Open-Meteo API: Respect rate limits (10,000 requests/day free tier)
├── Implement retry logic with exponential backoff
└── requests-cache to reduce redundant API calls

Security Layer: Input Validation
├── Feature value range validation (0-16, 0-20 scales)
├── Date picker constraints (1940 to today-5days)
├── City selection from predefined list (no arbitrary input)
└── Model selection from predefined options

Security Layer: Data Privacy
├── No user personal data collected
├── No authentication/authorization required (public tool)
└── No storage of user predictions (ephemeral sessions)

Security Layer: Model Integrity
├── Models serialized with joblib (trusted library)
├── Checksum validation on model loading (future enhancement)
└── Read-only model files (no runtime modification)
```

### 7.2 Performance Optimization

```
Performance Layer: Caching Strategy
├── @st.cache_resource for model loading
│   └── Load once, persist across sessions
├── @st.cache_data for API responses
│   └── TTL-based expiration (1-hour for forecasts)
├── requests-cache for HTTP responses
│   └── SQLite backend, configurable expiration
└── Browser-side caching (Streamlit default)

Performance Layer: Lazy Loading
├── Models loaded on-demand per page
├── SHAP computations triggered only when requested
└── Pydeck visualizations rendered incrementally

Performance Layer: Model Compression
├── joblib compress=9 for .pkl files
├── Reduced Random Forest trees (50 vs. default 100)
└── Optimized XGBoost tree_method='hist'

Performance Layer: Batch Predictions
├── Grid-based predictions (100 points) in single model call
├── Vectorized numpy operations
└── Avoid Python loops for feature engineering
```

---

## 8. ERROR HANDLING & RESILIENCE

### 8.1 Error Handling Flow

```
Error Type: API Failure (Open-Meteo unreachable)
├─→ Catch requests.exceptions.RequestException
├─→ Retry with exponential backoff (3 attempts)
├─→ If all retries fail:
│   ├─→ Use default/historical rainfall value
│   └─→ Display warning message to user
└─→ Log error for monitoring

Error Type: Model Loading Failure
├─→ Catch FileNotFoundError / pickle.UnpicklingError
├─→ Display error message: "Model file corrupted or missing"
├─→ Suggest retraining: "Run train_models.py"
└─→ Graceful degradation: Disable affected page

Error Type: Invalid Feature Values
├─→ Validate input ranges before prediction
├─→ Clip values to valid ranges (e.g., [0, 16])
├─→ Display warning: "Value adjusted to valid range"
└─→ Proceed with corrected values

Error Type: SHAP Computation Failure
├─→ Catch exceptions during TreeExplainer.shap_values()
├─→ Display message: "Explainability unavailable for this model"
├─→ Continue showing prediction without SHAP plot
└─→ Log error (model compatibility issue)

Error Type: Pydeck Rendering Failure
├─→ Catch JavaScript/WebGL errors
├─→ Fallback to tabular prediction display
├─→ Display message: "Map visualization unavailable"
└─→ Suggest browser update or WebGL enablement
```

### 8.2 Resilience Strategies

```
Strategy: Graceful Degradation
├── If XGBoost fails → Use LightGBM as fallback
├── If ensemble fails → Use single best model (XGBoost)
├── If live weather fails → Use historical average
└── If visualization fails → Show text-based results

Strategy: Data Validation
├── Schema validation for feature_columns.json
├── Model compatibility checks (sklearn version)
├── API response validation (expected JSON structure)
└── User input sanitization (slider constraints)

Strategy: Monitoring & Logging
├── Log model loading times (performance monitoring)
├── Log API response times (detect degradation)
├── Log prediction latencies (user experience tracking)
└── Log error frequencies (identify failure patterns)
```

---

## 9. SCALABILITY CONSIDERATIONS

### 9.1 Horizontal Scaling

```
Current Architecture: Single-instance Streamlit
└── Limitation: ~1000 concurrent users

Scaled Architecture: Multi-instance Deployment
├── Load Balancer distributes traffic across N Streamlit instances
├── Shared model storage (S3/Azure Blob)
├── Distributed cache (Redis Cluster)
└── Supports 10,000+ concurrent users

Implementation:
1. Containerize application (Docker)
2. Deploy to Kubernetes cluster or serverless platform
3. Configure auto-scaling (CPU/memory thresholds)
4. Implement session affinity (sticky sessions)
```

### 9.2 Vertical Scaling

```
CPU Optimization:
├── Current: Single-threaded prediction
├── Scaled: Multi-threaded batch predictions (joblib n_jobs=-1)
└── Impact: 2-4× speedup for large grids

Memory Optimization:
├── Current: All models loaded simultaneously (~500MB RAM)
├── Scaled: Lazy model loading per request
└── Impact: Reduce baseline memory by 80%

GPU Acceleration (Future):
├── XGBoost GPU training (tree_method='gpu_hist')
├── TensorFlow/PyTorch model conversion for inference
└── Impact: 10-100× speedup for training and inference
```

---

## 10. FUTURE ENHANCEMENTS

### 10.1 Planned Features

```
Enhancement: Real-time Alerting System
├── User subscription to specific cities
├── Email/SMS notifications when risk > threshold
├── Integration with SMTP server or Twilio API
└── Webhook support for third-party integrations

Enhancement: Historical Validation Dashboard
├── Compare model predictions vs. actual flood events (if available)
├── Precision/Recall metrics over time
├── Interactive timeline visualization
└── Model retraining trigger based on drift detection

Enhancement: Multi-region Support
├── Expand beyond India (Southeast Asia, Americas, Europe)
├── Localized weather API endpoints
├── Region-specific risk factor weights
└── Multi-language UI support

Enhancement: Advanced Ensemble Techniques
├── Weighted averaging (based on model confidence)
├── Stacking meta-learner (train combiner model)
├── Dynamic model selection (context-dependent)
└── Uncertainty quantification (prediction intervals)

Enhancement: Mobile Application
├── React Native / Flutter mobile app
├── Offline prediction capability (cached models)
├── GPS-based automatic location detection
└── Push notifications for local alerts
```

### 10.2 Technical Debt & Improvements

```
Code Quality:
├── Add comprehensive unit tests (pytest)
├── Implement integration tests (API mocking)
├── Add type hints (mypy validation)
└── Refactor duplicated code (DRY principle)

Documentation:
├── Add inline docstrings (Google/NumPy style)
├── Generate API documentation (Sphinx)
├── Create user manual (README expansion)
└── Video tutorials for deployment

Infrastructure:
├── CI/CD pipeline (GitHub Actions)
├── Automated model retraining (scheduled jobs)
├── A/B testing framework (model version comparison)
└── Monitoring dashboard (Grafana/Prometheus)
```

---

## 11. VISUAL DIAGRAM RECOMMENDATIONS

### 11.1 Suggested Diagrams to Create

**1. System Context Diagram**
- Actors: End Users, Administrators, External APIs
- System Boundary: Project AQUA Application
- External Systems: Open-Meteo API, GitHub Repository
- Purpose: High-level overview of system interactions

**2. Container Diagram (C4 Model)**
- Containers: Streamlit Web App, Model Storage, Dataset Storage, Cache
- Technology labels: Python 3.12, Streamlit, joblib, requests-cache
- Purpose: Show major deployable units and technologies

**3. Component Diagram**
- Within "Streamlit Web App" container
- Components: Homepage, Historical Analyzer, Calculator, Performance Dashboard, Model Manager, SHAP Engine
- Purpose: Internal structure of main application

**4. Sequence Diagrams** (Use textual descriptions from Section 4)
- Live Forecast Flow
- Historical Analysis Flow
- SHAP Explanation Flow

**5. Deployment Diagram**
- Show current local deployment
- Show proposed cloud deployment architecture
- Include network boundaries, firewalls, load balancers

**6. Data Flow Diagram (DFD)**
- Training Pipeline Flow
- Live Prediction Flow
- Show data transformations at each stage

**7. Entity-Relationship Diagram** (if applicable)
- flood.csv structure
- feature_columns.json schema
- City profiles structure

**8. State Machine Diagram**
- User session states: Landing → City Selection → Prediction → Explainability
- Model states: Unloaded → Loading → Loaded → Predicting

### 11.2 Diagramming Tools Recommendations

- **Draw.io (diagrams.net):** Free, web-based, extensive shape libraries
- **Lucidchart:** Professional, collaborative, templates for UML/C4
- **PlantUML:** Text-based, version-controllable, integrates with documentation
- **Mermaid:** Markdown-integrated, GitHub-rendered, simple syntax
- **Microsoft Visio:** Professional, comprehensive, Windows-native
- **Excalidraw:** Hand-drawn style, lightweight, open-source

---

## 12. APPENDIX: KEY DESIGN PATTERNS USED

### 12.1 Architectural Patterns

```
Pattern: Three-Tier Architecture
├── Separation of concerns: Presentation, Business Logic, Data
├── Benefit: Maintainability, independent scaling
└── Implementation: Streamlit (UI) | Python modules (logic) | Files/APIs (data)

Pattern: Model-View-Controller (MVC)
├── Model: ML models, data structures (City profiles)
├── View: Streamlit UI components (pages, visualizations)
└── Controller: Feature engineering, prediction routing

Pattern: Facade Pattern
├── Model Manager provides simple interface to 5 complex models
├── Feature Engineer abstracts preprocessing complexity
└── SHAP Engine hides explainability computation details

Pattern: Strategy Pattern
├── Prediction strategy: Ensemble vs. Single model
├── Visualization strategy: Heatmap vs. Hexagon vs. Scatter
└── Explainability strategy: Global vs. Local SHAP
```

### 12.2 Design Principles Applied

```
SOLID Principles:
├── Single Responsibility: Each page handles one concern
├── Open/Closed: Add new models without modifying core logic
├── Liskov Substitution: All models implement same predict() interface
├── Interface Segregation: Separate interfaces for classifiers vs. regressors
└── Dependency Inversion: Depend on abstractions (model interface) not concretions

DRY (Don't Repeat Yourself):
├── City profiles defined once, used across all pages
├── Model loading logic centralized in single function
└── Feature engineering utilities shared across modules

KISS (Keep It Simple, Stupid):
├── Straightforward linear prediction pipeline
├── Minimal external dependencies
└── Clear, readable code over clever optimizations
```

---

**End of System Design Specification**

This document provides all necessary textual information to create comprehensive visual system design diagrams. Use the component descriptions, data flows, and interaction patterns as a blueprint for your visual designs.

**Recommended Next Steps:**
1. Create high-level System Context Diagram
2. Design detailed Component Diagram
3. Document data flows with DFD
4. Visualize deployment architecture
5. Add sequence diagrams for key user flows

For questions or clarifications on any section, refer back to the PROJECT_STATUS_REPORT.md for implementation details.
