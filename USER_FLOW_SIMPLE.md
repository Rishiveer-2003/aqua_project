# Project AQUA: Simple User Flow
**A straightforward flowchart showing how users interact with the system**

---

## MAIN USER FLOW (Start to End)

```
START
  ↓
User opens browser and goes to: http://localhost:8501
  ↓
[Streamlit App Loads]
  ↓
System loads 5 trained ML models from disk (.pkl files)
  ↓
User sees Homepage: "India Live Flood Forecast"
  ↓
┌─────────────────────────────────────────────────────┐
│        USER CHOOSES ONE OF 4 OPTIONS:               │
├─────────────────────────────────────────────────────┤
│  Option 1: Use Homepage (Live Forecast)             │
│  Option 2: Go to Historical Event Analyzer          │
│  Option 3: Go to Model Performance Dashboard        │
│  Option 4: Go to Risk Calculator                    │
└─────────────────────────────────────────────────────┘
  ↓
[User selects one option - see detailed flows below]
  ↓
System shows prediction results
  ↓
User views results (map, probability, charts, etc.)
  ↓
User can:
  - Try different cities
  - Try different dates
  - Adjust parameters
  - Switch between pages
  - Close browser
  ↓
END
```

---

## OPTION 1: Live Forecast Flow (Homepage)

```
User on Homepage
  ↓
1. User selects a city from dropdown (e.g., "Mumbai")
  ↓
2. System fetches tomorrow's rainfall from weather API
  ↓
3. System converts rainfall to risk factors
  ↓
4. User chooses model: "Ensemble" or specific model (XGBoost, LightGBM, etc.)
  ↓
5. System predicts flood risk for 100 points around the city
  ↓
6. User sees:
   - Interactive 3D map (heatmap showing risk levels)
   - Average flood probability percentage
   - Risk statistics (min, max, mean)
  ↓
7. User can:
   - Change city → Go back to step 1
   - Change model → Go back to step 4
   - Switch map style (Heatmap/Hexagon/Scatter)
   - Go to another page
  ↓
Done with Homepage
```

**Visual Summary:**
```
Select City → Fetch Weather → Calculate Risk → Show Map → Done
```

---

## OPTION 2: Historical Event Analyzer Flow

```
User goes to "Historical Event Analyzer" page
  ↓
1. User selects a city (e.g., "Chennai")
  ↓
2. User selects a past date (e.g., "November 30, 2015")
  ↓
3. System fetches historical rainfall for that date from API
  ↓
4. User chooses a model (e.g., "XGBoost")
  ↓
5. System calculates what the flood risk was on that day
  ↓
6. User sees:
   - Rainfall amount (mm)
   - Flood probability (0-100%)
   - Risk level: LOW / MODERATE / HIGH
  ↓
7. User can:
   - Try different dates → Go back to step 2
   - Try different cities → Go back to step 1
   - Try different models → Go back to step 4
  ↓
Done with Historical Analysis
```

**Visual Summary:**
```
Select City → Select Date → Fetch Historical Data → Calculate Risk → Show Result → Done
```

---

## OPTION 3: Model Performance Dashboard Flow

```
User goes to "Model Performance Dashboard" page
  ↓
1. System shows performance table for all 5 models:
   - LightGBM: R² = 0.92
   - Random Forest: R² = 0.72
   - XGBoost: R² = 0.96 (BEST)
   - SVR: R² = 0.72
   - KNN: R² = 0.78
  ↓
2. User selects a model to see feature importance
  ↓
3. System shows SHAP chart:
   - Which factors matter most for predictions
   - Bar chart ranking features by importance
  ↓
4. User can:
   - Select different model → Go back to step 2
   - Read metrics and compare models
  ↓
Done with Performance Dashboard
```

**Visual Summary:**
```
View Metrics Table → Select Model → View Feature Importance → Done
```

---

## OPTION 4: Risk Calculator Flow

```
User goes to "Risk Calculator" page
  ↓
1. User sees 20 sliders on the left sidebar:
   - MonsoonIntensity (0-16)
   - TopographyDrainage (0-20)
   - RiverManagement (0-20)
   - Deforestation (0-20)
   - ... (16 more factors)
  ↓
2. User adjusts sliders to create custom scenario
   Example: Set MonsoonIntensity=14, Deforestation=18
  ↓
3. User chooses model (Ensemble or specific)
  ↓
4. System calculates flood probability in REAL-TIME
   (updates as user moves sliders)
  ↓
5. User sees:
   - Flood probability percentage
   - SHAP explanation chart showing:
     * Which factors increased risk (red)
     * Which factors decreased risk (blue)
  ↓
6. User can:
   - Adjust more sliders → System recalculates automatically
   - Change model → Go back to step 3
   - Use "Live Weather Helper" to auto-fill current weather
  ↓
Done with Risk Calculator
```

**Visual Summary:**
```
Adjust 20 Sliders → Select Model → See Probability + Explanation → Done
```

---

## COMPLETE SYSTEM FLOW (Bird's Eye View)

```
┌─────────────────────────────────────────────────────────────┐
│                    USER STARTS                              │
└────────────────────────┬────────────────────────────────────┘
                         ↓
                  Opens Web Browser
                         ↓
              Goes to localhost:8501
                         ↓
┌────────────────────────┴────────────────────────────────────┐
│              STREAMLIT APP LOADS                            │
│  (Loads 5 ML models: LightGBM, RF, XGBoost, SVR, KNN)      │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌────────────────────────┴────────────────────────────────────┐
│                   HOMEPAGE DISPLAYS                         │
│            "India Live Flood Forecast"                      │
└────────────────────────┬────────────────────────────────────┘
                         ↓
              ┌──────────┴──────────┐
              │   User Navigates    │
              └──────────┬──────────┘
                         ↓
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
   [Homepage]    [Historical]     [Performance]     [Calculator]
        │                │                │                │
        ↓                ↓                ↓                ↓
  Select City    Select City+Date   View Metrics    Adjust Sliders
        ↓                ↓                ↓                ↓
  Fetch Weather   Fetch History    Select Model    Select Model
        ↓                ↓                ↓                ↓
  Predict Risk    Predict Risk     View SHAP       Predict Risk
        ↓                ↓                ↓                ↓
  Show Map        Show Result      Show Chart      Show Result+SHAP
        │                │                │                │
        └────────────────┼────────────────┘                │
                         ↓                                 │
                 User Reviews Results ←────────────────────┘
                         ↓
                 ┌───────┴────────┐
                 │  Satisfied?    │
                 └───────┬────────┘
                         │
                    No ──┴─→ Try different inputs
                    │         (go back to any page)
                    │
                   Yes
                    ↓
┌────────────────────────────────────────────────────────────┐
│                    USER ENDS SESSION                        │
│                   (Closes Browser)                         │
└────────────────────────────────────────────────────────────┘
```

---

## BEHIND THE SCENES (What Happens Internally)

```
User Action                    →    System Response
─────────────────────────────────────────────────────────────

1. Opens app                   →    Load 5 models from .pkl files

2. Selects city                →    Get city lat/lon from database

3. Clicks predict              →    Fetch weather from API
                                    Convert weather to features
                                    Run model prediction
                                    
4. Views results               →    Create visualization
                                    Display on screen
                                    
5. Changes parameters          →    Recalculate prediction
                                    Update display instantly
                                    
6. Requests explanation        →    Run SHAP analysis
                                    Generate importance chart
```

---

## DATA FLOW (Simplified)

```
INPUT                      PROCESSING                    OUTPUT
─────────────────────────────────────────────────────────────────

User selects city      →   System gets:                →  3D Risk Map
(e.g., Mumbai)             - Latitude: 19.07              Probability: 65%
                           - Longitude: 72.87             Risk: MODERATE
                           ↓
                       Fetch tomorrow's
                       rainfall from API
                       (e.g., 45mm)
                       ↓
                       Convert to risk factors:
                       - MonsoonIntensity: 12
                       - Other factors: from
                         city profile
                       ↓
                       Feed into ML model
                       (20 features total)
                       ↓
                       Model calculates
                       flood probability
```

---

## QUICK SUMMARY FOR DIAGRAM

**If you want to draw just ONE simple flowchart, use this:**

```
START
  ↓
User opens app
  ↓
System loads ML models
  ↓
User sees 4 page options
  ↓
User picks one:
  • Live Forecast → Select city → See risk map
  • Historical → Select city+date → See past risk
  • Performance → View model accuracy metrics
  • Calculator → Adjust 20 sliders → See custom risk
  ↓
System shows results
  ↓
User satisfied? 
  No → Try again
  Yes → Exit
  ↓
END
```

---

## ONE-LINE SUMMARY PER PAGE

1. **Homepage:** User picks city → System predicts tomorrow's flood risk → Shows heatmap
2. **Historical:** User picks city + past date → System shows what risk was that day → Shows number
3. **Performance:** User views which models perform best → System shows accuracy table + charts
4. **Calculator:** User adjusts 20 risk sliders → System predicts risk in real-time → Shows percentage + explanation

---

**That's it! This is the complete user journey from opening the app to getting results.**

Each user interaction is 3-5 steps maximum, making it simple and intuitive.
