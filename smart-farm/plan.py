import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Standard daily water requirements (mm/day)
CROP_DAILY_WATER_NEEDS = {
    "rice": 8.0,
    "wheat": 4.2,
    "maize": 5.5,
    "cotton": 6.0,
    "soybean": 5.0,
    "millet": 3.5,
    "default": 4.5
}

def generate_crop_schedule(crop_name: str, duration: int = 100, forecast_csv_path: str = "data/future_prediction_results.csv") -> List[Dict]:
    """
    Generates a high-fidelity farming schedule for the full crop duration with an integrated
    irrigation recommendation system, growth lifecycle tracking, and climate risk mitigation.
    """
    crop_lower = crop_name.lower().strip()
    daily_water_need = CROP_DAILY_WATER_NEEDS.get(crop_lower, CROP_DAILY_WATER_NEEDS["default"])
    
    # Load daily weather predictions if available
    path = Path(forecast_csv_path)
    daily_weather_data = []
    
    if path.exists():
        try:
            df = pd.read_csv(path)
            # Normalize headers
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            daily_weather_data = df.to_dict(orient="records")
        except Exception as e:
            print(f"[PlanningEngine] Warning: Could not read forecast CSV ({e}). Using simulated weather baseline.")
            
    # Fallback to simulated forecast if file not present or empty
    if not daily_weather_data:
        np.random.seed(42)
        dates = pd.date_range(start=datetime.now().strftime("%Y-%m-%d"), periods=duration)
        daily_weather_data = [{
            "date": d.strftime('%Y-%m-%d'),
            "temp": np.round(np.random.normal(28, 2), 2),
            "rainfall": np.round(np.random.exponential(1.5), 2) if np.random.rand() > 0.8 else 0.0,
            "humidity": np.round(np.random.uniform(50, 80), 2),
            "similarity_score": 75.0,
            "success_probability": 80.0,
            "risk_level": "medium"
        } for d in dates]
        
    schedule = []
    
    # Proportional lifecycle thresholds
    germ_end = int(duration * 0.15)
    veg_end = int(duration * 0.45)
    flow_end = int(duration * 0.75)

    for i in range(min(duration, len(daily_weather_data))):
        day_info = daily_weather_data[i]
        day_num = i + 1
        
        # 1. Lifecycle Analysis (Proportional)
        if day_num <= germ_end:
            stage = "Germination & Emergence"
            core_activity = "Ensure stable moisture for seed sprouting; inspect soil crusting."
        elif day_num <= veg_end:
            stage = "Vegetative Growth"
            core_activity = "Monitor leaf area development; apply nitrogen-rich fertilizer."
        elif day_num <= flow_end:
            stage = "Flowering & Pollination"
            core_activity = "Critical reproductive phase: prevent water stress at all costs."
        else:
            stage = "Yield Formation & Ripening"
            core_activity = "Prepare for grain filling; gradually reduce watering frequency."
            
        # 2. Irrigation Recommendation System
        # Calculate water deficit: Deficit = Water Need - Forecast Rainfall
        rainfall = day_info.get("rainfall", 0.0)
        water_deficit = max(0.0, daily_water_need - rainfall)
        
        if water_deficit > 0:
            irrigation_needed = True
            irrigation_volume = np.round(water_deficit, 2)
            irrigation_action = f"Apply {irrigation_volume} mm of targeted irrigation."
        else:
            irrigation_needed = False
            irrigation_volume = 0.0
            irrigation_action = "No irrigation required: forecast rainfall meets crop requirements."
            
        # 3. Climate Risk Mitigation
        risk = day_info.get("risk_level", "medium").lower()
        actions = []
        
        # Add basic watering action
        actions.append(irrigation_action)
        
        # Risk specific alerts
        if risk == "high":
            actions.append("WARNING: Extreme weather deviation! Implement emergency water-saving mulching.")
        elif risk == "medium":
            actions.append("Alert: Moderate climatic stress detected. Monitor soil moisture levels closely.")
            
        # Day-specific scheduling activities
        if day_num in [7, 21]:
            actions.append("Apply NPK balanced fertilizers to support growth cycle.")
        if day_num in [12, 26]:
            actions.append("Perform manual or organic weeding around root zones.")
            
        actions.append(core_activity)
        
        schedule.append({
            "day": day_num,
            "date": day_info.get("date", "N/A"),
            "stage": stage,
            "temperature": day_info.get("temp", 25.0),
            "rainfall": rainfall,
            "humidity": day_info.get("humidity", 60.0),
            "similarity_score": day_info.get("similarity_score", 75.0),
            "risk_level": risk,
            "irrigation_needed": irrigation_needed,
            "irrigation_volume_mm": irrigation_volume,
            "actions": actions
        })
        
    return schedule

if __name__ == "__main__":
    plan = generate_crop_schedule("Wheat", duration=10)
    print(f"Generated advanced {len(plan)} day schedule.")
    print("Day 1 Plan Sample:")
    print(plan[0])
