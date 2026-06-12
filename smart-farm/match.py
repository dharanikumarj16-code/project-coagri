import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import modular components from the existing codebase
from crop_dataset_loader import CropDatasetLoader
from weather_fetcher import WeatherFetcher
from data_cleaner import DataCleaner
from prediction_engine import PredictionEngine

def collect_user_inputs():
    """
    Collects the 7 specific agricultural inputs requested by the user.
    """
    print("\n" + "="*50)
    print("  AGRICULTURE AI: CLIMATE SIMILARITY ENGINE")
    print("="*50)
    
    farmer_location = input("Enter Farmer Location (City/Region): ")
    crop_name = input("Enter Crop Name (e.g., Rice, Wheat, Millet): ")
    water_availability = input("Enter Water Availability (mm/season): ")
    land_area = input("Enter Land Area (acres): ")
    planting_date = input("Enter Planting Date (YYYY-MM-DD): ")
    forecast_days = input("Enter Forecast Days (e.g., 30): ")
    soil_ph = input("Enter Soil pH (0-14): ")
    
    return {
        "farmer_location": farmer_location,
        "crop_name": crop_name,
        "water_availability": float(water_availability) if water_availability else 500.0,
        "land_area": float(land_area) if land_area else 1.0,
        "planting_date": planting_date if planting_date else datetime.now().strftime("%Y-%m-%d"),
        "forecast_days": int(forecast_days) if forecast_days else 30,
        "soil_ph": float(soil_ph) if soil_ph else 6.5
    }

def run_similarity_check(inputs):
    """
    Processes inputs and returns formatted similarity results.
    """
    crop_target = inputs['crop_name'].lower().strip()
    
    # 1. Load Data
    loader = CropDatasetLoader(filepath="data/crop_data.csv")
    try:
        df_hist = loader.load()
    except Exception:
        # Fallback synthesis if data missing
        print("[System] Historical data missing. Initializing bootstrap...")
        from pipeline import run_pipeline
        run_pipeline(crop=crop_target, lat=28.61, lon=77.20, start_date="2026-05-01", end_date="2026-05-30")
        df_hist = loader.load()

    # 2. Setup Prediction Engine
    features = ["temp", "min_temp", "max_temp", "rainfall", "humidity", "soil_ph"]
    engine = PredictionEngine(features=features)
    engine.fit(df_hist)
    
    # 3. Fetch Forecast
    fetcher = WeatherFetcher(lat=28.6139, lon=77.2090)
    df_forecast = fetcher.fetch_forecast(days=inputs['forecast_days'])
    
    # Add user provided soil_ph to the forecast df for prediction
    df_forecast['soil_ph'] = inputs['soil_ph']
    
    # 4. Analyze Comparisons
    crops_to_compare = [crop_target, "rice", "wheat", "maize", "millet"]
    # Ensure uniqueness
    crops_to_compare = list(set([c.lower() for c in crops_to_compare]))
    
    comparisons = engine.analyze_crop_comparisons(df_forecast, crops_to_compare)
    
    print("\n" + "="*50)
    print(" SIMILARITY RESULTS")
    print("="*50)
    
    for comp in comparisons:
        crop_name = comp['crop'].capitalize()
        sim = comp['similarity_score']
        success = comp['success_probability']
        risk = comp['risk_level']
        
        # Recommendation Logic
        if sim >= 85 and success >= 80:
            rec = "yes"
        elif sim >= 70 and success >= 60:
            rec = "alternative option like this"
        else:
            rec = "no"
            
        print(f"\n{crop_name}:\n")
        print(f"climate similarity: {sim:.0f}%")
        print(f"success rate: {success:.0f}%")
        print(f"risk level: {risk}")
        print(f"recommended: {rec}")

if __name__ == "__main__":
    import os
    # Ensure we are in the right directory for local file imports and data paths
    os.chdir(Path(__file__).parent)
    
    try:
        user_data = collect_user_inputs()
        run_similarity_check(user_data)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"\nError: {e}")
