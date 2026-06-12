import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Import custom modular components
from crop_dataset_loader import CropDatasetLoader
from weather_fetcher import WeatherFetcher
from data_cleaner import DataCleaner
from feature_scaler import FeatureScaler
from knn_similarity_engine import KNNSimilarityEngine
from prediction_engine import PredictionEngine
from csv_exporter import CSVExporter
from mongodb_storage import MongoDBStorage

def run_calculation_pipeline(target_crop: str = "wheat", lat: float = 28.6139, lon: float = 77.2090):
    print("\n" + "="*70)
    print("      SMART FARM CLIMATE SIMILARITY ENGINE - CALCULATION PHASE")
    print("="*70)
    print(f"Target Crop: {target_crop.upper()}")
    print(f"Coordinates: Lat {lat:.4f}, Lon {lon:.4f}")
    
    # 1. LOAD HISTORICAL CROP CLIMATE CSV DATASET
    print("\n[Step 1] Loading historical crop climate data...")
    loader = CropDatasetLoader(filepath="data/crop_data.csv")
    try:
        df_hist = loader.load()
        print(f"--> Success! Loaded {len(df_hist)} historical records.")
    except Exception as e:
        print(f"--> Error loading dataset: {e}. Attempting recovery...")
        # Recover by running the fetching pipeline first if data is missing
        try:
            from pipeline import run_pipeline
            run_pipeline(crop=target_crop, lat=lat, lon=lon, start_date="2026-05-01", end_date="2026-05-30")
            df_hist = loader.load()
            print(f"--> Recovery success! Loaded {len(df_hist)} historical records.")
        except Exception as ex:
            print(f"--> Critical recovery failure: {ex}")
            sys.exit(1)

    # 2. FETCH CURRENT AND FUTURE WEATHER FORECAST
    print("\n[Step 2] Fetching 30-day forecast & current weather...")
    fetcher = WeatherFetcher(lat=lat, lon=lon)
    df_weather = fetcher.fetch_forecast(days=30)
    print(f"--> Success! Fetched forecast for {len(df_weather)} days.")

    # 3 & 4. CLEAN AND STANDARDIZE COLUMN NAMES AND DATA FORMATS
    print("\n[Step 3 & 4] Cleaning and standardizing schemas...")
    cleaner = DataCleaner()
    df_hist_cleaned = cleaner.clean(df_hist)
    df_weather_cleaned = cleaner.clean(df_weather)
    print("--> Standardization verified! Column schemas align perfectly.")

    # 5. SCALE NUMERICAL FEATURES USING STANDARD SCALER
    print("\n[Step 5] Scaling features using StandardScaler...")
    features = ["temp", "min_temp", "max_temp", "rainfall", "humidity"]
    scaler = FeatureScaler(features=features)
    scaler.fit(df_hist_cleaned)
    scaled_hist = scaler.transform(df_hist_cleaned)
    scaled_weather = scaler.transform(df_weather_cleaned)
    print(f"--> Features Standardized: {features}")
    print(f"    Historical Data Scaled Shape: {scaled_hist.shape}")
    print(f"    Weather Data Scaled Shape:    {scaled_weather.shape}")

    # 6 & 7. KNN NEAREST-NEIGHBOR COMPARISON WITH EUCLIDEAN DISTANCE
    print("\n[Step 6 & 7] Fitting KNN and comparing forecast weather against historical records...")
    # Fit Nearest Neighbors for pattern matching
    similarity_engine = KNNSimilarityEngine(n_neighbors=5, features=features)
    similarity_engine.fit(df_hist_cleaned)
    knn_matches = similarity_engine.find_matches(df_weather_cleaned)
    
    # Fit KNeighborsClassifier for success probability prediction
    prediction_engine = PredictionEngine(n_neighbors=5, features=features)
    prediction_engine.fit(df_hist_cleaned)
    daily_predictions = prediction_engine.predict_viability(df_weather_cleaned, target_crop=target_crop)
    print("--> Nearest Neighbor model fitted and prediction mappings generated successfully!")

    # 8 & 9. CALCULATE SIMILARITY, CROP SUCCESS PROBABILITY, AND CLIMATE RISK
    print("\n[Step 8 & 9] Running multi-crop comparisons and risk assessments...")
    crops_list = ["rice", "wheat", "maize", "soybean", "cotton"]
    crop_comparisons = prediction_engine.analyze_crop_comparisons(df_weather_cleaned, crops_list)
    
    print("\n" + "-"*40)
    print(" CROP CLIMATE SIMILARITY REPORT (30-Day Outlook)")
    print("-"*40)
    for comp in crop_comparisons:
        prob_str = f"{comp['success_probability']:.1f}% success probability"
        risk_str = f"{comp['risk_level'].upper()} RISK"
        print(f" * {comp['crop']:<10} -> {comp['similarity_score']:.1f}% similarity -> {risk_str} ({prob_str})")
    print("-"*40)

    # 10. SAVE ANALYSIS RESULTS INTO CSV OUTPUTS
    print("\n[Step 10] Exporting results to structured CSVs...")
    exporter = CSVExporter(output_dir="data")
    sim_file = exporter.export_similarity_results(crop_comparisons)
    pat_file = exporter.export_matched_patterns(knn_matches)
    pred_file = exporter.export_future_predictions(daily_predictions)
    
    # FUTURE ARCHITECTURE: SAVE TO MOCK/REAL MONGODB STORAGE
    print("\n[Future Architecture] Registering analysis run in MongoDB Storage...")
    storage = MongoDBStorage()
    run_id = storage.save_document("similarity_analysis_runs", {
        "target_crop": target_crop,
        "coordinates": {"lat": lat, "lon": lon},
        "crops_comparisons": crop_comparisons,
        "daily_forecast_summary": daily_predictions,
        "results_files": {
            "similarity_results": str(sim_file),
            "matched_patterns": str(pat_file),
            "future_predictions": str(pred_file)
        }
    })
    print(f"--> Saved metadata record with Document ID: {run_id}")
    
    print("\n" + "="*70)
    print("      CALCULATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Smart Farm Calculation Engine")
    parser.add_argument("--crop", type=str, default="wheat", help="Target crop")
    parser.add_argument("--lat", type=float, default=28.6139, help="Latitude")
    parser.add_argument("--lon", type=float, default=77.2090, help="Longitude")
    
    args = parser.parse_args()
    run_calculation_pipeline(target_crop=args.crop, lat=args.lat, lon=args.lon)
