from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import io
import contextlib

# Import modular components
from pipeline import run_pipeline, get_user_location
from crop_dataset_loader import CropDatasetLoader
from weather_fetcher import WeatherFetcher
from prediction_engine import PredictionEngine
from data_cleaner import DataCleaner
from plan import generate_crop_schedule

app = FastAPI(title="SmartFarm AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PipelineRequest(BaseModel):
    crop: str
    lat: float
    lon: float

# Comprehensive Master List of Indian Crops
MASTER_INDIAN_CROPS = [
    "Rice", "Basmati Rice", "Paddy", "Wheat", "Bajra (Pearl Millet)", "Ragi (Finger Millet)", 
    "Jowar (Sorghum)", "Foxtail Millet", "Kodo Millet", "Barnyard Millet", "Little Millet", 
    "Toor Dal (Pigeon Pea)", "Urad Dal", "Moong Dal", "Chana (Chickpea)", "Masoor Dal", 
    "Groundnut", "Mustard", "Sunflower", "Sesame", "Soybean", "Castor", "Sugarcane", 
    "Cotton", "Jute", "Tea", "Coffee", "Rubber", "Coconut", "Arecanut", "Tomato", 
    "Potato", "Onion", "Brinjal (Eggplant)", "Cabbage", "Cauliflower", "Okra (Lady Finger)", 
    "Carrot", "Beetroot", "Chilli", "Mango", "Banana", "Apple", "Orange", "Grapes", 
    "Guava", "Papaya", "Pomegranate", "Watermelon", "Turmeric", "Black Pepper", 
    "Cardamom", "Clove", "Coriander", "Cumin", "Fenugreek", "Ginger", "Maize (Corn)", 
    "Saffron", "Cocoa", "Aloe Vera", "Medicinal Plants"
]

@app.get("/api/crops")
def get_crops():
    try:
        loader = CropDatasetLoader()
        df = loader.load()
        dataset_crops = [c.capitalize() for c in df["crop"].unique().tolist()]
        # Merge and deduplicate
        all_crops = sorted(list(set(dataset_crops + MASTER_INDIAN_CROPS)))
        return {"status": "success", "crops": all_crops}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def read_root():
    index_path = Path("index.html")
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

@app.get("/api/similarity")
def get_similarity(lat: float, lon: float, crop: str, start_date: str = None):
    try:
        # Load Data
        loader = CropDatasetLoader(filepath="data/crop_data.csv")
        try:
            df_hist = loader.load()
        except Exception:
            print("[API] Historical data missing. Initializing bootstrap pipeline...")
            run_pipeline(crop=crop, lat=lat, lon=lon, start_date="2026-05-01", end_date="2026-05-30")
            df_hist = loader.load()

        cleaner = DataCleaner()
        df_hist_cleaned = cleaner.clean(df_hist)

        # Setup Prediction Engine
        features = ["temp", "min_temp", "max_temp", "rainfall", "humidity"]
        engine = PredictionEngine(features=features)
        engine.fit(df_hist_cleaned)

        # Determine dynamic duration for the target crop
        target_hist = df_hist_cleaned[df_hist_cleaned["crop"] == crop.lower()]
        if not target_hist.empty and "growth_duration" in target_hist.columns:
            duration = int(target_hist["growth_duration"].iloc[0])
        else:
            duration = 100 # Default if unknown

        # Fetch Forecast for the full duration
        fetcher = WeatherFetcher(lat=lat, lon=lon)
        df_forecast = fetcher.fetch_forecast(days=duration, start_date=start_date)
        df_weather_cleaned = cleaner.clean(df_forecast)

        # Dynamic Crop List from External Data
        available_crops = df_hist_cleaned["crop"].unique().tolist()
        crops_to_compare = list(set([crop.lower()] + available_crops))
        comparisons = engine.analyze_crop_comparisons(df_weather_cleaned, crops_to_compare)

        # Recommendations for alternative dates (+7 days, +14 days)
        recommendations = []
        if start_date:
            current_prob = next((c["success_probability"] for c in comparisons if c["crop"].lower() == crop.lower()), 0)
            from datetime import datetime, timedelta
            base_date = datetime.strptime(start_date, "%Y-%m-%d")
            for offset in [7, 14]:
                alt_date = (base_date + timedelta(days=offset)).strftime("%Y-%m-%d")
                df_alt = fetcher.fetch_forecast(days=duration, start_date=alt_date)
                df_alt_cleaned = cleaner.clean(df_alt)
                alt_comps = engine.analyze_crop_comparisons(df_alt_cleaned, [crop.lower()])
                if alt_comps:
                    alt_prob = alt_comps[0]["success_probability"]
                    if alt_prob > current_prob:
                        recommendations.append({
                            "start_date": alt_date,
                            "success_probability": alt_prob
                        })
            
            # Sort recommendations to put highest probability first
            recommendations = sorted(recommendations, key=lambda x: x["success_probability"], reverse=True)

        # Generate schedule for the active crop using full duration
        schedule = generate_crop_schedule(crop, duration=duration)

        # Prepare mathematical visualization data for the active crop
        active_comp = next((c for c in comparisons if c["crop"].lower() == crop.lower()), None)
        viz_data = {}
        if active_comp and "historical_match_sequence" in active_comp:
            # We already have df_weather_cleaned (forecast)
            # And historical_match_sequence (best successful matches)
            
            # 1. Raw comparison
            viz_data["forecast"] = df_weather_cleaned.to_dict(orient="records")
            viz_data["historical"] = active_comp["historical_match_sequence"]
            
            # 2. Scaled comparison (for normalization visualization)
            viz_data["scaled_forecast"] = engine.scaler.transform(df_weather_cleaned).tolist()
            # To get scaled historical, we use the same scaler
            hist_df = pd.DataFrame(active_comp["historical_match_sequence"])
            # Ensure hist_df has all features
            for feat in engine.features:
                if feat not in hist_df.columns: hist_df[feat] = 0.0
            viz_data["scaled_historical"] = engine.scaler.transform(hist_df[engine.features]).tolist()

        # Fetch dynamic location
        _, _, city, country = get_user_location()

        return {
            "status": "success",
            "crop_comparisons": comparisons,
            "schedule": schedule,
            "crop": crop,
            "location": {"city": city, "country": country},
            "recommendations": recommendations,
            "visualization_data": viz_data
        }
    except Exception as e:
        print(f"Error in /api/similarity: {e}")
        return {"status": "error", "message": str(e)}



@app.post("/api/run-pipeline")
def run_pipe(req: PipelineRequest):
    try:
        print(f"[API] Running pipeline for {req.crop} at {req.lat}, {req.lon}")
        
        # Capture standard output printed during pipeline execution
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            run_pipeline(crop=req.crop, lat=req.lat, lon=req.lon, start_date="2026-05-01", end_date="2026-05-30")
        output_logs = f.getvalue()
        
        # Also print to regular stdout so it shows up in local console
        print(output_logs)
        
        return {"status": "success", "logs": output_logs}
    except Exception as e:
        err_msg = f"Error in /api/run-pipeline: {e}"
        print(err_msg)
        return {"status": "error", "message": err_msg}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
