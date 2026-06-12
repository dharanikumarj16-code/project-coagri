import pandas as pd
import numpy as np
import requests
import io
from pathlib import Path

# External Raw Dataset URL (Indian Crop Recommendation Dataset)
DATASET_URL = "https://raw.githubusercontent.com/AbhishekKandoi/Crop-Yield-Prediction-based-on-Indian-Agriculture/main/Crop%20Recommendation%20dataset.csv"

class CropDatasetLoader:
    """
    Loads historical crop data from external resources (GitHub/APIs) 
    and standardizes it for similarity analysis.
    """
    def __init__(self, filepath: str = "data/crop_data.csv", use_external: bool = True):
        self.filepath = Path(filepath)
        self.use_external = use_external

    def load(self) -> pd.DataFrame:
        """
        Loads the dataset from an external URL or fallback local CSV.
        """
        df = None
        if self.use_external:
            try:
                print(f"[CropDatasetLoader] Fetching real-world Indian crop data from: {DATASET_URL}")
                response = requests.get(DATASET_URL, timeout=15)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text))
                print(f"[CropDatasetLoader] Successfully fetched {len(df)} records for {df['label'].nunique()} Indian crops.")
            except Exception as e:
                print(f"[CropDatasetLoader] External fetch failed: {e}. Falling back to local/synthetic data.")

        if df is None:
            if not self.filepath.exists():
                # If neither external nor local exists, create a minimal bootstrap dataset
                print("[CropDatasetLoader] No data source found. Creating bootstrap dataset...")
                df = self._create_bootstrap_df()
            else:
                df = pd.read_csv(self.filepath)
        
        return self._standardize_and_process(df)

    def _standardize_and_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps external columns to internal schema and ensures 'result' column.
        Internal Schema: [date, crop, temp, min_temp, max_temp, rainfall, humidity, result]
        """
        # Mapping for the GitHub dataset schema
        mapping = {
            "label": "crop",
            "temperature": "temp",
            "humidity": "humidity",
            "rainfall": "rainfall"
        }
        df = df.rename(columns=mapping)
        
        # Ensure 'crop' column is lowercase and clean
        if "crop" in df.columns:
            df["crop"] = df["crop"].str.lower().str.strip()
            
        # Synthesize missing columns required by the engines
        if "min_temp" not in df.columns and "temp" in df.columns:
            df["min_temp"] = df["temp"] - np.random.uniform(2, 5, len(df))
        if "max_temp" not in df.columns and "temp" in df.columns:
            df["max_temp"] = df["temp"] + np.random.uniform(2, 5, len(df))
        if "date" not in df.columns:
            df["date"] = pd.date_range(start="2025-01-01", periods=len(df), freq="h").strftime("%Y-%m-%d")
            
        # The external dataset represents 'Optimal' conditions (Success = 1)
        if "result" not in df.columns:
            df["result"] = 1
            
        # To make KNN/Classifier more robust, we synthesize some "Failure" cases
        # by creating records with extreme values for each crop.
        print("[CropDatasetLoader] Synthesizing negative samples for robust KNN classification...")
        failure_samples = []
        for crop in df["crop"].unique():
            crop_df = df[df["crop"] == crop]
            # Average conditions for this crop
            avg_temp = crop_df["temp"].mean()
            avg_hum = crop_df["humidity"].mean()
            avg_rain = crop_df["rainfall"].mean()
            
            # Create a few failure scenarios (Extreme drought, extreme heat, extreme cold)
            failure_samples.append({
                "crop": crop, "temp": avg_temp + 15, "min_temp": avg_temp + 10, "max_temp": avg_temp + 20,
                "rainfall": 0.0, "humidity": avg_hum / 2, "result": 0, "date": "2025-01-01"
            })
            failure_samples.append({
                "crop": crop, "temp": avg_temp - 15, "min_temp": avg_temp - 20, "max_temp": avg_temp - 10,
                "rainfall": avg_rain * 3, "humidity": 98.0, "result": 0, "date": "2025-01-01"
            })
            
        df_failures = pd.DataFrame(failure_samples)
        df = pd.concat([df, df_failures], ignore_index=True)
            
        # Final column selection
        cols = ["date", "crop", "temp", "min_temp", "max_temp", "rainfall", "humidity", "result"]
        return df[cols].copy()

    def _create_bootstrap_df(self) -> pd.DataFrame:
        """Fallback bootstrap data."""
        data = []
        crops = ["rice", "wheat", "maize", "cotton", "soybean", "millet"]
        for crop in crops:
            for _ in range(20):
                data.append({
                    "label": crop,
                    "temperature": np.random.uniform(20, 35),
                    "humidity": np.random.uniform(50, 90),
                    "rainfall": np.random.uniform(50, 200),
                    "ph": 6.5, "N": 50, "P": 50, "K": 50
                })
        return pd.DataFrame(data)

if __name__ == "__main__":
    loader = CropDatasetLoader()
    try:
        df = loader.load()
        print("\nProcessed Historical Crop Data Sample (including result):")
        print(df.head())
        print(f"\nTotal Records: {len(df)}")
        print(f"Crops Found: {df['crop'].unique()}")
        print("\nResult Distribution:")
        print(df["result"].value_counts())
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    loader = CropDatasetLoader()
    try:
        df = loader.load()
        print("Historical Crop Data Sample (including result):")
        print(df.head())
        print("Result Distribution:")
        print(df["result"].value_counts())
    except Exception as e:
        print(f"Error: {e}")
