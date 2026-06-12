import pandas as pd
from pathlib import Path
from typing import List, Dict

class CSVExporter:
    """
    Standardizes and exports execution outputs to required structured CSV format:
    - similarity_results.csv
    - matched_crop_patterns.csv
    - future_prediction_results.csv
    """
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def export_similarity_results(self, crop_comparisons: List[Dict]) -> Path:
        """
        Saves overall similarity comparisons across multiple crops.
        Columns: Crop, Similarity_Score, Success_Probability, Risk_Level
        """
        df = pd.DataFrame(crop_comparisons)
        # Standardize column headers and format
        df.columns = [c.replace("_", " ").title() for c in df.columns]
        
        filepath = self.output_dir / "similarity_results.csv"
        df.to_csv(filepath, index=False)
        print(f"[CSVExporter] Saved similarity results: {filepath.resolve()}")
        return filepath

    def export_matched_patterns(self, knn_matches: List[Dict]) -> Path:
        """
        Saves details of matched crop climate patterns.
        Columns match weather items paired side-by-side with historical equivalents.
        """
        records = []
        for match in knn_matches:
            q_date = match["query_data"].get("date", "Unknown")
            q_crop = match["query_data"].get("crop", "Unknown")
            
            # Extract first neighbor match
            best = match["best_match"]
            records.append({
                "forecast_date": q_date,
                "crop": q_crop,
                "forecast_temp": match["query_data"].get("temp"),
                "forecast_rain": match["query_data"].get("rainfall"),
                "forecast_hum": match["query_data"].get("humidity"),
                "matched_hist_date": best.get("date"),
                "matched_hist_temp": best.get("temp"),
                "matched_hist_rain": best.get("rainfall"),
                "matched_hist_hum": best.get("humidity"),
                "distance": best.get("distance"),
                "similarity_score": best.get("similarity")
            })
            
        df = pd.DataFrame(records)
        df.columns = [c.replace("_", " ").title() for c in df.columns]
        
        filepath = self.output_dir / "matched_crop_patterns.csv"
        df.to_csv(filepath, index=False)
        print(f"[CSVExporter] Saved matched crop patterns: {filepath.resolve()}")
        return filepath

    def export_future_predictions(self, daily_predictions: List[Dict]) -> Path:
        """
        Saves day-by-day 30-day forecast predictions showing crop suitability over time.
        Columns: Date, Crop, Similarity_Score, Success_Probability, Risk_Level
        """
        df = pd.DataFrame(daily_predictions)
        df.columns = [c.replace("_", " ").title() for c in df.columns]
        
        filepath = self.output_dir / "future_prediction_results.csv"
        df.to_csv(filepath, index=False)
        print(f"[CSVExporter] Saved future predictions: {filepath.resolve()}")
        return filepath

if __name__ == "__main__":
    exporter = CSVExporter()
    
    # Test dummy export
    dummy_sims = [
        {"crop": "Rice", "similarity_score": 92.4, "success_probability": 95.0, "risk_level": "low"},
        {"crop": "Millet", "similarity_score": 81.2, "success_probability": 78.0, "risk_level": "medium"},
        {"crop": "Maize", "similarity_score": 74.1, "success_probability": 65.0, "risk_level": "medium"}
    ]
    exporter.export_similarity_results(dummy_sims)
