import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from feature_scaler import FeatureScaler

class KNNSimilarityEngine:
    """
    Fits an unsupervised NearestNeighbors model using Euclidean distance 
    on standardized historical successful crop growth weather patterns.
    Matches current and future weather conditions to find closest historical equivalents.
    """
    def __init__(self, n_neighbors: int = 5, features=None):
        self.n_neighbors = n_neighbors
        self.features = features or ["temp", "min_temp", "max_temp", "rainfall", "humidity"]
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="auto")
        self.scaler = FeatureScaler(features=self.features)
        self.fitted = False
        
        # Keep references to the original historical records for matching lookups
        self.historical_df = None
        self.successful_patterns_df = None

    def fit(self, historical_df: pd.DataFrame):
        """
        Fits the scaling and NearestNeighbors model on historical data.
        Filters for successful records (result == 1) to match optimal crop patterns.
        """
        self.historical_df = historical_df.copy()
        
        # Filter for successful crop scenarios (result == 1 or true)
        # If 'result' is missing (already checked in loader, but safety first), fit on all
        if "result" in self.historical_df.columns:
            self.successful_patterns_df = self.historical_df[self.historical_df["result"] == 1].copy()
            # If for some reason there are no successful entries, fallback to all data
            if len(self.successful_patterns_df) == 0:
                self.successful_patterns_df = self.historical_df.copy()
        else:
            self.successful_patterns_df = self.historical_df.copy()
            
        if len(self.successful_patterns_df) < self.n_neighbors:
            self.n_neighbors = max(1, len(self.successful_patterns_df))
            self.model.set_params(n_neighbors=self.n_neighbors)
            
        # Fit feature scaler on successful patterns
        self.scaler.fit(self.successful_patterns_df)
        scaled_features = self.scaler.transform(self.successful_patterns_df)
        
        # Fit KNN model
        self.model.fit(scaled_features)
        self.fitted = True
        print(f"[KNNSimilarityEngine] Fitted model on {len(self.successful_patterns_df)} successful climate patterns.")
        return self

    def find_matches(self, query_df: pd.DataFrame) -> dict:
        """
        Finds top matches in historical crop records for each row in query_df (current/future weather).
        Returns a dictionary containing distances, matching indices, and detailed historical matches.
        """
        if not self.fitted:
            raise ValueError("KNNSimilarityEngine has not been fitted! Call fit() first.")
            
        # Standardize query features using the fitted scaler
        scaled_queries = self.scaler.transform(query_df)
        
        # Query Nearest Neighbors
        distances, indices = self.model.kneighbors(scaled_queries)
        
        # Calculate max possible distance in the scaled space to convert distance into similarity percentage
        # Distance = sqrt(sum((x - y)^2)). In standard scaled space, standard deviation of each feature is 1.0.
        # Max theoretical distance for 5 standardized features is roughly 5.0 to 10.0 in practical terms.
        # We can normalize distance relative to a scaling factor to get a percentage score.
        max_dist_reference = np.sqrt(len(self.features)) * 3.0 # ~6.7 for 5 features, representing 3 std dev apart
        
        results = []
        for i in range(len(query_df)):
            row_dists = distances[i]
            row_idxs = indices[i]
            
            # Find the actual matching records from successful patterns
            matched_records = self.successful_patterns_df.iloc[row_idxs].copy()
            matched_records["distance"] = row_dists
            
            # Calculate similarity score: smooth exponential decay
            similarities = np.exp(-row_dists / 6.0) * 100.0
            matched_records["similarity"] = np.round(similarities, 2)
            
            results.append({
                "query_index": i,
                "query_data": query_df.iloc[i].to_dict(),
                "matched_patterns": matched_records.to_dict(orient="records"),
                "best_match": matched_records.iloc[0].to_dict()
            })
            
        return results

if __name__ == "__main__":
    # Test Similarity matching
    from crop_dataset_loader import CropDatasetLoader
    from weather_fetcher import WeatherFetcher
    
    loader = CropDatasetLoader()
    try:
        hist_df = loader.load()
        engine = KNNSimilarityEngine(n_neighbors=3)
        engine.fit(hist_df)
        
        fetcher = WeatherFetcher()
        forecast_df = fetcher.fetch_forecast(days=5)
        
        matches = engine.find_matches(forecast_df)
        print(f"\nMatched successfully! Number of query rows: {len(matches)}")
        print("\nBest match details for first day's forecast:")
        print(matches[0]["best_match"])
    except Exception as e:
        print(f"Error during testing: {e}")
