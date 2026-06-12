import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from feature_scaler import FeatureScaler

class PredictionEngine:
    """
    Employs KNeighborsClassifier trained on historical weather outcomes 
    to predict crop success probability, similarity scores, and climate risk levels.
    """
    def __init__(self, n_neighbors: int = 5, features=None):
        self.n_neighbors = n_neighbors
        self.features = features or ["temp", "min_temp", "max_temp", "rainfall", "humidity"]
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        self.scaler = FeatureScaler(features=self.features)
        self.fitted = False
        
        # Keep track of crops in dataset to handle crop-specific models
        self.historical_df = None

    def fit(self, historical_df: pd.DataFrame):
        """
        Fits the scaler and training features onto the KNeighborsClassifier.
        """
        self.historical_df = historical_df.copy()
        
        # Extract features and targets
        X = self.historical_df[self.features]
        # Target is 'result' representing crop success (1) vs failure (0)
        y = self.historical_df["result"].astype(int)
        
        # Scaler fit and transform
        self.scaler.fit(self.historical_df)
        X_scaled = self.scaler.transform(self.historical_df)
        
        # Train classifier
        self.classifier.fit(X_scaled, y)
        self.fitted = True
        print(f"[PredictionEngine] Trained KNeighborsClassifier with {len(X)} instances.")
        return self

    def predict_viability(self, query_df: pd.DataFrame, target_crop: str = None) -> list:
        """
        Predicts crop success probability, similarity, and risk levels for current/future conditions.
        """
        if not self.fitted:
            raise ValueError("PredictionEngine must be fitted before predicting viability.")
            
        # Transform query weather features
        query_scaled = self.scaler.transform(query_df)
        
        # Predict probabilities
        # classes_ are [0, 1]. Prob of success is prob at index 1.
        probs = self.classifier.predict_proba(query_scaled)
        
        # If dataset only had one class, predict_proba might have shape (N, 1). Handle gracefully.
        if probs.shape[1] == 2:
            success_probs = probs[:, 1]
        else:
            # Fallback if only 1 label is present
            success_probs = np.ones(len(probs)) if self.classifier.classes_[0] == 1 else np.zeros(len(probs))
            
        # Get distances to closest neighbors to derive similarity score
        distances, _ = self.classifier.kneighbors(query_scaled)
        avg_dists = np.mean(distances, axis=1)
        
        # Real calculation: 0 distance = 100% probability. Drops exponentially as distance grows.
        similarities = np.exp(-avg_dists / 3.0) * 100.0
        
        results = []
        for i in range(len(query_df)):
            raw_prob = float(success_probs[i])
            similarity = float(similarities[i])
            
            # Real calculated probability based on distance
            prob = (similarity / 100.0) * raw_prob
            
            # Risk assessment is based on similarity and success probability thresholds
            if similarity >= 85.0 and prob >= 75.0:
                risk = "low"
            elif similarity >= 68.0:
                risk = "medium"
            else:
                risk = "high"
                
            results.append({
                "date": query_df.iloc[i].get("date", "Unknown"),
                "crop": target_crop or query_df.iloc[i].get("crop", "Unknown"),
                "similarity_score": np.round(similarity, 2),
                "success_probability": np.round(prob * 100, 2),
                "risk_level": risk
            })
            
        return results

    def analyze_crop_comparisons(self, query_conditions_df: pd.DataFrame, crops_list: list) -> list:
        """
        Compares multiple crops against a given set of weather conditions.
        Used for the expected output: Rice -> 92% similarity -> low risk, etc.
        """
        # Group historical dataset by crop to match specifically per crop
        comparisons = []
        
        for crop in crops_list:
            crop_clean = crop.strip().lower()
            crop_hist = self.historical_df[self.historical_df["crop"] == crop_clean]
            
            if len(crop_hist) == 0:
                # If no historical data for this specific crop, skip or fit generic model
                continue
                
            # Create a crop-specific sub-model for precise evaluation
            crop_engine = PredictionEngine(n_neighbors=min(5, len(crop_hist)), features=self.features)
            crop_engine.fit(crop_hist)
            
            # Predict for the average/median weather condition of query_conditions_df
            avg_weather = pd.DataFrame([{
                "temp": query_conditions_df["temp"].mean(),
                "min_temp": query_conditions_df["min_temp"].mean(),
                "max_temp": query_conditions_df["max_temp"].mean(),
                "rainfall": query_conditions_df["rainfall"].mean(),
                "humidity": query_conditions_df["humidity"].mean()
            }])
            
            res = crop_engine.predict_viability(avg_weather, target_crop=crop_clean.capitalize())
            if res:
                comp = res[0]
                
                # Get the absolute best matching historical sequence for visualization
                # We'll use the KNNSimilarityEngine to find the record closest to the query average
                from knn_similarity_engine import KNNSimilarityEngine
                knn = KNNSimilarityEngine(n_neighbors=1, features=self.features)
                knn.fit(crop_hist)
                match_data = knn.find_matches(query_conditions_df)
                
                # Extract the best historical match sequence
                # For visualization, we'll provide the 'best_match' record for each day of the query
                comp["historical_match_sequence"] = [m["best_match"] for m in match_data]
                
                comp["optimal_weather"] = {
                    "temp": float(crop_hist["temp"].median()),
                    "humidity": float(crop_hist["humidity"].median()),
                    "rainfall": float(crop_hist["rainfall"].median())
                }
                comp["forecast_weather"] = {
                    "temp": float(query_conditions_df["temp"].mean()),
                    "humidity": float(query_conditions_df["humidity"].mean()),
                    "rainfall": float(query_conditions_df["rainfall"].mean())
                }
                comparisons.append(comp)
                
        # Sort by similarity score descending
        comparisons = sorted(comparisons, key=lambda x: x["similarity_score"], reverse=True)
        return comparisons

if __name__ == "__main__":
    from crop_dataset_loader import CropDatasetLoader
    from weather_fetcher import WeatherFetcher
    
    loader = CropDatasetLoader()
    try:
        hist_df = loader.load()
        engine = PredictionEngine()
        engine.fit(hist_df)
        
        fetcher = WeatherFetcher()
        forecast_df = fetcher.fetch_forecast(days=5)
        
        predictions = engine.predict_viability(forecast_df, target_crop="wheat")
        print("\nPredictions for first 5 days:")
        for pred in predictions:
            print(f"Date: {pred['date']} | Similarity: {pred['similarity_score']}% | Prob: {pred['success_probability']}% | Risk: {pred['risk_level'].upper()}")
            
        print("\nMulti-crop Comparisons:")
        comparisons = engine.analyze_crop_comparisons(forecast_df, ["rice", "wheat", "maize"])
        for comp in comparisons:
            print(f"* {comp['crop']} -> {comp['similarity_score']:.0f}% similarity -> {comp['risk_level']} risk")
    except Exception as e:
        print(f"Error during testing: {e}")
