import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureScaler:
    """
    Standardizes numerical climate features using sklearn's StandardScaler.
    Ensures that distances calculated during KNN are not biased by feature magnitudes.
    """
    def __init__(self, features=None):
        self.features = features or ["temp", "min_temp", "max_temp", "rainfall", "humidity"]
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Fits the scaler on historical crop features.
        """
        # Ensure all columns are clean and numeric
        X = df[self.features].copy()
        for col in self.features:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
            
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms the given dataframe features using the fitted scaler parameters.
        """
        if not self.fitted:
            raise ValueError("FeatureScaler has not been fitted yet! Call fit() with historical data first.")
            
        X = df[self.features].copy()
        for col in self.features:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
            
        return self.scaler.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fits on the data and returns the transformed array.
        """
        self.fit(df)
        return self.transform(df)

if __name__ == "__main__":
    # Test scaler
    df_dummy = pd.DataFrame({
        "temp": [20, 25, 30],
        "min_temp": [15, 20, 25],
        "max_temp": [25, 30, 35],
        "rainfall": [100, 150, 200],
        "humidity": [80, 85, 90]
    })
    
    scaler = FeatureScaler()
    scaled_x = scaler.fit_transform(df_dummy)
    print("Fitted mean:", scaler.scaler.mean_)
    print("Scaled output:\n", scaled_x)
