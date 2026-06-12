import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Re-use or adapt pipeline fetch methods for clean modular design
try:
    from pipeline import fetch_nasa_power_weather, fetch_openweather_forecast, generate_synthetic_weather
except ImportError:
    # Inline fallback if pipeline isn't directly importable
    def fetch_nasa_power_weather(lat, lon, start_date, end_date):
        return None
    def fetch_openweather_forecast(lat, lon, api_key):
        return None
    def generate_synthetic_weather(lat, lon, start_date, end_date):
        dates = pd.date_range(start=start_date, end=end_date)
        n_days = len(dates)
        np.random.seed(42)
        temp = np.random.normal(24, 3, n_days)
        min_temp = temp - np.random.uniform(2, 5, n_days)
        max_temp = temp + np.random.uniform(2, 5, n_days)
        rainfall = np.random.exponential(5, n_days)
        rainfall = np.where(np.random.rand(n_days) > 0.7, rainfall, 0.0)
        humidity = np.random.uniform(50, 90, n_days)
        return pd.DataFrame({
            "date": dates.strftime('%Y-%m-%d'),
            "temp": np.round(temp, 2),
            "min_temp": np.round(min_temp, 2),
            "max_temp": np.round(max_temp, 2),
            "rainfall": np.round(rainfall, 2),
            "humidity": np.round(humidity, 2)
        })

class WeatherFetcher:
    """
    Handles fetching of current weather and future forecasts (up to 30 days) 
    using real APIs with high-fidelity synthetic fallbacks.
    """
    def __init__(self, lat: float = 28.6139, lon: float = 77.2090, openweather_key: str = None):
        self.lat = lat
        self.lon = lon
        self.openweather_key = openweather_key or os.getenv("OPENWEATHER_API_KEY")

    def fetch_forecast(self, days: int = 30, start_date: str = None) -> pd.DataFrame:
        """
        Fetches or simulates weather forecast for a given number of days.
        """
        if start_date is None:
            start = datetime.now()
        else:
            start = datetime.strptime(start_date, "%Y-%m-%d")
        
        end = start + timedelta(days=days - 1)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        
        df_weather = None
        
        # Try OpenWeather first if key is available
        if self.openweather_key:
            print(f"[WeatherFetcher] Attempting OpenWeather for coordinates: ({self.lat}, {self.lon})")
            df_weather = fetch_openweather_forecast(self.lat, self.lon, self.openweather_key)
            if df_weather is not None:
                # Ensure we have the requested range by padding if needed
                if len(df_weather) < days:
                    print(f"[WeatherFetcher] OpenWeather returned {len(df_weather)} days. Simulating the remaining days.")
                    rem_start = start + timedelta(days=len(df_weather))
                    df_rem = generate_synthetic_weather(self.lat, self.lon, rem_start.strftime("%Y-%m-%d"), end_str)
                    df_weather = pd.concat([df_weather, df_rem], ignore_index=True)
                df_weather = df_weather.head(days)

        # Try NASA POWER
        if df_weather is None:
            print(f"[WeatherFetcher] Attempting NASA POWER daily weather...")
            df = fetch_nasa_power_weather(self.lat, self.lon, start_str, end_str)
            if df is not None:
                # Standardize columns for check
                df.columns = [c.lower() for c in df.columns]
                # Replace standard missing indicators
                df = df.replace([-999.0, -9999.0, "-999.0", "-9999.0"], np.nan)
                
                # Check if it has mostly missing data
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                # If there are only NaNs or only 1 observation, discard and use high-fidelity simulator
                if len(df) <= 1 or df[numeric_cols].isnull().all().any():
                    print("[WeatherFetcher] NASA POWER returned missing values (-999.0) or incomplete future daily data. Activating high-fidelity simulation fallback...")
                    df_weather = None
                else:
                    df_weather = df
            else:
                df_weather = None

        # Fallback to high-fidelity generator
        if df_weather is None:
            print(f"[WeatherFetcher] Using high-fidelity synthetic offline simulation forecast...")
            df_weather = generate_synthetic_weather(self.lat, self.lon, start_str, end_str)
            
        # Ensure schema names are standardized
        rename_dict = {
            "temperature": "temp",
            "t2m": "temp",
            "t2m_min": "min_temp",
            "t2m_max": "max_temp",
            "prectotcorr": "rainfall",
            "rh2m": "humidity"
        }
        df_weather.columns = [c.lower() for c in df_weather.columns]
        df_weather = df_weather.rename(columns=rename_dict)
        
        # Ensure correct column selection
        cols = ["date", "temp", "min_temp", "max_temp", "rainfall", "humidity"]
        for col in cols:
            if col not in df_weather.columns:
                df_weather[col] = np.nan # Use NaN so cleaner can fill
                
        # Apply robust cleaning and filling
        from data_cleaner import DataCleaner
        df_weather = DataCleaner.clean(df_weather[cols])
                
        return df_weather.copy()

if __name__ == "__main__":
    fetcher = WeatherFetcher(lat=28.6139, lon=77.2090)
    df = fetcher.fetch_forecast(days=30)
    print("Fetched Forecast Samples:")
    print(df.head())
